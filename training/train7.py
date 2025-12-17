import os
import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

# Your project imports
from sliding_window_chan import SlidingWindowChan
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE, AUTYPE
from DataAPI.csvAPI import CSV_API


# ============================================================
# 0) Raw Kline helper (execution + buy&hold)
# ============================================================

def load_kline_index(csv_path: str, start_time: str, end_time: str):
    raw = pd.read_csv(csv_path)

    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    else:
        raw["timestamp"] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")

    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def pick_col(cands):
        for c in cands:
            if c in raw.columns:
                return c
        for c in raw.columns:
            if c.lower() in [x.lower() for x in cands]:
                return c
        return None

    open_col = pick_col(["open", "Open"])
    close_col = pick_col(["close", "Close", "adj_close", "Adj Close", "AdjClose"])
    if open_col is None or close_col is None:
        raise ValueError("Raw CSV must contain open and close columns.")

    mask = (raw["timestamp"] >= pd.to_datetime(start_time)) & (raw["timestamp"] <= pd.to_datetime(end_time))
    raw = raw.loc[mask].copy().reset_index(drop=True)

    raw["date"] = raw["timestamp"].dt.date
    raw["next_open"] = raw[open_col].shift(-1)
    raw["next_close"] = raw[close_col].shift(-1)

    next_open_by_idx = raw["next_open"].to_numpy()
    next_close_by_idx = raw["next_close"].to_numpy()
    cur_close_by_idx = raw[close_col].to_numpy()

    day_close_map = raw.groupby("date")[close_col].last().to_dict()
    all_days = sorted(raw["date"].unique())
    return raw, next_open_by_idx, next_close_by_idx, cur_close_by_idx, day_close_map, all_days


def compute_buy_hold_equity(day_close_map: dict, daily_dates: list, initial_capital: float) -> pd.Series:
    closes, dates = [], []
    for d in daily_dates:
        px = day_close_map.get(d)
        if px is None or pd.isna(px):
            continue
        dates.append(d)
        closes.append(float(px))
    if not closes:
        return pd.Series(dtype=float)
    first = closes[0]
    equity = [initial_capital * (c / first) for c in closes]
    return pd.Series(equity, index=pd.to_datetime(dates))


# ============================================================
# 1) Feature cleaning (safe encoding for bsp_type + direction)
# ============================================================

def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # --- categorical: bsp_type (can be '1','1p','2s','3a'...) ---
    # Never let strings leak into X
    if "bsp_type" in df.columns:
        df["bsp_type"] = df["bsp_type"].fillna("unknown").astype(str)
        # label-encode it
        le_bt = LabelEncoder()
        df["bsp_type_encoded"] = le_bt.fit_transform(df["bsp_type"])

    # direction encode
    if "direction" in df.columns:
        df["direction"] = df["direction"].fillna("unknown").astype(str)
        le = LabelEncoder()
        df["direction_encoded"] = le.fit_transform(df["direction"])

    # numeric
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    if numeric_cols:
        df[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(df[numeric_cols])

    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols_final = df.select_dtypes(include=[np.number]).columns
    df[num_cols_final] = df[num_cols_final].fillna(0.0)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    # keep only model-features; exclude IDs, timestamps, labels
    exclude_contains = ["timestamp", "date", "exit_", "snapshot_", "klu_idx"]
    exclude_exact = {
        "direction",  # keep direction_encoded instead
        "maturity_rate", "return_to_mature_pct",
        "chain_id", "chain_dir",
        "last_reverse_close", "matured_close",
        "is_chain_end_matured",
        "accum_phase",
        # keep bsp_type_encoded instead
        "bsp_type",
    }

    cols = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(x in c for x in exclude_contains):
            continue
        cols.append(c)

    # ensure we don't keep raw direction string
    if "direction" in cols and "direction_encoded" in cols:
        cols.remove("direction")

    return sorted(cols)


# ============================================================
# 2) Long+Short execution engine (fixed short accounting)
# ============================================================

class LiveExecutionEngine:
    """
    pos: +1 long, -1 short, 0 flat
    Cash accounting:
      - Long: spend notional on entry, receive on exit (fees applied).
      - Short: receive proceeds on entry, pay to buy back on exit (fees applied).
    Equity (mark-to-market):
      equity = cash + pos_value
        long:  pos_value = qty * px
        short: pos_value = - qty * px   (liability)
    """
    def __init__(self, initial_capital: float, position_size: float, fee_pct: float):
        self.cash = float(initial_capital)
        self.position_size = float(position_size)
        self.fee_pct = float(fee_pct)

        self.pos = 0
        self.qty = 0.0
        self.entry_px = None
        self.entry_idx = None

        self.pending_order = None
        self.trades = []

    def _exec_price_by_idx(self, seen_idx: int, next_open_by_idx, next_close_by_idx, mode="next_open"):
        arr = next_open_by_idx if mode == "next_open" else next_close_by_idx
        if not (0 <= seen_idx < len(arr)):
            return None
        px = arr[seen_idx]
        if px is None or (isinstance(px, float) and np.isnan(px)):
            return None
        return float(px)

    def place_order_for_next_bar(self, side: str, seen_idx: int, reason: str, overwrite: bool = False):
        if self.pending_order is not None and not overwrite:
            return
        self.pending_order = {"side": side, "seen_idx": int(seen_idx), "reason": reason}

    def maybe_execute_pending(self, next_open_by_idx, next_close_by_idx, execution_mode="next_open", verbose=False):
        if self.pending_order is None:
            return

        side = self.pending_order["side"]
        idx = self.pending_order["seen_idx"]
        reason = self.pending_order["reason"]

        px = self._exec_price_by_idx(idx, next_open_by_idx, next_close_by_idx, mode=execution_mode)
        if px is None:
            return

        fee = self.fee_pct

        # OPEN LONG
        if side == "buy" and self.pos == 0:
            notional = self.cash * self.position_size
            if notional <= 0:
                self.pending_order = None
                return
            # buy pays fee
            spend = notional * (1 + fee)
            if spend > self.cash:
                spend = self.cash
                notional = spend / (1 + fee)

            qty = notional / px
            self.cash -= spend

            self.pos = +1
            self.qty = qty
            self.entry_px = px
            self.entry_idx = idx
            self.trades.append({"side": "long", "entry_idx": idx, "entry_price": px, "entry_reason": reason})
            if verbose:
                print(f"[EXEC] OPEN LONG  buy @ {px:.4f} | idx={idx} | notional={notional:.2f} | {reason}")

        # CLOSE LONG
        elif side == "sell" and self.pos == +1 and self.entry_px is not None:
            # sell receives after fee
            proceeds = self.qty * px * (1 - fee)
            entry_cost = self.qty * self.entry_px * (1 + fee)  # approx for reporting
            pnl = proceeds - (self.qty * self.entry_px)  # pnl ignoring entry fee in this line (fees reflected in cash)
            ret = pnl / (self.qty * self.entry_px) * 100.0 if self.entry_px > 0 else 0.0

            self.cash += proceeds
            self.trades[-1].update({"exit_idx": idx, "exit_price": px, "pnl": pnl, "return_pct": ret, "exit_reason": reason})

            if verbose:
                print(f"[EXEC] CLOSE LONG sell @ {px:.4f} | idx={idx} | ret={ret:.2f}% | {reason}")

            self.pos = 0
            self.qty = 0.0
            self.entry_px = None
            self.entry_idx = None

        # OPEN SHORT
        elif side == "sell" and self.pos == 0:
            notional = self.cash * self.position_size
            if notional <= 0:
                self.pending_order = None
                return
            qty = notional / px
            # short sell receives proceeds after fee
            proceeds = notional * (1 - fee)
            self.cash += proceeds

            self.pos = -1
            self.qty = qty
            self.entry_px = px
            self.entry_idx = idx
            self.trades.append({"side": "short", "entry_idx": idx, "entry_price": px, "entry_reason": reason})
            if verbose:
                print(f"[EXEC] OPEN SHORT sell @ {px:.4f} | idx={idx} | notional={notional:.2f} | {reason}")

        # CLOSE SHORT
        elif side == "buy" and self.pos == -1 and self.entry_px is not None:
            # buy back costs, pay fee
            cost = self.qty * px * (1 + fee)
            self.cash -= cost

            pnl = (self.entry_px - px) * self.qty  # pnl ignoring fees (fees reflected in cash)
            ret = pnl / (self.qty * self.entry_px) * 100.0 if self.entry_px > 0 else 0.0

            self.trades[-1].update({"exit_idx": idx, "exit_price": px, "pnl": pnl, "return_pct": ret, "exit_reason": reason})

            if verbose:
                print(f"[EXEC] CLOSE SHORT buy @ {px:.4f} | idx={idx} | ret={ret:.2f}% | {reason}")

            self.pos = 0
            self.qty = 0.0
            self.entry_px = None
            self.entry_idx = None

        self.pending_order = None

    def mark_to_market(self, px: float):
        px = float(px)
        # equity = cash + position market value
        if self.pos == 0 or self.entry_px is None or self.qty == 0:
            return self.cash
        if self.pos == +1:
            return self.cash + self.qty * px
        else:
            return self.cash - self.qty * px


# ============================================================
# 3) Chain-based maturity labeling (YOUR definition)
# ============================================================

def _dir_of_snapshot(s: Dict[str, Any]) -> str:
    d = str(s.get("direction", "unknown")).lower()
    if d not in ("buy", "sell"):
        return "unknown"
    return d

def _close_of_snapshot(s: Dict[str, Any]) -> Optional[float]:
    v = s.get("klu_close", None)
    try:
        if v is None:
            return None
        v = float(v)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

def _ts_of_snapshot(s: Dict[str, Any]) -> str:
    return str(s.get("timestamp", ""))

@dataclass
class ChainState:
    chain_id: int = 0
    chain_dir: Optional[str] = None     # "buy" or "sell"
    chain_snap_ids: List[int] = None
    last_reverse_close: Optional[float] = None  # close of previous reverse BSP (required)
    last_reverse_snap_id: Optional[int] = None

    def __post_init__(self):
        if self.chain_snap_ids is None:
            self.chain_snap_ids = []

class BSChainMaturityLabeler:
    """
    Same-dir chain. When reverse BSP appears:
      - old chain end is the matured_close
      - maturity_rate for each point i:
          buy:  (close_i - last_rev) / (matured_close - last_rev)
          sell: (last_rev - close_i) / (last_rev - matured_close)
      - matured point has mr=1
      - return_to_mature_pct:
          buy:  (matured_close - close_i) / close_i * 100
          sell: (close_i - matured_close) / close_i * 100
    Then new chain starts, and its last_reverse_close becomes old matured_close.
    """
    def __init__(self):
        self.state = ChainState(chain_id=0)
        self.next_chain_id = 1

    def _finalize_chain(self, snapshots: List[Dict[str, Any]], verbose=False):
        st = self.state
        if st.chain_dir not in ("buy", "sell"):
            return
        if len(st.chain_snap_ids) == 0:
            return
        if st.last_reverse_close is None:
            if verbose:
                print("[CHAIN] finalize skipped: missing last_reverse_close (early history)")
            return

        last_id = st.chain_snap_ids[-1]
        matured_close = _close_of_snapshot(snapshots[last_id])
        if matured_close is None:
            if verbose:
                print("[CHAIN] finalize skipped: matured_close missing")
            return

        last_rev = float(st.last_reverse_close)
        if st.chain_dir == "buy":
            denom = (matured_close - last_rev)
        else:
            denom = (last_rev - matured_close)

        if abs(denom) < 1e-12:
            for sid in st.chain_snap_ids:
                snapshots[sid]["matured_close"] = matured_close
                snapshots[sid]["last_reverse_close"] = last_rev
                snapshots[sid]["chain_id"] = st.chain_id
                snapshots[sid]["chain_dir"] = st.chain_dir
                snapshots[sid]["is_chain_end_matured"] = 1 if sid == last_id else 0
                snapshots[sid]["maturity_rate"] = 1.0 if sid == last_id else 0.0
                snapshots[sid]["return_to_mature_pct"] = 0.0
            return

        for sid in st.chain_snap_ids:
            c = _close_of_snapshot(snapshots[sid])
            if c is None:
                continue

            if st.chain_dir == "buy":
                mr = (c - last_rev) / denom
                ret = (matured_close - c) / c * 100.0
            else:
                mr = (last_rev - c) / denom
                ret = (c - matured_close) / c * 100.0

            mr = float(np.clip(mr, 0.0, 1.0))

            snapshots[sid]["matured_close"] = float(matured_close)
            snapshots[sid]["last_reverse_close"] = float(last_rev)
            snapshots[sid]["chain_id"] = int(st.chain_id)
            snapshots[sid]["chain_dir"] = st.chain_dir
            snapshots[sid]["is_chain_end_matured"] = 1 if sid == last_id else 0
            snapshots[sid]["maturity_rate"] = mr
            snapshots[sid]["return_to_mature_pct"] = float(ret)

        if verbose:
            print(f"[CHAIN] finalized chain_id={st.chain_id} dir={st.chain_dir} "
                  f"len={len(st.chain_snap_ids)} last_rev={last_rev:.4f} matured_close={matured_close:.4f}")

    def on_new_bsp(self, snapshots: List[Dict[str, Any]], new_snap_id: int, verbose=False):
        st = self.state
        new_dir = _dir_of_snapshot(snapshots[new_snap_id])
        new_close = _close_of_snapshot(snapshots[new_snap_id])

        if new_dir not in ("buy", "sell") or new_close is None:
            return

        # first chain ever
        if st.chain_dir is None:
            st.chain_dir = new_dir
            st.chain_id = self.next_chain_id; self.next_chain_id += 1
            st.chain_snap_ids = [new_snap_id]
            st.last_reverse_close = None
            st.last_reverse_snap_id = None
            if verbose:
                print(f"[CHAIN] start initial chain_id={st.chain_id} dir={st.chain_dir} (no last_reverse_close yet)")
            return

        # same direction -> extend
        if new_dir == st.chain_dir:
            st.chain_snap_ids.append(new_snap_id)
            return

        # reverse -> finalize old chain
        self._finalize_chain(snapshots, verbose=verbose)

        old_last_id = st.chain_snap_ids[-1] if st.chain_snap_ids else None
        old_matured_close = _close_of_snapshot(snapshots[old_last_id]) if old_last_id is not None else None

        # start new chain
        st.chain_dir = new_dir
        st.chain_id = self.next_chain_id; self.next_chain_id += 1
        st.chain_snap_ids = [new_snap_id]
        st.last_reverse_close = old_matured_close
        st.last_reverse_snap_id = old_last_id

        if verbose:
            print(f"[CHAIN] reverse -> start chain_id={st.chain_id} dir={st.chain_dir} "
                  f"last_reverse_close={st.last_reverse_close if st.last_reverse_close is not None else None}")


# ============================================================
# 4) Models: predict maturity_rate + return_to_mature_pct
# ============================================================

@dataclass
class Models:
    mr_buy: Optional[xgb.XGBRegressor] = None
    mr_sell: Optional[xgb.XGBRegressor] = None
    ret_buy: Optional[xgb.XGBRegressor] = None
    ret_sell: Optional[xgb.XGBRegressor] = None
    feature_cols: Optional[List[str]] = None

def train_models_from_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_samples: int = 300,
    verbose: bool = True
) -> Optional[Models]:
    """
    Train 4 regressors:
      mr_buy, mr_sell on maturity_rate
      ret_buy, ret_sell on return_to_mature_pct
    """
    if df.empty:
        return None

    labeled = df.dropna(subset=["maturity_rate", "return_to_mature_pct"]).copy()
    labeled = labeled[labeled["direction"].isin(["buy", "sell"])].copy()
    if len(labeled) < min_samples:
        if verbose:
            print(f"[TRAIN-SKIP] labeled too small: {len(labeled)} < min_samples={min_samples}")
        return None

    for c in feature_cols:
        if c not in labeled.columns:
            labeled[c] = 0.0

    params = dict(
        max_depth=6,
        learning_rate=0.08,
        n_estimators=350,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    out = Models(feature_cols=feature_cols)

    buy = labeled[labeled["direction"] == "buy"].copy()
    sell = labeled[labeled["direction"] == "sell"].copy()

    if len(buy) < max(40, min_samples // 10) or len(sell) < max(40, min_samples // 10):
        if verbose:
            print(f"[TRAIN-SKIP] buy/sell balance too small: buy={len(buy)} sell={len(sell)}")
        return None

    out.mr_buy = xgb.XGBRegressor(**params)
    out.mr_sell = xgb.XGBRegressor(**params)
    out.ret_buy = xgb.XGBRegressor(**params)
    out.ret_sell = xgb.XGBRegressor(**params)

    out.mr_buy.fit(buy[feature_cols].values, buy["maturity_rate"].values)
    out.ret_buy.fit(buy[feature_cols].values, buy["return_to_mature_pct"].values)

    out.mr_sell.fit(sell[feature_cols].values, sell["maturity_rate"].values)
    out.ret_sell.fit(sell[feature_cols].values, sell["return_to_mature_pct"].values)

    if verbose:
        print(f"[TRAIN-OK] models trained | labeled={len(labeled)} buy={len(buy)} sell={len(sell)} feats={len(feature_cols)}")
    return out


def predict_one(models: Models, row_df: pd.DataFrame) -> Tuple[float, float]:
    """
    returns (pred_maturity_rate, pred_return_pct)
    """
    d = str(row_df.iloc[0].get("direction", "unknown")).lower()
    X = row_df[models.feature_cols].values
    if d == "buy":
        mr = float(models.mr_buy.predict(X)[0])
        ret = float(models.ret_buy.predict(X)[0])
    else:
        mr = float(models.mr_sell.predict(X)[0])
        ret = float(models.ret_sell.predict(X)[0])

    mr = float(np.clip(mr, 0.0, 1.0))
    return mr, ret


# ============================================================
# 5) Threshold optimization (expected return - penalty * trades)
# ============================================================

def optimize_thresholds_expected_return(
    snapshots: List[Dict[str, Any]],
    models: Models,
    asof_day: pd.Timestamp,
    lookback_days: int = 2,

    # candidate grids
    mr_grid: Optional[np.ndarray] = None,
    ret_grid_buy: Optional[np.ndarray] = None,
    ret_grid_sell: Optional[np.ndarray] = None,

    # eval filters
    min_label_mr_eval: float = 0.0,

    # objective controls
    min_trades: int = 3,
    penalty_per_trade: float = 0.01,

    verbose: bool = True,
) -> Dict[str, float]:
    """
    EOD calibration using only labeled BSPs in the last N days.

    Score:
      score = mean(realized_return_to_mature_pct of selected) - penalty_per_trade * trade_count
    """

    if mr_grid is None:
        mr_grid = np.round(np.arange(0.50, 0.91, 0.05), 2)
    if ret_grid_buy is None:
        ret_grid_buy = np.round(np.arange(0.00, 3.01, 0.10), 2)
    if ret_grid_sell is None:
        ret_grid_sell = np.round(np.arange(0.00, 3.01, 0.10), 2)

    df = pd.DataFrame(snapshots)
    if df.empty or models is None or models.feature_cols is None:
        return {}

    if "timestamp" not in df.columns:
        return {}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # Only consider labeled rows
    df = df.dropna(subset=["maturity_rate", "return_to_mature_pct"]).copy()
    df = df[df["direction"].isin(["buy", "sell"])].copy()
    df = df[df["maturity_rate"] >= float(min_label_mr_eval)].copy()

    if df.empty:
        if verbose:
            print(f"[TH-OPT] {asof_day.date()} no labeled BSPs available for threshold optimization.")
        return {}

    # last N days window
    day_end = pd.to_datetime(asof_day).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    day_start = pd.to_datetime(asof_day).normalize() - pd.Timedelta(days=lookback_days - 1)
    df = df[(df["timestamp"] >= day_start) & (df["timestamp"] <= day_end)].copy()

    if df.empty:
        if verbose:
            print(f"[TH-OPT] {asof_day.date()} no labeled BSPs in last {lookback_days} days.")
        return {}

    df = prepare_ml_dataset(df)

    # ensure features exist
    for c in models.feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[models.feature_cols].values
    dirs = df["direction"].astype(str).str.lower().values

    mr_hat = np.zeros(len(df), dtype=float)
    ret_hat = np.zeros(len(df), dtype=float)

    buy_mask = (dirs == "buy")
    sell_mask = (dirs == "sell")

    if buy_mask.any():
        mr_hat[buy_mask] = models.mr_buy.predict(X[buy_mask])
        ret_hat[buy_mask] = models.ret_buy.predict(X[buy_mask])
    if sell_mask.any():
        mr_hat[sell_mask] = models.mr_sell.predict(X[sell_mask])
        ret_hat[sell_mask] = models.ret_sell.predict(X[sell_mask])

    mr_hat = np.clip(mr_hat, 0.0, 1.0)

    df["mr_hat"] = mr_hat
    df["ret_hat"] = ret_hat

    realized = df["return_to_mature_pct"].astype(float).values

    best_score = -1e18
    best = None

    for mr_th in mr_grid:
        for rtb in ret_grid_buy:
            for rts in ret_grid_sell:
                sel = (
                    ((df["direction"] == "buy") & (df["mr_hat"] >= mr_th) & (df["ret_hat"] >= rtb)) |
                    ((df["direction"] == "sell") & (df["mr_hat"] >= mr_th) & (df["ret_hat"] >= rts))
                )
                trade_cnt = int(sel.sum())
                if trade_cnt < min_trades:
                    continue

                mean_ret = float(realized[sel.values].mean())
                score = mean_ret - float(penalty_per_trade) * trade_cnt

                if score > best_score:
                    best_score = score
                    best = (float(mr_th), float(rtb), float(rts), trade_cnt, mean_ret)

    if best is None:
        if verbose:
            print(f"[TH-OPT] {asof_day.date()} cannot meet min_trades={min_trades} in window={lookback_days}d.")
        return {}

    mr_th, rtb, rts, trade_cnt, mean_ret = best
    if verbose:
        print(f"[TH-OPT] {asof_day.date()} window={lookback_days}d "
              f"best: mr_th={mr_th:.2f} ret_buy={rtb:.2f} ret_sell={rts:.2f} "
              f"| mean_ret={mean_ret:.3f} trades={trade_cnt} "
              f"| score={best_score:.3f} (penalty={penalty_per_trade})")

    return {"mr_th": mr_th, "ret_th_buy": rtb, "ret_th_sell": rts}


# ============================================================
# 6) Main: accumulate then trade, train only starting sim_start
# ============================================================

def run_chain_maturity_system(
    kline_csv_path: str,
    code: str,
    lv: KL_TYPE,

    accum_start: str,
    sim_start: str,
    end_time: str,

    # optional prebuilt bsp dataset
    bsp_dataset_csv: Optional[str] = None,

    chan_window_size: int = 500,

    # training (starts only at sim_start)
    retrain_every_days: int = 5,
    min_train_samples: int = 300,
    train_lookback_days: Optional[int] = 365,     # None = use all history
    min_label_mr_train: float = 0.0,              # only train using labels with mr >= this

    # threshold optimization
    enable_threshold_opt: bool = True,
    thr_lookback_days: int = 2,
    thr_min_trades: int = 3,
    thr_penalty_per_trade: float = 0.01,
    thr_min_label_mr_eval: float = 0.0,

    # initial thresholds (will be updated if enable_threshold_opt)
    mr_th: float = 0.65,
    ret_th_buy: float = 0.30,
    ret_th_sell: float = 0.30,

    # trading
    initial_capital: float = 100_000,
    position_size: float = 1.0,
    fee_pct: float = 0.0,
    execution_mode: str = "next_open",

    output_dir: str = "./output/chain_maturity_system",
    verbose: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load kline index for execution + daily closes
    raw, next_open_by_idx, next_close_by_idx, cur_close_by_idx, day_close_map, all_days = load_kline_index(
        kline_csv_path, accum_start, end_time
    )

    # Chan config
    config = CChanConfig({
        "cal_demark": True,
        "cal_kdj": True,
        "cal_dmi": True,
        "cal_rsi": True,
        "cal_rsl": True,
        "cal_demand_index": True,
        "cal_adline": True,
        "cal_bb_vals": True,
        "cal_kc_vals": True,
        "cal_starc_vals": True,
        "bi_strict": True,
        "trigger_step": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": True,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": "1,2,3a,1p,2s,3b",
        "print_warning": False,
        "zs_algo": "normal",
    })

    chan = SlidingWindowChan(
        code=code,
        begin_time=None,
        end_time=None,
        data_src=None,
        lv_list=[lv],
        config=config,
        autype=AUTYPE.QFQ,
        max_klines=chan_window_size,
    )

    csv_api = CSV_API(
        code=code,
        k_type=lv,
        begin_date=accum_start,
        end_date=end_time,
        autype=AUTYPE.QFQ,
    )

    # snapshots list + chain labeler
    snapshots: List[Dict[str, Any]] = []
    labeler = BSChainMaturityLabeler()

    # models and features
    models: Optional[Models] = None
    feature_cols: Optional[List[str]] = None

    # if prebuilt dataset provided, load and keep it
    if bsp_dataset_csv and os.path.exists(bsp_dataset_csv):
        if verbose:
            print(f"[LOAD] prebuilt bsp_dataset_csv={bsp_dataset_csv}")
        df0 = pd.read_csv(bsp_dataset_csv)
        df0 = prepare_ml_dataset(df0)
        feature_cols = get_feature_columns(df0)
        # Do NOT train here (training must start at sim_start). Just load snapshots.
        snapshots = df0.to_dict("records")
        if verbose:
            labeled_cnt = int(pd.DataFrame(snapshots).dropna(subset=["maturity_rate","return_to_mature_pct"]).shape[0])
            print(f"[LOAD] loaded snapshots={len(snapshots)} labeled={labeled_cnt} (training will start at sim_start)")
    else:
        if verbose:
            print("[LOAD] no bsp_dataset_csv (or not found) -> will build dataset from accum_start")

    engine = LiveExecutionEngine(initial_capital, position_size, fee_pct)

    sim_start_dt = pd.to_datetime(sim_start)
    accum_start_dt = pd.to_datetime(accum_start)
    end_dt = pd.to_datetime(end_time)

    # daily logs
    daily_log = []
    current_day = None
    last_retrain_day = None

    # bsp counters by day
    bsp_count_today = 0
    bsp_count_today_sim = 0

    def get_train_df(asof_day: pd.Timestamp) -> pd.DataFrame:
        df = pd.DataFrame(snapshots)
        if df.empty:
            return df
        if "timestamp" not in df.columns:
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()

        # Only use data available up to EOD(asof_day)
        day_end = pd.to_datetime(asof_day).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df["timestamp"] <= day_end].copy()

        # optional rolling lookback
        if train_lookback_days is not None:
            day_start = pd.to_datetime(asof_day).normalize() - pd.Timedelta(days=int(train_lookback_days) - 1)
            df = df[df["timestamp"] >= day_start].copy()

        # must be labeled for training
        df = df.dropna(subset=["maturity_rate","return_to_mature_pct"]).copy()
        df = df[df["direction"].isin(["buy","sell"])].copy()
        df = df[df["maturity_rate"] >= float(min_label_mr_train)].copy()

        df = prepare_ml_dataset(df)
        return df

    def maybe_retrain(asof_day: pd.Timestamp):
        nonlocal models, feature_cols, last_retrain_day

        if asof_day is None:
            return
        # training starts only at sim_start
        if asof_day < sim_start_dt.normalize():
            return

        if last_retrain_day is not None:
            if (asof_day - last_retrain_day).days < retrain_every_days:
                return

        df = get_train_df(asof_day)
        if df.empty:
            if verbose:
                print(f"[TRAIN-SKIP] {asof_day.date()} train_df empty (labels not enough yet?)")
            return

        feature_cols = get_feature_columns(df)
        if not feature_cols:
            if verbose:
                print(f"[TRAIN-SKIP] {asof_day.date()} feature_cols empty")
            return

        m = train_models_from_dataset(df, feature_cols, min_samples=min_train_samples, verbose=verbose)
        if m is not None:
            models = m
            last_retrain_day = asof_day.normalize()

    if verbose:
        print("=" * 110)
        print("CHAIN MATURITY SYSTEM (with EOD threshold optimization)")
        print(f"accum: {accum_start} -> {sim_start}  (accumulate BSP only, NO training, NO trading)")
        print(f"sim:   {sim_start} -> {end_time}  (trade live + training)")
        print(f"train_lookback_days={train_lookback_days} | min_label_mr_train={min_label_mr_train}")
        print(f"retrain_every_days={retrain_every_days} | min_train_samples={min_train_samples}")
        print(f"threshold_opt: {enable_threshold_opt} | thr_lookback_days={thr_lookback_days} "
              f"| penalty={thr_penalty_per_trade} | min_trades={thr_min_trades}")
        print(f"initial thresholds: mr_th={mr_th:.2f} ret_buy={ret_th_buy:.2f} ret_sell={ret_th_sell:.2f}")
        print("=" * 110)

    # Iterate kline stream
    for klu_idx, klu in enumerate(csv_api.get_kl_data()):
        klu.kl_type = lv
        klu.set_idx(klu_idx)

        bar_ts = pd.to_datetime(str(klu.time))
        if bar_ts < accum_start_dt or bar_ts > end_dt:
            continue

        bar_day = bar_ts.date()
        in_sim = (bar_ts >= sim_start_dt)

        if current_day is None:
            current_day = bar_day

        # day rollover
        if bar_day != current_day:
            prev_day = current_day
            prev_day_ts = pd.to_datetime(prev_day)

            # execute pending (created on last bar)
            if in_sim:
                engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode, verbose=verbose)

            # mark-to-market
            day_close = day_close_map.get(prev_day)
            equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash

            daily_log.append({
                "date": prev_day,
                "equity": equity,
                "cash": engine.cash,
                "pos": engine.pos,
                "qty": engine.qty,
                "has_model": int(models is not None),
                "snapshots": len(snapshots),
                "bsp_count": int(bsp_count_today),
                "bsp_count_sim": int(bsp_count_today_sim),
                "mr_th": mr_th,
                "ret_th_buy": ret_th_buy,
                "ret_th_sell": ret_th_sell,
            })

            if verbose:
                tag = "SIM" if (pd.to_datetime(prev_day) >= sim_start_dt.normalize()) else "ACCUM"
                print(f"[EOD-{tag}] {prev_day} equity={equity:.2f} cash={engine.cash:.2f} pos={engine.pos} qty={engine.qty:.6f} "
                      f"| bsp_count={bsp_count_today} (sim={bsp_count_today_sim}) "
                      f"| has_model={int(models is not None)} "
                      f"| thr(mr={mr_th:.2f}, buy={ret_th_buy:.2f}, sell={ret_th_sell:.2f})")

            # training + threshold optimization only after sim_start
            maybe_retrain(pd.to_datetime(prev_day))

            if enable_threshold_opt and models is not None and pd.to_datetime(prev_day) >= sim_start_dt.normalize():
                new_th = optimize_thresholds_expected_return(
                    snapshots=snapshots,
                    models=models,
                    asof_day=pd.to_datetime(prev_day),
                    lookback_days=int(thr_lookback_days),
                    min_trades=int(thr_min_trades),
                    penalty_per_trade=float(thr_penalty_per_trade),
                    min_label_mr_eval=float(thr_min_label_mr_eval),
                    verbose=verbose
                )
                if new_th:
                    mr_th = new_th["mr_th"]
                    ret_th_buy = new_th["ret_th_buy"]
                    ret_th_sell = new_th["ret_th_sell"]

            # reset daily counters
            bsp_count_today = 0
            bsp_count_today_sim = 0
            current_day = bar_day

        # execute pending each bar during sim
        if in_sim:
            engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode, verbose=False)

        # step chan
        window_chan, new_bsp_list = chan.process_new_kline(klu)

        # handle new BSPs
        if new_bsp_list:
            bsp_count_today += len(new_bsp_list)
            if in_sim:
                bsp_count_today_sim += len(new_bsp_list)

            for s in new_bsp_list:
                s = dict(s)
                s.setdefault("timestamp", str(bar_ts))
                s.setdefault("klu_idx", int(klu_idx))
                s.setdefault("accum_phase", 0 if in_sim else 1)  # 1=accum only, 0=sim/trade

                snap_id = len(snapshots)
                snapshots.append(s)

                # update chain labels
                labeler.on_new_bsp(snapshots, snap_id, verbose=False)

                # during sim: trade immediately on new BSP
                if in_sim:
                    if models is None or models.feature_cols is None:
                        if verbose:
                            print(f"[NEW] BSP (no model yet) dir={s.get('direction')} type={s.get('bsp_type')} ts={_ts_of_snapshot(s)}")
                        continue

                    row_df = pd.DataFrame([s])
                    row_df = prepare_ml_dataset(row_df)

                    # fill missing features
                    for c in models.feature_cols:
                        if c not in row_df.columns:
                            row_df[c] = 0.0

                    d = str(row_df.iloc[0].get("direction", "unknown")).lower()
                    if d not in ("buy", "sell"):
                        continue

                    mr_hat, ret_hat = predict_one(models, row_df)

                    if verbose:
                        print(f"[PRED] {d.upper():4s} idx={klu_idx} type={s.get('bsp_type')} "
                              f"mr={mr_hat:.3f} ret={ret_hat:.3f} | pos={engine.pos} "
                              f"th_mr={mr_th:.2f} th_ret={(ret_th_buy if d=='buy' else ret_th_sell):.2f}")

                    # gating thresholds
                    if mr_hat < mr_th:
                        continue
                    if d == "buy" and ret_hat < ret_th_buy:
                        continue
                    if d == "sell" and ret_hat < ret_th_sell:
                        continue

                    # action: open/close based on direction & current pos
                    if d == "buy":
                        if engine.pos == 0:
                            engine.place_order_for_next_bar("buy", int(klu_idx), reason=f"OPEN_LONG mr={mr_hat:.2f} ret={ret_hat:.2f}")
                        elif engine.pos == -1:
                            engine.place_order_for_next_bar("buy", int(klu_idx), reason=f"CLOSE_SHORT mr={mr_hat:.2f} ret={ret_hat:.2f}")
                    else:
                        if engine.pos == 0:
                            engine.place_order_for_next_bar("sell", int(klu_idx), reason=f"OPEN_SHORT mr={mr_hat:.2f} ret={ret_hat:.2f}")
                        elif engine.pos == +1:
                            engine.place_order_for_next_bar("sell", int(klu_idx), reason=f"CLOSE_LONG mr={mr_hat:.2f} ret={ret_hat:.2f}")

    # finalize last day
    if current_day is not None:
        # execute pending if in sim end
        engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode, verbose=False)

        day_close = day_close_map.get(current_day)
        equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash

        daily_log.append({
            "date": current_day,
            "equity": equity,
            "cash": engine.cash,
            "pos": engine.pos,
            "qty": engine.qty,
            "has_model": int(models is not None),
            "snapshots": len(snapshots),
            "bsp_count": int(bsp_count_today),
            "bsp_count_sim": int(bsp_count_today_sim),
            "mr_th": mr_th,
            "ret_th_buy": ret_th_buy,
            "ret_th_sell": ret_th_sell,
        })

        if verbose:
            tag = "SIM" if (pd.to_datetime(current_day) >= sim_start_dt.normalize()) else "ACCUM"
            print(f"[EOD-{tag}] {current_day} equity={equity:.2f} cash={engine.cash:.2f} pos={engine.pos} qty={engine.qty:.6f} "
                  f"| bsp_count={bsp_count_today} (sim={bsp_count_today_sim}) "
                  f"| has_model={int(models is not None)}")

    # save outputs
    daily_df = pd.DataFrame(daily_log)
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

    trades_df = pd.DataFrame(engine.trades)

    bsp_df = pd.DataFrame(snapshots)
    bsp_df = prepare_ml_dataset(bsp_df)

    out_bsp = os.path.join(output_dir, "bsp_dataset.csv")
    out_daily = os.path.join(output_dir, "daily_equity.csv")
    out_trades = os.path.join(output_dir, "trades.csv")

    bsp_df.to_csv(out_bsp, index=False)
    daily_df.to_csv(out_daily, index=False)
    trades_df.to_csv(out_trades, index=False)

    # plot strategy vs buy&hold (from sim_start onward)
    if not daily_df.empty:
        daily_df_plot = daily_df.copy()
        daily_df_plot["date_dt"] = pd.to_datetime(daily_df_plot["date"])

        start_dt = pd.to_datetime(sim_start).date()
        daily_df_plot = daily_df_plot[daily_df_plot["date"] >= start_dt].copy()

        strat = pd.Series(daily_df_plot["equity"].values, index=daily_df_plot["date_dt"])
        bh = compute_buy_hold_equity(day_close_map, list(daily_df_plot["date"].values), initial_capital)

        plt.figure(figsize=(12, 6))
        plt.plot(strat.index, strat.values, label="Strategy Equity", linewidth=2)
        if len(bh) > 0:
            plt.plot(bh.index, bh.values, label="Buy & Hold Equity", linewidth=2)
        plt.title("Chain Maturity Strategy vs Buy&Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "strategy_vs_buyhold.png")
        plt.savefig(fig_path, dpi=150)
        if verbose:
            print(f"[PLOT] Saved: {fig_path}")
        plt.show()

    if verbose:
        labeled_cnt = int(pd.DataFrame(snapshots).dropna(subset=["maturity_rate", "return_to_mature_pct"]).shape[0]) if snapshots else 0
        print("=" * 110)
        print("DONE")
        print(f"BSP dataset saved: {out_bsp}  (rows={len(bsp_df)} labeled={labeled_cnt})")
        print(f"Daily saved:       {out_daily}")
        print(f"Trades saved:      {out_trades}")
        print("=" * 110)

    return {
        "bsp_df": bsp_df,
        "daily_df": daily_df,
        "trades_df": trades_df,
    }


# ============================================================
# 7) Run example
# ============================================================

if __name__ == "__main__":
    # Use your previously generated bsp_dataset.csv this time:
    # bsp_dataset_csv="./output/chain_maturity_system/bsp_dataset.csv"
    results = run_chain_maturity_system(
        kline_csv_path="DataAPI/data/QQQ_5M.csv",
        code="QQQ",
        lv=KL_TYPE.K_5M,

        accum_start="2018-01-01",
        sim_start="2021-01-01",
        end_time="2023-12-30",

        bsp_dataset_csv="./output/chain_maturity_system/bsp_dataset.csv",

        chan_window_size=500,

        # training (starts only at sim_start)
        retrain_every_days=5,
        min_train_samples=300,
        train_lookback_days=365,      # <-- range parameter for training
        min_label_mr_train=0.20,      # <-- only train on points with label mr>=0.20

        # threshold optimization (EOD rolling window)
        enable_threshold_opt=True,
        thr_lookback_days=2,          # <-- window parameter
        thr_min_trades=3,
        thr_penalty_per_trade=0.01,   # <-- expected return minus penalty*count
        thr_min_label_mr_eval=0.0,

        # initial thresholds (will be updated after threshold opt starts working)
        mr_th=0.65,
        ret_th_buy=0.30,
        ret_th_sell=0.30,

        initial_capital=100_000,
        position_size=1.0,
        fee_pct=0.0,
        execution_mode="next_open",

        output_dir="./output/chain_maturity_system_run2",
        verbose=True,
    )

    print(results["daily_df"].head(5))
    print(results["trades_df"].head(10))
    print(results["bsp_df"][["timestamp","direction","bsp_type","maturity_rate","return_to_mature_pct",
                             "chain_id","last_reverse_close","matured_close"]].head(10))
