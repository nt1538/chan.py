import os
import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Project imports
from sliding_window_chan import SlidingWindowChan
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE, AUTYPE
from DataAPI.csvAPI import CSV_API


# ============================================================
# 0) Raw Kline helper (execution + buy&hold + time->raw_idx map)
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

    day_close_map = raw.groupby("date")[close_col].last().to_dict()

    # time -> raw_idx mapping
    ts_idx = pd.to_datetime(raw["timestamp"]).reset_index(drop=True)
    time_to_raw_idx = {pd.Timestamp(t): int(i) for i, t in enumerate(ts_idx)}

    return raw, next_open_by_idx, next_close_by_idx, day_close_map, time_to_raw_idx


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
# 1) Encoders + strict numeric feature pipeline (HARD FIX)
# ============================================================

@dataclass
class Encoders:
    dir_le: Optional[LabelEncoder] = None
    type_le: Optional[LabelEncoder] = None
    fitted: bool = False


def _fit_or_apply_label_encoder(le: Optional[LabelEncoder], series: pd.Series, fit: bool):
    if le is None:
        le = LabelEncoder()
    s = series.fillna("unknown").astype(str)

    if fit:
        le.fit(s.values)
        enc = le.transform(s.values)
        return le, enc

    known = set(le.classes_)
    mapped = []
    for v in s.values:
        if v in known:
            mapped.append(int(le.transform([v])[0]))
        else:
            mapped.append(-1)
    return le, np.array(mapped, dtype=int)


def _robust_numeric_impute(out: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Fix for sklearn SimpleImputer shape mismatch when some numeric columns are entirely NaN.
    Sklearn can drop/skip those columns -> output array has fewer columns than numeric_cols.
    """
    if not numeric_cols:
        return out

    # Ensure we only operate on existing cols
    numeric_cols = [c for c in numeric_cols if c in out.columns]
    if not numeric_cols:
        return out

    # Replace inf first (so they count as missing)
    out.loc[:, numeric_cols] = out.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)

    all_missing = [c for c in numeric_cols if out[c].isna().all()]
    has_any = [c for c in numeric_cols if c not in all_missing]

    # Fill all-missing with 0 (or any constant you prefer)
    if all_missing:
        out.loc[:, all_missing] = 0.0

    if has_any:
        imputer = SimpleImputer(strategy="mean")
        arr = imputer.fit_transform(out.loc[:, has_any].to_numpy())
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # Assign back with matching shapes
        out.loc[:, has_any] = pd.DataFrame(arr, columns=has_any, index=out.index)

    # Final numeric NaN -> 0 (safety)
    out.loc[:, numeric_cols] = out.loc[:, numeric_cols].fillna(0.0)
    return out


def prepare_ml_dataset(df: pd.DataFrame, enc: Encoders, fit_encoders: bool) -> Tuple[pd.DataFrame, Encoders]:
    """
    Robust:
      - de-duplicates columns (prevents pandas assignment mismatch)
      - replaces inf->nan before impute
      - imputes numeric columns robustly even if some columns are all-missing
      - encodes direction/bsp_type safely
      - selects numeric-only features later for XGBoost
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df, enc

    out = df.copy()

    # Deduplicate columns (critical for pandas assignment stability)
    if not out.columns.is_unique:
        out = out.loc[:, ~out.columns.duplicated()].copy()

    # timestamp normalize (if exists)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # bool -> int early
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)

    # replace inf early (so imputer sees NaN)
    out = out.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

    # encode direction
    if "direction" in out.columns:
        enc.dir_le, out["direction_encoded"] = _fit_or_apply_label_encoder(
            enc.dir_le, out["direction"], fit=fit_encoders
        )

    # encode bsp_type (string like 1p/2s/3a)
    if "bsp_type" in out.columns:
        enc.type_le, out["bsp_type_encoded"] = _fit_or_apply_label_encoder(
            enc.type_le, out["bsp_type"], fit=fit_encoders
        )

    # numeric impute (ONLY numeric cols) - robust even with all-missing cols
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    out = _robust_numeric_impute(out, numeric_cols)

    if fit_encoders:
        enc.fitted = True

    return out, enc


def get_feature_columns_numeric_only(df: pd.DataFrame) -> List[str]:
    """
    STRICT: return only numeric feature columns.
    Prevents raw strings like '1p' from entering XGBoost.
    """
    exclude_contains = ["timestamp", "date", "exit_", "snapshot_", "klu_idx", "raw_idx"]
    exclude_exact = {
        "direction", "bsp_type",
        "maturity_rate", "return_to_mature_pct",
        "chain_id", "chain_dir",
        "last_reverse_close", "matured_close",
        "is_chain_end_matured",
        "accum_phase",
    }

    cols = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(x in c for x in exclude_contains):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)

    return sorted(cols)


# ============================================================
# 2) Execution engine (fixed equity for shorts)
# ============================================================

class LiveExecutionEngine:
    """
    Equity:
      - long:  cash + qty*px
      - short: cash - qty*px (liability)
    """
    def __init__(self, initial_capital: float, position_size: float, fee_pct: float):
        self.cash = float(initial_capital)
        self.position_size = float(position_size)
        self.fee_pct = float(fee_pct)

        self.pos = 0
        self.qty = 0.0
        self.entry_px = None
        self.entry_raw_idx = None

        self.pending_order = None
        self.trades = []

    def _exec_price_by_raw_idx(self, seen_raw_idx: int, next_open_by_idx, next_close_by_idx, mode="next_open"):
        arr = next_open_by_idx if mode == "next_open" else next_close_by_idx
        if not (0 <= seen_raw_idx < len(arr)):
            return None
        px = arr[seen_raw_idx]
        if px is None or (isinstance(px, float) and np.isnan(px)):
            return None
        return float(px)

    def place_order_for_next_bar(self, side: str, seen_raw_idx: int, reason: str, overwrite: bool = False):
        if (self.pending_order is not None) and (not overwrite):
            return
        self.pending_order = {"side": side, "seen_raw_idx": int(seen_raw_idx), "reason": reason}

    def maybe_execute_pending(self, next_open_by_idx, next_close_by_idx, execution_mode="next_open", verbose=False):
        if self.pending_order is None:
            return

        side = self.pending_order["side"]
        ridx = self.pending_order["seen_raw_idx"]
        reason = self.pending_order["reason"]

        px = self._exec_price_by_raw_idx(ridx, next_open_by_idx, next_close_by_idx, mode=execution_mode)
        if px is None:
            return

        # OPEN LONG
        if side == "buy" and self.pos == 0:
            trade_cap = self.cash * self.position_size
            if trade_cap <= 0:
                self.pending_order = None
                return
            fee = trade_cap * self.fee_pct
            net = trade_cap - fee
            qty = net / px
            self.cash -= trade_cap
            self.pos = +1
            self.qty = qty
            self.entry_px = px
            self.entry_raw_idx = ridx
            self.trades.append({"side": "long", "entry_raw_idx": ridx, "entry_price": px, "entry_reason": reason})
            if verbose:
                print(f"[EXEC] OPEN LONG  @ {px:.4f} | raw_idx={ridx} | {reason}")

        # CLOSE LONG
        elif side == "sell" and self.pos == +1 and self.entry_px is not None:
            gross = self.qty * px
            fee = gross * self.fee_pct
            net = gross - fee
            entry_val = self.qty * self.entry_px
            pnl = net - entry_val
            ret = pnl / entry_val * 100.0 if entry_val > 0 else 0.0
            self.cash += net
            self.trades[-1].update({"exit_raw_idx": ridx, "exit_price": px, "pnl": pnl, "return_pct": ret, "exit_reason": reason})
            if verbose:
                print(f"[EXEC] CLOSE LONG @ {px:.4f} | raw_idx={ridx} | ret={ret:.2f}% | {reason}")
            self.pos = 0
            self.qty = 0.0
            self.entry_px = None
            self.entry_raw_idx = None

        # OPEN SHORT (receive proceeds into cash)
        elif side == "sell" and self.pos == 0:
            trade_cap = self.cash * self.position_size
            if trade_cap <= 0:
                self.pending_order = None
                return
            gross_proceeds = trade_cap
            fee = gross_proceeds * self.fee_pct
            net_proceeds = gross_proceeds - fee
            qty = trade_cap / px
            self.cash += net_proceeds
            self.pos = -1
            self.qty = qty
            self.entry_px = px
            self.entry_raw_idx = ridx
            self.trades.append({"side": "short", "entry_raw_idx": ridx, "entry_price": px, "entry_reason": reason})
            if verbose:
                print(f"[EXEC] OPEN SHORT @ {px:.4f} | raw_idx={ridx} | {reason}")

        # CLOSE SHORT (pay to cover)
        elif side == "buy" and self.pos == -1 and self.entry_px is not None:
            gross_cost = self.qty * px
            fee = gross_cost * self.fee_pct
            total_cost = gross_cost + fee
            entry_val = self.qty * self.entry_px
            pnl = entry_val - gross_cost - fee
            ret = pnl / entry_val * 100.0 if entry_val > 0 else 0.0
            self.cash -= total_cost
            self.trades[-1].update({"exit_raw_idx": ridx, "exit_price": px, "pnl": pnl, "return_pct": ret, "exit_reason": reason})
            if verbose:
                print(f"[EXEC] CLOSE SHORT @ {px:.4f} | raw_idx={ridx} | ret={ret:.2f}% | {reason}")
            self.pos = 0
            self.qty = 0.0
            self.entry_px = None
            self.entry_raw_idx = None

        self.pending_order = None

    def mark_to_market(self, px: float):
        px = float(px)
        if self.pos == 0:
            return self.cash
        if self.pos == +1:
            return self.cash + self.qty * px
        else:
            return self.cash - self.qty * px


# ============================================================
# 3) Chain maturity labeler
# ============================================================

def _dir_of_snapshot(s: Dict[str, Any]) -> str:
    d = str(s.get("direction", "unknown")).lower()
    return d if d in ("buy", "sell") else "unknown"


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


@dataclass
class ChainState:
    chain_id: int = 0
    chain_dir: Optional[str] = None
    chain_snap_ids: List[int] = None
    last_reverse_close: Optional[float] = None
    last_reverse_snap_id: Optional[int] = None

    def __post_init__(self):
        if self.chain_snap_ids is None:
            self.chain_snap_ids = []


class BSChainMaturityLabeler:
    def __init__(self):
        self.state = ChainState(chain_id=0)
        self.next_chain_id = 1

    def _finalize_chain(self, snapshots: List[Dict[str, Any]], verbose=False):
        st = self.state
        if st.chain_dir not in ("buy", "sell") or len(st.chain_snap_ids) == 0:
            return
        if st.last_reverse_close is None:
            return

        last_id = st.chain_snap_ids[-1]
        matured_close = _close_of_snapshot(snapshots[last_id])
        if matured_close is None:
            return

        last_rev = float(st.last_reverse_close)
        denom = (matured_close - last_rev) if st.chain_dir == "buy" else (last_rev - matured_close)
        if abs(denom) < 1e-12:
            for sid in st.chain_snap_ids:
                snapshots[sid]["matured_close"] = float(matured_close)
                snapshots[sid]["last_reverse_close"] = float(last_rev)
                snapshots[sid]["chain_id"] = int(st.chain_id)
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

    def on_new_bsp(self, snapshots: List[Dict[str, Any]], new_snap_id: int, verbose=False):
        st = self.state
        new_dir = _dir_of_snapshot(snapshots[new_snap_id])
        new_close = _close_of_snapshot(snapshots[new_snap_id])

        if new_dir not in ("buy", "sell") or new_close is None:
            return

        if st.chain_dir is None:
            st.chain_dir = new_dir
            st.chain_id = self.next_chain_id
            self.next_chain_id += 1
            st.chain_snap_ids = [new_snap_id]
            st.last_reverse_close = None
            st.last_reverse_snap_id = None
            return

        if new_dir == st.chain_dir:
            st.chain_snap_ids.append(new_snap_id)
            return

        # reverse => finalize, then start new
        self._finalize_chain(snapshots, verbose=verbose)
        old_last_id = st.chain_snap_ids[-1] if st.chain_snap_ids else None
        old_matured_close = _close_of_snapshot(snapshots[old_last_id]) if old_last_id is not None else None

        st.chain_dir = new_dir
        st.chain_id = self.next_chain_id
        self.next_chain_id += 1
        st.chain_snap_ids = [new_snap_id]
        st.last_reverse_close = old_matured_close
        st.last_reverse_snap_id = old_last_id


# ============================================================
# 4) Models + threshold optimization
# ============================================================

@dataclass
class Models:
    mr_buy: Optional[xgb.XGBRegressor] = None
    mr_sell: Optional[xgb.XGBRegressor] = None
    ret_buy: Optional[xgb.XGBRegressor] = None
    ret_sell: Optional[xgb.XGBRegressor] = None
    feature_cols: Optional[List[str]] = None
    enc: Optional[Encoders] = None


def _ensure_numeric_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    if not out.columns.is_unique:
        out = out.loc[:, ~out.columns.duplicated()].copy()

    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def train_models_from_dataset(
    df: pd.DataFrame,
    enc: Encoders,
    min_samples_total: int,
    min_samples_each_side: int,
    min_mr_for_ret_train: float,
    verbose: bool = True
) -> Optional[Models]:
    if df is None or len(df) == 0:
        return None

    df2, enc = prepare_ml_dataset(df, enc, fit_encoders=(not enc.fitted))

    labeled = df2.dropna(subset=["maturity_rate", "return_to_mature_pct"]).copy()
    labeled = labeled[labeled["direction"].isin(["buy", "sell"])].copy()

    if len(labeled) < min_samples_total:
        if verbose:
            print(f"[TRAIN-SKIP] labeled too small: {len(labeled)} < {min_samples_total}")
        return None

    feats = get_feature_columns_numeric_only(labeled)
    if not feats:
        if verbose:
            print("[TRAIN-SKIP] no numeric features")
        return None

    buy = labeled[labeled["direction"] == "buy"].copy()
    sell = labeled[labeled["direction"] == "sell"].copy()
    if len(buy) < min_samples_each_side or len(sell) < min_samples_each_side:
        if verbose:
            print(f"[TRAIN-SKIP] buy/sell too small: buy={len(buy)} sell={len(sell)}")
        return None

    buy_ret = buy[buy["maturity_rate"] >= min_mr_for_ret_train].copy()
    sell_ret = sell[sell["maturity_rate"] >= min_mr_for_ret_train].copy()
    if len(buy_ret) < max(20, min_samples_each_side // 2):
        buy_ret = buy
    if len(sell_ret) < max(20, min_samples_each_side // 2):
        sell_ret = sell

    params = dict(
        max_depth=6,
        learning_rate=0.08,
        n_estimators=350,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    out = Models(
        mr_buy=xgb.XGBRegressor(**params),
        mr_sell=xgb.XGBRegressor(**params),
        ret_buy=xgb.XGBRegressor(**params),
        ret_sell=xgb.XGBRegressor(**params),
        feature_cols=feats,
        enc=enc,
    )

    buy = _ensure_numeric_features(buy, feats)
    sell = _ensure_numeric_features(sell, feats)
    buy_ret = _ensure_numeric_features(buy_ret, feats)
    sell_ret = _ensure_numeric_features(sell_ret, feats)

    out.mr_buy.fit(buy[feats].values, buy["maturity_rate"].values)
    out.mr_sell.fit(sell[feats].values, sell["maturity_rate"].values)
    out.ret_buy.fit(buy_ret[feats].values, buy_ret["return_to_mature_pct"].values)
    out.ret_sell.fit(sell_ret[feats].values, sell_ret["return_to_mature_pct"].values)

    if verbose:
        print(f"[TRAIN-OK] feats={len(feats)} labeled={len(labeled)} "
              f"buy={len(buy)} sell={len(sell)} ret_min_mr={min_mr_for_ret_train:.2f}")
    return out


def predict_one(models: Models, row_df: pd.DataFrame) -> Tuple[float, float]:
    d = str(row_df.iloc[0].get("direction", "unknown")).lower()
    X = _ensure_numeric_features(row_df, models.feature_cols)[models.feature_cols].values
    if d == "buy":
        mr = float(models.mr_buy.predict(X)[0])
        ret = float(models.ret_buy.predict(X)[0])
    else:
        mr = float(models.mr_sell.predict(X)[0])
        ret = float(models.ret_sell.predict(X)[0])
    mr = float(np.clip(mr, 0.0, 1.0))
    return mr, ret


@dataclass
class Thresholds:
    mr_th: float = 0.65
    ret_th_buy: float = 0.30
    ret_th_sell: float = 0.30


def optimize_thresholds(
    eval_df: pd.DataFrame,
    mr_grid: List[float],
    ret_grid: List[float],
    min_trades: int,
    trade_penalty: float,
    verbose: bool = True
) -> Thresholds:
    if eval_df is None or len(eval_df) == 0:
        return Thresholds()

    best = Thresholds()
    best_score = -1e18

    b = eval_df[eval_df["direction"] == "buy"].copy()
    s = eval_df[eval_df["direction"] == "sell"].copy()

    for mr_th in mr_grid:
        b0 = b[b["pred_mr"] >= mr_th]
        s0 = s[s["pred_mr"] >= mr_th]
        for bt in ret_grid:
            vb = b0[b0["pred_ret"] >= bt]
            for st in ret_grid:
                vs = s0[s0["pred_ret"] >= st]
                n = len(vb) + len(vs)
                if n < min_trades:
                    continue
                realized = []
                if len(vb):
                    realized.append(vb["return_to_mature_pct"].values)
                if len(vs):
                    realized.append(vs["return_to_mature_pct"].values)
                realized = np.concatenate(realized) if realized else np.array([], dtype=float)
                if realized.size == 0:
                    continue
                mean_ret = float(np.mean(realized))
                score = mean_ret - trade_penalty * float(n)
                if score > best_score:
                    best_score = score
                    best = Thresholds(mr_th=float(mr_th), ret_th_buy=float(bt), ret_th_sell=float(st))

    if verbose:
        print(f"[THR-OPT] mr={best.mr_th:.2f} buy={best.ret_th_buy:.2f} sell={best.ret_th_sell:.2f} score={best_score:.4f}")
    return best


# ============================================================
# 5) Main system
# ============================================================

def run_chain_maturity_system(
    kline_csv_path: str,
    code: str,
    lv: KL_TYPE,
    sim_start: str,
    end_time: str,
    bsp_dataset_csv: Optional[str] = None,

    # Chan warm-start
    chan_window_size: int = 500,
    warmup_bars: int = 500,
    bar_interval_minutes: int = 5,

    # Train schedule
    retrain_every_days: int = 5,
    train_lookback_days: int = 365,
    min_samples_total: int = 300,
    min_samples_each_side: int = 60,
    min_mr_for_ret_train: float = 0.30,

    # Threshold optimization
    do_threshold_opt: bool = True,
    threshold_opt_window_days: int = 2,
    threshold_opt_min_trades: int = 10,
    threshold_trade_penalty: float = 0.02,
    mr_grid: Optional[List[float]] = None,
    ret_grid: Optional[List[float]] = None,

    # Initial thresholds
    init_mr_th: float = 0.65,
    init_ret_th_buy: float = 0.30,
    init_ret_th_sell: float = 0.30,

    # Trading
    initial_capital: float = 100_000,
    position_size: float = 1.0,
    fee_pct: float = 0.0,
    execution_mode: str = "next_open",

    output_dir: str = "./output/chain_maturity_system",
    verbose: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    sim_start_dt = pd.to_datetime(sim_start)
    end_dt = pd.to_datetime(end_time)

    warmup_start_dt = sim_start_dt - pd.Timedelta(minutes=int(warmup_bars) * int(bar_interval_minutes))
    kline_load_start = str(warmup_start_dt)

    raw, next_open_by_idx, next_close_by_idx, day_close_map, time_to_raw_idx = load_kline_index(
        kline_csv_path, kline_load_start, end_time
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
        begin_date=kline_load_start,
        end_date=end_time,
        autype=AUTYPE.QFQ,
    )

    enc = Encoders()
    labeler = BSChainMaturityLabeler()
    snapshots: List[Dict[str, Any]] = []
    full_bsp_df = None

    if bsp_dataset_csv and os.path.exists(bsp_dataset_csv):
        full_bsp_df = pd.read_csv(bsp_dataset_csv)
        if not full_bsp_df.columns.is_unique:
            full_bsp_df = full_bsp_df.loc[:, ~full_bsp_df.columns.duplicated()].copy()

        full_bsp_df["timestamp"] = pd.to_datetime(full_bsp_df["timestamp"], errors="coerce")
        full_bsp_df = full_bsp_df.dropna(subset=["timestamp"]).copy()
        pre_sim_df = full_bsp_df[full_bsp_df["timestamp"] < sim_start_dt].copy()
        pre_sim_df = pre_sim_df.sort_values("timestamp").reset_index(drop=True)
        snapshots = pre_sim_df.to_dict("records")
        for i in range(len(snapshots)):
            labeler.on_new_bsp(snapshots, i, verbose=False)
        if verbose:
            print(f"[LOAD] bsp_dataset_csv={bsp_dataset_csv} total={len(full_bsp_df)} pre_sim={len(pre_sim_df)}")
    else:
        if verbose:
            print("[LOAD] No bsp_dataset_csv provided/found. Model will start later once enough new labeled points exist.")

    engine = LiveExecutionEngine(initial_capital, position_size, fee_pct)

    models: Optional[Models] = None
    thr = Thresholds(init_mr_th, init_ret_th_buy, init_ret_th_sell)

    if mr_grid is None:
        mr_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    if ret_grid is None:
        ret_grid = list(np.round(np.linspace(-0.5, 2.0, 26), 2))

    daily_log = []
    current_day = None
    last_retrain_day = None
    bsp_count_today = 0

    def _asof_train_df(asof_ts: pd.Timestamp) -> pd.DataFrame:
        parts = []
        if full_bsp_df is not None:
            parts.append(full_bsp_df[full_bsp_df["timestamp"] <= asof_ts].copy())
        if snapshots:
            d1 = pd.DataFrame(snapshots)
            if not d1.columns.is_unique:
                d1 = d1.loc[:, ~d1.columns.duplicated()].copy()
            if "timestamp" in d1.columns:
                d1["timestamp"] = pd.to_datetime(d1["timestamp"], errors="coerce")
                d1 = d1.dropna(subset=["timestamp"]).copy()
                d1 = d1[d1["timestamp"] <= asof_ts].copy()
            parts.append(d1)

        if not parts:
            return pd.DataFrame()

        df = pd.concat(parts, ignore_index=True)

        if not df.columns.is_unique:
            df = df.loc[:, ~df.columns.duplicated()].copy()

        subset_cols = [c for c in ["timestamp", "direction", "bsp_type", "klu_close"] if c in df.columns]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep="last")

        if train_lookback_days and train_lookback_days > 0 and "timestamp" in df.columns:
            cutoff = asof_ts - pd.Timedelta(days=int(train_lookback_days))
            df = df[df["timestamp"] >= cutoff].copy()

        return df

    def maybe_train_models(today: pd.Timestamp):
        nonlocal models, enc, last_retrain_day
        if today < sim_start_dt.normalize():
            return
        if last_retrain_day is not None and (today - last_retrain_day).days < retrain_every_days:
            return
        asof_ts = pd.Timestamp(today.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        train_df = _asof_train_df(asof_ts)
        if train_df.empty:
            return
        m = train_models_from_dataset(
            train_df, enc=enc,
            min_samples_total=min_samples_total,
            min_samples_each_side=min_samples_each_side,
            min_mr_for_ret_train=min_mr_for_ret_train,
            verbose=verbose
        )
        if m is not None:
            models = m
            enc = m.enc
            last_retrain_day = today

    def maybe_optimize_thresholds(today: pd.Timestamp):
        nonlocal thr
        if not do_threshold_opt or models is None:
            return
        if today < sim_start_dt.normalize():
            return

        asof_ts = pd.Timestamp(today.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        window_start = asof_ts - pd.Timedelta(days=int(threshold_opt_window_days))

        eval_df = _asof_train_df(asof_ts)
        if eval_df.empty:
            return
        if "timestamp" in eval_df.columns:
            eval_df = eval_df[(eval_df["timestamp"] >= window_start) & (eval_df["timestamp"] <= asof_ts)].copy()

        # Must have labels
        need_cols = [c for c in ["maturity_rate", "return_to_mature_pct", "direction"] if c in eval_df.columns]
        if len(need_cols) < 3:
            return

        eval_df = eval_df.dropna(subset=["maturity_rate", "return_to_mature_pct"]).copy()
        if eval_df.empty:
            return

        eval_df2, _ = prepare_ml_dataset(eval_df, models.enc, fit_encoders=False)
        eval_df2 = _ensure_numeric_features(eval_df2, models.feature_cols)

        preds_mr, preds_ret = [], []
        for _, r in eval_df2.iterrows():
            row = pd.DataFrame([r])
            mr_hat, ret_hat = predict_one(models, row)
            preds_mr.append(mr_hat)
            preds_ret.append(ret_hat)

        eval_df2["pred_mr"] = preds_mr
        eval_df2["pred_ret"] = preds_ret

        thr = optimize_thresholds(
            eval_df2,
            mr_grid=mr_grid,
            ret_grid=ret_grid,
            min_trades=threshold_opt_min_trades,
            trade_penalty=threshold_trade_penalty,
            verbose=verbose
        )

    if verbose:
        print("=" * 110)
        print("CHAIN MATURITY SYSTEM (IMPUTER SHAPE FIXED)")
        print(f"Warmup: {kline_load_start} -> {sim_start} (Chan only)")
        print(f"Sim:    {sim_start} -> {end_time} (trade)")
        print(f"init thr: mr={thr.mr_th:.2f} buy={thr.ret_th_buy:.2f} sell={thr.ret_th_sell:.2f}")
        print("=" * 110)

    for _, klu in enumerate(csv_api.get_kl_data()):
        bar_ts = pd.to_datetime(str(klu.time))
        if bar_ts < pd.to_datetime(kline_load_start) or bar_ts > end_dt:
            continue

        ts_key = pd.Timestamp(bar_ts)
        if ts_key not in time_to_raw_idx:
            continue
        raw_idx = time_to_raw_idx[ts_key]
        klu.kl_type = lv
        klu.set_idx(raw_idx)

        bar_day = bar_ts.date()
        in_sim = (bar_ts >= sim_start_dt)

        if current_day is None:
            current_day = bar_day

        # day rollover
        if bar_day != current_day:
            prev_day = current_day

            engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode,
                                         verbose=bool(verbose and in_sim))

            day_close = day_close_map.get(prev_day)
            equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash

            daily_log.append({
                "date": prev_day,
                "equity": equity,
                "cash": engine.cash,
                "pos": engine.pos,
                "qty": engine.qty,
                "has_model": int(models is not None),
                "bsp_count": int(bsp_count_today),
                "mr_th": thr.mr_th,
                "ret_th_buy": thr.ret_th_buy,
                "ret_th_sell": thr.ret_th_sell,
            })

            if verbose:
                print(f"[EOD] {prev_day} equity={equity:.2f} pos={engine.pos} cash={engine.cash:.2f} "
                      f"| bsp_count={bsp_count_today} model={int(models is not None)}")

            bsp_count_today = 0

            prev_day_ts = pd.to_datetime(prev_day)
            maybe_train_models(prev_day_ts)
            maybe_optimize_thresholds(prev_day_ts)

            current_day = bar_day

        if in_sim:
            engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx,
                                         execution_mode=execution_mode, verbose=bool(verbose))

        # step chan
        _, new_bsp_list = chan.process_new_kline(klu)

        if new_bsp_list and in_sim:
            bsp_count_today += len(new_bsp_list)

            for s in new_bsp_list:
                s = dict(s)
                s["timestamp"] = str(bar_ts)
                s["raw_idx"] = int(raw_idx)
                s["accum_phase"] = 0

                snap_id = len(snapshots)
                snapshots.append(s)
                labeler.on_new_bsp(snapshots, snap_id, verbose=False)

                if models is None:
                    if verbose:
                        print(f"[NEW] BSP (no model yet) dir={s.get('direction')} type={s.get('bsp_type')} ts={s.get('timestamp')}")
                    continue

                row_df = pd.DataFrame([s])
                row_df, _ = prepare_ml_dataset(row_df, models.enc, fit_encoders=False)
                row_df = _ensure_numeric_features(row_df, models.feature_cols)

                d = str(row_df.iloc[0].get("direction", "unknown")).lower()
                if d not in ("buy", "sell"):
                    continue

                mr_hat, ret_hat = predict_one(models, row_df)

                if verbose:
                    print(f"[PRED] {d.upper():4s} raw_idx={raw_idx} type={s.get('bsp_type')} "
                          f"mr={mr_hat:.3f} ret={ret_hat:.3f} | pos={engine.pos} "
                          f"th_mr={thr.mr_th:.2f} th_ret={(thr.ret_th_buy if d=='buy' else thr.ret_th_sell):.2f}")

                if mr_hat < thr.mr_th:
                    continue

                if d == "buy":
                    if ret_hat < thr.ret_th_buy:
                        continue
                    if engine.pos == 0:
                        engine.place_order_for_next_bar("buy", int(raw_idx), reason=f"OPEN_LONG mr={mr_hat:.2f} ret={ret_hat:.2f}")
                    elif engine.pos == -1:
                        engine.place_order_for_next_bar("buy", int(raw_idx), reason=f"CLOSE_SHORT mr={mr_hat:.2f} ret={ret_hat:.2f}")
                else:
                    if ret_hat < thr.ret_th_sell:
                        continue
                    if engine.pos == 0:
                        engine.place_order_for_next_bar("sell", int(raw_idx), reason=f"OPEN_SHORT mr={mr_hat:.2f} ret={ret_hat:.2f}")
                    elif engine.pos == +1:
                        engine.place_order_for_next_bar("sell", int(raw_idx), reason=f"CLOSE_LONG mr={mr_hat:.2f} ret={ret_hat:.2f}")

    # finalize last day
    if current_day is not None:
        engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode, verbose=bool(verbose))
        day_close = day_close_map.get(current_day)
        equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash
        daily_log.append({
            "date": current_day,
            "equity": equity,
            "cash": engine.cash,
            "pos": engine.pos,
            "qty": engine.qty,
            "has_model": int(models is not None),
            "bsp_count": int(bsp_count_today),
            "mr_th": thr.mr_th,
            "ret_th_buy": thr.ret_th_buy,
            "ret_th_sell": thr.ret_th_sell,
        })
        if verbose:
            print(f"[EOD] {current_day} equity={equity:.2f} pos={engine.pos} cash={engine.cash:.2f} "
                  f"| bsp_count={bsp_count_today} model={int(models is not None)}")

    daily_df = pd.DataFrame(daily_log)
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    trades_df = pd.DataFrame(engine.trades)

    # output BSP dataset = old full + new snapshots
    out_parts = []
    if full_bsp_df is not None:
        out_parts.append(full_bsp_df.copy())
    if snapshots:
        out_parts.append(pd.DataFrame(snapshots))
    bsp_out_df = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()

    if not bsp_out_df.empty:
        if not bsp_out_df.columns.is_unique:
            bsp_out_df = bsp_out_df.loc[:, ~bsp_out_df.columns.duplicated()].copy()
        if "timestamp" in bsp_out_df.columns:
            bsp_out_df["timestamp"] = pd.to_datetime(bsp_out_df["timestamp"], errors="coerce")
            bsp_out_df = bsp_out_df.dropna(subset=["timestamp"]).copy()

        subset_cols = [c for c in ["timestamp", "direction", "bsp_type", "klu_close"] if c in bsp_out_df.columns]
        if subset_cols:
            bsp_out_df = bsp_out_df.drop_duplicates(subset=subset_cols, keep="last")

        if "timestamp" in bsp_out_df.columns:
            bsp_out_df = bsp_out_df.sort_values("timestamp").reset_index(drop=True)

    out_bsp = os.path.join(output_dir, "bsp_dataset_out.csv")
    out_daily = os.path.join(output_dir, "daily_equity.csv")
    out_trades = os.path.join(output_dir, "trades.csv")

    bsp_out_df.to_csv(out_bsp, index=False)
    daily_df.to_csv(out_daily, index=False)
    trades_df.to_csv(out_trades, index=False)

    # plot
    if not daily_df.empty:
        plot_df = daily_df.copy()
        plot_df["date_dt"] = pd.to_datetime(plot_df["date"])
        plot_df = plot_df[plot_df["date_dt"] >= sim_start_dt.normalize()].copy()

        strat = pd.Series(plot_df["equity"].values, index=plot_df["date_dt"])
        bh = compute_buy_hold_equity(day_close_map, list(plot_df["date"].values), initial_capital)

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
        print("=" * 110)
        print("DONE")
        print(f"BSP dataset saved: {out_bsp} (rows={len(bsp_out_df)})")
        print(f"Daily saved:       {out_daily}")
        print(f"Trades saved:      {out_trades}")
        print("=" * 110)

    return {"bsp_df": bsp_out_df, "daily_df": daily_df, "trades_df": trades_df}


# ============================================================
# 6) Run example
# ============================================================

if __name__ == "__main__":
    results = run_chain_maturity_system(
        kline_csv_path="DataAPI/data/QQQ_5M.csv",
        code="QQQ",
        lv=KL_TYPE.K_5M,

        sim_start="2021-01-01",
        end_time="2021-12-30",

        bsp_dataset_csv="./output/chain_maturity_system/bsp_dataset.csv",

        chan_window_size=500,
        warmup_bars=500,
        bar_interval_minutes=5,

        retrain_every_days=5,
        train_lookback_days=365,
        min_samples_total=300,
        min_samples_each_side=60,
        min_mr_for_ret_train=0.30,

        do_threshold_opt=True,
        threshold_opt_window_days=2,
        threshold_opt_min_trades=10,
        threshold_trade_penalty=0.02,
        mr_grid=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        ret_grid=list(np.round(np.linspace(-0.5, 2.0, 26), 2)),

        init_mr_th=0.65,
        init_ret_th_buy=0.30,
        init_ret_th_sell=0.30,

        initial_capital=100_000,
        position_size=1.0,
        fee_pct=0.0,
        execution_mode="next_open",

        output_dir="./output/chain_maturity_system_run_fixed",
        verbose=True,
    )

    print(results["daily_df"].head())
    print(results["trades_df"].head())