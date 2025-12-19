# ============================================================
# Chain maturity + best-lookahead return + ONLINE threshold selection (UNLABELED)
# - Thresholds are selected using the last N days of BUY BSPs that do NOT yet have best_return_pct
# - Uses model predictions only (no future realized return leakage)
# ============================================================

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Project imports (must exist in your repo)
from sliding_window_chan import SlidingWindowChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_FIELD, KL_TYPE, AUTYPE
from KLine.KLine_Unit import CKLine_Unit
from Common.CTime import CTime


# ============================================================
# 0) Helpers: CTime + CKLine_Unit builder
# ============================================================

def to_ctime(ts) -> CTime:
    """Convert pandas Timestamp/str/datetime to your repo's CTime robustly."""
    if isinstance(ts, CTime):
        return ts

    dt = pd.to_datetime(ts).to_pydatetime()
    # Most common in chan repos: CTime(y,m,d,h,mi,s, auto=?)
    try:
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, auto=False)
    except Exception:
        pass
    try:
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    except Exception:
        pass

    # Some repos accept (datetime, auto=?)
    try:
        return CTime(dt, auto=False)
    except Exception:
        pass
    try:
        return CTime(dt)
    except Exception:
        pass

    # String fallback
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    try:
        return CTime(s, auto=False)
    except Exception:
        return CTime(s)


def build_klu(ts, o, h, l, c, v=0.0) -> CKLine_Unit:
    ct = to_ctime(ts)

    kl_dict = {
        DATA_FIELD.FIELD_TIME: ct,               # usually "time_key"
        DATA_FIELD.FIELD_OPEN: float(o),
        DATA_FIELD.FIELD_HIGH: float(h),
        DATA_FIELD.FIELD_LOW:  float(l),
        DATA_FIELD.FIELD_CLOSE: float(c),
        DATA_FIELD.FIELD_VOLUME: float(v),

        # redundancy keys (harmless)
        "time": ct,
        "timestamp": ct,
        "datetime": ct,
        "dt": ct,
        "volume": float(v),
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
    }

    klu = CKLine_Unit(kl_dict)

    # enforce time as CTime
    try:
        klu.time = ct
    except Exception:
        pass

    return klu


# ============================================================
# 1) Raw Kline loader (execution + buy&hold + label arrays)
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

    open_col  = pick_col(["open", "Open"])
    high_col  = pick_col(["high", "High"])
    low_col   = pick_col(["low", "Low"])
    close_col = pick_col(["close", "Close", "adj_close", "Adj Close", "AdjClose"])

    if open_col is None or close_col is None:
        raise ValueError("Raw CSV must contain open and close columns.")
    if high_col is None: high_col = close_col
    if low_col  is None: low_col  = close_col

    mask = (raw["timestamp"] >= pd.to_datetime(start_time)) & (raw["timestamp"] <= pd.to_datetime(end_time))
    raw = raw.loc[mask].copy().reset_index(drop=True)

    raw["date"] = raw["timestamp"].dt.date
    raw["next_open"] = raw[open_col].shift(-1)
    raw["next_close"] = raw[close_col].shift(-1)

    next_open_by_idx  = raw["next_open"].to_numpy()
    next_close_by_idx = raw["next_close"].to_numpy()
    cur_close_by_idx  = raw[close_col].to_numpy()

    day_close_map = raw.groupby("date")[close_col].last().to_dict()
    all_days = sorted(raw["date"].unique())

    highs  = raw[high_col].to_numpy(dtype=float)
    lows   = raw[low_col].to_numpy(dtype=float)
    closes = raw[close_col].to_numpy(dtype=float)

    return raw, next_open_by_idx, next_close_by_idx, cur_close_by_idx, day_close_map, all_days, highs, lows, closes, open_col, high_col, low_col, close_col


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
# 2) Feature cleaning (safe: DO NOT impute labels)
# ============================================================

LABEL_COLS = {"best_return_pct", "maturity_rate"}

def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    if "direction" in df.columns:
        df["direction"] = df["direction"].fillna("unknown").astype(str)
        le = LabelEncoder()
        df["direction_encoded"] = le.fit_transform(df["direction"].astype(str))

    if "bsp_type" in df.columns:
        df["bsp_type"] = df["bsp_type"].fillna("unknown").astype(str)
        le2 = LabelEncoder()
        df["bsp_type_encoded"] = le2.fit_transform(df["bsp_type"].astype(str))

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in LABEL_COLS]  # exclude labels

    if numeric_cols:
        imputer = SimpleImputer(strategy="mean")
        imputed = imputer.fit_transform(df[numeric_cols])
        imputed_df = pd.DataFrame(imputed, columns=numeric_cols, index=df.index).fillna(0.0)
        df[numeric_cols] = imputed_df[numeric_cols]

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude_exact = set(LABEL_COLS) | {
        "timestamp", "code", "direction", "bsp_type",
        "has_best_exit", "best_exit_type", "best_exit_klu_idx", "best_exit_price",
        "klu_idx",
    }
    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cols = [c for c in num_cols if c not in exclude_exact]
    return sorted(cols)


def to_float_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=np.float32, copy=False)


# ============================================================
# 3) Execution engine (long-only)
# ============================================================

class LongOnlyExecutionEngine:
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

    def place_order_for_next_bar(self, side: str, seen_idx: int, reason: str, overwrite: bool = True):
        if self.pending_order is not None and not overwrite:
            return
        self.pending_order = {"side": side, "seen_idx": int(seen_idx), "reason": reason}

    def maybe_execute_pending(self, next_open_by_idx, next_close_by_idx, execution_mode="next_open"):
        if self.pending_order is None:
            return

        side = self.pending_order["side"]
        idx = self.pending_order["seen_idx"]
        reason = self.pending_order["reason"]

        px = self._exec_price_by_idx(idx, next_open_by_idx, next_close_by_idx, mode=execution_mode)
        if px is None:
            return

        fee = self.fee_pct

        if side == "buy" and self.pos == 0:
            notional = self.cash * self.position_size
            if notional <= 0:
                self.pending_order = None
                return

            spend = notional * (1 + fee)
            if spend > self.cash:
                spend = self.cash
                notional = spend / (1 + fee)

            qty = notional / px
            self.cash -= spend
            self.pos = 1
            self.qty = qty
            self.entry_px = px
            self.entry_idx = idx

            self.trades.append({
                "side": "buy",
                "seen_idx": idx,
                "exec_px": px,
                "qty": qty,
                "fee": notional * fee,
                "reason": reason
            })

        elif side == "sell" and self.pos == 1:
            notional = self.qty * px
            recv = notional * (1 - fee)
            self.cash += recv

            pnl = (px - self.entry_px) * self.qty - (notional * fee)

            self.trades.append({
                "side": "sell",
                "seen_idx": idx,
                "exec_px": px,
                "qty": self.qty,
                "fee": notional * fee,
                "reason": reason,
                "pnl": pnl,
                "entry_px": self.entry_px,
                "entry_idx": self.entry_idx
            })

            self.pos = 0
            self.qty = 0.0
            self.entry_px = None
            self.entry_idx = None

        self.pending_order = None

    def mark_to_market(self, last_close: float) -> float:
        if self.pos == 0:
            return float(self.cash)
        return float(self.cash + self.qty * float(last_close))


# ============================================================
# 4) Labels: best-lookahead return
# ============================================================

def _bars_from_days(days: float, bar_interval_minutes: int) -> int:
    minutes = float(days) * 24.0 * 60.0
    return max(1, int(round(minutes / float(bar_interval_minutes))))

def label_bestlookahead_for_ready_points(
    bsp_rows: List[Dict[str, Any]],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookahead_days: float,
    bar_interval_minutes: int,
    current_bar_idx: int,
):
    if not bsp_rows:
        return
    lookahead_bars = _bars_from_days(lookahead_days, bar_interval_minutes)
    n = len(closes)

    for r in bsp_rows:
        if not np.isnan(r.get("best_return_pct", np.nan)):
            continue
        i = r.get("klu_idx", None)
        if i is None:
            continue
        i = int(i)
        end_i = i + lookahead_bars
        if end_i >= n:
            continue
        if end_i > current_bar_idx:
            continue

        d = str(r.get("direction", "buy")).lower()
        c0 = float(closes[i])

        if d == "buy":
            mx = float(np.nanmax(highs[i+1:end_i+1]))
            r["best_return_pct"] = (mx - c0) / c0 * 100.0
        else:
            mn = float(np.nanmin(lows[i+1:end_i+1]))
            r["best_return_pct"] = (c0 - mn) / c0 * 100.0


# ============================================================
# 5) Maturity chain manager (your reverse-trigger definition)
# ============================================================

class MaturityChainManager:
    """
    Reverse BSP appears => finalize previous same-direction chain.
    Uses last reverse close as anchor, mature close = last in chain.

    BUY chain: mr = (close_i - rev_close) / (mature_close - rev_close)
    SELL chain: mr = (rev_close - close_i) / (rev_close - mature_close)
    """
    def __init__(self):
        self.last_bsp: Optional[Dict[str, Any]] = None
        self.last_reverse_close: Optional[float] = None
        self.active_dir: Optional[str] = None
        self.active_indices: List[int] = []

    def _dir(self, r: Dict[str, Any]) -> str:
        d = str(r.get("direction", "buy")).lower()
        return "buy" if d == "buy" else "sell"

    def _close(self, r: Dict[str, Any]) -> Optional[float]:
        v = r.get("klu_close", None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            return float(v)
        except Exception:
            return None

    def on_new_bsp(self, bsp_rows: List[Dict[str, Any]], new_row_index: int):
        r = bsp_rows[new_row_index]
        d = self._dir(r)

        if self.last_bsp is None:
            self.last_bsp = r
            self.active_dir = d
            self.active_indices = [new_row_index]
            return

        prev_d = self._dir(self.last_bsp)

        if d != prev_d:
            # reverse: finalize previous chain
            self._finalize_chain(bsp_rows)
            # new chain anchor = previous bsp close (reverse of new chain start)
            self.last_reverse_close = self._close(self.last_bsp)
            self.active_dir = d
            self.active_indices = [new_row_index]
        else:
            if self.active_dir != d:
                self.active_dir = d
                self.active_indices = [new_row_index]
            else:
                self.active_indices.append(new_row_index)

        self.last_bsp = r

    def _finalize_chain(self, bsp_rows: List[Dict[str, Any]]):
        if not self.active_indices:
            return
        if self.last_reverse_close is None:
            return

        mature_idx = self.active_indices[-1]
        mature_close = self._close(bsp_rows[mature_idx])
        if mature_close is None:
            return

        c_rev = float(self.last_reverse_close)

        if self.active_dir == "buy":
            denom = (mature_close - c_rev)
            if denom == 0:
                for j in self.active_indices:
                    bsp_rows[j]["maturity_rate"] = 0.0
                bsp_rows[mature_idx]["maturity_rate"] = 1.0
                return

            for j in self.active_indices:
                cj = self._close(bsp_rows[j])
                if cj is None:
                    bsp_rows[j]["maturity_rate"] = np.nan
                    continue
                mr = (float(cj) - c_rev) / denom
                bsp_rows[j]["maturity_rate"] = float(np.clip(mr, 0.0, 1.0))
            bsp_rows[mature_idx]["maturity_rate"] = 1.0

        else:
            denom = (c_rev - mature_close)
            if denom == 0:
                for j in self.active_indices:
                    bsp_rows[j]["maturity_rate"] = 0.0
                bsp_rows[mature_idx]["maturity_rate"] = 1.0
                return

            for j in self.active_indices:
                cj = self._close(bsp_rows[j])
                if cj is None:
                    bsp_rows[j]["maturity_rate"] = np.nan
                    continue
                mr = (c_rev - float(cj)) / denom
                bsp_rows[j]["maturity_rate"] = float(np.clip(mr, 0.0, 1.0))
            bsp_rows[mature_idx]["maturity_rate"] = 1.0


# ============================================================
# 6) Models: BUY return + BUY maturity
# ============================================================

@dataclass
class ModelPack:
    feature_cols: List[str]
    model_ret: xgb.XGBRegressor
    model_mr: xgb.XGBRegressor


def train_models_buy(bsp_df: pd.DataFrame, feature_cols: List[str], min_samples_total: int = 300) -> Optional[ModelPack]:
    if bsp_df.empty or "direction" not in bsp_df.columns:
        return None

    df = bsp_df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    buy_df = df[df["direction"].astype(str).str.lower() == "buy"].copy()
    if buy_df.empty:
        return None

    ret_df = buy_df.dropna(subset=["best_return_pct"]).copy()
    mr_df  = buy_df.dropna(subset=["maturity_rate"]).copy()

    if len(ret_df) < min_samples_total or len(mr_df) < min_samples_total:
        return None

    X_ret = to_float_matrix(ret_df, feature_cols)
    y_ret = ret_df["best_return_pct"].to_numpy(dtype=np.float32)

    X_mr = to_float_matrix(mr_df, feature_cols)
    y_mr = mr_df["maturity_rate"].to_numpy(dtype=np.float32)

    model_ret = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )
    model_mr = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=43,
        n_jobs=4,
    )

    model_ret.fit(X_ret, y_ret)
    model_mr.fit(X_mr, y_mr)

    return ModelPack(feature_cols=feature_cols, model_ret=model_ret, model_mr=model_mr)


def predict_buy(models: ModelPack, row_df: pd.DataFrame) -> Tuple[float, float]:
    X = to_float_matrix(row_df, models.feature_cols)
    pr = float(models.model_ret.predict(X)[0])
    pm = float(models.model_mr.predict(X)[0])
    return pr, float(np.clip(pm, 0.0, 1.0))


# ============================================================
# 7) NEW: Threshold selection using UNLABELED recent BSPs (NO realized returns)
# ============================================================

def _safe_ts(x) -> pd.Timestamp:
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def choose_thresholds_from_unlabeled(
    bsp_rows: List[Dict[str, Any]],
    models: ModelPack,
    asof_day: pd.Timestamp,
    window_days: float = 2.0,
    target_trades_per_day: int = 3,
    penalty_weight: float = 1.0,
    ret_grid: Optional[List[float]] = None,
    mr_grid: Optional[List[float]] = None,
) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """
    Use ONLY unlabeled recent BUY BSPs (best_return_pct is NaN) within [asof_day-window_days, asof_day]
    and model predictions to pick (ret_th, mr_th).

    Objective (no realized returns available):
      score = sum(pred_ret * pred_mr for selected) - penalty_weight * |count - target|
    """
    if models is None or not bsp_rows:
        return None

    if ret_grid is None:
        ret_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]
    if mr_grid is None:
        mr_grid = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    start_ts = asof_day - pd.Timedelta(days=float(window_days))
    end_ts = asof_day + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # collect unlabeled BUY rows in window
    cand = []
    for r in bsp_rows:
        if str(r.get("direction", "")).lower() != "buy":
            continue
        if not np.isnan(r.get("best_return_pct", np.nan)):
            continue  # labeled => not allowed for threshold selection
        t = _safe_ts(r.get("timestamp", None))
        if pd.isna(t):
            continue
        if not (start_ts <= t <= end_ts):
            continue
        cand.append(r)

    if len(cand) < 10:
        return None

    df = pd.DataFrame(cand)
    df = prepare_ml_dataset(df)

    # ensure feature cols exist
    for c in models.feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = to_float_matrix(df, models.feature_cols)
    pred_ret = models.model_ret.predict(X).astype(float)
    pred_mr  = models.model_mr.predict(X).astype(float)
    pred_mr = np.clip(pred_mr, 0.0, 1.0)

    # choose thresholds
    best = None
    best_info = None

    target = max(0, int(target_trades_per_day))
    for rt in ret_grid:
        for mt in mr_grid:
            mask = (pred_ret >= float(rt)) & (pred_mr >= float(mt))
            cnt = int(mask.sum())
            if cnt <= 0:
                score = -penalty_weight * abs(cnt - target)
            else:
                score = float(np.sum(pred_ret[mask] * pred_mr[mask])) - penalty_weight * abs(cnt - target)

            if (best is None) or (score > best):
                best = score
                best_info = {
                    "ret_th": float(rt),
                    "mr_th": float(mt),
                    "count": cnt,
                    "target": target,
                    "sum_ret_x_mr": float(np.sum(pred_ret[mask] * pred_mr[mask])) if cnt > 0 else 0.0,
                    "n_unlabeled_window": int(len(cand)),
                }

    if best_info is None:
        return None
    return best_info["ret_th"], best_info["mr_th"], best_info


# ============================================================
# 8) Main runner
# ============================================================

def run_chain_maturity_system(
    kline_csv_path: str,
    code: str,
    lv: KL_TYPE,

    accumulation_start: str,
    sim_start: str,
    end_time: str,

    chan_window_size: int = 500,
    warmup_bars: int = 500,
    bar_interval_minutes: int = 5,

    lookahead_days: float = 2.0,

    retrain_every_days: int = 5,
    min_samples_total: int = 300,

    # initial thresholds
    init_ret_th_buy: float = 0.30,
    init_mr_th_buy: float = 0.70,

    # NEW: threshold selection (unlabeled) params
    threshold_window_days: float = 2.0,
    threshold_target_trades_per_day: int = 3,
    threshold_penalty_weight: float = 1.0,
    threshold_ret_grid: Optional[List[float]] = None,
    threshold_mr_grid: Optional[List[float]] = None,

    initial_capital: float = 100000.0,
    position_size: float = 1.0,
    fee_pct: float = 0.0,
    execution_mode: str = "next_open",

    bsp_dataset_csv: Optional[str] = None,
    output_dir: str = "output_chain_maturity",
    verbose: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    raw, next_open_by_idx, next_close_by_idx, cur_close_by_idx, day_close_map, all_days, highs, lows, closes, open_col, high_col, low_col, close_col = load_kline_index(
        kline_csv_path, accumulation_start, end_time
    )
    buy_hold = compute_buy_hold_equity(day_close_map, all_days, initial_capital)

    sim_start_dt = pd.to_datetime(sim_start)

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

    engine = LongOnlyExecutionEngine(initial_capital=initial_capital, position_size=position_size, fee_pct=fee_pct)

    # BSP storage
    bsp_rows: List[Dict[str, Any]] = []
    seen_keys = set()

    def bsp_key(r: Dict[str, Any]) -> Tuple:
        return (str(r.get("timestamp")), int(r.get("klu_idx", -1)), str(r.get("direction")), str(r.get("bsp_type")))

    if bsp_dataset_csv is not None and os.path.exists(bsp_dataset_csv):
        df0 = pd.read_csv(bsp_dataset_csv)
        for _, row in df0.iterrows():
            r = row.to_dict()
            if "timestamp" in r:
                r["timestamp"] = str(r["timestamp"])
            if "direction" in r and pd.notna(r["direction"]):
                r["direction"] = str(r["direction"]).lower()
            if "bsp_type" in r and pd.notna(r["bsp_type"]):
                r["bsp_type"] = str(r["bsp_type"])
            r.setdefault("best_return_pct", np.nan)
            r.setdefault("maturity_rate", np.nan)
            k = bsp_key(r)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            bsp_rows.append(r)

    mr_mgr = MaturityChainManager()

    models: Optional[ModelPack] = None
    last_train_day: Optional[pd.Timestamp] = None

    ret_th_buy = float(init_ret_th_buy)
    mr_th_buy  = float(init_mr_th_buy)

    daily_log = []
    current_day = None

    def maybe_retrain(day_ts: pd.Timestamp):
        nonlocal models, last_train_day
        if models is None:
            do = True
        else:
            do = (last_train_day is None) or ((day_ts - last_train_day).days >= int(retrain_every_days))
        if not do:
            return

        df = pd.DataFrame(bsp_rows)
        if df.empty:
            return
        df2 = prepare_ml_dataset(df)
        feature_cols = get_feature_columns(df2)

        pack = train_models_buy(df2, feature_cols, min_samples_total=min_samples_total)
        if pack is None:
            return

        models = pack
        last_train_day = day_ts
        if verbose:
            print(f"[TRAIN-OK] buy models trained | rows={len(df2)} feats={len(feature_cols)}")

    def maybe_select_thresholds(day_ts: pd.Timestamp):
        """NEW: select thresholds using last N days unlabeled BSPs (predictions only)."""
        nonlocal ret_th_buy, mr_th_buy
        if models is None:
            return
        out = choose_thresholds_from_unlabeled(
            bsp_rows=bsp_rows,
            models=models,
            asof_day=day_ts,
            window_days=threshold_window_days,
            target_trades_per_day=threshold_target_trades_per_day,
            penalty_weight=threshold_penalty_weight,
            ret_grid=threshold_ret_grid,
            mr_grid=threshold_mr_grid,
        )
        if out is None:
            return
        ret_th_buy, mr_th_buy, info = out
        if verbose:
            print(f"[TH-OPT] asof={day_ts.date()} ret_th_buy={ret_th_buy:.2f} mr_th_buy={mr_th_buy:.2f} "
                  f"| unlabeled_window={info['n_unlabeled_window']} selected_cnt={info['count']} target={info['target']}")

    # main loop
    for i in range(len(raw)):
        bar_ts = raw.loc[i, "timestamp"]
        bar_day = bar_ts.date()

        if current_day is None:
            current_day = bar_day

        in_sim = (bar_ts >= sim_start_dt)

        # day rollover
        if bar_day != current_day:
            prev_day = current_day
            prev_day_ts = pd.to_datetime(prev_day)

            # label returns for points whose lookahead is now available
            label_bestlookahead_for_ready_points(
                bsp_rows=bsp_rows,
                highs=highs, lows=lows, closes=closes,
                lookahead_days=lookahead_days,
                bar_interval_minutes=bar_interval_minutes,
                current_bar_idx=i,
            )

            # retrain EOD
            maybe_retrain(prev_day_ts)

            # NEW: threshold selection using unlabeled recent BSPs (no realized returns)
            maybe_select_thresholds(prev_day_ts)

            # execute pending (EOD)
            if in_sim:
                engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode)

            day_close = day_close_map.get(prev_day)
            equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash

            daily_log.append({
                "date": prev_day,
                "equity": equity,
                "cash": engine.cash,
                "pos": engine.pos,
                "qty": engine.qty,
                "has_model": int(models is not None),
                "bsp_rows": len(bsp_rows),
                "ret_th_buy": ret_th_buy,
                "mr_th_buy": mr_th_buy,
            })

            if verbose:
                tag = "SIM" if (pd.to_datetime(prev_day) >= sim_start_dt.normalize()) else "ACCUM"
                print(f"[EOD-{tag}] {prev_day} equity={equity:.2f} cash={engine.cash:.2f} pos={engine.pos} "
                      f"| rows={len(bsp_rows)} model={int(models is not None)} ret_th_buy={ret_th_buy:.2f} mr_th_buy={mr_th_buy:.2f}")

            current_day = bar_day

        # execute pending each bar during sim
        if in_sim:
            engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode)

        # feed chan
        o = float(raw.loc[i, open_col])
        h = float(raw.loc[i, high_col]) if high_col in raw.columns else float(raw.loc[i, close_col])
        l = float(raw.loc[i, low_col])  if low_col  in raw.columns else float(raw.loc[i, close_col])
        c = float(raw.loc[i, close_col])
        v = float(raw.loc[i, "Volume"]) if "Volume" in raw.columns else 0.0

        klu = build_klu(bar_ts, o, h, l, c, v)
        _window_chan, _ = chan.process_new_kline(klu)

        new_rows = chan.export_new_historical_bsp_to_list()
        if new_rows:
            for r0 in new_rows:
                r = dict(r0)

                # timestamp
                r.setdefault("timestamp", str(bar_ts))

                # direction
                if "direction" not in r or r["direction"] is None:
                    if r.get("is_buy", None) is not None:
                        r["direction"] = "buy" if bool(r["is_buy"]) else "sell"
                    else:
                        r["direction"] = "buy"
                r["direction"] = str(r["direction"]).lower()

                # ensure close
                if "klu_close" not in r or r["klu_close"] is None or (isinstance(r["klu_close"], float) and np.isnan(r["klu_close"])):
                    ki = r.get("klu_idx", None)
                    if ki is not None and 0 <= int(ki) < len(closes):
                        r["klu_close"] = float(closes[int(ki)])
                    else:
                        r["klu_close"] = float(c)

                # labels
                r.setdefault("best_return_pct", np.nan)
                r.setdefault("maturity_rate", np.nan)

                k = bsp_key(r)
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                bsp_rows.append(r)

                # maturity update
                mr_mgr.on_new_bsp(bsp_rows, len(bsp_rows) - 1)

                # print during accumulation too
                if verbose and (pd.to_datetime(r["timestamp"]) < sim_start_dt):
                    print(f"[NEW] {r['direction'].upper()} BSP but no model yet | "
                          f"idx={r.get('klu_idx')} type={r.get('bsp_type')} ts={r.get('timestamp')} close={r.get('klu_close')}")

                # trading (BUY only)
                if in_sim:
                    d = str(r.get("direction", "buy")).lower()

                    # close on sell BSP
                    if d == "sell" and engine.pos == 1:
                        engine.place_order_for_next_bar("sell", int(r.get("klu_idx", i)), reason="CLOSE_LONG (sell BSP)")
                        continue

                    if d != "buy":
                        continue

                    if models is None:
                        if verbose:
                            print(f"[NEW] BUY BSP but no model yet | idx={r.get('klu_idx')} type={r.get('bsp_type')} ts={r.get('timestamp')} close={r.get('klu_close')}")
                        continue

                    row_df = pd.DataFrame([r])
                    row_df = prepare_ml_dataset(row_df)
                    for cc in models.feature_cols:
                        if cc not in row_df.columns:
                            row_df[cc] = 0.0

                    pred_ret, pred_mr = predict_buy(models, row_df)

                    if verbose:
                        print(f"[PRED] BUY idx={r.get('klu_idx')} type={r.get('bsp_type')} "
                              f"pred_ret={pred_ret:.3f}% pred_mr={pred_mr:.3f} | pos={engine.pos} th_ret={ret_th_buy:.2f} th_mr={mr_th_buy:.2f}")

                    if pred_ret >= ret_th_buy and pred_mr >= mr_th_buy and engine.pos == 0:
                        engine.place_order_for_next_bar(
                            "buy",
                            int(r.get("klu_idx", i)),
                            reason=f"OPEN_LONG pred_ret={pred_ret:.2f}% pred_mr={pred_mr:.2f}"
                        )

    # finalize labels
    label_bestlookahead_for_ready_points(
        bsp_rows=bsp_rows,
        highs=highs, lows=lows, closes=closes,
        lookahead_days=lookahead_days,
        bar_interval_minutes=bar_interval_minutes,
        current_bar_idx=len(raw) - 1,
    )

    engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode)

    # final day log
    if current_day is not None:
        day_close = day_close_map.get(current_day)
        equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash
        daily_log.append({
            "date": current_day,
            "equity": equity,
            "cash": engine.cash,
            "pos": engine.pos,
            "qty": engine.qty,
            "has_model": int(models is not None),
            "bsp_rows": len(bsp_rows),
            "ret_th_buy": ret_th_buy,
            "mr_th_buy": mr_th_buy,
        })

    daily_df = pd.DataFrame(daily_log)
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    trades_df = pd.DataFrame(engine.trades)

    bsp_df = pd.DataFrame(bsp_rows)
    bsp_df = prepare_ml_dataset(bsp_df)

    out_bsp    = os.path.join(output_dir, f"bsp_dataset_{code}_{lv.name}.csv")
    out_daily  = os.path.join(output_dir, f"daily_equity_{code}_{lv.name}.csv")
    out_trades = os.path.join(output_dir, f"trades_{code}_{lv.name}.csv")

    bsp_df.to_csv(out_bsp, index=False)
    daily_df.to_csv(out_daily, index=False)
    trades_df.to_csv(out_trades, index=False)

    try:
        # full-period equity from daily_df
        eq_full = pd.Series(daily_df["equity"].values, index=pd.to_datetime(daily_df["date"]))
        bh_full = buy_hold.copy()

        # --- Full period ---
        plt.figure()
        plt.plot(eq_full.index, eq_full.values, label="Strategy")
        if not bh_full.empty:
            plt.plot(bh_full.index, bh_full.values, label="Buy&Hold")
        plt.legend()
        plt.title(f"{code} {lv.name} Strategy vs Buy&Hold (Full)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"equity_curve_full_{code}_{lv.name}.png"), dpi=150)
        plt.close()

        # --- Simulation period only ---
        sim_start_day = pd.to_datetime(sim_start).normalize()
        sim_end_day = pd.to_datetime(end_time).normalize()

        eq_sim = eq_full[(eq_full.index >= sim_start_day) & (eq_full.index <= sim_end_day)].copy()
        bh_sim = bh_full[(bh_full.index >= sim_start_day) & (bh_full.index <= sim_end_day)].copy()

        # Align indices (so both series share same dates where possible)
        if not eq_sim.empty and not bh_sim.empty:
            common_idx = eq_sim.index.intersection(bh_sim.index)
            eq_sim_aligned = eq_sim.loc[common_idx]
            bh_sim_aligned = bh_sim.loc[common_idx]

            # Normalize both to 1.0 at simulation start (clean comparison)
            if len(eq_sim_aligned) > 0:
                eq_sim_norm = eq_sim_aligned / float(eq_sim_aligned.iloc[0])
            else:
                eq_sim_norm = eq_sim_aligned

            if len(bh_sim_aligned) > 0:
                bh_sim_norm = bh_sim_aligned / float(bh_sim_aligned.iloc[0])
            else:
                bh_sim_norm = bh_sim_aligned

            plt.figure()
            plt.plot(eq_sim_norm.index, eq_sim_norm.values, label="Strategy (Normalized)")
            plt.plot(bh_sim_norm.index, bh_sim_norm.values, label="Buy&Hold (Normalized)")
            plt.legend()
            plt.title(f"{code} {lv.name} Strategy vs Buy&Hold (Simulation Only)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"equity_curve_sim_{code}_{lv.name}.png"), dpi=150)
            plt.close()

        else:
            # Fallback: still create a sim plot even if one side is missing
            plt.figure()
            if not eq_sim.empty:
                eq_sim_norm = eq_sim / float(eq_sim.iloc[0])
                plt.plot(eq_sim_norm.index, eq_sim_norm.values, label="Strategy (Normalized)")
            if not bh_sim.empty:
                bh_sim_norm = bh_sim / float(bh_sim.iloc[0])
                plt.plot(bh_sim_norm.index, bh_sim_norm.values, label="Buy&Hold (Normalized)")
            plt.legend()
            plt.title(f"{code} {lv.name} Strategy vs Buy&Hold (Simulation Only)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"equity_curve_sim_{code}_{lv.name}.png"), dpi=150)
            plt.close()

    except Exception as e:
        if verbose:
            print("[PLOT-ERR]", repr(e))

    return {"bsp_csv": out_bsp, "daily_csv": out_daily, "trades_csv": out_trades, "output_dir": output_dir}


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    results = run_chain_maturity_system(
        kline_csv_path="DataAPI/data/QQQ_5M.csv",
        code="QQQ",
        lv=KL_TYPE.K_5M,

        accumulation_start="2023-01-01",
        sim_start="2023-04-01",
        end_time="2023-06-30",

        chan_window_size=500,
        warmup_bars=500,
        bar_interval_minutes=5,

        lookahead_days=2.0,
        retrain_every_days=5,
        min_samples_total=300,

        init_ret_th_buy=0.30,
        init_mr_th_buy=0.70,

        # NEW threshold selection (UNLABELED last 2 days)
        threshold_window_days=2.0,
        threshold_target_trades_per_day=3,
        threshold_penalty_weight=1.0,
        threshold_ret_grid=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00],
        threshold_mr_grid=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90],

        initial_capital=100000.0,
        position_size=1.0,
        fee_pct=0.0,
        execution_mode="next_open",

        bsp_dataset_csv=None,
        output_dir="output_chain_maturity",
        verbose=True,
    )
    print(results)
