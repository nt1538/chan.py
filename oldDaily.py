# ============================================================
# DAILY Kline "Extreme Confirmation" Probability (ONE MODEL) + ONLINE Trading
# + Macro Indices features from local CSVs (NO FRED pulling)
# + Feature importance (Logistic Regression coefficients)
#
# Uses these files in SAME folder as QQQ_DAY.csv (same structure, no volume ok):
#   VIX.csv, DXY.csv, US10Y.csv, US30Y.csv, XAU.csv, NYXBT.csv
#
# Label rule (same as before, depends on base_dir at t0):
#   If base_dir == 'sell': y=1 if max(high next N) < high(t0) else 0
#   If base_dir == 'buy' : y=1 if min(low  next N) > low(t0)  else 0
#
# One model trained on BOTH buy+sell labeled days (base_dir is a feature).
# Trading (kept consistent with your last thresholds direction):
#   - SELL when pos==1 and p_day > p_sell_level
#   - BUY  when pos==0 and p_day < p_buy_level
#
# Outputs:
#   - daily_df_out (output window)
#   - trades_df_out
#   - bsp_df_out
#   - feat_importance_df (top abs coefficients)
#   - Plots (only output window)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from sliding_window_chan import SlidingWindowChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_FIELD, KL_TYPE, AUTYPE

try:
    from Common.CEnum import DATA_SRC
except Exception:
    class DATA_SRC:
        CSV = "CSV"

from KLine.KLine_Unit import CKLine_Unit
from Common.CTime import CTime

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

# ----------------------------
# Chan bar builders
# ----------------------------
def to_ctime(ts) -> CTime:
    if isinstance(ts, CTime):
        return ts
    dt = pd.to_datetime(ts).to_pydatetime()
    try:
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, auto=False)
    except Exception:
        pass
    try:
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    except Exception:
        pass
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    try:
        return CTime(s, auto=False)
    except Exception:
        return CTime(s)

def build_klu(ts, o, h, l, c, v=0.0) -> CKLine_Unit:
    ct = to_ctime(ts)
    kl_dict = {
        DATA_FIELD.FIELD_TIME: ct,
        DATA_FIELD.FIELD_OPEN: float(o),
        DATA_FIELD.FIELD_HIGH: float(h),
        DATA_FIELD.FIELD_LOW:  float(l),
        DATA_FIELD.FIELD_CLOSE: float(c),
        DATA_FIELD.FIELD_VOLUME: float(v),
        "time": ct, "timestamp": ct, "datetime": ct, "dt": ct,
        "open": float(o), "high": float(h), "low": float(l), "close": float(c),
        "volume": float(v),
    }
    klu = CKLine_Unit(kl_dict)
    try:
        klu.time = ct
    except Exception:
        pass
    return klu

# ----------------------------
# CSV loader
# ----------------------------
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def load_daily_csv(daily_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(daily_csv_path)

    ts_col = _pick_col(df, ["timestamp", "date", "datetime", "time"])
    if ts_col is None:
        ts_col = df.columns[0]

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "timestamp"})

    open_col  = _pick_col(df, ["open", "Open", "o", "O"])
    high_col  = _pick_col(df, ["high", "High", "h", "H"])
    low_col   = _pick_col(df, ["low", "Low", "l", "L"])
    close_col = _pick_col(df, ["close", "Close", "adj_close", "Adj Close", "AdjClose", "c", "C"])
    if open_col is None or close_col is None:
        raise ValueError(f"CSV must contain open and close columns: {daily_csv_path}")
    if high_col is None:
        high_col = close_col
    if low_col is None:
        low_col = close_col

    vol_col = _pick_col(df, ["volume", "Volume", "vol", "Vol", "v", "V"])

    df["_open"]  = pd.to_numeric(df[open_col], errors="coerce").astype(float)
    df["_high"]  = pd.to_numeric(df[high_col], errors="coerce").astype(float)
    df["_low"]   = pd.to_numeric(df[low_col], errors="coerce").astype(float)
    df["_close"] = pd.to_numeric(df[close_col], errors="coerce").astype(float)
    df["_vol"]   = pd.to_numeric(df[vol_col], errors="coerce").astype(float) if vol_col is not None else 0.0

    df = df.dropna(subset=["_open", "_high", "_low", "_close"]).reset_index(drop=True)
    df["ts_norm"] = df["timestamp"].dt.normalize()
    return df

# ----------------------------
# Kline features (generic)
# ----------------------------
def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def compute_kline_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("timestamp").reset_index(drop=True)

    o = d["_open"].astype(float)
    h = d["_high"].astype(float)
    l = d["_low"].astype(float)
    c = d["_close"].astype(float)
    v = d["_vol"].astype(float) if "_vol" in d.columns else pd.Series([0.0]*len(d), index=d.index)

    prev_c = c.shift(1)
    d["ret1"] = (_safe_div(c, prev_c) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    d["tr"] = tr.fillna(0.0)
    d["atr_14"] = d["tr"].rolling(14).mean().bfill().fillna(0.0)

    d["range"] = (h - l).fillna(0.0)
    d["range_over_atr"] = _safe_div(d["range"], d["atr_14"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rng = (h - l).replace(0.0, np.nan)
    d["close_pos"] = ((c - l) / rng).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    d["gap"] = (_safe_div(o, prev_c) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for w in [5, 10, 20, 40]:
        d[f"ret_{w}"] = (_safe_div(c, c.shift(w)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        d[f"vol_{w}"] = d["ret1"].rolling(w).std().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for w in [20, 50, 100]:
        ma = c.rolling(w).mean()
        d[f"above_ma_{w}"] = (_safe_div(c, ma) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for w in [5, 20, 60]:
        m = v.rolling(w).mean()
        s = v.rolling(w).std()
        d[f"vol_ratio_{w}"] = _safe_div(v, m).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        d[f"vol_z_{w}"] = ((v - m) / (s + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    d["vol_jump"] = _safe_div(v, v.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    def _slope_log(x):
        x = np.asarray(x, dtype=float)
        x = np.log(np.maximum(x, 1e-12))
        t = np.arange(len(x), dtype=float)
        t = t - t.mean()
        x = x - x.mean()
        den = (t*t).sum()
        return 0.0 if den <= 0 else float((t*x).sum() / den)

    d["slope40"] = d["_close"].rolling(40).apply(_slope_log, raw=False).fillna(0.0)
    return d

KLINE_KEYS = [
    "ret1","ret_5","ret_10","ret_20","ret_40",
    "vol_5","vol_10","vol_20","vol_40",
    "atr_14","range_over_atr","close_pos","gap",
    "vol_ratio_5","vol_ratio_20","vol_ratio_60",
    "vol_z_20","vol_z_60","vol_jump",
    "above_ma_20","above_ma_50","above_ma_100",
    "slope40",
]

def make_kline_dict(kline_row: pd.Series, prefix: str) -> Dict[str, float]:
    d = {}
    for k in KLINE_KEYS:
        val = kline_row[k] if k in kline_row.index else 0.0
        d[f"{prefix}{k}"] = float(val) if np.isfinite(val) else 0.0
    return d

# ----------------------------
# Load & align macro indices (NO volume required)
# ----------------------------
def load_macro_features_from_folder(
    folder: str,
    files: Dict[str, str],
    chan_start: str,
) -> pd.DataFrame:
    """
    files: dict prefix -> filename, e.g. {"vix_":"VIX.csv"}
    returns: DataFrame indexed by ts_norm with prefixed KLINE_KEYS columns
    """
    out = None
    for pref, fn in files.items():
        path = os.path.join(folder, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Macro file not found: {path}")
        dfm = load_daily_csv(path)
        dfm = dfm[dfm["timestamp"] >= pd.to_datetime(chan_start)].copy().reset_index(drop=True)
        dfm_feat = compute_kline_features(dfm)
        cols = ["ts_norm"] + KLINE_KEYS
        dfm_feat = dfm_feat[cols].copy()
        rename = {k: f"{pref}{k}" for k in KLINE_KEYS}
        dfm_feat = dfm_feat.rename(columns=rename)
        if out is None:
            out = dfm_feat.copy()
        else:
            out = out.merge(dfm_feat, on="ts_norm", how="outer")
    if out is None:
        out = pd.DataFrame(columns=["ts_norm"])
    out = out.sort_values("ts_norm").reset_index(drop=True)
    return out

# ----------------------------
# BSP helpers
# ----------------------------
def normalize_bsp_row(r: Dict[str, Any]) -> Dict[str, Any]:
    rr = dict(r)
    if "direction" not in rr or rr["direction"] is None:
        if rr.get("is_buy", None) is not None:
            rr["direction"] = "buy" if bool(rr["is_buy"]) else "sell"
        else:
            rr["direction"] = "buy"
    rr["direction"] = str(rr["direction"]).lower()
    if "bsp_type" in rr and rr["bsp_type"] is not None:
        rr["bsp_type"] = str(rr["bsp_type"]).lower()
    else:
        rr["bsp_type"] = "?"
    return rr

def extract_bsp_rows_from_chan(chan_obj) -> List[Dict[str, Any]]:
    if hasattr(chan_obj, "export_new_historical_bsp_to_list"):
        out = chan_obj.export_new_historical_bsp_to_list()
        return out if out else []
    if hasattr(chan_obj, "export_new_bsp_to_list"):
        out = chan_obj.export_new_bsp_to_list()
        return out if out else []
    return []

def latest_bsp_dir_up_to(bsp_rows: List[Dict[str, Any]], ts: pd.Timestamp) -> Optional[str]:
    past = [r for r in bsp_rows if pd.to_datetime(r["timestamp"]) <= ts]
    if not past:
        return None
    past = sorted(past, key=lambda x: pd.to_datetime(x["timestamp"]))
    return str(past[-1].get("direction", "")).lower()

# ----------------------------
# BSP context features
# ----------------------------
def make_daily_bsp_context(bsp_hist: List[Dict[str, Any]], cur_ts: pd.Timestamp, window_days: int = 60) -> Dict[str, float]:
    past = [r for r in bsp_hist if pd.to_datetime(r["timestamp"]) <= cur_ts]
    past = sorted(past, key=lambda x: pd.to_datetime(x["timestamp"]))
    if not past:
        return {
            "ctx_has_bsp": 0.0,
            "ctx_last_dir_buy": 0.0,
            "ctx_last_dir_sell": 0.0,
            "ctx_days_since_last_bsp": 999.0,
            "ctx_days_since_last_buy": 999.0,
            "ctx_days_since_last_sell": 999.0,
            "ctx_density_total": 0.0,
            "ctx_density_buy": 0.0,
            "ctx_density_sell": 0.0,
            "ctx_density_imb": 0.0,
        }
    last = past[-1]
    last_dir = str(last.get("direction", "")).lower()
    days_since_last = float((cur_ts.normalize() - pd.to_datetime(last["timestamp"]).normalize()).days)

    def _days_since(target: str) -> float:
        for r in reversed(past):
            if str(r.get("direction", "")).lower() == target:
                return float((cur_ts.normalize() - pd.to_datetime(r["timestamp"]).normalize()).days)
        return 999.0

    start = cur_ts.normalize() - pd.Timedelta(days=int(window_days))
    recent = [r for r in past if pd.to_datetime(r["timestamp"]).normalize() >= start]
    buy = sum(1 for r in recent if str(r.get("direction", "")).lower() == "buy")
    sell = sum(1 for r in recent if str(r.get("direction", "")).lower() == "sell")
    tot = max(1.0, float(buy + sell))

    return {
        "ctx_has_bsp": 1.0,
        "ctx_last_dir_buy": 1.0 if last_dir == "buy" else 0.0,
        "ctx_last_dir_sell": 1.0 if last_dir == "sell" else 0.0,
        "ctx_days_since_last_bsp": days_since_last,
        "ctx_days_since_last_buy": _days_since("buy"),
        "ctx_days_since_last_sell": _days_since("sell"),
        "ctx_density_total": float(len(recent)),
        "ctx_density_buy": float(buy),
        "ctx_density_sell": float(sell),
        "ctx_density_imb": float((sell - buy) / tot),
    }

# ----------------------------
# Confirmation label (depends on base_dir)
# ----------------------------
def label_top_bottom_for_day(df_feat: pd.DataFrame, day_idx: int, N: int, base_dir: str) -> Optional[int]:
    if N <= 0:
        return None
    if day_idx + N >= len(df_feat):
        return None
    h0 = float(df_feat.loc[day_idx, "_high"])
    l0 = float(df_feat.loc[day_idx, "_low"])
    fut = df_feat.loc[day_idx+1:day_idx+N]
    if fut.empty:
        return None
    if base_dir == "sell":
        mx = float(fut["_high"].max())
        return 1 if mx < h0 else 0
    if base_dir == "buy":
        mn = float(fut["_low"].min())
        return 1 if mn > l0 else 0
    return None

# ----------------------------
# Chain regimes
# ----------------------------
def compute_chain_endpoints(bsp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not bsp_rows:
        return []
    bsps = sorted(bsp_rows, key=lambda r: pd.to_datetime(r["timestamp"]))
    ends = []
    cur_dir = str(bsps[0].get("direction", "")).lower()
    cur_end = bsps[0]
    cur_end_i = 0
    for i in range(1, len(bsps)):
        d = str(bsps[i].get("direction", "")).lower()
        if d == cur_dir:
            cur_end = bsps[i]
            cur_end_i = i
        else:
            ends.append({"end_ts": pd.to_datetime(cur_end["timestamp"]).normalize(), "end_dir": cur_dir, "end_i": cur_end_i})
            cur_dir = d
            cur_end = bsps[i]
            cur_end_i = i
    ends.append({"end_ts": pd.to_datetime(cur_end["timestamp"]).normalize(), "end_dir": cur_dir, "end_i": cur_end_i})
    return ends

def regime_for_day_from_ends(day_norm: pd.Timestamp, ends: List[Dict[str, Any]]) -> str:
    if len(ends) < 2:
        return "unknown"
    for k in range(len(ends) - 1):
        a = ends[k]; b = ends[k+1]
        a_ts, a_dir = a["end_ts"], a["end_dir"]
        b_ts, b_dir = b["end_ts"], b["end_dir"]
        if (day_norm >= a_ts) and (day_norm <= b_ts):
            if a_dir == "buy" and b_dir == "sell":
                return "up"
            if a_dir == "sell" and b_dir == "buy":
                return "down"
    return "unknown"

# ----------------------------
# Model helpers (convergence-safe)
# ----------------------------
def fit_prob_model_dicts(X_dicts: List[Dict[str, float]], y: np.ndarray):
    base = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("scaler", MaxAbsScaler()),
        ("lr", LogisticRegression(
            max_iter=8000,
            class_weight="balanced",
            solver="saga",
            C=0.5,
            n_jobs=-1
        ))
    ])
    uniq, counts = np.unique(y, return_counts=True)
    if len(uniq) < 2:
        raise ValueError("Only one class in y so far.")
    min_count = int(counts.min())

    if min_count >= 5:
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        cal.fit(X_dicts, y)
        return cal
    elif min_count >= 3:
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=2)
        cal.fit(X_dicts, y)
        return cal
    else:
        base.fit(X_dicts, y)
        return base

def predict_prob(model, X_dicts: List[Dict[str, float]]) -> np.ndarray:
    return model.predict_proba(X_dicts)[:, 1]

def _get_underlying_pipeline(model):
    # If calibrated, take first calibrated classifier's estimator (pipeline)
    if hasattr(model, "calibrated_classifiers_") and len(getattr(model, "calibrated_classifiers_", [])) > 0:
        cc0 = model.calibrated_classifiers_[0]
        if hasattr(cc0, "estimator") and cc0.estimator is not None:
            return cc0.estimator
    return model

def feature_importance_from_lr(model, top_n: int = 80) -> pd.DataFrame:
    pipe = _get_underlying_pipeline(model)
    if not (hasattr(pipe, "named_steps") and "vec" in pipe.named_steps and "lr" in pipe.named_steps):
        return pd.DataFrame(columns=["feature", "coef", "abs_coef"])
    vec = pipe.named_steps["vec"]
    lr = pipe.named_steps["lr"]
    try:
        names = np.array(vec.get_feature_names_out(), dtype=object)
    except Exception:
        names = np.array(getattr(vec, "feature_names_", []), dtype=object)
    coef = lr.coef_.ravel()
    if len(names) != len(coef):
        return pd.DataFrame(columns=["feature", "coef", "abs_coef"])
    df_imp = pd.DataFrame({"feature": names, "coef": coef})
    df_imp["abs_coef"] = df_imp["coef"].abs()
    df_imp = df_imp.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return df_imp.head(int(top_n)).copy()

@dataclass
class OnlineState:
    model: Optional[Any] = None
    new_labels: int = 0
    trained_n: int = 0

# ----------------------------
# Build daily features (includes QQQ + macro + BSP context + dp + regime + base_dir one-hot)
# ----------------------------
def make_daily_features(
    qqq_row: pd.Series,
    macro_row: Optional[pd.Series],
    bsp_hist_up_to_day: List[Dict[str, Any]],
    p_val: float,
    dp_minK: float,
    dp_maxK: float,
    regime: str,
    base_dir: Optional[str],
) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    # QQQ kline
    feats.update(make_kline_dict(qqq_row, prefix="q_"))

    # Macro indices kline features (already prefixed in columns)
    if macro_row is not None:
        for col in macro_row.index:
            if col == "ts_norm":
                continue
            val = macro_row[col]
            feats[str(col)] = float(val) if np.isfinite(val) else 0.0

    # BSP context
    ts = pd.to_datetime(qqq_row["timestamp"])
    feats.update(make_daily_bsp_context(bsp_hist_up_to_day, ts, window_days=60))

    # dp features
    feats["p"] = float(p_val) if np.isfinite(p_val) else 0.0
    feats["dp_minK"] = float(dp_minK) if np.isfinite(dp_minK) else 0.0
    feats["dp_maxK"] = float(dp_maxK) if np.isfinite(dp_maxK) else 0.0

    # regime one-hot
    rg = str(regime).lower()
    feats["rg_up"] = 1.0 if rg == "up" else 0.0
    feats["rg_down"] = 1.0 if rg == "down" else 0.0
    feats["rg_unknown"] = 1.0 if rg not in ("up", "down") else 0.0

    # base_dir one-hot (THIS is key for single-model mixing buy/sell)
    bd = str(base_dir).lower() if base_dir is not None else "none"
    feats["bd_buy"] = 1.0 if bd == "buy" else 0.0
    feats["bd_sell"] = 1.0 if bd == "sell" else 0.0
    feats["bd_none"] = 1.0 if bd not in ("buy", "sell") else 0.0

    return feats

# ----------------------------
# Main runner (ONE MODEL)
# ----------------------------
def run_daily_prob_and_trade_one_model_with_macro(
    daily_csv_path: str,
    code: str = "QQQ",

    daily_chan_start: str = "2010-06-01",

    plot_start: str = "2011-09-01",
    plot_end: str   = "2024-12-31",

    output_start: str = "2015-09-01",
    output_end: str   = "2024-12-31",

    daily_chan_max_klines: int = 800,

    N: int = 5,

    min_labeled_days_to_train: int = 200,
    retrain_every_new_labels: int = 25,

    dp_lookback: int = 5,

    p_sell_level: float = 0.35,  # SELL when p_day > this
    p_buy_level: float  = 0.35,  # BUY  when p_day < this  (kept as your current direction)

    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    trade_at: str = "next_open",

    start_in_market: bool = True,

    macro_files: Optional[Dict[str, str]] = None,  # prefix->filename
    verbose: bool = True,
    shade_regimes: bool = True,
):
    # ----------------------------
    # Load QQQ + compute features
    # ----------------------------
    df = load_daily_csv(daily_csv_path)
    df = df[df["timestamp"] >= pd.to_datetime(daily_chan_start)].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No daily bars after daily_chan_start.")
    df_feat = compute_kline_features(df)

    plot_s = pd.to_datetime(plot_start)
    plot_e = pd.to_datetime(plot_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    out_s = pd.to_datetime(output_start)
    out_e = pd.to_datetime(output_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # ----------------------------
    # Load macro features (aligned by date)
    # ----------------------------
    folder = os.path.dirname(os.path.abspath(daily_csv_path))
    if macro_files is None:
        macro_files = {
            "vix_":   "VIX.csv",
            "dxy_":   "DXY.csv",
            "us10y_": "US10Y.csv",
            "us30y_": "US30Y.csv",
            "xau_":   "XAU.csv",
            "nyxbt_": "NYXBT.csv",
        }
    macro_feat = load_macro_features_from_folder(folder, macro_files, chan_start=daily_chan_start)
    # merge to qqq by ts_norm
    df_feat = df_feat.merge(macro_feat, on="ts_norm", how="left")
    df_feat = df_feat.sort_values("timestamp").reset_index(drop=True)

    # A quick index for macro row access
    macro_cols = [c for c in df_feat.columns if any(c.startswith(pref) for pref in macro_files.keys())]
    # per-row macro series view (optional)
    # we'll pass df_feat.loc[i, macro_cols] as macro_row

    # ----------------------------
    # Chan config
    # ----------------------------
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

    daily_chan = SlidingWindowChan(
        code=code,
        begin_time=None,
        end_time=None,
        data_src=getattr(DATA_SRC, "CSV", "CSV"),
        lv_list=[KL_TYPE.K_DAY],
        config=config,
        autype=AUTYPE.QFQ,
        max_klines=int(daily_chan_max_klines),
    )

    # ----------------------------
    # Online buffers (ONE model)
    # ----------------------------
    bsp_rows: List[Dict[str, Any]] = []
    seen_bsp = set()

    X_all: List[Dict[str, float]] = []
    y_all: List[int] = []

    st = OnlineState()
    pending_day_idx: List[int] = []

    p_day = np.full(len(df_feat), np.nan, dtype=float)

    dp_vs_minK = np.full(len(df_feat), np.nan, dtype=float)
    dp_vs_maxK = np.full(len(df_feat), np.nan, dtype=float)
    p_minK = np.full(len(df_feat), np.nan, dtype=float)
    p_maxK = np.full(len(df_feat), np.nan, dtype=float)

    regime_series = np.array(["unknown"] * len(df_feat), dtype=object)
    base_dir_series = np.array([None] * len(df_feat), dtype=object)

    # ----------------------------
    # Trading state
    # ----------------------------
    def exec_price(i: int) -> float:
        if trade_at == "close":
            return float(df_feat.loc[i, "_close"])
        if i + 1 >= len(df_feat):
            return np.nan
        return float(df_feat.loc[i + 1, "_open"])

    pos = 1 if start_in_market else 0
    if pos == 1:
        cash = 0.0
        qty = float(initial_capital) / float(df_feat.loc[0, "_close"])
    else:
        cash = float(initial_capital)
        qty = 0.0

    trades: List[Dict[str, Any]] = []

    # ----------------------------
    # Train helpers
    # ----------------------------
    def maybe_train():
        if len(y_all) < int(min_labeled_days_to_train):
            return
        if st.model is None or st.new_labels >= int(retrain_every_new_labels):
            y = np.asarray(y_all, dtype=int)
            if len(np.unique(y)) < 2:
                return
            try:
                st.model = fit_prob_model_dicts(X_all, y)
                st.trained_n = len(y)
                if verbose:
                    print(f"[TRAIN][ONE] n={len(y)} pos={int(y.sum())} ({y.mean():.2%})")
                st.new_labels = 0
            except ValueError as e:
                if verbose:
                    print(f"[TRAIN][ONE] skipped: {e}")

    def compute_dp_series(p_arr, i, lb):
        p_val = p_arr[i]
        if not (lb > 0 and i >= 1 and np.isfinite(p_val)):
            return np.nan, np.nan, np.nan, np.nan
        start = max(0, i - lb)
        prev = pd.Series(p_arr[start:i]).dropna()
        if len(prev) == 0:
            return np.nan, np.nan, np.nan, np.nan
        pmin = float(prev.min())
        pmax = float(prev.max())
        return pmin, pmax, (p_val - pmin), (p_val - pmax)

    # ----------------------------
    # Walk-forward loop
    # ----------------------------
    for i in range(len(df_feat)):
        ts = pd.to_datetime(df_feat.loc[i, "timestamp"])
        day_norm = ts.normalize()

        # feed chan
        daily_chan.process_new_kline(build_klu(
            ts=ts,
            o=float(df_feat.loc[i, "_open"]),
            h=float(df_feat.loc[i, "_high"]),
            l=float(df_feat.loc[i, "_low"]),
            c=float(df_feat.loc[i, "_close"]),
            v=float(df_feat.loc[i, "_vol"]) if "_vol" in df_feat.columns else 0.0,
        ))

        # collect BSPs
        new_bsp = extract_bsp_rows_from_chan(daily_chan)
        if new_bsp:
            for r0 in new_bsp:
                r = normalize_bsp_row(dict(r0))
                r.setdefault("timestamp", ts)
                key = (pd.to_datetime(r["timestamp"]).strftime("%Y-%m-%d"), r["direction"], r.get("bsp_type", "?"))
                if key in seen_bsp:
                    continue
                seen_bsp.add(key)
                bsp_rows.append(r)

        # regime + base_dir today
        ends = compute_chain_endpoints(bsp_rows)
        regime = regime_for_day_from_ends(day_norm, ends)
        regime_series[i] = regime
        base_dir_today = latest_bsp_dir_up_to(bsp_rows, ts)
        base_dir_series[i] = base_dir_today

        # predict p_day (if model exists)
        p_val_today = np.nan
        if st.model is not None:
            bsp_hist = [r for r in bsp_rows if pd.to_datetime(r["timestamp"]) <= ts]
            macro_row = df_feat.loc[i, macro_cols] if len(macro_cols) else None
            feat_i = make_daily_features(
                qqq_row=df_feat.loc[i],
                macro_row=macro_row,
                bsp_hist_up_to_day=bsp_hist,
                p_val=0.0, dp_minK=0.0, dp_maxK=0.0,
                regime=regime,
                base_dir=base_dir_today,
            )
            p_val_today = float(predict_prob(st.model, [feat_i])[0])
            p_day[i] = p_val_today

        # dp vs min/max prev K
        lb = int(dp_lookback)
        pmin, pmax, dmin, dmax = compute_dp_series(p_day, i, lb)
        p_minK[i] = pmin; p_maxK[i] = pmax
        dp_vs_minK[i] = dmin; dp_vs_maxK[i] = dmax

        # finalize labels for older days -> add to dataset
        pending_day_idx.append(i)
        while pending_day_idx and i >= pending_day_idx[0] + int(N):
            j = pending_day_idx.pop(0)
            t0 = pd.to_datetime(df_feat.loc[j, "timestamp"])
            base_dir_j = latest_bsp_dir_up_to(bsp_rows, t0)
            if base_dir_j not in ("buy", "sell"):
                continue

            y = label_top_bottom_for_day(df_feat, j, int(N), base_dir_j)
            if y is None:
                continue

            ends_j = compute_chain_endpoints([r for r in bsp_rows if pd.to_datetime(r["timestamp"]) <= t0])
            regime_j = regime_for_day_from_ends(t0.normalize(), ends_j)
            bsp_hist_j = [r for r in bsp_rows if pd.to_datetime(r["timestamp"]) <= t0]
            macro_row_j = df_feat.loc[j, macro_cols] if len(macro_cols) else None

            feat_j = make_daily_features(
                qqq_row=df_feat.loc[j],
                macro_row=macro_row_j,
                bsp_hist_up_to_day=bsp_hist_j,
                p_val=float(p_day[j]) if np.isfinite(p_day[j]) else 0.0,
                dp_minK=float(dp_vs_minK[j]) if np.isfinite(dp_vs_minK[j]) else 0.0,
                dp_maxK=float(dp_vs_maxK[j]) if np.isfinite(dp_vs_maxK[j]) else 0.0,
                regime=regime_j,
                base_dir=base_dir_j,
            )
            X_all.append(feat_j)
            y_all.append(int(y))
            st.new_labels += 1

        # train (walk-forward)
        maybe_train()

        # ============================
        # TRADING (online) - ONE p_day
        # ============================
        px = exec_price(i)
        if not (np.isfinite(px) and px > 0):
            continue

        # SELL
        if pos == 1 and np.isfinite(p_val_today) and (p_val_today > float(p_sell_level)):
            notional = qty * px
            fee = notional * float(fee_pct)
            cash += (notional - fee)
            qty = 0.0
            pos = 0

            trades.append({
                "day": day_norm, "ts": ts, "side": "SELL", "exec_px": float(px),
                "p_day": float(p_val_today),
                "dp_vs_minK": float(dp_vs_minK[i]) if np.isfinite(dp_vs_minK[i]) else np.nan,
                "dp_vs_maxK": float(dp_vs_maxK[i]) if np.isfinite(dp_vs_maxK[i]) else np.nan,
                "regime": str(regime), "base_dir": str(base_dir_today),
                "reason": f"SELL: p_day>{float(p_sell_level):.2f} | p_day={p_val_today:.2f}"
            })

        # BUY
        if pos == 0 and np.isfinite(p_val_today) and (p_val_today < float(p_buy_level)):
            notional = cash
            fee = notional * float(fee_pct)
            spend = notional + fee
            if spend > cash:
                notional = cash / (1.0 + float(fee_pct))
                fee = notional * float(fee_pct)
                spend = notional + fee

            if notional > 0:
                qty = notional / px
                cash -= spend
                pos = 1

                trades.append({
                    "day": day_norm, "ts": ts, "side": "BUY", "exec_px": float(px),
                    "p_day": float(p_val_today),
                    "dp_vs_minK": float(dp_vs_minK[i]) if np.isfinite(dp_vs_minK[i]) else np.nan,
                    "dp_vs_maxK": float(dp_vs_maxK[i]) if np.isfinite(dp_vs_maxK[i]) else np.nan,
                    "regime": str(regime), "base_dir": str(base_dir_today),
                    "reason": f"BUY: p_day<{float(p_buy_level):.2f} | p_day={p_val_today:.2f}"
                })

    # ----------------------------
    # Build daily_df (full)
    # ----------------------------
    daily_df = pd.DataFrame({
        "timestamp": pd.to_datetime(df_feat["timestamp"]),
        "open": df_feat["_open"].astype(float).to_numpy(),
        "high": df_feat["_high"].astype(float).to_numpy(),
        "low":  df_feat["_low"].astype(float).to_numpy(),
        "close": df_feat["_close"].astype(float).to_numpy(),

        f"p_confirm_within_{N}d": p_day,

        f"p_min_prev_{dp_lookback}": p_minK,
        f"p_max_prev_{dp_lookback}": p_maxK,
        f"dp_vs_min_prev_{dp_lookback}": dp_vs_minK,
        f"dp_vs_max_prev_{dp_lookback}": dp_vs_maxK,

        "regime_chain": regime_series,
        "base_dir": base_dir_series,
    })

    trades_df = pd.DataFrame(trades)

    # ----------------------------
    # Replay equity (align daily series)
    # ----------------------------
    d = daily_df.copy().sort_values("timestamp").reset_index(drop=True)
    d["day"] = d["timestamp"].dt.normalize()
    d["action"] = ""
    d["exec_px"] = np.nan

    if not trades_df.empty:
        tmap = trades_df.groupby("day").last().reset_index()
        mp_px = dict(zip(tmap["day"], tmap["exec_px"]))
        mp_side = dict(zip(tmap["day"], tmap["side"]))
        d["exec_px"] = d["day"].map(mp_px)
        d["action"] = d["day"].map(mp_side).fillna("")

    cash2 = float(initial_capital)
    pos2 = 1 if start_in_market else 0
    qty2 = (cash2 / float(d.loc[0, "close"])) if pos2 == 1 else 0.0
    if pos2 == 1:
        cash2 = 0.0

    equity = []
    position = []
    for i in range(len(d)):
        close = float(d.loc[i, "close"])
        side = str(d.loc[i, "action"])
        px = d.loc[i, "exec_px"]

        if side == "SELL" and pos2 == 1 and np.isfinite(px):
            notional = qty2 * float(px)
            fee = notional * float(fee_pct)
            cash2 += (notional - fee)
            qty2 = 0.0
            pos2 = 0

        if side == "BUY" and pos2 == 0 and np.isfinite(px):
            notional = cash2
            fee = notional * float(fee_pct)
            spend = notional + fee
            if spend > cash2:
                notional = cash2 / (1.0 + float(fee_pct))
                fee = notional * float(fee_pct)
                spend = notional + fee
            qty2 = notional / float(px) if float(px) > 0 else 0.0
            cash2 -= spend
            pos2 = 1

        eq = cash2 + qty2 * close
        equity.append(eq)
        position.append(pos2)

    d["position"] = position
    d["equity"] = equity

    bh0 = float(d.loc[0, "close"])
    d["bh_equity"] = float(initial_capital) * (d["close"] / (bh0 + 1e-12))

    # ----------------------------
    # Slice OUTPUT range
    # ----------------------------
    mask_out = (d["timestamp"] >= out_s) & (d["timestamp"] <= out_e)
    d_out = d.loc[mask_out].copy().reset_index(drop=True)

    if not trades_df.empty:
        trades_out = trades_df[(trades_df["ts"] >= out_s) & (trades_df["ts"] <= out_e)].copy().reset_index(drop=True)
    else:
        trades_out = trades_df

    bsp_all = pd.DataFrame(bsp_rows)
    if not bsp_all.empty:
        bsp_all["timestamp"] = pd.to_datetime(bsp_all["timestamp"], errors="coerce")
        bsp_out = bsp_all[(bsp_all["timestamp"] >= out_s) & (bsp_all["timestamp"] <= out_e)].copy().reset_index(drop=True)
    else:
        bsp_out = bsp_all

    # ----------------------------
    # Feature importance (final trained model snapshot)
    # ----------------------------
    feat_imp = feature_importance_from_lr(st.model, top_n=120) if st.model is not None else pd.DataFrame()

    # ----------------------------
    # Plot only OUTPUT window
    # ----------------------------
    plot_mask = (d_out["timestamp"] >= plot_s) & (d_out["timestamp"] <= plot_e)
    dp = d_out.loc[plot_mask].copy()

    pcol = f"p_confirm_within_{N}d"

    plt.figure(figsize=(16, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(dp["timestamp"], dp["close"], linewidth=1.0, label="Close")
    ax2.plot(dp["timestamp"], dp[pcol], linewidth=1.2, alpha=0.85, label=pcol)

    if shade_regimes and len(dp):
        up_mask = dp["regime_chain"].astype(str).str.lower().eq("up").to_numpy()
        dn_mask = dp["regime_chain"].astype(str).str.lower().eq("down").to_numpy()

        def shade(mask, alpha):
            in_span = False
            start = None
            for k in range(len(mask)):
                if mask[k] and not in_span:
                    in_span = True
                    start = dp["timestamp"].iloc[k]
                if in_span and (k == len(mask)-1 or not mask[k+1]):
                    end = dp["timestamp"].iloc[k]
                    ax1.axvspan(start, end, alpha=alpha)
                    in_span = False

        shade(up_mask, 0.06)
        shade(dn_mask, 0.06)

    if not trades_out.empty:
        buys = trades_out[trades_out["side"] == "BUY"]
        sells = trades_out[trades_out["side"] == "SELL"]
        if len(buys):
            ax1.scatter(buys["ts"], buys["exec_px"], marker="^", s=90, alpha=0.85, label="BUY")
        if len(sells):
            ax1.scatter(sells["ts"], sells["exec_px"], marker="v", s=90, alpha=0.85, label="SELL")

    ax1.set_title(f"{code} Close + ONE p_confirm + Macro Indices + Online Trades | dp compares to MIN/MAX of prev {dp_lookback}")
    ax1.set_ylabel("Close")
    ax2.set_ylabel("Probability (0-1)")
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left")
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 6))
    plt.plot(dp["timestamp"], dp["equity"] / float(dp["equity"].iloc[0]), label="Strategy (normalized)")
    plt.plot(dp["timestamp"], dp["bh_equity"] / float(dp["bh_equity"].iloc[0]), label="Buy&Hold (normalized)")
    plt.title("Equity: Strategy vs Buy&Hold (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    if verbose:
        y = np.asarray(y_all, dtype=int) if len(y_all) else np.array([], dtype=int)
        if len(y):
            print(f"[LABELS][ONE] n={len(y)} pos={int(y.sum())} ({y.mean():.2%})")
        print(f"[MODEL][ONE] trained={'yes' if st.model is not None else 'no'} trained_n={st.trained_n}")
        print(f"[BSP] total_bsp_all={len(bsp_rows)} | bsp_out={len(bsp_out)}")
        print(f"[TRADES] all={len(trades_df)} | out={len(trades_out)}")
        if st.model is not None and not feat_imp.empty:
            print("[FEATURE_IMPORTANCE] shown = abs(LR coef) top rows")

    return d_out, trades_out, bsp_out, feat_imp


# ============================================================
# RUN
# ============================================================
daily_df_out, trades_df_out, bsp_df_out, feat_importance_df = run_daily_prob_and_trade_one_model_with_macro(
    daily_csv_path="DataAPI/data/SPY_DAY.csv",
    code="SPY",

    daily_chan_start="2015-06-01",

    plot_start="2016-09-01",
    plot_end="2024-12-31",

    output_start="2016-09-01",
    output_end="2024-12-31",

    daily_chan_max_klines=800,

    N=5,

    min_labeled_days_to_train=200,
    retrain_every_new_labels=25,

    dp_lookback=5,

    p_sell_level=0.30,
    p_buy_level=0.20,

    initial_capital=100000.0,
    fee_pct=0.0,
    trade_at="next_open",
    start_in_market=True,

    macro_files={
        "vix_":   "VIX.csv",
        "dxy_":   "DXY.csv",
        "us10y_": "US10Y.csv",
        "us30y_": "US30Y.csv",
        "xau_":   "XAU.csv",
        "nyxbt_": "NYXBT.csv",
    },

    verbose=True,
    shade_regimes=True,
)

display(trades_df_out.head(50))
display(daily_df_out.head(10))
display(daily_df_out.tail(10))

# Feature importance (abs coefficients). Higher abs => more influential in LR.
display(feat_importance_df.head(80))

