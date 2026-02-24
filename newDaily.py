# ============================================================
# UNIFIED DAILY "RISK / INSTABILITY" P (ONE MEANING)
# + DAILY CHAN BSP STRUCTURE FEATURES (key improvement)
# + POOLED TRAINING ON SPY + QQQ (instrument_id)
# + 3-STATE GATE: HOLD / FREE / RISK_OFF
#
# p interpretation:
#   p low  -> stable -> HOLD (ensure long)
#   p mid  -> choppy/uncertain -> FREE (delegate to 5m system)
#   p high -> unstable -> RISK_OFF (ensure flat)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

# ---- your Chan imports (same as your code)
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


# ============================================================
# 1) CSV loader + kline features (same as your style)
# ============================================================
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

def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

KLINE_KEYS = [
    "ret1","ret_5","ret_10","ret_20","ret_40",
    "vol_5","vol_10","vol_20","vol_40",
    "atr_14","range_over_atr","close_pos","gap",
    "above_ma_20","above_ma_50","above_ma_100",
    "slope40",
]

def compute_kline_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("timestamp").reset_index(drop=True)

    o = d["_open"].astype(float)
    h = d["_high"].astype(float)
    l = d["_low"].astype(float)
    c = d["_close"].astype(float)

    prev_c = c.shift(1)
    d["ret1"] = (_safe_div(c, prev_c) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    d["tr"] = tr.fillna(0.0)
    d["atr_14"] = d["tr"].rolling(14).mean().bfill().fillna(0.0)

    d["range_over_atr"] = _safe_div((h-l).abs(), d["atr_14"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rng = (h - l).replace(0.0, np.nan)
    d["close_pos"] = ((c - l) / rng).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    d["gap"] = (_safe_div(o, prev_c) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for w in [5, 10, 20, 40]:
        d[f"ret_{w}"] = (_safe_div(c, c.shift(w)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        d[f"vol_{w}"] = d["ret1"].rolling(w).std().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for w in [20, 50, 100]:
        ma = c.rolling(w).mean()
        d[f"above_ma_{w}"] = (_safe_div(c, ma) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

def make_kline_dict(kline_row: pd.Series, prefix: str) -> Dict[str, float]:
    out = {}
    for k in KLINE_KEYS:
        val = kline_row[k] if k in kline_row.index else 0.0
        out[f"{prefix}{k}"] = float(val) if np.isfinite(val) else 0.0
    return out


# ============================================================
# 2) Macro features loader (daily)
# ============================================================
def load_macro_features_from_folder(folder: str, files: Dict[str, str], start: str) -> pd.DataFrame:
    out = None
    for pref, fn in files.items():
        path = os.path.join(folder, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Macro file not found: {path}")
        dfm = load_daily_csv(path)
        dfm = dfm[dfm["timestamp"] >= pd.to_datetime(start)].copy().reset_index(drop=True)
        dfm_feat = compute_kline_features(dfm)
        cols = ["ts_norm"] + KLINE_KEYS
        dfm_feat = dfm_feat[cols].copy()
        rename = {k: f"{pref}{k}" for k in KLINE_KEYS}
        dfm_feat = dfm_feat.rename(columns=rename)
        out = dfm_feat.copy() if out is None else out.merge(dfm_feat, on="ts_norm", how="outer")

    if out is None:
        out = pd.DataFrame(columns=["ts_norm"])
    return out.sort_values("ts_norm").reset_index(drop=True)


# ============================================================
# 3) DAILY CHAN BSP extraction + structure features
# ============================================================
def to_ctime(ts) -> CTime:
    dt = pd.to_datetime(ts).to_pydatetime()
    try:
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, auto=False)
    except Exception:
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

def build_klu(ts, o, h, l, c, v=0.0) -> CKLine_Unit:
    ct = to_ctime(ts)
    kl_dict = {
        DATA_FIELD.FIELD_TIME: ct,
        DATA_FIELD.FIELD_OPEN: float(o),
        DATA_FIELD.FIELD_HIGH: float(h),
        DATA_FIELD.FIELD_LOW:  float(l),
        DATA_FIELD.FIELD_CLOSE: float(c),
        DATA_FIELD.FIELD_VOLUME: float(v),
        "time": ct, "timestamp": ct, "datetime": ct,
        "open": float(o), "high": float(h), "low": float(l), "close": float(c),
        "volume": float(v),
    }
    klu = CKLine_Unit(kl_dict)
    try:
        klu.time = ct
    except Exception:
        pass
    return klu

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

def compute_chain_endpoints(bsp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not bsp_rows:
        return []
    bsps = sorted(bsp_rows, key=lambda r: pd.to_datetime(r["timestamp"]))
    ends = []
    cur_dir = str(bsps[0].get("direction", "")).lower()
    cur_end = bsps[0]
    for i in range(1, len(bsps)):
        d = str(bsps[i].get("direction", "")).lower()
        if d == cur_dir:
            cur_end = bsps[i]
        else:
            ends.append({"end_ts": pd.to_datetime(cur_end["timestamp"]).normalize(), "end_dir": cur_dir})
            cur_dir = d
            cur_end = bsps[i]
    ends.append({"end_ts": pd.to_datetime(cur_end["timestamp"]).normalize(), "end_dir": cur_dir})
    return ends

def regime_for_day(day_norm: pd.Timestamp, ends: List[Dict[str, Any]]) -> str:
    if len(ends) < 2:
        return "unknown"
    for k in range(len(ends) - 1):
        a = ends[k]; b = ends[k+1]
        if (day_norm >= a["end_ts"]) and (day_norm <= b["end_ts"]):
            if a["end_dir"] == "buy" and b["end_dir"] == "sell":
                return "up"
            if a["end_dir"] == "sell" and b["end_dir"] == "buy":
                return "down"
    return "unknown"

def bsp_context_features(bsp_rows: List[Dict[str, Any]], cur_ts: pd.Timestamp, window_days: int = 60) -> Dict[str, float]:
    past = [r for r in bsp_rows if pd.to_datetime(r["timestamp"]) <= cur_ts]
    past = sorted(past, key=lambda x: pd.to_datetime(x["timestamp"]))
    if not past:
        return {
            "bsp_has": 0.0,
            "bsp_last_buy": 0.0,
            "bsp_last_sell": 0.0,
            "bsp_days_since_last": 999.0,
            "bsp_density": 0.0,
            "bsp_density_buy": 0.0,
            "bsp_density_sell": 0.0,
            "bsp_imbalance": 0.0,
            "bsp_alternations_20": 0.0,
        }

    last = past[-1]
    last_dir = str(last.get("direction", "")).lower()
    days_since_last = float((cur_ts.normalize() - pd.to_datetime(last["timestamp"]).normalize()).days)

    start = cur_ts.normalize() - pd.Timedelta(days=int(window_days))
    recent = [r for r in past if pd.to_datetime(r["timestamp"]).normalize() >= start]
    buy = sum(1 for r in recent if str(r.get("direction","")).lower() == "buy")
    sell = sum(1 for r in recent if str(r.get("direction","")).lower() == "sell")
    tot = max(1.0, float(buy + sell))

    # alternations in last 20 signals (structure churn proxy)
    last20 = recent[-20:] if len(recent) > 0 else []
    alt = 0
    for i in range(1, len(last20)):
        if str(last20[i]["direction"]).lower() != str(last20[i-1]["direction"]).lower():
            alt += 1

    return {
        "bsp_has": 1.0,
        "bsp_last_buy": 1.0 if last_dir == "buy" else 0.0,
        "bsp_last_sell": 1.0 if last_dir == "sell" else 0.0,
        "bsp_days_since_last": days_since_last,
        "bsp_density": float(len(recent)),
        "bsp_density_buy": float(buy),
        "bsp_density_sell": float(sell),
        "bsp_imbalance": float((sell - buy) / tot),
        "bsp_alternations_20": float(alt),
    }


# ============================================================
# 4) Unified LABEL = future instability (NOT base_dir confirm)
# ============================================================
def label_instability_for_day(
    df_feat: pd.DataFrame,
    day_idx: int,
    N: int,
    dd_thresh: float = -0.03,      # unstable if next N days sees <= -3% dd
    rv_quantile: float = 0.80,     # or future vol is >= recent history top 20%
    rv_ref_lookback: int = 252,
) -> Optional[int]:
    if N <= 0 or day_idx + N >= len(df_feat):
        return None

    c0 = float(df_feat.loc[day_idx, "_close"])
    if not (np.isfinite(c0) and c0 > 0):
        return None

    fut = df_feat.loc[day_idx+1:day_idx+N].copy()
    if fut.empty:
        return None

    min_low = float(fut["_low"].min())
    dd = (min_low / c0) - 1.0  # negative

    fut_rets = fut["ret1"].astype(float).to_numpy()
    rv = float(np.std(fut_rets)) if len(fut_rets) > 1 else 0.0

    if day_idx < rv_ref_lookback:
        return 1 if dd <= dd_thresh else 0

    past = df_feat.loc[day_idx-rv_ref_lookback:day_idx].copy()
    past_rv_proxy = past["ret1"].rolling(N).std().dropna()
    if len(past_rv_proxy) < 50:
        return 1 if dd <= dd_thresh else 0

    rv_cut = float(past_rv_proxy.quantile(rv_quantile))
    unstable = (dd <= dd_thresh) or (rv >= rv_cut)
    return 1 if unstable else 0


# ============================================================
# 5) Model (LR + calibration)
# ============================================================
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


# ============================================================
# 6) Feature builder = daily kline + macro + DAILY CHAN BSP + instrument_id
# ============================================================
def make_daily_features_unified(
    asset_row: pd.Series,
    macro_row: Optional[pd.Series],
    instrument_id: int,
    bsp_rows_up_to_day: List[Dict[str, Any]],
    regime: str,
) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    feats.update(make_kline_dict(asset_row, prefix="d_"))

    if macro_row is not None:
        for col in macro_row.index:
            if col == "ts_norm":
                continue
            val = macro_row[col]
            feats[str(col)] = float(val) if np.isfinite(val) else 0.0

    # DAILY CHAN structure features
    ts = pd.to_datetime(asset_row["timestamp"])
    feats.update(bsp_context_features(bsp_rows_up_to_day, ts, window_days=60))

    rg = str(regime).lower()
    feats["rg_up"] = 1.0 if rg == "up" else 0.0
    feats["rg_down"] = 1.0 if rg == "down" else 0.0
    feats["rg_unknown"] = 1.0 if rg not in ("up","down") else 0.0

    feats["instrument_id"] = float(instrument_id)
    return feats


# ============================================================
# 7) Gate state machine (3 states + hysteresis)
# ============================================================
class GateState:
    HOLD = "HOLD"
    FREE = "FREE"
    RISK_OFF = "RISK_OFF"

def next_gate_state_with_hysteresis(
    cur_state: str,
    p: float,
    p_low_enter: float,
    p_low_exit: float,
    p_high_enter: float,
    p_high_exit: float,
) -> str:
    if not np.isfinite(p):
        return cur_state

    if cur_state == GateState.HOLD:
        if p >= p_high_enter:
            return GateState.RISK_OFF
        if p > p_low_exit:
            return GateState.FREE
        return GateState.HOLD

    if cur_state == GateState.RISK_OFF:
        if p <= p_low_enter:
            return GateState.HOLD
        if p < p_high_exit:
            return GateState.FREE
        return GateState.RISK_OFF

    # FREE
    if p <= p_low_enter:
        return GateState.HOLD
    if p >= p_high_enter:
        return GateState.RISK_OFF
    return GateState.FREE


@dataclass
class OnlineState:
    model: Optional[Any] = None
    new_labels: int = 0
    trained_n: int = 0


# ============================================================
# 8) Main pooled runner: SPY+QQQ daily Chan BSP-aware p
# ============================================================
def run_unified_daily_gate_with_daily_chan_bsp_pooled(
    asset_csvs: Dict[str, str],  # {"SPY": ".../SPY_DAY.csv", "QQQ":".../QQQ_DAY.csv"}
    daily_start: str = "2015-06-01",

    plot_start: str = "2016-09-01",
    plot_end: str   = "2024-12-31",
    output_start: str = "2016-09-01",
    output_end: str   = "2024-12-31",

    N: int = 5,
    dd_thresh: float = -0.03,
    rv_quantile: float = 0.80,
    rv_ref_lookback: int = 252,

    min_labeled_days_to_train: int = 400,
    retrain_every_new_labels: int = 50,

    p_low_enter: float = 0.30,
    p_low_exit:  float = 0.36,
    p_high_enter: float = 0.70,
    p_high_exit:  float = 0.62,

    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    trade_at: str = "next_open",
    start_in_market: bool = True,

    macro_files: Optional[Dict[str, str]] = None,
    daily_chan_max_klines: int = 800,
    verbose: bool = True,
):
    if macro_files is None:
        macro_files = {
            "vix_":   "VIX.csv",
            "dxy_":   "DXY.csv",
            "us10y_": "US10Y.csv",
            "us30y_": "US30Y.csv",
            "xau_":   "XAU.csv",
            "nyxbt_": "NYXBT.csv",
        }

    # ---- macro once (folder from first csv)
    first_path = list(asset_csvs.values())[0]
    folder = os.path.dirname(os.path.abspath(first_path))
    macro_feat = load_macro_features_from_folder(folder, macro_files, start=daily_start)
    macro_cols = [c for c in macro_feat.columns if c != "ts_norm"]

    # deterministic instrument map
    inst_map = {sym: i for i, sym in enumerate(sorted(asset_csvs.keys()))}

    # ---- Chan config (daily)
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

    # ---- load assets, compute daily kline feats + merge macro
    assets = []
    for sym, path in asset_csvs.items():
        df0 = load_daily_csv(path)
        df0 = df0[df0["timestamp"] >= pd.to_datetime(daily_start)].copy().reset_index(drop=True)
        df_feat = compute_kline_features(df0)
        df_feat = df_feat.merge(macro_feat, on="ts_norm", how="left").sort_values("timestamp").reset_index(drop=True)
        df_feat["symbol"] = sym
        df_feat["instrument_id"] = inst_map[sym]
        assets.append(df_feat)

    # ---- union of trading days across assets
    day_set = set()
    for df in assets:
        day_set |= set(pd.to_datetime(df["timestamp"]).dt.normalize().tolist())
    all_days = sorted(day_set)

    # ---- day -> index map per symbol
    idx_map = {}
    for df in assets:
        sym = df["symbol"].iloc[0]
        idx_map[sym] = {pd.to_datetime(df.loc[i, "timestamp"]).normalize(): i for i in range(len(df))}

    # ---- Online buffers (pooled)
    st = OnlineState()
    X_all, y_all = [], []
    pending = {sym: [] for sym in asset_csvs.keys()}

    # ---- per asset: daily Chan objects + BSP store
    chan_objs = {}
    bsp_rows = {sym: [] for sym in asset_csvs.keys()}
    seen_bsp = {sym: set() for sym in asset_csvs.keys()}
    regime_series = {sym: np.array(["unknown"]*len(df), dtype=object) for sym, df in [(d["symbol"].iloc[0], d) for d in assets]}
    p_pred = {sym: np.full(len(df), np.nan, dtype=float) for sym, df in [(d["symbol"].iloc[0], d) for d in assets]}
    y_label = {}

    for df in assets:
        sym = df["symbol"].iloc[0]
        chan_objs[sym] = SlidingWindowChan(
            code=sym,
            begin_time=None,
            end_time=None,
            data_src=getattr(DATA_SRC, "CSV", "CSV"),
            lv_list=[KL_TYPE.K_DAY],
            config=config,
            autype=AUTYPE.QFQ,
            max_klines=int(daily_chan_max_klines),
        )

        # precompute instability labels (revealed with delay N)
        y = np.array([np.nan]*len(df), dtype=float)
        for i in range(len(df)):
            lab = label_instability_for_day(df, i, N, dd_thresh, rv_quantile, rv_ref_lookback)
            y[i] = np.nan if lab is None else float(lab)
        y_label[sym] = y

    def maybe_train():
        if len(y_all) < int(min_labeled_days_to_train):
            return
        if st.model is None or st.new_labels >= int(retrain_every_new_labels):
            y = np.asarray(y_all, dtype=int)
            if len(np.unique(y)) < 2:
                return
            st.model = fit_prob_model_dicts(X_all, y)
            st.trained_n = len(y)
            st.new_labels = 0
            if verbose:
                print(f"[TRAIN][POOLED] n={len(y)} pos={int(y.sum())} ({y.mean():.2%})")

    # ---- ONLINE LOOP day-by-day across both assets
    for day in all_days:
        for df in assets:
            sym = df["symbol"].iloc[0]
            if day not in idx_map[sym]:
                continue
            i = idx_map[sym][day]
            ts = pd.to_datetime(df.loc[i, "timestamp"])

            # feed DAILY Chan
            chan = chan_objs[sym]
            chan.process_new_kline(build_klu(
                ts=ts,
                o=float(df.loc[i, "_open"]),
                h=float(df.loc[i, "_high"]),
                l=float(df.loc[i, "_low"]),
                c=float(df.loc[i, "_close"]),
                v=float(df.loc[i, "_vol"]) if "_vol" in df.columns else 0.0,
            ))

            # collect BSP
            new_bsp = extract_bsp_rows_from_chan(chan)
            if new_bsp:
                for r0 in new_bsp:
                    r = normalize_bsp_row(dict(r0))
                    r.setdefault("timestamp", ts)
                    key = (pd.to_datetime(r["timestamp"]).strftime("%Y-%m-%d"), r["direction"], r.get("bsp_type","?"))
                    if key in seen_bsp[sym]:
                        continue
                    seen_bsp[sym].add(key)
                    bsp_rows[sym].append(r)

            # compute regime for day (from endpoints)
            ends = compute_chain_endpoints(bsp_rows[sym])
            rg = regime_for_day(day, ends)
            regime_series[sym][i] = rg

            # (A) predict p if model exists (using BSP context up to today)
            if st.model is not None:
                macro_row = df.loc[i, macro_cols] if len(macro_cols) else None
                past_bsp = [r for r in bsp_rows[sym] if pd.to_datetime(r["timestamp"]) <= ts]
                feat_i = make_daily_features_unified(
                    asset_row=df.loc[i],
                    macro_row=macro_row,
                    instrument_id=int(df.loc[i, "instrument_id"]),
                    bsp_rows_up_to_day=past_bsp,
                    regime=rg,
                )
                p_pred[sym][i] = float(predict_prob(st.model, [feat_i])[0])

            # (B) reveal label after N days
            pending[sym].append(i)
            while pending[sym] and i >= pending[sym][0] + int(N):
                j = pending[sym].pop(0)
                lab = y_label[sym][j]
                if not np.isfinite(lab):
                    continue

                tsj = pd.to_datetime(df.loc[j, "timestamp"])
                macro_row_j = df.loc[j, macro_cols] if len(macro_cols) else None
                past_bsp_j = [r for r in bsp_rows[sym] if pd.to_datetime(r["timestamp"]) <= tsj]
                rgj = regime_series[sym][j]

                feat_j = make_daily_features_unified(
                    asset_row=df.loc[j],
                    macro_row=macro_row_j,
                    instrument_id=int(df.loc[j, "instrument_id"]),
                    bsp_rows_up_to_day=past_bsp_j,
                    regime=rgj,
                )
                X_all.append(feat_j)
                y_all.append(int(lab))
                st.new_labels += 1

        maybe_train()

    # ============================================================
    # PER-ASSET: apply gate + daily trading sim (FREE => leave to 5m engine)
    # ============================================================
    results = {}
    for df in assets:
        sym = df["symbol"].iloc[0]
        inst_id = int(df["instrument_id"].iloc[0])

        plot_s = pd.to_datetime(plot_start)
        plot_e = pd.to_datetime(plot_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        out_s  = pd.to_datetime(output_start)
        out_e  = pd.to_datetime(output_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

        p_arr = p_pred[sym]
        rg_arr = regime_series[sym]

        # gate series
        gate = np.array([GateState.FREE]*len(df), dtype=object)
        state = GateState.HOLD if start_in_market else GateState.RISK_OFF
        for i in range(len(df)):
            state = next_gate_state_with_hysteresis(
                state, p_arr[i], p_low_enter, p_low_exit, p_high_enter, p_high_exit
            )
            gate[i] = state

        # trading sim: HOLD ensure long, RISK_OFF ensure flat, FREE no action
        def exec_price(i: int) -> float:
            if trade_at == "close":
                return float(df.loc[i, "_close"])
            if i + 1 >= len(df):
                return np.nan
            return float(df.loc[i + 1, "_open"])

        pos = 1 if start_in_market else 0
        if pos == 1:
            cash = 0.0
            qty = float(initial_capital) / float(df.loc[0, "_close"])
        else:
            cash = float(initial_capital)
            qty = 0.0

        trades = []
        for i in range(len(df)):
            px = exec_price(i)
            if not (np.isfinite(px) and px > 0):
                continue

            want = gate[i]
            day_norm = pd.to_datetime(df.loc[i, "timestamp"]).normalize()
            ts = pd.to_datetime(df.loc[i, "timestamp"])
            p = p_arr[i]

            if want == GateState.FREE:
                continue

            if want == GateState.HOLD and pos == 0:
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
                    trades.append({"day": day_norm, "ts": ts, "side": "BUY", "exec_px": float(px),
                                   "p": float(p) if np.isfinite(p) else np.nan, "gate": want, "regime": rg_arr[i]})

            if want == GateState.RISK_OFF and pos == 1:
                notional = qty * px
                fee = notional * float(fee_pct)
                cash += (notional - fee)
                qty = 0.0
                pos = 0
                trades.append({"day": day_norm, "ts": ts, "side": "SELL", "exec_px": float(px),
                               "p": float(p) if np.isfinite(p) else np.nan, "gate": want, "regime": rg_arr[i]})

        trades_df = pd.DataFrame(trades)

        daily_df = pd.DataFrame({
            "timestamp": pd.to_datetime(df["timestamp"]),
            "open": df["_open"].astype(float).to_numpy(),
            "high": df["_high"].astype(float).to_numpy(),
            "low":  df["_low"].astype(float).to_numpy(),
            "close": df["_close"].astype(float).to_numpy(),
            f"p_unstable_next_{N}d": p_arr,
            "gate_state": gate,
            "regime_chain": rg_arr,
            "instrument_id": inst_id,
        })

        # replay equity
        d = daily_df.copy().sort_values("timestamp").reset_index(drop=True)
        d["day"] = d["timestamp"].dt.normalize()
        d["action"] = ""
        d["exec_px"] = np.nan

        if not trades_df.empty:
            tmap = trades_df.groupby("day").last().reset_index()
            d["exec_px"] = d["day"].map(dict(zip(tmap["day"], tmap["exec_px"])))
            d["action"] = d["day"].map(dict(zip(tmap["day"], tmap["side"]))).fillna("")

        cash2 = float(initial_capital)
        pos2 = 1 if start_in_market else 0
        qty2 = (cash2 / float(d.loc[0, "close"])) if pos2 == 1 else 0.0
        if pos2 == 1:
            cash2 = 0.0

        equity, position = [], []
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

            equity.append(cash2 + qty2 * close)
            position.append(pos2)

        d["position"] = position
        d["equity"] = equity
        bh0 = float(d.loc[0, "close"])
        d["bh_equity"] = float(initial_capital) * (d["close"] / (bh0 + 1e-12))

        mask_out = (d["timestamp"] >= out_s) & (d["timestamp"] <= out_e)
        d_out = d.loc[mask_out].copy().reset_index(drop=True)
        trades_out = trades_df[(trades_df["ts"] >= out_s) & (trades_df["ts"] <= out_e)].copy().reset_index(drop=True) if not trades_df.empty else trades_df

        # plots
        dp = d_out[(d_out["timestamp"] >= plot_s) & (d_out["timestamp"] <= plot_e)].copy()
        if len(dp):
            pcol = f"p_unstable_next_{N}d"
            plt.figure(figsize=(16, 8))
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax1.plot(dp["timestamp"], dp["close"], linewidth=1.0, label=f"{sym} Close")
            ax2.plot(dp["timestamp"], dp[pcol], linewidth=1.2, alpha=0.85, label=pcol)
            ax2.axhline(p_low_enter, linestyle="--", linewidth=0.8, alpha=0.6)
            ax2.axhline(p_high_enter, linestyle="--", linewidth=0.8, alpha=0.6)
            ax1.set_title(f"{sym} Close + Daily-Chan-BSP-aware Unified Instability p + Gate")
            ax1.grid(True, alpha=0.25)
            h1,l1 = ax1.get_legend_handles_labels()
            h2,l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc="upper left")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(16, 6))
            plt.plot(dp["timestamp"], dp["equity"]/dp["equity"].iloc[0], label="Daily Gate (normalized)")
            plt.plot(dp["timestamp"], dp["bh_equity"]/dp["bh_equity"].iloc[0], label="Buy&Hold (normalized)")
            plt.title(f"{sym} Equity: Daily Gate vs Buy&Hold")
            plt.legend()
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.show()

        if verbose and len(d_out):
            print(f"[{sym}] out_days={len(d_out)} trades={len(trades_out)} final_eq={d_out['equity'].iloc[-1]:.2f}")

        results[sym] = (d_out, trades_out)

    if verbose:
        y = np.asarray(y_all, dtype=int) if len(y_all) else np.array([], dtype=int)
        if len(y):
            print(f"[LABELS][POOLED] n={len(y)} pos={int(y.sum())} ({y.mean():.2%})")
        print(f"[MODEL][POOLED] trained={'yes' if st.model is not None else 'no'} trained_n={st.trained_n}")
        for sym in asset_csvs.keys():
            print(f"[{sym}] total_daily_bsp={len(bsp_rows[sym])}")

    return results, st.model


# ============================================================
# RUN
# ============================================================
asset_results, trained_model = run_unified_daily_gate_with_daily_chan_bsp_pooled(
    asset_csvs={
        "SPY": "DataAPI/data/SPY_DAY.csv",
        "QQQ": "DataAPI/data/QQQ_DAY.csv",
    },
    daily_start="2015-06-01",

    plot_start="2016-09-01",
    plot_end="2024-12-31",
    output_start="2016-09-01",
    output_end="2024-12-31",

    N=5,

    dd_thresh=-0.03,
    rv_quantile=0.80,
    rv_ref_lookback=252,

    min_labeled_days_to_train=400,
    retrain_every_new_labels=50,

    p_low_enter=0.30,
    p_low_exit=0.36,
    p_high_enter=0.70,
    p_high_exit=0.62,

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
)

# Access:
# spy_daily, spy_trades = asset_results["SPY"]
# qqq_daily, qqq_trades = asset_results["QQQ"]
