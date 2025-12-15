import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import timedelta

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sliding_window_chan import SlidingWindowChan
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE, DATA_SRC, AUTYPE
from DataAPI.csvAPI import CSV_API  # ✅ we stream K-lines ourselves


# ============================================================
# 1. Labeling: BEST profit target within configurable window
# ============================================================

def calculate_profit_targets_best_window(
    bs_points_with_features: list,
    lookahead_days: float = 1.0,
) -> dict:
    """
    For each BSP, find the BEST reverse signal within a configurable time window.

    Parameters
    ----------
    bs_points_with_features : list[dict]
        BSP snapshots, each row must contain:
            - 'timestamp'
            - 'is_buy' (1 for buy, 0 for sell)
            - 'klu_idx'
            - 'klu_close'
    lookahead_days : float
        How many *calendar days* ahead to look for exits.
        - 1.0  => 24 hours
        - 0.5  => 12 hours
        - 2.0  => 48 hours, etc.

    Logic
    -----
    BUY BSP:
        - search later SELL BSPs with
              entry_ts < ts <= entry_ts + lookahead_days
        - choose EXIT with MAX klu_close (max profit)

    SELL BSP:
        - search later BUY BSPs within same time window
        - choose EXIT with MIN klu_close (max profit)

    If no reverse BSP within the window => has_profit_target = 0.
    """
    if not bs_points_with_features:
        return {}

    # --- normalize timestamps & sort ---
    enriched = []
    for row in bs_points_with_features:
        ts = row.get("timestamp")
        ts = pd.to_datetime(str(ts))
        new_row = dict(row)
        new_row["_ts"] = ts
        enriched.append(new_row)

    enriched.sort(key=lambda x: x["_ts"])

    profit_targets = {}
    n = len(enriched)

    for i in range(n):
        cur = enriched[i]
        entry_idx = cur["klu_idx"]
        entry_price = cur["klu_close"]
        is_buy = cur["is_buy"]
        entry_ts = cur["_ts"]

        horizon_ts = entry_ts + timedelta(days=lookahead_days)
        best_exit = None

        # scan forward within horizon
        for j in range(i + 1, n):
            future = enriched[j]
            future_ts = future["_ts"]

            # stop if beyond horizon
            if future_ts > horizon_ts:
                break

            # only reverse direction
            if future["is_buy"] == is_buy:
                continue

            if is_buy:
                # BUY -> want SELL with highest price
                if best_exit is None or future["klu_close"] > best_exit["klu_close"]:
                    best_exit = future
            else:
                # SELL -> want BUY with lowest price
                if best_exit is None or future["klu_close"] < best_exit["klu_close"]:
                    best_exit = future

        if best_exit is None:
            profit_targets[entry_idx] = {
                "profit_target_pct": None,
                "profit_target_distance": None,
                "has_profit_target": 0,
                "exit_type": None,
                "exit_klu_idx": None,
                "exit_price": None,
            }
            continue

        exit_price = best_exit["klu_close"]
        if is_buy:
            profit_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            profit_pct = (entry_price - exit_price) / entry_price * 100.0

        profit_targets[entry_idx] = {
            "profit_target_pct": profit_pct,
            "profit_target_distance": best_exit["klu_idx"] - entry_idx,
            "has_profit_target": 1,
            "exit_type": f"best_within_{lookahead_days}d",
            "exit_klu_idx": best_exit["klu_idx"],
            "exit_price": exit_price,
        }

    return profit_targets


# ============================================================
# 2. Dataset preparation (no future leak in FEATURES)
# ============================================================

def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare BSP dataset for ML:

    - Impute numeric & binary columns
    - Encode 'direction' as categorical if present
    - Replace inf with nan then 0
    """
    if df.empty:
        return df

    print("[🔧] Preparing ML dataset...")

    # Columns considered binary flags
    binary_cols = [
        col for col in df.columns
        if col.endswith(("_signal", "_oversold", "_overbought", "_positive", "_up", "_trend_up"))
    ]
    # extend explicit binary
    for col in ["is_buy", "is_bullish_candle", "has_profit_target"]:
        if col in df.columns:
            binary_cols.append(col)
    binary_cols = sorted(set(binary_cols))

    # numeric columns
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in []  # keep all numeric for now
    ]

    # Impute numeric
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Impute binary (if exists)
    if binary_cols:
        existing_binary = [c for c in binary_cols if c in df.columns]
        if existing_binary:
            bin_imputer = SimpleImputer(strategy="most_frequent")
            df[existing_binary] = bin_imputer.fit_transform(df[existing_binary])

    # Categorical: only 'direction' here
    if "direction" in df.columns:
        df["direction"] = df["direction"].fillna("unknown").astype(str)
        le = LabelEncoder()
        df["direction_encoded"] = le.fit_transform(df["direction"])

    # Replace infinities then re-impute numeric as 0 on leftover nans
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols_final = df.select_dtypes(include=[np.number]).columns
    df[num_cols_final] = df[num_cols_final].fillna(0.0)

    print(f"[✅] Dataset prepared: {len(df)} samples, {len(df.columns)} columns")
    return df


# ============================================================
# 3. Feature selection & threshold optimization
# ============================================================

def get_feature_columns(df: pd.DataFrame, target_col: str = "profit_target_pct") -> list:
    """
    Choose feature columns from BSP dataset, excluding:
      - timestamps, IDs, explicit labels
      - exit_* fields, etc.
    """
    exclude_patterns = [
        "timestamp",
        "bsp_type",
        "snapshot_",
        "exit_",
        "klu_idx",
        "date",
    ]
    extra_exclude = [
        target_col,
        "profit_target_pct",
        "profit_target_distance",
        "has_profit_target",
        "price_change_pct",
        "direction",  # keep encoded version only
    ]
    cols = []
    for col in df.columns:
        if col in extra_exclude:
            continue
        if any(p in col for p in exclude_patterns):
            continue
        cols.append(col)
    # prefer encoded direction if available
    if "direction" in cols and "direction_encoded" in cols:
        cols.remove("direction")
    return sorted(cols)


def optimize_thresholds_with_fee(signals_df: pd.DataFrame,
                                 transaction_fee_pct: float,
                                 min_trades: int = 5) -> tuple:
    """
    Grid search thresholds using *realized* profit_target_pct
    (works only on past days with labels).

    We separate buy/sell BSPs, scan thresholds on predicted_profit_pct,
    and choose the pair that maximizes average net return after fees.
    """
    if len(signals_df) == 0:
        return 0.0, 0.0

    if "profit_target_pct" not in signals_df.columns:
        print("[⚠️] optimize_thresholds_with_fee: missing profit_target_pct, using 0/0 thresholds.")
        return 0.0, 0.0

    thresholds = np.linspace(-0.5, 3.0, 29)  # tune as needed
    best_score = -1e9
    best_bt = 0.0
    best_st = 0.0
    roundtrip_fee_pct = 2 * transaction_fee_pct * 100.0

    buy_df = signals_df[signals_df["direction"] == "buy"]
    sell_df = signals_df[signals_df["direction"] == "sell"]

    for bt in thresholds:
        for st in thresholds:
            val_b = buy_df[buy_df["predicted_profit_pct"] >= bt]
            val_s = sell_df[sell_df["predicted_profit_pct"] >= st]

            n_b, n_s = len(val_b), len(val_s)
            tot = n_b + n_s
            if tot < min_trades:
                continue

            b_ret = (val_b["profit_target_pct"] - roundtrip_fee_pct).mean() if n_b > 0 else 0.0
            s_ret = (val_s["profit_target_pct"] - roundtrip_fee_pct).mean() if n_s > 0 else 0.0

            combined = (n_b * b_ret + n_s * s_ret) / tot

            if combined > best_score:
                best_score = combined
                best_bt = bt
                best_st = st

    return best_bt, best_st


# ============================================================
# 3.5. Market regime adjustment for thresholds
# ============================================================

def apply_market_regime_adjustment(buy_th: float,
                                   sell_th: float,
                                   regime: str = "neutral",
                                   base_shift: float = 0.25) -> tuple[float, float]:
    """
    Adjust thresholds based on discretionary market regime.

    Parameters
    ----------
    buy_th, sell_th : float
        Original thresholds from optimize_thresholds_with_fee (in % predicted profit).
    regime : str
        One of: "neutral", "bullish", "strong_bullish",
                "bearish", "strong_bearish"
    base_shift : float
        The basic shift magnitude in percentage points.

    Returns
    -------
    (adj_buy_th, adj_sell_th)
    """
    regime = (regime or "neutral").lower()

    if regime == "neutral":
        d_buy = 0.0
        d_sell = 0.0
    elif regime == "bullish":
        d_buy = -base_shift          # easier to buy
        d_sell = +base_shift / 2.0   # slightly harder to sell
    elif regime == "strong_bullish":
        d_buy = -2 * base_shift
        d_sell = +base_shift
    elif regime == "bearish":
        d_buy = +base_shift          # harder to buy
        d_sell = -base_shift / 2.0   # easier to sell
    elif regime == "strong_bearish":
        d_buy = +2 * base_shift
        d_sell = -base_shift
    else:
        # unknown regime -> no adjustment
        d_buy = 0.0
        d_sell = 0.0

    adj_buy = buy_th + d_buy
    adj_sell = sell_th + d_sell

    # clamp to reasonable range
    adj_buy = max(-1.0, min(4.0, adj_buy))
    adj_sell = max(-1.0, min(4.0, adj_sell))

    return adj_buy, adj_sell


# ============================================================
# 4. Per-day backtest (intraday BSP execution) + Stop-loss
# ============================================================

def backtest_one_day_signals(signals_df: pd.DataFrame,
                             buy_threshold: float,
                             sell_threshold: float,
                             initial_capital: float,
                             position_size: float,
                             transaction_fee_pct: float,
                             stop_loss_pct: float | None = None,
                             verbose: bool = False):
    """
    Intraday backtest using BSP signals with predicted_profit_pct.

    Simple one-position strategy:
      - Only one long position at a time.
      - BUY BSP opens, SELL BSP closes (if thresholds passed).
      - Optional stop loss based on % loss from entry price.

    stop_loss_pct:
      - If None: no stop loss
      - If 2.0: close position when return <= -2% from entry

    IMPORTANT CHANGE:
      - We DO NOT force closing at end-of-day anymore.
      - We mark-to-market open positions: final_value = cash + shares * last_price.
      - Open trades with no exit are left in the trades list without exit_time/exit_price.
    """
    if len(signals_df) == 0:
        return {
            "final_value": initial_capital,
            "return_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
        }, []

    df = signals_df.sort_values("timestamp").reset_index(drop=True)

    cash = initial_capital
    shares = 0.0
    entry_price = None
    entry_time = None
    trades = []
    per_side_fee = transaction_fee_pct
    last_price_in_day = None  # for mark-to-market

    def open_position(price, time_):
        nonlocal cash, shares, entry_price, entry_time
        trade_cap = cash * position_size
        if trade_cap <= 0:
            return False
        effective_capital = trade_cap * (1 - per_side_fee)
        shares = effective_capital / price
        cash -= trade_cap
        entry_price = price
        entry_time = time_
        trades.append({
            "entry_time": time_,
            "entry_price": price,
            "direction": "buy",
        })
        if verbose:
            print(f"[TRADE] BUY at {time_} | price = {price:.4f}")
        return True

    def close_position(price, time_, reason: str = "signal"):
        nonlocal cash, shares, entry_price, entry_time
        if shares <= 0:
            return
        gross_exit = shares * price
        net_exit = gross_exit * (1 - per_side_fee)
        trade_cap = shares * entry_price if entry_price is not None else 0.0
        pnl = net_exit - trade_cap
        ret_pct = (pnl / trade_cap * 100) if trade_cap > 0 else 0.0
        cash += net_exit
        if trades:
            trades[-1].update({
                "exit_time": time_,
                "exit_price": price,
                "pnl": pnl,
                "return_pct": ret_pct,
                "exit_reason": reason,
            })
        if verbose:
            print(
                f"[TRADE] SELL ({reason}) at {time_} | price = {price:.4f} | "
                f"trade return = {ret_pct:.2f}%"
            )
        shares = 0.0
        entry_price = None
        entry_time = None

    for _, row in df.iterrows():
        ts = row["timestamp"]
        price = row["klu_close"]
        pred = row["predicted_profit_pct"]
        direction = row["direction"]

        # update last price of day for mark-to-market
        last_price_in_day = price

        # 🔹 Check stop loss first
        if shares > 0 and stop_loss_pct is not None and entry_price is not None:
            current_ret = (price - entry_price) / entry_price * 100.0
            if current_ret <= -abs(stop_loss_pct):
                close_position(price, ts, reason="stop_loss")
                # Once stopped out, skip using this bar as a new signal
                continue

        # Then follow model signals
        if direction == "buy":
            if shares == 0 and pred >= buy_threshold:
                open_position(price, ts)
        elif direction == "sell":
            if shares > 0 and pred >= sell_threshold:
                close_position(price, ts, reason="signal")

    # ❌ No forced end-of-day close here.
    # ✅ Mark-to-market open positions:
    if last_price_in_day is None:
        last_price_in_day = df["klu_close"].iloc[-1]

    final_value = cash + (shares * last_price_in_day if shares > 0 else 0.0)
    ret_pct = (final_value / initial_capital - 1) * 100 if initial_capital > 0 else 0.0

    completed = [t for t in trades if "exit_price" in t]
    win_trades = [t for t in completed if t.get("return_pct", 0) > 0]
    win_rate = (len(win_trades) / len(completed) * 100) if completed else 0.0

    buy_signals = (df["direction"] == "buy").sum()
    sell_signals = (df["direction"] == "sell").sum()

    result = {
        "final_value": final_value,
        "return_pct": ret_pct,
        "trades": len(completed),
        "win_rate": win_rate,
        "signals": len(df),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
    }
    if verbose:
        print(
            f"    Day summary: trades={len(completed)}, "
            f"win_rate={win_rate:.1f}%, day_return={ret_pct:.2f}%"
        )
    return result, trades


# ============================================================
# 5. Buy & Hold benchmark
# ============================================================

def calc_buy_hold_daily(csv_path: str,
                        start_time: str,
                        end_time: str) -> pd.DataFrame:
    """
    Compute daily open/close and cumulative return for Buy&Hold from raw kline CSV.
    """
    raw = pd.read_csv(csv_path)

    # Detect timestamp column
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    else:
        first_col = raw.columns[0]
        raw["timestamp"] = pd.to_datetime(raw[first_col], errors="coerce")

    raw = raw.dropna(subset=["timestamp"])

    # Detect close column
    close_col = None
    for c in ["close", "Close", "adj_close", "Adj Close", "AdjClose"]:
        if c in raw.columns:
            close_col = c
            break
    if close_col is None:
        # fallback: any column ending with 'close'
        for c in raw.columns:
            if c.lower().endswith("close"):
                close_col = c
                break
    if close_col is None:
        raise ValueError("Could not find a close column in CSV.")

    mask = (raw["timestamp"] >= pd.to_datetime(start_time)) & (raw["timestamp"] <= pd.to_datetime(end_time))
    raw = raw[mask].copy()

    raw["date"] = raw["timestamp"].dt.date
    daily = raw.groupby("date")[close_col].agg(["first", "last"])
    if daily.empty:
        return pd.DataFrame(columns=["date", "open", "close", "return_pct", "cum_return_pct"])

    daily.rename(columns={"first": "open", "last": "close"}, inplace=True)
    daily["return_pct"] = (daily["close"] / daily["open"] - 1) * 100.0
    first_price = daily["open"].iloc[0]
    daily["cum_return_pct"] = (daily["close"] / first_price - 1) * 100.0

    return daily.reset_index().rename(columns={"date": "date"})


# ============================================================
# 5.5. Build dataset from snapshots with lookahead window
# ============================================================

def _build_dataset_from_snapshots(
    all_bsp_snapshots: list,
    label_lookahead_days: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From the accumulated BSP snapshots (up to 'now'), compute:
      - best profit labels within a configurable time window
      - a full BSP DataFrame (bsp_df)
      - a labeled subset (bsp_valid: has_profit_target == 1)

    label_lookahead_days controls how far we look for exits (24h, 48h, etc.).
    """
    if not all_bsp_snapshots:
        return pd.DataFrame(), pd.DataFrame()

    # 1) Label with best exits within N-day horizon
    profit_targets = calculate_profit_targets_best_window(
        all_bsp_snapshots,
        lookahead_days=label_lookahead_days,
    )
    for row in all_bsp_snapshots:
        idx = row["klu_idx"]
        if idx in profit_targets:
            row.update(profit_targets[idx])
        else:
            row["profit_target_pct"] = None
            row["profit_target_distance"] = None
            row["has_profit_target"] = 0
            row["exit_type"] = None
            row["exit_klu_idx"] = None
            row["exit_price"] = None

    # 2) Build DataFrame & time columns
    bsp_df_raw = (
        pd.DataFrame(all_bsp_snapshots)
        .sort_values("klu_idx")
        .reset_index(drop=True)
    )
    bsp_df_raw["timestamp"] = bsp_df_raw["timestamp"].apply(
        lambda t: pd.to_datetime(str(t))
    )
    bsp_df_raw["date"] = bsp_df_raw["timestamp"].dt.date

    # 3) Prepare ML dataset
    bsp_df = prepare_ml_dataset(bsp_df_raw.copy())

    # 4) Labeled subset
    if "has_profit_target" in bsp_df.columns:
        bsp_valid = (
            bsp_df[bsp_df["has_profit_target"] == 1]
            .copy()
            .sort_values("timestamp")
        )
    else:
        bsp_valid = pd.DataFrame()

    return bsp_df, bsp_valid


# ============================================================
# 6. Main: ONLINE-style walk-forward with warm-up
# ============================================================

def run_realtime_chan_xgb_same_day_online(
    csv_path: str,
    code: str = "SPY",
    start_time: str = "2022-01-01",
    end_time: str = "2022-03-31",
    lv: KL_TYPE = KL_TYPE.K_5M,
    chan_window_size: int = 1000,
    warmup_trading_days: int = 40,          # ≈2 months
    threshold_days_for_selection: int = 3,  # last N days for threshold tuning
    min_train_samples: int = 100,
    min_valid_samples: int = 20,
    initial_capital: float = 100_000.0,
    position_size: float = 1.0,
    transaction_fee_pct: float = 0.001,
    output_dir: str = "./output/chan_xgb_online",
    plot_results: bool = True,
    verbose: bool = True,
    xgb_train_days: int | None = None,
    label_lookahead_days: float = 1.0,
    stop_loss_pct: float | None = None,
    market_regime: str = "neutral",
):
    """
    ONE-PASS version:

    - Stream K-lines from CSV_API.
    - Feed every K-line into SlidingWindowChan.process_new_kline().
    - Accumulate BSP snapshots in all_bsp_snapshots.
    - At each *day boundary*:
        - Build dataset from all BSP up to that day (no future days involved).
        - Train XGBoost on PREVIOUS labeled days (respecting warm-up).
        - Tune thresholds on last N candidate training days.
        - Optionally adjust thresholds by human market regime.
        - Backtest the just-finished day using the models and thresholds.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Realtime Chan + XGBoost (ONLINE style, day-by-day interleaved)")
    print("=" * 80)
    print(f"Code: {code}")
    print(f"Period: {start_time} → {end_time}")
    print(f"Warm-up trading days (no trading): {warmup_trading_days}")
    print(f"Initial capital: {initial_capital}, Position size: {position_size*100:.0f}%")
    print(f"Transaction fee per side: {transaction_fee_pct*100:.3f}%")
    print(f"Label lookahead days: {label_lookahead_days}")
    print(f"Stop loss pct: {stop_loss_pct}")
    print(f"Market regime: {market_regime}")
    print("=" * 80)

    # ---------------- Chan config & SlidingWindowChan ----------------
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
        begin_time=None,       # we stream manually
        end_time=None,
        data_src=None,         # not using internal loader
        lv_list=[lv],
        config=config,
        autype=AUTYPE.QFQ,
        max_klines=chan_window_size,
    )

    # ---------------- Stream K-lines from CSV_API ----------------
    csv_api = CSV_API(
        code=code,
        k_type=lv,
        begin_date=start_time,
        end_date=end_time,
        autype=AUTYPE.QFQ,
    )

    from datetime import datetime as _dt

    all_bsp_snapshots = []   # all BSP snapshots up to "now"
    daily_results = []       # result per test day
    all_trades = []          # all trades from all test days

    current_day = None

    # To compute Buy&Hold later
    all_dates_seen = set()

    # ------------- inner helper: finalize a completed day -------------
    def finalize_day(finished_day, all_bsp_snapshots_local):
        """
        Called when we've just finished streaming all K-lines for `finished_day`.
        We:
          - build dataset from all BSP up to this moment,
          - decide if we can train,
          - if yes, train XGBoost and backtest this finished_day.
        """
        nonlocal daily_results, all_trades

        print(f"\n{'-'*60}")
        print(f"[DAY DONE] {finished_day} – building dataset & training...")

        bsp_df, bsp_valid = _build_dataset_from_snapshots(
            all_bsp_snapshots_local,
            label_lookahead_days=label_lookahead_days,
        )

        if bsp_df.empty or bsp_valid.empty:
            if verbose:
                print("    [Info] No labeled BSPs yet, skipping this day.")
            return bsp_df  # still return last snapshot of dataset

        # unique labeled days up to now
        labeled_days = np.sort(bsp_valid["date"].unique())
        if finished_day not in labeled_days:
            if verbose:
                print("    [Info] No labeled BSPs for this day itself, skipping.")
            return bsp_df

        # training days must be strictly BEFORE finished_day
        all_prior_days = [d for d in labeled_days if d < finished_day]

        # Warm-up condition uses ALL prior labeled days
        if len(all_prior_days) < warmup_trading_days:
            if verbose:
                print(
                    f"    [Warmup] Only {len(all_prior_days)} training days "
                    f"(< {warmup_trading_days}). No trading on {finished_day}."
                )
            return bsp_df

        # If xgb_train_days is set, restrict to most recent candidate days
        if xgb_train_days is not None and len(all_prior_days) > xgb_train_days:
            candidate_days = all_prior_days[-xgb_train_days:]
        else:
            candidate_days = all_prior_days

        # Split candidate_days into:
        #   - thr_days: last N threshold_days_for_selection days
        #   - train_days: the earlier part (exclude thr_days)
        if len(candidate_days) <= threshold_days_for_selection:
            thr_days = candidate_days[:]  # all used as threshold days
            train_days = []
        else:
            thr_days = candidate_days[-threshold_days_for_selection:]
            train_days = candidate_days[:-threshold_days_for_selection]

        if verbose:
            print(f"    Candidate days: {candidate_days[0]} → {candidate_days[-1]} ({len(candidate_days)})")
            if train_days:
                print(f"    Train days:    {train_days[0]} → {train_days[-1]} ({len(train_days)})")
            else:
                print(f"    Train days:    [NONE; all used as threshold days]")
            print(f"    Threshold days:{thr_days[0]} → {thr_days[-1]} ({len(thr_days)})")

        # If no training days after excluding thr_days, skip
        if len(train_days) == 0:
            if verbose:
                print("    [Skip] No remaining train days after excluding threshold days.")
            return bsp_df

        train_df = bsp_valid[bsp_valid["date"].isin(train_days)].copy()
        thr_df = bsp_valid[bsp_valid["date"].isin(thr_days)].copy()
        test_df = bsp_df[bsp_df["date"] == finished_day].copy()  # BSPs of the finished day

        if len(train_df) < min_train_samples or len(thr_df) < min_valid_samples:
            if verbose:
                print("    [Skip] Not enough samples for train/threshold.")
            return bsp_df

        train_buy = train_df[train_df["direction"] == "buy"]
        train_sell = train_df[train_df["direction"] == "sell"]
        if len(train_buy) < 10 or len(train_sell) < 10:
            if verbose:
                print("    [Skip] Not enough buy/sell samples.")
            return bsp_df

        feature_cols = get_feature_columns(train_df, target_col="profit_target_pct")
        if verbose:
            print(f"    [🧬] Using {len(feature_cols)} feature columns.")

        X_train_buy = train_buy[feature_cols].values
        y_train_buy = train_buy["profit_target_pct"].values
        X_train_sell = train_sell[feature_cols].values
        y_train_sell = train_sell["profit_target_pct"].values

        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 150,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        buy_model = xgb.XGBRegressor(**params)
        sell_model = xgb.XGBRegressor(**params)

        buy_model.fit(X_train_buy, y_train_buy)
        sell_model.fit(X_train_sell, y_train_sell)

        print(
            f"[MODEL] {finished_day} | XGBoost trained on "
            f"buy={len(train_buy)} / sell={len(train_sell)} samples"
        )

        # Threshold tuning on thr_days
        thr_buy = thr_df[thr_df["direction"] == "buy"].copy()
        thr_sell = thr_df[thr_df["direction"] == "sell"].copy()
        if len(thr_buy) > 0:
            thr_buy["predicted_profit_pct"] = buy_model.predict(thr_buy[feature_cols].values)
        if len(thr_sell) > 0:
            thr_sell["predicted_profit_pct"] = sell_model.predict(thr_sell[feature_cols].values)
        thr_signals = pd.concat([thr_buy, thr_sell], ignore_index=True)

        buy_th_raw, sell_th_raw = optimize_thresholds_with_fee(
            thr_signals,
            transaction_fee_pct=transaction_fee_pct,
            min_trades=5,
        )

        # Apply discretionary regime overlay
        buy_th_adj, sell_th_adj = apply_market_regime_adjustment(
            buy_th_raw,
            sell_th_raw,
            regime=market_regime,
            base_shift=0.25,
        )

        if verbose:
            print(f"    [Thresholds] raw: Buy={buy_th_raw:.2f}%, Sell={sell_th_raw:.2f}%")
            if (buy_th_adj, sell_th_adj) != (buy_th_raw, sell_th_raw):
                print(f"                 adj: Buy={buy_th_adj:.2f}%, Sell={sell_th_adj:.2f}% "
                      f"(regime={market_regime})")

        # Backtest the finished day using only that day's BSPs
        test_buy = test_df[test_df["direction"] == "buy"].copy()
        test_sell = test_df[test_df["direction"] == "sell"].copy()
        if len(test_buy) > 0:
            test_buy["predicted_profit_pct"] = buy_model.predict(test_buy[feature_cols].values)
        if len(test_sell) > 0:
            test_sell["predicted_profit_pct"] = sell_model.predict(test_sell[feature_cols].values)
        test_signals = pd.concat([test_buy, test_sell], ignore_index=True)

        if len(test_signals) == 0:
            if verbose:
                print("    [Info] No BSP signals on test day.")
            return bsp_df

        bt_res, trades = backtest_one_day_signals(
            test_signals,
            buy_threshold=buy_th_adj,
            sell_threshold=sell_th_adj,
            initial_capital=initial_capital,
            position_size=position_size,
            transaction_fee_pct=transaction_fee_pct,
            stop_loss_pct=stop_loss_pct,
            verbose=verbose,
        )

        for t in trades:
            t["test_date"] = finished_day
        all_trades.extend(trades)

        daily_results.append({
            "date": finished_day,
            "train_start": train_days[0],
            "train_end": train_days[-1],
            "strategy_return": bt_res["return_pct"],
            "signals": bt_res["signals"],
            "trades": bt_res["trades"],
            "win_rate": bt_res["win_rate"],
            "buy_threshold": buy_th_adj,
            "sell_threshold": sell_th_adj,
        })

        return bsp_df  # return last dataset snapshot for final export

    # ---------------- main streaming loop ----------------
    print("[🧪] Streaming K-lines into SlidingWindowChan and training per day...")
    last_log = _dt.now()
    bsp_df_final = pd.DataFrame()  # last snapshot, for return / CSV

    for klu_idx, klu in enumerate(csv_api.get_kl_data()):
        # basic metadata for Chan
        klu.kl_type = lv
        klu.set_idx(klu_idx)

        raw_ts = klu.time
        ts = pd.to_datetime(str(raw_ts))  # always via string
        d = ts.date()
        all_dates_seen.add(d)

        # Day rollover?
        if current_day is None:
            current_day = d
        elif d != current_day:
            # We've just finished current_day
            bsp_df_final = finalize_day(current_day, all_bsp_snapshots)
            current_day = d  # start new day

        # Process this K-line through Chan
        window_chan, new_bsp_list = chan.process_new_kline(klu)

        # Accumulate BSP snapshots
        if new_bsp_list:
            all_bsp_snapshots.extend(new_bsp_list)

        # For logging
        now = _dt.now()
        if klu_idx % 500 == 0 or (now - last_log).total_seconds() > 5:
            stats = chan.get_stats()
            print(
                f"[📈] K-line {klu_idx}: unique BSP={stats['unique_bsp_count']}, "
                f"window_start={stats['window_start_idx']}, "
                f"window_size={stats['current_window_size']}"
            )
            last_log = now

    # After loop: finalize the last day
    if current_day is not None and all_bsp_snapshots:
        bsp_df_final = finalize_day(current_day, all_bsp_snapshots)

    # ---------------- Build daily_results_df & benchmark ----------------
    if len(daily_results) == 0:
        print("[❌] No valid daily trading results (warm-up might be too long).")
        return {
            "daily_results_df": pd.DataFrame(),
            "trades_df": pd.DataFrame(),
            "bsp_df": bsp_df_final,
            "bh_df": pd.DataFrame(),
        }

    daily_results_df = (
        pd.DataFrame(daily_results)
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Buy & Hold benchmark (full period)
    bh_df = calc_buy_hold_daily(csv_path, start_time, end_time)

    if not bh_df.empty:
        # Map original buy&hold cumulative returns to the test days
        bh_map = bh_df.set_index("date")["cum_return_pct"].to_dict()
        daily_results_df["buy_hold_cum_return"] = daily_results_df["date"].map(bh_map)
    else:
        daily_results_df["buy_hold_cum_return"] = np.nan

    # Strategy cumulative return
    daily_results_df["strategy_cum_return"] = (
        1 + daily_results_df["strategy_return"] / 100.0
    ).cumprod() * 100.0 - 100.0

    # 🔁 Rebase BOTH curves so the first test day is 0%
    if not daily_results_df.empty:
        # Strategy
        first_strat = daily_results_df["strategy_cum_return"].iloc[0]
        daily_results_df["strategy_cum_return"] = (
            daily_results_df["strategy_cum_return"] - first_strat
        )

        # Buy & Hold (only if it exists)
        if daily_results_df["buy_hold_cum_return"].notna().any():
            first_bh = daily_results_df["buy_hold_cum_return"].dropna().iloc[0]
            daily_results_df["buy_hold_cum_return"] = (
                daily_results_df["buy_hold_cum_return"] - first_bh
            )

    # Plot
    if plot_results and not daily_results_df.empty:
        plt.figure(figsize=(12, 6))
        dates = pd.to_datetime(daily_results_df["date"])
        plt.plot(dates, daily_results_df["strategy_cum_return"], label="Strategy", linewidth=2)
        if "buy_hold_cum_return" in daily_results_df.columns:
            plt.plot(dates, daily_results_df["buy_hold_cum_return"], label="Buy & Hold", linewidth=2)
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.title("Cumulative Return: Strategy vs Buy & Hold")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    trades_df = pd.DataFrame(all_trades)

    if verbose:
        print("\n" + "=" * 80)
        print("Realtime Chan + XGBoost (ONLINE style, day-by-day) Complete")
        print("=" * 80)
        print(f"Total test days: {len(daily_results_df)}")
        print(f"Total completed trades: {len(trades_df)}")
        if "strategy_cum_return" in daily_results_df.columns:
            print(f"Final strategy cumulative return: "
                  f"{daily_results_df['strategy_cum_return'].iloc[-1]:.2f}%")
        if "buy_hold_cum_return" in daily_results_df.columns:
            print(f"Final buy & hold cumulative return: "
                  f"{daily_results_df['buy_hold_cum_return'].iloc[-1]:.2f}%")
        print("=" * 80)

    # Save CSVs
    daily_results_df.to_csv(os.path.join(output_dir, "daily_results_online.csv"), index=False)
    trades_df.to_csv(os.path.join(output_dir, "trades_online.csv"), index=False)
    if not bsp_df_final.empty:
        bsp_df_final.to_csv(os.path.join(output_dir, "bsp_dataset_used_online.csv"), index=False)

    return {
        "daily_results_df": daily_results_df,
        "trades_df": trades_df,
        "bsp_df": bsp_df_final,
        "bh_df": bh_df,
    }


if __name__ == "__main__":
    from Common.CEnum import KL_TYPE

    results = run_realtime_chan_xgb_same_day_online(
        csv_path="DataAPI/data/QQQ_5M.csv",
        code="QQQ",
        start_time="2023-01-01",
        end_time="2023-12-30",
        lv=KL_TYPE.K_5M,
        chan_window_size=500,
        warmup_trading_days=30,
        threshold_days_for_selection=2,
        min_train_samples=100,
        min_valid_samples=20,
        initial_capital=100_000,
        position_size=1.0,
        transaction_fee_pct=0,
        output_dir="./output/chan_xgb_online",
        plot_results=True,
        verbose=True,
        xgb_train_days=60,
        label_lookahead_days=2.0,
        stop_loss_pct=3.0,             # 2% stop loss
        market_regime="bullish",       # "neutral", "bullish", "strong_bullish", "bearish", "strong_bearish"
    )

    trades_df = results["trades_df"]
    daily_df = results["daily_results_df"]
    print(trades_df.head())
    print(daily_df.head())
