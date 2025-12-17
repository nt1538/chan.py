import os
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sliding_window_chan import SlidingWindowChan
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE, AUTYPE
from DataAPI.csvAPI import CSV_API
import matplotlib.pyplot as plt


# ============================================================
# 0) Raw Kline helper: next-bar execution + day close + calendar days
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
    low_col = pick_col(["low", "Low"])  # optional, for stop-loss "low" trigger
    if open_col is None or close_col is None:
        raise ValueError("Raw CSV must contain open and close columns.")

    mask = (raw["timestamp"] >= pd.to_datetime(start_time)) & (raw["timestamp"] <= pd.to_datetime(end_time))
    raw = raw.loc[mask].copy().reset_index(drop=True)

    raw["date"] = raw["timestamp"].dt.date
    raw["next_open"] = raw[open_col].shift(-1)
    raw["next_close"] = raw[close_col].shift(-1)

    next_open_by_idx = raw["next_open"].to_numpy()
    next_close_by_idx = raw["next_close"].to_numpy()

    # also keep current close/low arrays for stop checks
    cur_close_by_idx = raw[close_col].to_numpy()
    cur_low_by_idx = raw[low_col].to_numpy() if low_col is not None else None

    day_close_map = raw.groupby("date")[close_col].last().to_dict()
    all_days = sorted(raw["date"].unique())

    return raw, next_open_by_idx, next_close_by_idx, cur_close_by_idx, cur_low_by_idx, day_close_map, all_days


# ============================================================
# 1) Labeling: BEST reverse within lookahead window (uses timestamps)
# ============================================================

def calculate_profit_targets_best_window(
    bs_points_with_features: list,
    lookahead_days: float = 1.0,
) -> dict:
    """
    For each BSP, find the BEST reverse signal within lookahead_days.
    BUY -> pick SELL with max close
    SELL -> pick BUY with min close
    """
    if not bs_points_with_features:
        return {}

    enriched = []
    for row in bs_points_with_features:
        ts = pd.to_datetime(str(row.get("timestamp")))
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
        for j in range(i + 1, n):
            future = enriched[j]
            if future["_ts"] > horizon_ts:
                break
            if future["is_buy"] == is_buy:
                continue

            if is_buy:
                if best_exit is None or future["klu_close"] > best_exit["klu_close"]:
                    best_exit = future
            else:
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
# 2) Dataset prep (feature cleaning)
# ============================================================

def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    binary_cols = [
        col for col in df.columns
        if col.endswith(("_signal", "_oversold", "_overbought", "_positive", "_up", "_trend_up"))
    ]
    for col in ["is_buy", "is_bullish_candle", "has_profit_target"]:
        if col in df.columns:
            binary_cols.append(col)
    binary_cols = sorted(set(binary_cols))

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    if numeric_cols:
        df[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(df[numeric_cols])

    existing_binary = [c for c in binary_cols if c in df.columns]
    if existing_binary:
        df[existing_binary] = SimpleImputer(strategy="most_frequent").fit_transform(df[existing_binary])

    if "direction" in df.columns:
        df["direction"] = df["direction"].fillna("unknown").astype(str)
        le = LabelEncoder()
        df["direction_encoded"] = le.fit_transform(df["direction"])

    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols_final = df.select_dtypes(include=[np.number]).columns
    df[num_cols_final] = df[num_cols_final].fillna(0.0)

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "profit_target_pct") -> list:
    exclude_patterns = ["timestamp", "bsp_type", "snapshot_", "exit_", "klu_idx", "date"]
    extra_exclude = [
        target_col,
        "profit_target_pct",
        "profit_target_distance",
        "has_profit_target",
        "direction",
        "snapshot_first_seen_ts",  # do NOT include as feature
    ]
    cols = []
    for col in df.columns:
        if col in extra_exclude:
            continue
        if any(p in col for p in exclude_patterns):
            continue
        cols.append(col)
    if "direction" in cols and "direction_encoded" in cols:
        cols.remove("direction")
    return sorted(cols)


# ============================================================
# 3) Threshold optimizer (fee optional)
# ============================================================

def optimize_thresholds(signals_df: pd.DataFrame,
                        transaction_fee_pct: float = 0.0,
                        min_trades: int = 10) -> tuple[float, float]:
    if len(signals_df) == 0 or "profit_target_pct" not in signals_df.columns:
        return 0.0, 0.0

    thresholds = np.linspace(-0.5, 3.0, 29)
    roundtrip_fee_pct = 2 * transaction_fee_pct * 100.0

    best_score = -1e18
    best_bt, best_st = 0.0, 0.0

    buy_df = signals_df[signals_df["direction"] == "buy"]
    sell_df = signals_df[signals_df["direction"] == "sell"]

    for bt in thresholds:
        for st in thresholds:
            vb = buy_df[buy_df["predicted_profit_pct"] >= bt]
            vs = sell_df[sell_df["predicted_profit_pct"] >= st]
            tot = len(vb) + len(vs)
            if tot < min_trades:
                continue

            b_ret = (vb["profit_target_pct"] - roundtrip_fee_pct).mean() if len(vb) else 0.0
            s_ret = (vs["profit_target_pct"] - roundtrip_fee_pct).mean() if len(vs) else 0.0
            combined = (len(vb) * b_ret + len(vs) * s_ret) / tot

            if combined > best_score:
                best_score = combined
                best_bt, best_st = bt, st

    return best_bt, best_st


# ============================================================
# 4) Intraday LIVE-like execution engine (next bar open/close) + STOP LOSS
# ============================================================

class LiveExecutionEngine:
    def __init__(self,
                 initial_capital: float,
                 position_size: float,
                 fee_pct: float,
                 stop_loss_pct: float = 0.0):
        """
        stop_loss_pct: e.g. 2.0 means -2% from entry triggers a stop (long-only).
        """
        self.cash = float(initial_capital)
        self.position_size = float(position_size)
        self.fee_pct = float(fee_pct)

        self.shares = 0.0
        self.entry_px = None
        self.entry_idx = None

        self.stop_loss_pct = float(stop_loss_pct)

        self.pending_order = None  # {"side":..., "seen_idx":..., "reason":...}
        self.trades = []

    def _exec_price_by_idx(self, seen_idx: int, next_open_by_idx, next_close_by_idx, mode="next_open"):
        if mode == "next_open":
            nxt = next_open_by_idx[seen_idx] if 0 <= seen_idx < len(next_open_by_idx) else None
        else:
            nxt = next_close_by_idx[seen_idx] if 0 <= seen_idx < len(next_close_by_idx) else None

        if nxt is None or (isinstance(nxt, float) and np.isnan(nxt)):
            return None
        return float(nxt)

    def place_order_for_next_bar(self, side: str, seen_idx: int, reason: str, overwrite: bool = False):
        """
        overwrite=True lets STOP override an existing pending order.
        """
        if (self.pending_order is not None) and (not overwrite):
            return
        self.pending_order = {"side": side, "seen_idx": int(seen_idx), "reason": reason}

    def has_pending_sell(self) -> bool:
        return self.pending_order is not None and self.pending_order.get("side") == "sell"

    def maybe_execute_pending(self, next_open_by_idx, next_close_by_idx, execution_mode="next_open", verbose=False):
        if self.pending_order is None:
            return

        side = self.pending_order["side"]
        seen_idx = self.pending_order["seen_idx"]
        reason = self.pending_order["reason"]

        px = self._exec_price_by_idx(seen_idx, next_open_by_idx, next_close_by_idx, mode=execution_mode)
        if px is None:
            return

        if side == "buy" and self.shares == 0:
            trade_cap = self.cash * self.position_size
            if trade_cap <= 0:
                self.pending_order = None
                return
            net_cap = trade_cap * (1 - self.fee_pct)
            self.shares = net_cap / px
            self.cash -= trade_cap
            self.entry_px = px
            self.entry_idx = seen_idx
            self.trades.append({"entry_idx": seen_idx, "entry_price": px})
            if verbose:
                print(f"[EXEC] BUY @ {px:.4f} | seen_idx={seen_idx} | reason={reason}")

        elif side == "sell" and self.shares > 0 and self.entry_px is not None:
            gross = self.shares * px
            net = gross * (1 - self.fee_pct)
            trade_cap = self.shares * self.entry_px
            pnl = net - trade_cap
            ret = pnl / trade_cap * 100.0 if trade_cap > 0 else 0.0

            self.cash += net
            if self.trades:
                self.trades[-1].update({
                    "exit_idx": seen_idx,
                    "exit_price": px,
                    "pnl": pnl,
                    "return_pct": ret,
                    "exit_reason": reason,
                })

            if verbose:
                print(f"[EXEC] SELL @ {px:.4f} | seen_idx={seen_idx} | reason={reason} | ret={ret:.2f}%")

            self.shares = 0.0
            self.entry_px = None
            self.entry_idx = None

        self.pending_order = None

    def check_stop_and_place_sell(self,
                                 seen_idx: int,
                                 current_price: float,
                                 execution_mode: str,
                                 stop_price_source: str = "close",
                                 verbose: bool = False):
        """
        Long-only stop-loss:
        - Trigger when (current_price - entry_px)/entry_px <= -stop_loss_pct/100.
        - If triggered, place SELL for next bar execution.
        stop_price_source is just for labeling/debug; current_price is what you pass in.
        """
        if self.stop_loss_pct <= 0:
            return
        if self.shares <= 0 or self.entry_px is None:
            return
        if self.has_pending_sell():
            return

        entry = float(self.entry_px)
        cp = float(current_price)
        dd_pct = (cp - entry) / entry * 100.0

        if dd_pct <= -abs(self.stop_loss_pct):
            reason = f"STOP({stop_price_source}) dd={dd_pct:.2f}% <= -{abs(self.stop_loss_pct):.2f}%"
            # STOP should override any pending buy (rare), so overwrite=True
            self.place_order_for_next_bar("sell", seen_idx, reason=reason, overwrite=True)
            if verbose:
                print(f"[STOP] place SELL seen_idx={seen_idx} | {reason} | exec={execution_mode}")

    def force_liquidate(self, px: float, last_idx: int, reason: str = "forced_liquidation_end", verbose=False):
        if self.shares <= 0 or self.entry_px is None:
            return

        px = float(px)
        gross = self.shares * px
        net = gross * (1 - self.fee_pct)
        trade_cap = self.shares * self.entry_px
        pnl = net - trade_cap
        ret = pnl / trade_cap * 100.0 if trade_cap > 0 else 0.0

        self.cash += net
        if self.trades:
            self.trades[-1].update({
                "exit_idx": int(last_idx),
                "exit_price": px,
                "pnl": pnl,
                "return_pct": ret,
                "exit_reason": reason,
            })

        if verbose:
            print(f"[LIQ] SELL(all) @ {px:.4f} | idx={last_idx} | reason={reason} | ret={ret:.2f}%")

        self.shares = 0.0
        self.entry_px = None
        self.entry_idx = None

    def mark_to_market(self, day_close_price: float):
        return self.cash + (self.shares * float(day_close_price) if self.shares > 0 else 0.0)


def apply_market_regime_adjustment(buy_th: float,
                                   sell_th: float,
                                   regime: str = "neutral",
                                   base_shift: float = 0.25) -> tuple[float, float]:
    regime = (regime or "neutral").lower()

    if regime == "neutral":
        d_buy = 0.0
        d_sell = 0.0
    elif regime == "bullish":
        d_buy = -base_shift
        d_sell = +base_shift / 2.0
    elif regime == "strong_bullish":
        d_buy = -2 * base_shift
        d_sell = +base_shift
    elif regime == "bearish":
        d_buy = +base_shift
        d_sell = -base_shift / 2.0
    elif regime == "strong_bearish":
        d_buy = +2 * base_shift
        d_sell = -base_shift
    else:
        d_buy = 0.0
        d_sell = 0.0

    adj_buy = max(-1.0, min(4.0, buy_th + d_buy))
    adj_sell = max(-1.0, min(4.0, sell_th + d_sell))
    return adj_buy, adj_sell


def compute_buy_hold_equity(day_close_map: dict, daily_dates: list, initial_capital: float) -> pd.Series:
    closes = []
    dates = []
    for d in daily_dates:
        px = day_close_map.get(d)
        if px is None or pd.isna(px):
            continue
        dates.append(d)
        closes.append(float(px))

    if len(closes) == 0:
        return pd.Series(dtype=float)

    first = closes[0]
    equity = [initial_capital * (c / first) for c in closes]
    return pd.Series(equity, index=pd.to_datetime(dates))


# ============================================================
# 5) Main loop: intraday trade on new BSPs, train at day end
# ============================================================

def run_intraday_trade_daily_train(
    csv_path: str,
    code: str,
    start_time: str,
    end_time: str,
    lv: KL_TYPE,
    market_regime: str = "neutral",
    regime_shift: float = 0.25,
    chan_window_size: int = 500,
    label_lookahead_days: float = 1.0,

    warmup_mature_days: int = 30,
    threshold_days: int = 2,
    xgb_train_days: int = 60,

    min_train_samples: int = 200,
    min_thr_samples: int = 80,

    initial_capital: float = 100_000,
    position_size: float = 1.0,
    transaction_fee_pct: float = 0.0,

    execution_mode: str = "next_open",  # "next_open" or "next_close"
    output_dir: str = "./output/intraday_live_like",
    verbose: bool = True,

    # ---------------- STOP LOSS ----------------
    stop_loss_pct: float = 0.0,          # e.g. 2.0 means -2% from entry triggers stop
    stop_loss_trigger: str = "close",    # "close" or "low" (if raw csv has low column)
):
    os.makedirs(output_dir, exist_ok=True)

    (raw_kline_df,
     next_open_by_idx,
     next_close_by_idx,
     cur_close_by_idx,
     cur_low_by_idx,
     day_close_map,
     all_days) = load_kline_index(csv_path, start_time, end_time)

    # ---- Chan config ----
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
        begin_date=start_time,
        end_date=end_time,
        autype=AUTYPE.QFQ,
    )

    # ---- State ----
    all_bsp_snapshots = []
    buy_model = None
    sell_model = None
    buy_th = 0.0
    sell_th = 0.0
    feature_cols = None

    engine = LiveExecutionEngine(initial_capital, position_size, transaction_fee_pct, stop_loss_pct=stop_loss_pct)

    daily_log = []
    current_day = None

    had_bsp_today = False
    bsp_count_today = 0

    stop_loss_trigger = (stop_loss_trigger or "close").lower()
    if stop_loss_trigger not in ("close", "low"):
        stop_loss_trigger = "close"

    def maturity_cutoff_date(finished_day):
        return finished_day - timedelta(days=label_lookahead_days)

    def train_models_and_thresholds(finished_day):
        nonlocal buy_model, sell_model, buy_th, sell_th, feature_cols

        # label ALL snapshots so far
        profit_targets = calculate_profit_targets_best_window(all_bsp_snapshots, lookahead_days=label_lookahead_days)
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

        bsp_df_raw = pd.DataFrame(all_bsp_snapshots)
        if bsp_df_raw.empty:
            return False, bsp_df_raw

        bsp_df_raw = bsp_df_raw.sort_values("timestamp").reset_index(drop=True)
        bsp_df_raw["timestamp"] = pd.to_datetime(bsp_df_raw["timestamp"].astype(str))
        bsp_df_raw["date"] = bsp_df_raw["timestamp"].dt.date

        bsp_df = prepare_ml_dataset(bsp_df_raw.copy())

        # mature labels only
        cutoff = maturity_cutoff_date(finished_day)
        bsp_valid = bsp_df[(bsp_df["has_profit_target"] == 1) & (bsp_df["date"] <= cutoff)].copy()
        if bsp_valid.empty:
            return False, bsp_df

        labeled_days = np.sort(bsp_valid["date"].unique())
        if len(labeled_days) < warmup_mature_days:
            return False, bsp_df

        days = labeled_days[-xgb_train_days:] if len(labeled_days) > xgb_train_days else labeled_days
        if len(days) <= threshold_days:
            return False, bsp_df

        thr_days = days[-threshold_days:]
        train_days = days[:-threshold_days]

        train_df = bsp_valid[bsp_valid["date"].isin(train_days)].copy()
        thr_df = bsp_valid[bsp_valid["date"].isin(thr_days)].copy()

        if len(train_df) < min_train_samples or len(thr_df) < min_thr_samples:
            return False, bsp_df

        train_buy = train_df[train_df["direction"] == "buy"]
        train_sell = train_df[train_df["direction"] == "sell"]
        if len(train_buy) < 30 or len(train_sell) < 30:
            return False, bsp_df

        feature_cols = get_feature_columns(train_df, target_col="profit_target_pct")

        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        buy_model = xgb.XGBRegressor(**params)
        sell_model = xgb.XGBRegressor(**params)

        buy_model.fit(train_buy[feature_cols].values, train_buy["profit_target_pct"].values)
        sell_model.fit(train_sell[feature_cols].values, train_sell["profit_target_pct"].values)

        thr_buy = thr_df[thr_df["direction"] == "buy"].copy()
        thr_sell = thr_df[thr_df["direction"] == "sell"].copy()
        if len(thr_buy):
            thr_buy["predicted_profit_pct"] = buy_model.predict(thr_buy[feature_cols].values)
        if len(thr_sell):
            thr_sell["predicted_profit_pct"] = sell_model.predict(thr_sell[feature_cols].values)
        thr_signals = pd.concat([thr_buy, thr_sell], ignore_index=True)

        buy_th_raw, sell_th_raw = optimize_thresholds(thr_signals, transaction_fee_pct=transaction_fee_pct, min_trades=20)
        buy_th_adj, sell_th_adj = apply_market_regime_adjustment(
            buy_th_raw, sell_th_raw, regime=market_regime, base_shift=regime_shift
        )

        buy_th, sell_th = buy_th_adj, sell_th_adj

        if verbose:
            print(f"[THRESH] raw: buy={buy_th_raw:.2f}, sell={sell_th_raw:.2f} | "
                  f"adj({market_regime}): buy={buy_th:.2f}, sell={sell_th:.2f}")
            print(f"[TRAIN] {finished_day} trained | train_days={len(train_days)} thr_days={len(thr_days)} "
                  f"| buy_th={buy_th:.2f} sell_th={sell_th:.2f} | feats={len(feature_cols)}")

        return True, bsp_df

    if verbose:
        print("=" * 80)
        print("INTRADAY TRADE (on new BSP) + DAILY TRAIN (end-of-day) + STOP LOSS")
        print("=" * 80)
        if stop_loss_pct > 0:
            print(f"[STOP] Enabled: stop_loss_pct={stop_loss_pct:.2f}% trigger={stop_loss_trigger} "
                  f"(execution={execution_mode} next bar)")
        else:
            print("[STOP] Disabled")

    for klu_idx, klu in enumerate(csv_api.get_kl_data()):
        klu.kl_type = lv
        klu.set_idx(klu_idx)

        bar_ts = pd.to_datetime(str(klu.time))
        bar_day = bar_ts.date()

        if current_day is None:
            current_day = bar_day

        # day rollover
        if bar_day != current_day:
            prev_day = current_day

            # execute pending
            engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode, verbose=verbose)

            # EOD mark-to-market
            day_close = day_close_map.get(prev_day)
            equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash

            daily_log.append({
                "date": prev_day,
                "equity": equity,
                "cash": engine.cash,
                "shares": engine.shares,
                "buy_th": buy_th,
                "sell_th": sell_th,
                "has_model": int(buy_model is not None and sell_model is not None),
                "had_bsp": int(had_bsp_today),
                "bsp_count": int(bsp_count_today),
            })

            if verbose:
                print(f"[EOD] {prev_day} equity={equity:.2f} cash={engine.cash:.2f} shares={engine.shares:.6f} "
                      f"| had_bsp={int(had_bsp_today)} bsp_count={int(bsp_count_today)}")

            trained, bsp_df_snapshot = train_models_and_thresholds(prev_day)

            out_bsp = os.path.join(output_dir, f"bsp_until_{prev_day}.csv")
            if isinstance(bsp_df_snapshot, pd.DataFrame):
                bsp_df_snapshot.to_csv(out_bsp, index=False)

            # reset day flags
            had_bsp_today = False
            bsp_count_today = 0
            current_day = bar_day

        # execute pending if possible (from previous bar decisions)
        engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode, verbose=verbose)

        # ---------------- STOP LOSS CHECK (each bar) ----------------
        # Use raw CSV's current close/low (no future info).
        if engine.shares > 0 and engine.entry_px is not None:
            cp = None
            if stop_loss_trigger == "low" and cur_low_by_idx is not None:
                v = cur_low_by_idx[klu_idx] if 0 <= klu_idx < len(cur_low_by_idx) else None
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    cp = float(v)
            if cp is None:
                v = cur_close_by_idx[klu_idx] if 0 <= klu_idx < len(cur_close_by_idx) else None
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    cp = float(v)

            if cp is not None:
                engine.check_stop_and_place_sell(
                    seen_idx=int(klu_idx),
                    current_price=cp,
                    execution_mode=execution_mode,
                    stop_price_source=stop_loss_trigger if (stop_loss_trigger == "low" and cur_low_by_idx is not None) else "close",
                    verbose=verbose
                )

        # update chan
        window_chan, new_bsp_list = chan.process_new_kline(klu)

        # trade on newly discovered BSPs
        if new_bsp_list:
            had_bsp_today = True
            bsp_count_today += len(new_bsp_list)
            discover_idx = int(klu_idx)

            for s in new_bsp_list:
                s.setdefault("snapshot_first_seen_ts", str(bar_ts))
                all_bsp_snapshots.append(s)

                if buy_model is None or sell_model is None or feature_cols is None:
                    continue

                row_df = prepare_ml_dataset(pd.DataFrame([s]))
                for c in feature_cols:
                    if c not in row_df.columns:
                        row_df[c] = 0.0

                direction = row_df.iloc[0].get("direction", "unknown")
                X = row_df[feature_cols].values

                # If stop already wants to sell, don't place other sells
                if engine.has_pending_sell():
                    continue

                if direction == "buy":
                    pred = float(buy_model.predict(X)[0])
                    if engine.shares == 0 and pred >= buy_th:
                        engine.place_order_for_next_bar("buy", discover_idx, reason=f"pred={pred:.2f}>=th={buy_th:.2f}")
                        if verbose:
                            print(f"[SIGNAL] BUY seen_idx={discover_idx} pred={pred:.2f} th={buy_th:.2f}")

                elif direction == "sell":
                    pred = float(sell_model.predict(X)[0])
                    if engine.shares > 0 and pred >= sell_th:
                        engine.place_order_for_next_bar("sell", discover_idx, reason=f"pred={pred:.2f}>=th={sell_th:.2f}")
                        if verbose:
                            print(f"[SIGNAL] SELL seen_idx={discover_idx} pred={pred:.2f} th={sell_th:.2f}")

    # finalize last day: execute pending, then FORCE LIQUIDATION, then log EOD
    if current_day is not None:
        engine.maybe_execute_pending(next_open_by_idx, next_close_by_idx, execution_mode=execution_mode, verbose=verbose)

        # force liquidation at end
        if engine.shares > 0:
            last_idx = len(next_open_by_idx) - 1
            px = engine._exec_price_by_idx(last_idx, next_open_by_idx, next_close_by_idx, mode=execution_mode)

            # fallback to day close
            if px is None:
                px = day_close_map.get(current_day)

            # final fallback: last available close in raw_kline_df
            if px is None and "next_close" in raw_kline_df.columns:
                tail_px = raw_kline_df.iloc[-1]["next_close"]
                if tail_px is not None and not pd.isna(tail_px):
                    px = float(tail_px)

            if px is not None:
                engine.force_liquidate(px=float(px), last_idx=last_idx, verbose=verbose)

        day_close = day_close_map.get(current_day)
        equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash

        daily_log.append({
            "date": current_day,
            "equity": equity,
            "cash": engine.cash,
            "shares": engine.shares,
            "buy_th": buy_th,
            "sell_th": sell_th,
            "has_model": int(buy_model is not None and sell_model is not None),
            "had_bsp": int(had_bsp_today),
            "bsp_count": int(bsp_count_today),
        })

        if verbose:
            print(f"[EOD] {current_day} equity={equity:.2f} cash={engine.cash:.2f} shares={engine.shares:.6f} "
                  f"| had_bsp={int(had_bsp_today)} bsp_count={int(bsp_count_today)}")

    # ---------------- Build daily_df that includes NO-BSP days ----------------
    daily_df = pd.DataFrame(daily_log)
    if daily_df.empty:
        daily_df = pd.DataFrame(columns=["date","equity","cash","shares","buy_th","sell_th","has_model","had_bsp","bsp_count"])

    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    full_days_df = pd.DataFrame({"date": all_days})

    daily_df = full_days_df.merge(daily_df, on="date", how="left").sort_values("date").reset_index(drop=True)

    daily_df["had_bsp"] = daily_df["had_bsp"].fillna(0).astype(int)
    daily_df["bsp_count"] = daily_df["bsp_count"].fillna(0).astype(int)
    daily_df["has_model"] = daily_df["has_model"].fillna(0).astype(int)

    # forward fill continuous values
    daily_df[["equity","cash","shares","buy_th","sell_th"]] = daily_df[["equity","cash","shares","buy_th","sell_th"]].ffill()
    daily_df["equity"] = daily_df["equity"].fillna(initial_capital)
    daily_df["cash"] = daily_df["cash"].fillna(initial_capital)
    daily_df["shares"] = daily_df["shares"].fillna(0.0)
    daily_df["buy_th"] = daily_df["buy_th"].fillna(0.0)
    daily_df["sell_th"] = daily_df["sell_th"].fillna(0.0)

    trades_df = pd.DataFrame(engine.trades)

    daily_df.to_csv(os.path.join(output_dir, "daily_equity.csv"), index=False)
    trades_df.to_csv(os.path.join(output_dir, "trades.csv"), index=False)

    # ---------------- Plot Strategy vs Buy&Hold (start after warmup/model ready) ----------------
    if not daily_df.empty:
        daily_df_plot = daily_df.copy()
        daily_df_plot["date_dt"] = pd.to_datetime(daily_df_plot["date"])

        # Start plotting at first day model becomes available (warmup done)
        if (daily_df_plot["has_model"] == 1).any():
            start_dt = daily_df_plot.loc[daily_df_plot["has_model"] == 1, "date_dt"].min()
            daily_df_plot = daily_df_plot[daily_df_plot["date_dt"] >= start_dt].copy()

        strat_equity = pd.Series(daily_df_plot["equity"].values, index=daily_df_plot["date_dt"])

        bh_equity = compute_buy_hold_equity(
            day_close_map=day_close_map,
            daily_dates=list(daily_df_plot["date"].values),
            initial_capital=initial_capital
        )

        plt.figure(figsize=(12, 6))
        plt.plot(strat_equity.index, strat_equity.values, label="Strategy Equity", linewidth=2)
        if len(bh_equity) > 0:
            plt.plot(bh_equity.index, bh_equity.values, label="Buy & Hold Equity", linewidth=2)

        title = "Strategy vs Buy & Hold (Equity Curve) — Start After Warmup"
        if stop_loss_pct > 0:
            title += f" | StopLoss={stop_loss_pct:.2f}%"
        plt.title(title)
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
        print("=" * 80)
        print("DONE")
        print(f"Daily saved:  {os.path.join(output_dir, 'daily_equity.csv')}")
        print(f"Trades saved: {os.path.join(output_dir, 'trades.csv')}")
        print("=" * 80)

    return {"daily_df": daily_df, "trades_df": trades_df}


# ============================================================
# 6) Run
# ============================================================

if __name__ == "__main__":
    results = run_intraday_trade_daily_train(
        csv_path="DataAPI/data/QQQ_5M.csv",
        code="QQQ",
        start_time="2022-10-01",
        end_time="2023-12-30",
        lv=KL_TYPE.K_5M,

        chan_window_size=500,
        label_lookahead_days=2.0,

        warmup_mature_days=90,
        threshold_days=2,
        xgb_train_days=90,

        min_train_samples=200,
        min_thr_samples=80,

        initial_capital=100_000,
        position_size=1.0,
        transaction_fee_pct=0.0,

        execution_mode="next_open",
        output_dir="./output/intraday_live_like",
        verbose=True,

        market_regime="bullish",
        regime_shift=0.25,

        # ---------- STOP LOSS ----------
        stop_loss_pct=2.0,          # <-- set to 0 to disable
        stop_loss_trigger="low",    # "close" or "low" (needs 'low' column in csv for 'low')
    )

    print(results["daily_df"].head(10))
    print(results["trades_df"].head(10))
