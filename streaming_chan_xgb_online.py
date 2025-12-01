import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE, DATA_SRC, AUTYPE

from chan_streamer import ChanStreamer
from streaming_labels import StreamingBest24hLabeler


def get_numeric_feature_columns(bsp_df: pd.DataFrame, target_col: str = "profit_target_pct") -> List[str]:
    exclude = {
        "klu_idx",
        "profit_target_pct",
        "profit_target_distance",
        "has_profit_target",
        "exit_klu_idx",
        "exit_price",
    }
    exclude_patterns = ["timestamp", "bsp_type", "bsp_types", "direction", "snapshot_first_seen", "snapshot_last_seen"]

    cols = []
    for c in bsp_df.select_dtypes(include=[np.number]).columns:
        if c in exclude:
            continue
        if any(pat in c for pat in exclude_patterns):
            continue
        cols.append(c)

    return sorted(cols)


def calc_buy_hold_daily(csv_path: str,
                        start_time: str,
                        end_time: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    if 'timestamp' in raw.columns:
        raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    else:
        raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0])
        raw.rename(columns={raw.columns[0]: 'timestamp'}, inplace=True)

    close_col = 'close'
    for c in ['close', 'Close', 'adj_close', 'Adj Close']:
        if c in raw.columns:
            close_col = c
            break

    mask = (raw['timestamp'] >= pd.to_datetime(start_time)) & (raw['timestamp'] <= pd.to_datetime(end_time))
    raw = raw[mask].copy()
    raw['date'] = raw['timestamp'].dt.date

    daily = raw.groupby('date')[close_col].agg(['first', 'last'])
    daily.rename(columns={'first': 'open', 'last': 'close'}, inplace=True)
    daily['return_pct'] = (daily['close'] / daily['open'] - 1) * 100

    first_price = daily['open'].iloc[0]
    daily['cum_return_pct'] = (daily['close'] / first_price - 1) * 100

    return daily.reset_index().rename(columns={'date': 'date'})


def run_streaming_chan_xgb_online(
    csv_path: str,
    code: str = "SPY",
    start_time: str = "2022-01-01",
    end_time: str = "2022-03-31",
    lv: KL_TYPE = KL_TYPE.K_5M,
    chan_window_size: int = 500,
    train_days_for_model: int = 30,
    threshold_days_for_selection: int = 1,
    min_train_samples: int = 100,
    min_valid_samples: int = 20,
    initial_capital: float = 100_000.0,
    position_size: float = 1.0,
    transaction_fee_pct: float = 0.001,
    output_dir: str = "./output/chan_xgb_streaming_online",
    plot_results: bool = True,
    verbose: bool = True,
) -> Dict:

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Streaming Chan + XGBoost (online training & trading)")
    print("=" * 80)

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
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": False,
        "zs_algo": "normal",
    })

    time_windows = {
        KL_TYPE.K_5M: 288,
        KL_TYPE.K_15M: 96,
        KL_TYPE.K_30M: 48,
        KL_TYPE.K_60M: 24,
        KL_TYPE.K_DAY: 1,
    }
    time_window_bars = time_windows.get(lv, 288)

    streamer = ChanStreamer(
        code=code,
        begin_time=start_time,
        end_time=end_time,
        data_src=DATA_SRC.CSV,
        lv=lv,
        config=config,
        autype=AUTYPE.QFQ,
        max_klines=chan_window_size,
    )

    labeler = StreamingBest24hLabeler(time_window_bars=time_window_bars)

    completed_days: List = []
    cash = initial_capital
    shares = 0.0
    entry_price = None
    entry_time = None

    daily_results: List[Dict] = []
    all_trades: List[Dict] = []

    feature_cols: List[str] = []
    buy_model: xgb.XGBRegressor = None
    sell_model: xgb.XGBRegressor = None
    buy_th = 0.0
    sell_th = 0.0
    model_ready = False

    per_side_fee = transaction_fee_pct

    def close_position(price, ts, force=False):
        nonlocal cash, shares, entry_price, entry_time, all_trades
        if shares <= 0:
            return
        gross_exit = shares * price
        net_exit = gross_exit * (1 - per_side_fee)
        trade_cap = shares * entry_price if entry_price is not None else 0.0
        pnl = net_exit - trade_cap
        ret_pct = (pnl / trade_cap * 100) if trade_cap > 0 else 0.0

        cash += net_exit

        all_trades.append({
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": ts,
            "exit_price": price,
            "pnl": pnl,
            "return_pct": ret_pct,
            "force_close": force,
        })

        print(f"[TRADE] SELL confirmed at {ts} | close={price:.4f} | return={ret_pct:.2f}%"
              + (" (force close end-of-day)" if force else ""))

        shares = 0.0
        entry_price = None
        entry_time = None

    def open_position(price, ts):
        nonlocal cash, shares, entry_price, entry_time
        trade_cap = cash * position_size
        if trade_cap <= 0:
            return
        effective_capital = trade_cap * (1 - per_side_fee)
        shares = effective_capital / price
        cash -= trade_cap
        entry_price = price
        entry_time = ts

        print(f"[TRADE] BUY confirmed at {ts} | close={price:.4f}")

    def on_day_finished(day):
        daily_results.append({"date": day})

    def train_models_if_possible(df_all: pd.DataFrame):
        nonlocal feature_cols, buy_model, sell_model, buy_th, sell_th, model_ready, completed_days

        if len(completed_days) < train_days_for_model + threshold_days_for_selection:
            return

        df_lab = df_all[df_all["has_profit_target"] == 1].copy()
        if df_lab.empty:
            return

        thr_days = completed_days[-threshold_days_for_selection:]
        train_days = completed_days[-(train_days_for_model + threshold_days_for_selection):-threshold_days_for_selection]

        train_df = df_lab[df_lab["date"].isin(train_days)]
        thr_df = df_lab[df_lab["date"].isin(thr_days)]

        if len(train_df) < min_train_samples or len(thr_df) < min_valid_samples:
            return

        feature_cols_ = get_numeric_feature_columns(df_lab, target_col="profit_target_pct")
        if not feature_cols_:
            return
        feature_cols = feature_cols_

        train_buy = train_df[train_df["direction"] == "buy"]
        train_sell = train_df[train_df["direction"] == "sell"]

        if len(train_buy) < 10 or len(train_sell) < 10:
            return

        X_train_buy = train_buy[feature_cols].fillna(0.0).values
        y_train_buy = train_buy["profit_target_pct"].values
        X_train_sell = train_sell[feature_cols].fillna(0.0).values
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

        print(f"[MODEL] XGBoost models trained on {len(train_days)} days "
              f"({train_days[0]} → {train_days[-1]}), "
              f"samples: buy={len(train_buy)}, sell={len(train_sell)}")

        thr_buy = thr_df[thr_df["direction"] == "buy"].copy()
        thr_sell = thr_df[thr_df["direction"] == "sell"].copy()

        if not thr_buy.empty:
            thr_buy["predicted"] = buy_model.predict(thr_buy[feature_cols].fillna(0.0).values)
        if not thr_sell.empty:
            thr_sell["predicted"] = sell_model.predict(thr_sell[feature_cols].fillna(0.0).values)

        thr_signals = pd.concat([thr_buy, thr_sell], ignore_index=True)
        if thr_signals.empty:
            return

        thresh_grid = np.linspace(-0.5, 3.0, 29)
        best_score = -1e9
        best_bt = 0.0
        best_st = 0.0
        roundtrip_fee_pct = 2 * transaction_fee_pct * 100.0

        for bt in thresh_grid:
            for st in thresh_grid:
                b_sel = thr_buy[thr_buy["predicted"] >= bt] if not thr_buy.empty else pd.DataFrame()
                s_sel = thr_sell[thr_sell["predicted"] >= st] if not thr_sell.empty else pd.DataFrame()

                if b_sel.empty and s_sel.empty:
                    continue

                b_ret = (b_sel["profit_target_pct"] - roundtrip_fee_pct).mean() if not b_sel.empty else 0.0
                s_ret = (s_sel["profit_target_pct"] - roundtrip_fee_pct).mean() if not s_sel.empty else 0.0

                n_b, n_s = len(b_sel), len(s_sel)
                tot = n_b + n_s
                combined = (n_b * b_ret + n_s * s_ret) / tot if tot > 0 else 0.0

                if combined > best_score:
                    best_score = combined
                    best_bt = bt
                    best_st = st

        buy_th = best_bt
        sell_th = best_st
        model_ready = True

        print(f"[MODEL] Thresholds set: buy_th={buy_th:.2f}%, sell_th={sell_th:.2f}%, "
              f"expected net avg label={best_score:.2f}%")

    last_date = None

    for global_idx, klu, new_bsp_list in streamer.stream_from_source():
        ts = getattr(klu, "time")
        ts_str = str(ts)

        labeler.add_new_bsp_list(new_bsp_list)
        labeler.update_labels_until(current_klu_idx=klu.idx)

        dt = pd.to_datetime(ts_str)
        this_date = dt.date()

        if last_date is None:
            last_date = this_date

        if this_date != last_date:
            if shares > 0:
                close_position(price=entry_price, ts=str(last_date), force=True)

            completed_days.append(last_date)
            on_day_finished(last_date)
            last_date = this_date

            df_all = labeler.get_labeled_bsp_df()
            if not df_all.empty:
                train_models_if_possible(df_all)

        if model_ready and new_bsp_list:
            df_all = labeler.get_labeled_bsp_df()
            if df_all.empty:
                continue

            if not feature_cols:
                feature_cols_ = get_numeric_feature_columns(df_all)
                if not feature_cols_:
                    continue
                feature_cols = feature_cols_

            df_all["key"] = df_all["timestamp"].astype(str) + "_" + df_all["bsp_type"].astype(str)
            df_all = df_all.set_index("key")

            for bsp in new_bsp_list:
                bsp_ts = pd.to_datetime(bsp["timestamp"])
                if bsp_ts.date() != this_date:
                    continue

                key = f"{bsp['timestamp']}_{bsp['bsp_type']}"
                if key not in df_all.index:
                    continue

                row = df_all.loc[key]
                x = row[feature_cols].fillna(0.0).values.reshape(1, -1)
                direction = row["direction"]
                price = float(row["klu_close"])

                if direction == "buy":
                    pred = float(buy_model.predict(x)[0])
                    if shares == 0 and pred >= buy_th:
                        open_position(price=price, ts=bsp["timestamp"])
                else:
                    pred = float(sell_model.predict(x)[0])
                    if shares > 0 and pred >= sell_th:
                        close_position(price=price, ts=bsp["timestamp"], force=False)

        if global_idx % 500 == 0:
            stats = streamer.get_stats()
            print(f"[📈] Bar {global_idx}: {stats['unique_bsp_count']} BSP collected, buffer={stats['buffer_size']}")

    if shares > 0:
        close_position(price=entry_price, ts=str(last_date), force=True)
    if last_date is not None and (not completed_days or completed_days[-1] != last_date):
        completed_days.append(last_date)
        on_day_finished(last_date)

    bsp_df = labeler.get_labeled_bsp_df()
    if not bsp_df.empty:
        bsp_df.to_csv(os.path.join(output_dir, "bsp_streaming_dataset.csv"), index=False)

    if not all_trades:
        print("[❌] No trades executed in streaming run.")
        return {}

    trades_df = pd.DataFrame(all_trades)
    trades_df["date"] = pd.to_datetime(trades_df["exit_time"]).dt.date
    daily_pnl = trades_df.groupby("date")["pnl"].sum().reset_index()
    daily_pnl = daily_pnl.sort_values("date")

    equity = initial_capital
    equity_curve = []
    for _, r in daily_pnl.iterrows():
        equity += r["pnl"]
        ret_pct = (equity / initial_capital - 1) * 100.0
        equity_curve.append({"date": r["date"], "strategy_cum_return": ret_pct})

    daily_results_df = pd.DataFrame(equity_curve)

    bh_df = calc_buy_hold_daily(csv_path, start_time, end_time)
    bh_map = bh_df.set_index("date")["cum_return_pct"].to_dict()
    daily_results_df["buy_hold_cum_return"] = daily_results_df["date"].map(bh_map)

    if plot_results and not daily_results_df.empty:
        plt.figure(figsize=(12, 6))
        dates = pd.to_datetime(daily_results_df["date"])
        plt.plot(dates, daily_results_df["strategy_cum_return"], label="Strategy", linewidth=2)
        plt.plot(dates, daily_results_df["buy_hold_cum_return"], label="Buy & Hold", linewidth=2)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.title("Cumulative Return: Strategy vs Buy & Hold")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "strategy_vs_buyhold_streaming.png"), dpi=150)
        plt.show()

    print("\n" + "=" * 80)
    print("Streaming Chan + XGBoost Online Complete")
    print("=" * 80)
    print(f"Total trades: {len(trades_df)}")
    print(f"Final strategy cumulative return: {daily_results_df['strategy_cum_return'].iloc[-1]:.2f}%")
    if not daily_results_df["buy_hold_cum_return"].isna().all():
        print(f"Final Buy & Hold cumulative return: {daily_results_df['buy_hold_cum_return'].iloc[-1]:.2f}%")
    print("=" * 80)

    trades_df.to_csv(os.path.join(output_dir, "trades_streaming.csv"), index=False)
    daily_results_df.to_csv(os.path.join(output_dir, "daily_results_streaming.csv"), index=False)

    return {
        "daily_results_df": daily_results_df,
        "trades_df": trades_df,
        "bsp_df": bsp_df,
        "bh_df": bh_df,
    }
