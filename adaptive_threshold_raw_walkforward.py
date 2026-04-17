from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_trade_extensions import (
    RollingThresholdConfig,
    ThresholdPairBandit,
    ThresholdPairBanditConfig,
    gate_from_levels,
    make_daily_threshold_state,
    make_threshold_grid,
    select_oracle_thresholds_from_daily_rewards,
)
from pipelineCurrent import (
    AUTYPE,
    CChanConfig,
    DATA_SRC,
    KL_TYPE,
    DailyProbState,
    ExecutionEngine,
    RetModelPack,
    SlidingWindowChan,
    build_klu,
    compute_buy_hold_equity,
    compute_chain_endpoints,
    compute_daily_kline_features,
    choose_thresholds_global_realized,
    extract_bsp_rows_from_chan,
    feature_importance_from_lr,
    feed_chan_one,
    fit_prob_model_dicts,
    get_feature_columns,
    label_bestlookahead_for_ready_points,
    label_confirm_extreme,
    latest_bsp_dir_up_to,
    load_5m_index,
    load_macro_features_from_folder,
    load_ohlcv_csv,
    make_daily_features_one_model,
    make_ret_grid,
    normalize_bsp_row,
    predict_prob,
    predict_ret,
    prepare_ml_dataset,
    regime_for_day_from_ends,
    train_models_two_sided_ret_only,
)


def clone_engine_from_state(engine_state: dict, fee_pct: float) -> ExecutionEngine:
    eng = ExecutionEngine(initial_capital=1.0, fee_pct=fee_pct)
    eng.load_state_dict(copy.deepcopy(engine_state))
    eng.fee_pct = float(fee_pct)
    return eng


def simulate_one_day_under_gate_from_events(
    engine_state: dict,
    day_events: List[Dict[str, Any]],
    gate_action: str,
    day_start_idx: int,
    day_end_idx: int,
    df_5m_idx: pd.DataFrame,
    next_open_by_idx: np.ndarray,
    closes: np.ndarray,
    buy_pack: Optional[RetModelPack],
    sell_pack: Optional[RetModelPack],
    buy_ret_th_live: float,
    sell_ret_th_live: float,
    fee_pct: float = 0.0,
) -> dict:
    eng = clone_engine_from_state(engine_state, fee_pct=fee_pct)
    eng.maybe_execute_pending(next_open_by_idx)

    first_open = float(df_5m_idx.loc[day_start_idx, "Open"])
    eq_start = eng.cash if eng.pos == 0 else (eng.cash + eng.qty * first_open)

    allow_buy = True
    allow_sell = True
    must_trade_dir = None

    if gate_action == "FORCE_BUY":
        allow_sell = False
        if eng.pos == 0:
            must_trade_dir = "buy"
    elif gate_action == "FORCE_SELL":
        allow_buy = False
        if eng.pos == 1:
            must_trade_dir = "sell"

    events_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    for r in day_events:
        ki = int(r.get("klu_idx", -1))
        if day_start_idx <= ki <= day_end_idx:
            events_by_idx.setdefault(ki, []).append(r)

    for i in range(day_start_idx, day_end_idx + 1):
        eng.maybe_execute_pending(next_open_by_idx)
        for r in events_by_idx.get(i, []):
            d = str(r.get("direction", "buy")).lower()
            if d == "buy" and not allow_buy:
                continue
            if d == "sell" and not allow_sell:
                continue
            if must_trade_dir is not None and d != must_trade_dir:
                continue

            if d == "buy" and eng.pos == 0 and buy_pack is not None:
                row_df = prepare_ml_dataset(pd.DataFrame([r]))
                for cc in buy_pack.feature_cols:
                    if cc not in row_df.columns:
                        row_df[cc] = 0.0
                pr = predict_ret(buy_pack, row_df)
                if pr >= float(buy_ret_th_live):
                    eng.place_order_for_next_bar(
                        side="buy",
                        seen_idx=i,
                        reason=f"{gate_action} day sim buy",
                        meta={"pred": float(pr), "th": float(buy_ret_th_live), "gate": gate_action},
                    )
                    if must_trade_dir == "buy":
                        must_trade_dir = None

            elif d == "sell" and eng.pos == 1 and sell_pack is not None:
                row_df = prepare_ml_dataset(pd.DataFrame([r]))
                for cc in sell_pack.feature_cols:
                    if cc not in row_df.columns:
                        row_df[cc] = 0.0
                pr = predict_ret(sell_pack, row_df)
                if pr >= float(sell_ret_th_live):
                    eng.place_order_for_next_bar(
                        side="sell",
                        seen_idx=i,
                        reason=f"{gate_action} day sim sell",
                        meta={"pred": float(pr), "th": float(sell_ret_th_live), "gate": gate_action},
                    )
                    if must_trade_dir == "sell":
                        must_trade_dir = None

    eq_end = eng.mark_to_market(closes[day_end_idx])
    day_ret = 0.0 if eq_start <= 0 else (eq_end - eq_start) / eq_start
    return {
        "equity_start": float(eq_start),
        "equity_end": float(eq_end),
        "day_return": float(day_ret),
    }


def evaluate_three_day_rewards_for_logging(
    engine_state: dict,
    day_events: List[Dict[str, Any]],
    day_start_idx: int,
    day_end_idx: int,
    df_5m_idx: pd.DataFrame,
    next_open_by_idx: np.ndarray,
    closes: np.ndarray,
    buy_pack: Optional[RetModelPack],
    sell_pack: Optional[RetModelPack],
    buy_ret_th_live: float,
    sell_ret_th_live: float,
    fee_pct: float = 0.0,
) -> dict:
    out = {}
    for gate_action in ["FORCE_BUY", "FREE", "FORCE_SELL"]:
        out[gate_action] = simulate_one_day_under_gate_from_events(
            engine_state=engine_state,
            day_events=day_events,
            gate_action=gate_action,
            day_start_idx=day_start_idx,
            day_end_idx=day_end_idx,
            df_5m_idx=df_5m_idx,
            next_open_by_idx=next_open_by_idx,
            closes=closes,
            buy_pack=buy_pack,
            sell_pack=sell_pack,
            buy_ret_th_live=buy_ret_th_live,
            sell_ret_th_live=sell_ret_th_live,
            fee_pct=fee_pct,
        )
    return out


def _make_chan_config() -> CChanConfig:
    return CChanConfig({
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


def run_raw_adaptive_threshold_walkforward(
    daily_csv_path: str,
    k5m_csv_path: str,
    code: str = "QQQ",
    daily_chan_start: str = "2014-06-01",
    accumulation_start: str = "2016-10-01",
    sim_start: str = "2019-01-01",
    end_time: str = "2021-12-31",
    N_confirm: int = 5,
    min_labeled_days_to_train: int = 200,
    retrain_every_new_labels: int = 25,
    dp_lookback: int = 5,
    lookahead_days_5m: float = 2.0,
    retrain_every_days_5m: int = 5,
    min_samples_total_5m: int = 300,
    threshold_window_days: float = 2.0,
    threshold_ret_grid=None,
    threshold_min_open_signals: int = 10,
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    daily_chan_max_klines: int = 500,
    five_chan_max_klines: int = 500,
    macro_files: Optional[dict] = None,
    policy_mode: str = "adaptive_reward",
    static_buy_level: float = 0.20,
    static_sell_level: float = 0.30,
    daily_threshold_config: Optional[RollingThresholdConfig] = None,
    threshold_pair_bandit_config: Optional[ThresholdPairBanditConfig] = None,
    output_dir: str = "output_raw_adaptive_threshold_walkforward",
    verbose: bool = True,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    if macro_files is None:
        macro_files = {"vix_": "VIX.csv"}
    if threshold_ret_grid is None:
        threshold_ret_grid = make_ret_grid(-0.5, 2.5, 0.05)
    if daily_threshold_config is None:
        daily_threshold_config = RollingThresholdConfig(
            lookback_days=252,
            buy_grid=make_threshold_grid(0.05, 0.35, 0.005),
            sell_grid=make_threshold_grid(0.15, 0.60, 0.005),
            min_gap=0.02,
            min_obs=60,
            switch_penalty=0.0,
        )

    df_day_raw = load_ohlcv_csv(daily_csv_path, "DAILY")
    df_5m_raw = load_ohlcv_csv(k5m_csv_path, "5M")

    daily_s = pd.to_datetime(daily_chan_start)
    acc_s = pd.to_datetime(accumulation_start)
    sim_s = pd.to_datetime(sim_start)
    end_t = pd.to_datetime(end_time)

    df_day = df_day_raw[(df_day_raw["timestamp"] >= daily_s) & (df_day_raw["timestamp"] <= end_t)].copy().reset_index(drop=True)
    if df_day.empty:
        raise ValueError("No daily bars in requested range.")
    df_day_feat = compute_daily_kline_features(df_day)
    df_day_feat["ts_norm"] = pd.to_datetime(df_day_feat["timestamp"]).dt.normalize()

    macro_folder = os.path.dirname(os.path.abspath(daily_csv_path))
    macro_feat = load_macro_features_from_folder(folder=macro_folder, files=macro_files, start=daily_chan_start)
    df_day_feat = df_day_feat.merge(macro_feat, on="ts_norm", how="left").sort_values("timestamp").reset_index(drop=True)
    macro_cols = [c for c in df_day_feat.columns if any(c.startswith(pref) for pref in macro_files.keys())]

    df_5m = df_5m_raw[(df_5m_raw["timestamp"] >= acc_s) & (df_5m_raw["timestamp"] <= end_t + pd.Timedelta(days=1))].copy().reset_index(drop=True)
    df_5m = df_5m.sort_values("timestamp").reset_index(drop=True)
    df_5m_idx, next_open_by_idx, _, closes, highs, lows, day_close_map, all_days = load_5m_index(df_5m, accumulation_start, end_time)
    buy_hold = compute_buy_hold_equity(day_close_map, all_days, initial_capital)

    config = _make_chan_config()
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
    chan_5m = SlidingWindowChan(
        code=code,
        begin_time=None,
        end_time=None,
        data_src=getattr(DATA_SRC, "CSV", "CSV"),
        lv_list=[KL_TYPE.K_5M],
        config=config,
        autype=AUTYPE.QFQ,
        max_klines=int(five_chan_max_klines),
    )

    bsp_rows_daily: List[Dict[str, Any]] = []
    seen_bsp_daily = set()
    X_days, y_days = [], []
    st = DailyProbState()
    pending_idx = []
    p_series = np.full(len(df_day_feat), np.nan, dtype=float)
    dp_vs_minK_series = np.full(len(df_day_feat), np.nan, dtype=float)
    dp_vs_maxK_series = np.full(len(df_day_feat), np.nan, dtype=float)
    p_by_day: Dict[pd.Timestamp, float] = {}

    for i in range(len(df_day_feat)):
        r = df_day_feat.loc[i]
        ts = pd.to_datetime(r["timestamp"])
        day = ts.normalize()
        daily_chan.process_new_kline(build_klu(ts, r["_open"], r["_high"], r["_low"], r["_close"], r["_vol"]))

        for rr0 in extract_bsp_rows_from_chan(daily_chan) or []:
            rr = normalize_bsp_row(dict(rr0))
            rr.setdefault("timestamp", ts)
            key = (pd.to_datetime(rr["timestamp"]).strftime("%Y-%m-%d"), rr["direction"], rr.get("bsp_type", "?"))
            if key not in seen_bsp_daily:
                seen_bsp_daily.add(key)
                bsp_rows_daily.append(rr)

        ends = compute_chain_endpoints(bsp_rows_daily)
        regime = regime_for_day_from_ends(day, ends)
        base_dir_today = latest_bsp_dir_up_to(bsp_rows_daily, ts)

        p_val = np.nan
        if st.model is not None:
            bsp_hist = [b for b in bsp_rows_daily if pd.to_datetime(b["timestamp"]) <= ts]
            feat_i = make_daily_features_one_model(
                kline_row=r,
                bsp_hist_up_to_day=bsp_hist,
                p_val=0.0,
                dp_minK=0.0,
                dp_maxK=0.0,
                regime=regime,
                base_dir=base_dir_today,
                macro_cols=macro_cols,
            )
            p_val = float(predict_prob(st.model, [feat_i])[0])
            p_series[i] = p_val
            p_by_day[day] = p_val

        lb = int(dp_lookback)
        if lb > 0 and i >= 1 and np.isfinite(p_val):
            prev = pd.Series(p_series[max(0, i - lb):i]).dropna()
            if len(prev) > 0:
                dp_vs_minK_series[i] = p_val - float(prev.min())
                dp_vs_maxK_series[i] = p_val - float(prev.max())

        pending_idx.append(i)
        while pending_idx and i >= pending_idx[0] + int(N_confirm):
            j = pending_idx.pop(0)
            t0 = pd.to_datetime(df_day_feat.loc[j, "timestamp"])
            base_dir_j = latest_bsp_dir_up_to(bsp_rows_daily, t0)
            if base_dir_j not in ("buy", "sell"):
                continue
            y = label_confirm_extreme(df_day_feat, j, int(N_confirm), base_dir_j)
            if y is None:
                continue
            ends_j = compute_chain_endpoints([b for b in bsp_rows_daily if pd.to_datetime(b["timestamp"]) <= t0])
            regime_j = regime_for_day_from_ends(t0.normalize(), ends_j)
            bsp_hist_j = [b for b in bsp_rows_daily if pd.to_datetime(b["timestamp"]) <= t0]
            feat_j = make_daily_features_one_model(
                kline_row=df_day_feat.loc[j],
                bsp_hist_up_to_day=bsp_hist_j,
                p_val=float(p_series[j]) if np.isfinite(p_series[j]) else 0.0,
                dp_minK=float(dp_vs_minK_series[j]) if np.isfinite(dp_vs_minK_series[j]) else 0.0,
                dp_maxK=float(dp_vs_maxK_series[j]) if np.isfinite(dp_vs_maxK_series[j]) else 0.0,
                regime=regime_j,
                base_dir=base_dir_j,
                macro_cols=macro_cols,
            )
            X_days.append(feat_j)
            y_days.append(int(y))
            st.new_labels += 1

        if len(y_days) >= int(min_labeled_days_to_train) and (st.model is None or st.new_labels >= int(retrain_every_new_labels)):
            y_arr = np.asarray(y_days, dtype=int)
            if len(np.unique(y_arr)) >= 2:
                st.model = fit_prob_model_dicts(X_days, y_arr)
                st.trained_n = len(y_arr)
                st.new_labels = 0
                if verbose:
                    print(f"[TRAIN][DAILY-PROB] n={len(y_arr)} pos={int(y_arr.sum())} ({y_arr.mean():.2%})")

    if st.model is not None:
        try:
            feature_importance_from_lr(st.model, top_n=120).to_csv(os.path.join(output_dir, "daily_lr_feature_importance.csv"), index=False)
        except Exception:
            pass

    engine = ExecutionEngine(initial_capital=initial_capital, fee_pct=fee_pct)
    bsp_rows_5m: List[Dict[str, Any]] = []
    seen_keys_5m = set()
    buy_pack = None
    sell_pack = None
    last_train_day = None
    buy_ret_th_live = 0.30
    sell_ret_th_live = 0.30
    current_day = None
    last_day_end_idx = None
    day_gate = "FREE"
    allow_buy = True
    allow_sell = True
    must_trade_dir = None
    day_start_engine_state = None
    day_events_today: List[Dict[str, Any]] = []
    day_start_idx = None
    daily_log = []
    daily_reward_log = []
    equity_peak = initial_capital
    oracle_equity = initial_capital
    threshold_pair_bandit = None
    if policy_mode == "threshold_pair_bandit":
        threshold_pair_bandit = ThresholdPairBandit(
            n_features=6,
            config=threshold_pair_bandit_config or ThresholdPairBanditConfig(),
        )
    current_buy_level = float(static_buy_level)
    current_sell_level = float(static_sell_level)
    current_pair_action_idx = None
    current_pair_state_x = None

    def maybe_retrain_5m(day_ts: pd.Timestamp):
        nonlocal buy_pack, sell_pack, last_train_day
        if last_train_day is not None and (day_ts - last_train_day).days < int(retrain_every_days_5m):
            return
        dfb = pd.DataFrame(bsp_rows_5m)
        if dfb.empty:
            return
        dfb2 = prepare_ml_dataset(dfb)
        feat_cols = get_feature_columns(dfb2)
        bp, sp = train_models_two_sided_ret_only(dfb2, feat_cols, min_samples_total=min_samples_total_5m)
        if bp is not None:
            buy_pack = bp
        if sp is not None:
            sell_pack = sp
        if (bp is not None) or (sp is not None):
            last_train_day = day_ts
            if verbose:
                print(
                    f"[TRAIN][5M] asof={day_ts.date()} feats={len(feat_cols)} "
                    f"buy={'YES' if bp else 'NO'} sell={'YES' if sp else 'NO'} rows={len(dfb2)}"
                )

    def maybe_opt_5m_thresholds(asof_bar_idx: int):
        nonlocal buy_ret_th_live, sell_ret_th_live
        if buy_pack is None or sell_pack is None:
            return
        out = choose_thresholds_global_realized(
            df_5m=df_5m_idx,
            bsp_rows=bsp_rows_5m,
            buy_pack=buy_pack,
            sell_pack=sell_pack,
            asof_bar_idx=asof_bar_idx,
            window_days=threshold_window_days,
            ret_grid=threshold_ret_grid,
            next_open_by_idx=next_open_by_idx,
            closes=closes,
            fee_pct=fee_pct,
            min_open_signals=threshold_min_open_signals,
        )
        if out is not None:
            buy_ret_th_live, sell_ret_th_live = out

    def choose_daily_gate_for_day(bar_day: pd.Timestamp) -> dict:
        nonlocal current_buy_level, current_sell_level, current_pair_action_idx, current_pair_state_x
        p_day_val = float(p_by_day.get(bar_day, np.nan))
        hist_df = pd.DataFrame(daily_reward_log)

        if policy_mode == "static":
            gate = gate_from_levels(p_day_val, static_buy_level, static_sell_level)
            current_buy_level = float(static_buy_level)
            current_sell_level = float(static_sell_level)
            current_pair_action_idx = None
            current_pair_state_x = None
            return {"gate": gate, "buy_level": current_buy_level, "sell_level": current_sell_level, "p_day": p_day_val}

        if policy_mode in ("adaptive_reward", "adaptive_accuracy"):
            obj = "reward" if policy_mode == "adaptive_reward" else "accuracy"
            out = select_oracle_thresholds_from_daily_rewards(
                history_df=hist_df,
                current_p_day=p_day_val,
                config=daily_threshold_config,
                prev_buy_level=current_buy_level,
                prev_sell_level=current_sell_level,
                objective=obj,
            )
            current_buy_level = out.buy_level
            current_sell_level = out.sell_level
            current_pair_action_idx = None
            current_pair_state_x = None
            return {"gate": out.gate, "buy_level": out.buy_level, "sell_level": out.sell_level, "p_day": p_day_val}

        if policy_mode == "threshold_pair_bandit":
            dd_rel = 0.0 if equity_peak <= 0 else max(0.0, (equity_peak - engine.mark_to_market(day_close_map.get(bar_day.date(), closes[0]))) / equity_peak)
            x = make_daily_threshold_state(
                p_day=p_day_val,
                dp_min=0.0,
                dp_max=0.0,
                realized_vol_20=0.0,
                drawdown_rel=dd_rel,
                current_pos=engine.pos,
            )
            decision = threshold_pair_bandit.decide_gate(x=x, p_day=p_day_val)
            current_pair_action_idx = int(decision["action_idx"])
            current_pair_state_x = x
            current_buy_level = float(decision["buy_level"])
            current_sell_level = float(decision["sell_level"])
            return {"gate": decision["gate"], "buy_level": current_buy_level, "sell_level": current_sell_level, "p_day": p_day_val}

        raise ValueError(f"Unknown policy_mode: {policy_mode}")

    def begin_day(bar_day: pd.Timestamp, bar_idx: int):
        nonlocal day_gate, allow_buy, allow_sell, must_trade_dir
        nonlocal day_start_engine_state, day_events_today, day_start_idx
        info = choose_daily_gate_for_day(bar_day)
        day_gate = info["gate"]
        allow_buy = True
        allow_sell = True
        must_trade_dir = None
        if day_gate == "FORCE_BUY":
            allow_sell = False
            if engine.pos == 0:
                must_trade_dir = "buy"
        elif day_gate == "FORCE_SELL":
            allow_buy = False
            if engine.pos == 1:
                must_trade_dir = "sell"
        day_start_engine_state = copy.deepcopy(engine.state_dict())
        day_events_today = []
        day_start_idx = int(bar_idx)
        return info

    begin_day(pd.to_datetime(df_5m_idx.loc[0, "timestamp"]).normalize(), 0)
    current_day = pd.to_datetime(df_5m_idx.loc[0, "timestamp"]).normalize().date()

    for i in range(len(df_5m_idx)):
        bar_ts = pd.to_datetime(df_5m_idx.loc[i, "timestamp"])
        bar_day = bar_ts.normalize()
        in_sim = bar_ts >= sim_s

        if bar_day.date() != current_day:
            prev_day = pd.to_datetime(current_day)
            prev_day_ts = pd.to_datetime(current_day)

            if day_start_engine_state is not None and day_start_idx is not None and last_day_end_idx is not None:
                reward_map = evaluate_three_day_rewards_for_logging(
                    engine_state=day_start_engine_state,
                    day_events=day_events_today,
                    day_start_idx=day_start_idx,
                    day_end_idx=last_day_end_idx,
                    df_5m_idx=df_5m_idx,
                    next_open_by_idx=next_open_by_idx,
                    closes=closes,
                    buy_pack=buy_pack,
                    sell_pack=sell_pack,
                    buy_ret_th_live=buy_ret_th_live,
                    sell_ret_th_live=sell_ret_th_live,
                    fee_pct=fee_pct,
                )
                chosen_reward = reward_map[day_gate]["day_return"]
                best_action_ex_post = max(
                    ["FORCE_BUY", "FREE", "FORCE_SELL"],
                    key=lambda k: reward_map[k]["day_return"],
                )
                oracle_equity *= (1.0 + reward_map[best_action_ex_post]["day_return"])
                daily_reward_log.append({
                    "date": prev_day,
                    "p_day": float(p_by_day.get(prev_day.normalize(), np.nan)),
                    "buy_level": current_buy_level,
                    "sell_level": current_sell_level,
                    "chosen_action": day_gate,
                    "reward_force_buy": reward_map["FORCE_BUY"]["day_return"],
                    "reward_free": reward_map["FREE"]["day_return"],
                    "reward_force_sell": reward_map["FORCE_SELL"]["day_return"],
                    "chosen_reward": chosen_reward,
                    "best_action_ex_post": best_action_ex_post,
                    "oracle_equity": oracle_equity,
                    "close": float(day_close_map.get(prev_day.date(), np.nan)),
                    "buy_th_5m": buy_ret_th_live,
                    "sell_th_5m": sell_ret_th_live,
                })
                if policy_mode == "threshold_pair_bandit" and threshold_pair_bandit is not None and current_pair_action_idx is not None and current_pair_state_x is not None:
                    threshold_pair_bandit.update(current_pair_action_idx, current_pair_state_x, chosen_reward)

            label_bestlookahead_for_ready_points(
                bsp_rows=bsp_rows_5m,
                highs=highs,
                lows=lows,
                closes=closes,
                lookahead_days=lookahead_days_5m,
                bar_interval_minutes=5,
                current_bar_idx=i,
            )
            maybe_retrain_5m(prev_day_ts)
            if last_day_end_idx is not None:
                maybe_opt_5m_thresholds(last_day_end_idx)

            day_close = day_close_map.get(prev_day.date())
            equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash
            equity_peak = max(equity_peak, equity)
            daily_log.append({
                "date": prev_day,
                "equity": equity,
                "cash": engine.cash,
                "pos": engine.pos,
                "buy_th": buy_ret_th_live,
                "sell_th": sell_ret_th_live,
                "p_day": float(p_by_day.get(prev_day.normalize(), np.nan)),
                "daily_action": day_gate,
                "daily_buy_level": current_buy_level,
                "daily_sell_level": current_sell_level,
            })

            current_day = bar_day.date()
            begin_day(bar_day, i)

        last_day_end_idx = i
        if in_sim:
            engine.maybe_execute_pending(next_open_by_idx)

        klu = build_klu(
            df_5m_idx.loc[i, "timestamp"],
            df_5m_idx.loc[i, "Open"],
            df_5m_idx.loc[i, "High"],
            df_5m_idx.loc[i, "Low"],
            df_5m_idx.loc[i, "Close"],
            df_5m_idx.loc[i, "Volume"],
        )
        feed_chan_one(chan_5m, klu)

        new_rows = extract_bsp_rows_from_chan(chan_5m)
        if not new_rows:
            continue

        for r0 in new_rows:
            r = dict(r0)
            r.setdefault("timestamp", str(bar_ts))
            r.setdefault("klu_idx", i)
            if "direction" not in r or r["direction"] is None:
                if r.get("is_buy", None) is not None:
                    r["direction"] = "buy" if bool(r["is_buy"]) else "sell"
                else:
                    r["direction"] = "buy"
            r["direction"] = str(r["direction"]).lower()
            if "bsp_type" in r and r["bsp_type"] is not None:
                r["bsp_type"] = str(r["bsp_type"]).lower()
            r.setdefault("best_return_pct", np.nan)

            k = (int(r.get("klu_idx", -1)), str(r.get("direction")), str(r.get("bsp_type")))
            if k in seen_keys_5m:
                continue
            seen_keys_5m.add(k)
            bsp_rows_5m.append(r)
            if in_sim:
                day_events_today.append(copy.deepcopy(r))

            if not in_sim:
                continue

            d = str(r.get("direction", "buy")).lower()
            ki = int(r.get("klu_idx", i))
            if d == "buy" and not allow_buy:
                continue
            if d == "sell" and not allow_sell:
                continue
            if must_trade_dir is not None and d != must_trade_dir:
                continue

            if d == "buy" and engine.pos == 0 and buy_pack is not None:
                row_df = prepare_ml_dataset(pd.DataFrame([r]))
                for cc in buy_pack.feature_cols:
                    if cc not in row_df.columns:
                        row_df[cc] = 0.0
                pr = predict_ret(buy_pack, row_df)
                if pr >= float(buy_ret_th_live):
                    engine.place_order_for_next_bar(
                        side="buy",
                        seen_idx=ki,
                        reason=("ADAPTIVE_FORCE_BUY->first acceptable 5m signal" if day_gate == "FORCE_BUY" else "5m BUY signal"),
                        meta={
                            "ts": str(bar_ts),
                            "p_day": float(p_by_day.get(bar_day, np.nan)),
                            "pred": float(pr),
                            "th": float(buy_ret_th_live),
                            "gate": day_gate,
                        },
                    )
                    if must_trade_dir == "buy":
                        must_trade_dir = None

            elif d == "sell" and engine.pos == 1 and sell_pack is not None:
                row_df = prepare_ml_dataset(pd.DataFrame([r]))
                for cc in sell_pack.feature_cols:
                    if cc not in row_df.columns:
                        row_df[cc] = 0.0
                pr = predict_ret(sell_pack, row_df)
                if pr >= float(sell_ret_th_live):
                    engine.place_order_for_next_bar(
                        side="sell",
                        seen_idx=ki,
                        reason=("ADAPTIVE_FORCE_SELL->first acceptable 5m signal" if day_gate == "FORCE_SELL" else "5m SELL signal"),
                        meta={
                            "ts": str(bar_ts),
                            "p_day": float(p_by_day.get(bar_day, np.nan)),
                            "pred": float(pr),
                            "th": float(sell_ret_th_live),
                            "gate": day_gate,
                        },
                    )
                    if must_trade_dir == "sell":
                        must_trade_dir = None

    if current_day is not None and day_start_engine_state is not None and day_start_idx is not None and last_day_end_idx is not None:
        prev_day = pd.to_datetime(current_day)
        reward_map = evaluate_three_day_rewards_for_logging(
            engine_state=day_start_engine_state,
            day_events=day_events_today,
            day_start_idx=day_start_idx,
            day_end_idx=last_day_end_idx,
            df_5m_idx=df_5m_idx,
            next_open_by_idx=next_open_by_idx,
            closes=closes,
            buy_pack=buy_pack,
            sell_pack=sell_pack,
            buy_ret_th_live=buy_ret_th_live,
            sell_ret_th_live=sell_ret_th_live,
            fee_pct=fee_pct,
        )
        chosen_reward = reward_map[day_gate]["day_return"]
        best_action_ex_post = max(
            ["FORCE_BUY", "FREE", "FORCE_SELL"],
            key=lambda k: reward_map[k]["day_return"],
        )
        oracle_equity *= (1.0 + reward_map[best_action_ex_post]["day_return"])
        daily_reward_log.append({
            "date": prev_day,
            "p_day": float(p_by_day.get(prev_day.normalize(), np.nan)),
            "buy_level": current_buy_level,
            "sell_level": current_sell_level,
            "chosen_action": day_gate,
            "reward_force_buy": reward_map["FORCE_BUY"]["day_return"],
            "reward_free": reward_map["FREE"]["day_return"],
            "reward_force_sell": reward_map["FORCE_SELL"]["day_return"],
            "chosen_reward": chosen_reward,
            "best_action_ex_post": best_action_ex_post,
            "oracle_equity": oracle_equity,
            "close": float(day_close_map.get(prev_day.date(), np.nan)),
            "buy_th_5m": buy_ret_th_live,
            "sell_th_5m": sell_ret_th_live,
        })

        day_close = day_close_map.get(prev_day.date())
        equity = engine.mark_to_market(day_close) if day_close is not None else engine.cash
        daily_log.append({
            "date": prev_day,
            "equity": equity,
            "cash": engine.cash,
            "pos": engine.pos,
            "buy_th": buy_ret_th_live,
            "sell_th": sell_ret_th_live,
            "p_day": float(p_by_day.get(prev_day.normalize(), np.nan)),
            "daily_action": day_gate,
            "daily_buy_level": current_buy_level,
            "daily_sell_level": current_sell_level,
        })

    trades_df = pd.DataFrame(engine.trades)
    daily_df = pd.DataFrame(daily_log)
    reward_df = pd.DataFrame(daily_reward_log)

    trades_df.to_csv(os.path.join(output_dir, "trades.csv"), index=False)
    daily_df.to_csv(os.path.join(output_dir, "daily_log.csv"), index=False)
    reward_df.to_csv(os.path.join(output_dir, "daily_reward_log.csv"), index=False)

    if not daily_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(daily_df["date"]), daily_df["equity"], label="Strategy")
        if len(buy_hold) > 0:
            plt.plot(pd.to_datetime(buy_hold.index), buy_hold.values, label="Buy&Hold")
        plt.legend()
        plt.title("Equity vs Buy&Hold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "equity_vs_buyhold.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(daily_df["date"]), daily_df["p_day"], label="p_day")
        plt.plot(pd.to_datetime(daily_df["date"]), daily_df["daily_buy_level"], label="daily_buy_level")
        plt.plot(pd.to_datetime(daily_df["date"]), daily_df["daily_sell_level"], label="daily_sell_level")
        plt.legend()
        plt.title("Daily Probability and Adaptive Thresholds")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "daily_thresholds.png"), dpi=160)
        plt.close()

    return {
        "trades_df": trades_df,
        "daily_log_df": daily_df,
        "daily_reward_df": reward_df,
        "p_by_day": p_by_day,
        "buy_hold": buy_hold,
        "output_dir": output_dir,
    }
