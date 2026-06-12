from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DIR = Path("output_adaptive_reward_resumed_fresh_TQQQ_252days_at_2020")
OUTPUT_DIR = Path("output_replay_new_daily_gate_old_5m_signals_lookback30")

DAILY_REWARD_PATH = SOURCE_DIR / "daily_reward_log.csv"
OLD_TRADES_PATH = SOURCE_DIR / "trades.csv"
FIVE_MIN_PATH = Path("DataAPI/data/TQQQ_5M.csv")

INITIAL_CAPITAL = 100000.0
FEE_PCT = 0.0

LOOKBACK_DAYS = 30
MIN_OBS = 30
MIN_GAP = 0.02


def action_from_p(p_val: float, buy_level: float, sell_level: float) -> str:
    if not np.isfinite(p_val):
        return "NO_P"
    if p_val <= buy_level:
        return "FORCE_BUY"
    if p_val >= sell_level:
        return "FORCE_SELL"
    return "FREE"


def build_daily_gate_replay() -> pd.DataFrame:
    df = pd.read_csv(DAILY_REWARD_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["p_day"] = pd.to_numeric(df["p_day"], errors="coerce")

    for c in ["reward_force_buy", "reward_free", "reward_force_sell"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = df.sort_values("date").reset_index(drop=True)

    p = df["p_day"].to_numpy(float)
    r_buy = df["reward_force_buy"].to_numpy(float)
    r_free = df["reward_free"].to_numpy(float)
    r_sell = df["reward_force_sell"].to_numpy(float)

    buy_grid = np.round(np.arange(0.05, 0.3501, 0.005), 6)
    sell_grid = np.round(np.arange(0.15, 0.6001, 0.005), 6)

    pairs = np.array(
        [(b, s) for b, s in product(buy_grid, sell_grid) if s - b >= MIN_GAP],
        dtype=float,
    )
    buy_levels = pairs[:, 0]
    sell_levels = pairs[:, 1]

    rows = []
    for i in range(len(df)):
        start = max(0, i - LOOKBACK_DAYS)
        hist_p = p[start:i]
        hist_buy = r_buy[start:i]
        hist_free = r_free[start:i]
        hist_sell = r_sell[start:i]

        valid = np.isfinite(hist_p)
        hist_p = hist_p[valid]
        hist_buy = hist_buy[valid]
        hist_free = hist_free[valid]
        hist_sell = hist_sell[valid]

        if len(hist_p) < MIN_OBS:
            buy_level = 0.20
            sell_level = 0.30
            best_score = np.nan
            window_size = len(hist_p)
        else:
            buy_mask = hist_p[:, None] <= buy_levels[None, :]
            sell_mask = hist_p[:, None] >= sell_levels[None, :]
            rewards = np.where(
                buy_mask,
                hist_buy[:, None],
                np.where(sell_mask, hist_sell[:, None], hist_free[:, None]),
            )
            scores = rewards.sum(axis=0)
            best_idx = int(np.nanargmax(scores))
            buy_level = float(buy_levels[best_idx])
            sell_level = float(sell_levels[best_idx])
            best_score = float(scores[best_idx])
            window_size = len(hist_p)

        current_p = float(p[i]) if np.isfinite(p[i]) else np.nan
        daily_action = action_from_p(current_p, buy_level, sell_level)
        if daily_action == "FORCE_BUY":
            daily_reward = float(r_buy[i])
        elif daily_action == "FORCE_SELL":
            daily_reward = float(r_sell[i])
        else:
            daily_reward = float(r_free[i])

        rows.append(
            {
                "date": df.loc[i, "date"],
                "p_day": current_p,
                "daily_buy_level": buy_level,
                "daily_sell_level": sell_level,
                "threshold_gap": sell_level - buy_level,
                "daily_action": daily_action,
                "daily_reward_from_reward_log": daily_reward,
                "score_on_lookback": best_score,
                "window_size": window_size,
                "reward_force_buy": float(r_buy[i]),
                "reward_free": float(r_free[i]),
                "reward_force_sell": float(r_sell[i]),
                "best_action_ex_post": df.loc[i, "best_action_ex_post"]
                if "best_action_ex_post" in df.columns
                else None,
            }
        )

    return pd.DataFrame(rows)


def load_daily_closes() -> pd.Series:
    bars = pd.read_csv(FIVE_MIN_PATH)
    ts_col = next((c for c in bars.columns if c.lower() in {"timestamp", "date", "datetime", "time"}), bars.columns[0])
    close_col = next((c for c in bars.columns if c.lower() in {"close", "adj close", "adj_close", "c"}), None)
    if close_col is None:
        raise ValueError(f"{FIVE_MIN_PATH} must contain a Close column")
    bars[ts_col] = pd.to_datetime(bars[ts_col], errors="coerce")
    bars[close_col] = pd.to_numeric(bars[close_col], errors="coerce")
    bars = bars.dropna(subset=[ts_col, close_col]).sort_values(ts_col)
    bars["date"] = bars[ts_col].dt.normalize()
    return bars.groupby("date")[close_col].last()


def gate_allows_trade(gate: str, side: str) -> bool:
    gate = str(gate)
    side = str(side).lower()
    if gate == "FORCE_BUY" and side == "sell":
        return False
    if gate == "FORCE_SELL" and side == "buy":
        return False
    return True


def replay_old_trade_points_with_new_gate(gates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(OLD_TRADES_PATH)
    trades["exec_ts"] = pd.to_datetime(trades.get("exec_ts", trades.get("ts")), errors="coerce")
    trades["ts"] = pd.to_datetime(trades["ts"], errors="coerce")
    trades["exec_px"] = pd.to_numeric(trades["exec_px"], errors="coerce")
    trades = trades.dropna(subset=["exec_ts", "exec_px"]).sort_values("exec_ts").reset_index(drop=True)
    trades["date"] = trades["exec_ts"].dt.normalize()

    gate_by_day = gates.set_index("date").to_dict("index")
    closes = load_daily_closes()
    all_days = sorted(set(gates["date"]).union(set(trades["date"])).union(set(closes.index)))

    cash = float(INITIAL_CAPITAL)
    pos = 0
    qty = 0.0
    entry_px = np.nan
    entry_ts = pd.NaT
    replay_trades = []
    daily_rows = []
    trade_i = 0

    for day in all_days:
        day_trades = trades[trades["date"].eq(day)]
        for _, tr in day_trades.iterrows():
            side = str(tr["side"]).lower()
            px = float(tr["exec_px"])
            gate_info = gate_by_day.get(day, {})
            gate = gate_info.get("daily_action", "FREE")

            reason = "new daily gate allows old 5m trade point"
            if not gate_allows_trade(gate, side):
                continue

            if side == "buy" and pos == 0:
                notional = cash
                spend = notional * (1.0 + FEE_PCT)
                if spend > cash:
                    spend = cash
                    notional = spend / (1.0 + FEE_PCT)
                qty = notional / px if px > 0 else 0.0
                cash -= spend
                pos = 1
                entry_px = px
                entry_ts = tr["exec_ts"]
                replay_trades.append(
                    {
                        "side": "buy",
                        "signal_ts": tr["ts"],
                        "exec_ts": tr["exec_ts"],
                        "exec_px": px,
                        "qty": qty,
                        "fee": spend - notional,
                        "gate": gate,
                        "p_day": gate_info.get("p_day", np.nan),
                        "daily_buy_level": gate_info.get("daily_buy_level", np.nan),
                        "daily_sell_level": gate_info.get("daily_sell_level", np.nan),
                        "old_pred": tr.get("pred", np.nan),
                        "old_th": tr.get("th", np.nan),
                        "old_gate": tr.get("gate", None),
                        "reason": reason,
                    }
                )
            elif side == "sell" and pos == 1:
                proceeds = qty * px * (1.0 - FEE_PCT)
                fee = qty * px - proceeds
                pnl = proceeds - (qty * float(entry_px))
                cash += proceeds
                replay_trades.append(
                    {
                        "side": "sell",
                        "signal_ts": tr["ts"],
                        "exec_ts": tr["exec_ts"],
                        "exec_px": px,
                        "qty": qty,
                        "fee": fee,
                        "gate": gate,
                        "p_day": gate_info.get("p_day", np.nan),
                        "daily_buy_level": gate_info.get("daily_buy_level", np.nan),
                        "daily_sell_level": gate_info.get("daily_sell_level", np.nan),
                        "old_pred": tr.get("pred", np.nan),
                        "old_th": tr.get("th", np.nan),
                        "old_gate": tr.get("gate", None),
                        "entry_px": entry_px,
                        "entry_ts": entry_ts,
                        "pnl": pnl,
                        "reason": reason,
                    }
                )
                pos = 0
                qty = 0.0
                entry_px = np.nan
                entry_ts = pd.NaT

        close_px = float(closes.get(day, np.nan))
        equity = cash if pos == 0 or not np.isfinite(close_px) else cash + qty * close_px
        gate_info = gate_by_day.get(day, {})
        daily_rows.append(
            {
                "date": day,
                "equity": equity,
                "cash": cash,
                "pos": pos,
                "qty": qty,
                "entry_px": entry_px,
                "close_px": close_px,
                "p_day": gate_info.get("p_day", np.nan),
                "daily_action": gate_info.get("daily_action", None),
                "daily_buy_level": gate_info.get("daily_buy_level", np.nan),
                "daily_sell_level": gate_info.get("daily_sell_level", np.nan),
                "threshold_gap": gate_info.get("threshold_gap", np.nan),
            }
        )
        trade_i += len(day_trades)

    return pd.DataFrame(daily_rows), pd.DataFrame(replay_trades)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gates = build_daily_gate_replay()
    daily_log, trades = replay_old_trade_points_with_new_gate(gates)

    gates.to_csv(OUTPUT_DIR / "daily_gate_replay_lookback30.csv", index=False)
    daily_log.to_csv(OUTPUT_DIR / "daily_log.csv", index=False)
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "initial_capital": INITIAL_CAPITAL,
                "final_equity": daily_log["equity"].dropna().iloc[-1] if not daily_log.empty else np.nan,
                "num_trades": len(trades),
                "num_buys": int((trades["side"] == "buy").sum()) if not trades.empty else 0,
                "num_sells": int((trades["side"] == "sell").sum()) if not trades.empty else 0,
                "lookback_days": LOOKBACK_DAYS,
                "min_obs": MIN_OBS,
                "min_gap": MIN_GAP,
                "source_daily_reward": str(DAILY_REWARD_PATH),
                "source_old_trades": str(OLD_TRADES_PATH),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print(f"saved folder: {OUTPUT_DIR}")
    print(summary.to_string(index=False))
    print(daily_log.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
