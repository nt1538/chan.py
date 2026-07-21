import argparse
from pathlib import Path

import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise ValueError(
        f"Could not find {label} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_enriched_trades(
    trades_csv: str | Path,
    output_csv: str | Path,
    bars_csv: str | Path | None = None,
    initial_capital: float | None = None,
) -> pd.DataFrame:
    """
    Write a new trade-level CSV with daily gate rewards appended to every trade row.

    The input trades.csv must contain:
        exec_ts, pnl, fee, exec_px, qty

    If bars_csv is provided, force-buy and force-sell rewards are calculated from
    the first open and last close of each day. If bars_csv is omitted, only the
    free-trade reward can be calculated from trades.csv.

    Reward units:
        reward_free_trade = daily sum of trade pnl - daily fees
        reward_force_buy  = daily capital * open-to-close return
        reward_force_sell = -daily capital * open-to-close return

    Daily capital is inferred from that day's max trade notional:
        abs(exec_px * qty)

    If a date has no inferable notional and initial_capital is provided, the
    script uses initial_capital for that date.
    """
    trades = _read_csv(trades_csv)

    exec_ts_col = _pick_column(trades, ["exec_ts", "timestamp", "ts", "time"], "trade timestamp")
    pnl_col = _pick_column(trades, ["pnl", "profit", "profit_loss"], "trade pnl")
    fee_col = _pick_column(trades, ["fee", "fees", "commission"], "trade fee")
    exec_px_col = _pick_column(trades, ["exec_px", "price", "exec_price"], "execution price")
    qty_col = _pick_column(trades, ["qty", "quantity", "shares"], "quantity")

    trades[exec_ts_col] = pd.to_datetime(trades[exec_ts_col], errors="coerce")
    trades["_reward_date"] = trades[exec_ts_col].dt.normalize()

    trades[pnl_col] = pd.to_numeric(trades[pnl_col], errors="coerce").fillna(0.0)
    trades[fee_col] = pd.to_numeric(trades[fee_col], errors="coerce").fillna(0.0)
    trades[exec_px_col] = pd.to_numeric(trades[exec_px_col], errors="coerce")
    trades[qty_col] = pd.to_numeric(trades[qty_col], errors="coerce")

    trades["_notional"] = (trades[exec_px_col] * trades[qty_col]).abs()

    daily_rewards = (
        trades.groupby("_reward_date", dropna=False)
        .agg(
            reward_free_trade=(pnl_col, "sum"),
            daily_fee=(fee_col, "sum"),
            daily_trade_count=(pnl_col, "size"),
            daily_capital=("_notional", "max"),
        )
        .reset_index()
    )
    daily_rewards["reward_free_trade"] = daily_rewards["reward_free_trade"] - daily_rewards["daily_fee"]

    if initial_capital is not None:
        daily_rewards["daily_capital"] = daily_rewards["daily_capital"].fillna(initial_capital)

    if bars_csv is not None:
        bars = _read_csv(bars_csv)
        bar_ts_col = _pick_column(bars, ["timestamp", "datetime", "date", "ts", "time"], "bar timestamp")
        open_col = _pick_column(bars, ["Open", "open"], "bar open")
        close_col = _pick_column(bars, ["Close", "close"], "bar close")

        bars[bar_ts_col] = pd.to_datetime(bars[bar_ts_col], errors="coerce")
        bars[open_col] = pd.to_numeric(bars[open_col], errors="coerce")
        bars[close_col] = pd.to_numeric(bars[close_col], errors="coerce")
        bars["_reward_date"] = bars[bar_ts_col].dt.normalize()

        daily_bars = (
            bars.dropna(subset=[bar_ts_col, open_col, close_col])
            .sort_values(bar_ts_col)
            .groupby("_reward_date")
            .agg(day_open=(open_col, "first"), day_close=(close_col, "last"))
            .reset_index()
        )

        daily_rewards = daily_rewards.merge(daily_bars, on="_reward_date", how="left")
        day_return = daily_rewards["day_close"] / daily_rewards["day_open"] - 1.0
        daily_rewards["reward_force_buy"] = daily_rewards["daily_capital"] * day_return
        daily_rewards["reward_force_sell"] = -daily_rewards["daily_capital"] * day_return
    else:
        daily_rewards["day_open"] = pd.NA
        daily_rewards["day_close"] = pd.NA
        daily_rewards["reward_force_buy"] = pd.NA
        daily_rewards["reward_force_sell"] = pd.NA

    reward_cols = ["reward_force_buy", "reward_force_sell", "reward_free_trade"]
    rewards_for_choice = daily_rewards[reward_cols].apply(pd.to_numeric, errors="coerce")
    daily_rewards["best_reward"] = rewards_for_choice.max(axis=1, skipna=True)
    daily_rewards["best_gate"] = rewards_for_choice.idxmax(axis=1, skipna=True).map(
        {
            "reward_force_buy": "FORCE_BUY",
            "reward_force_sell": "FORCE_SELL",
            "reward_free_trade": "FREE",
        }
    )

    enriched = trades.merge(
        daily_rewards[
            [
                "_reward_date",
                "reward_force_buy",
                "reward_force_sell",
                "reward_free_trade",
                "best_gate",
                "best_reward",
                "daily_capital",
                "daily_fee",
                "daily_trade_count",
                "day_open",
                "day_close",
            ]
        ],
        on="_reward_date",
        how="left",
    )

    enriched = enriched.drop(columns=["_reward_date", "_notional"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append daily force-buy, force-sell, and free-trade rewards to each trade row."
    )
    parser.add_argument("--trades", required=True, help="Path to trades.csv.")
    parser.add_argument("--out", required=True, help="Path for enriched output CSV.")
    parser.add_argument(
        "--bars",
        default=None,
        help="Optional 5m or daily OHLC CSV used to calculate force-buy/force-sell rewards.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=None,
        help="Fallback capital when daily notional cannot be inferred from trades.",
    )
    args = parser.parse_args()

    enriched = build_enriched_trades(
        trades_csv=args.trades,
        output_csv=args.out,
        bars_csv=args.bars,
        initial_capital=args.initial_capital,
    )
    print(f"Wrote {len(enriched):,} rows to {args.out}")


if __name__ == "__main__":
    main()
