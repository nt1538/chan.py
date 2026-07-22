from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestChanConfig:
    initial_capital: float = 100_000.0
    fee_pct: float = 0.0005
    slippage_pct: float = 0.0
    timestamp_col: str = "timestamp"
    open_col: str = "open"
    close_col: str = "close"
    close_at_end: bool = True
    bars_per_year: int = 19_656  # approximately 252 * 78 for US 5-minute bars
