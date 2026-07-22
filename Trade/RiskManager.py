from dataclasses import dataclass

from CustomBuySellPoint.Signal import OrderIntent
from CustomBuySellPoint.Strategy import PositionView


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    max_holding_bars: int | None = None


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def evaluate(self, position: PositionView, price: float, bar_index: int) -> OrderIntent | None:
        if not position.is_long or position.entry_price is None:
            return None
        entry = position.entry_price
        pnl_pct = price / entry - 1.0
        if self.config.stop_loss_pct is not None and pnl_pct <= -abs(self.config.stop_loss_pct):
            return OrderIntent.sell("stop_loss")
        if self.config.take_profit_pct is not None and pnl_pct >= abs(self.config.take_profit_pct):
            return OrderIntent.sell("take_profit")
        if (
            self.config.trailing_stop_pct is not None
            and position.peak_price is not None
            and price / position.peak_price - 1.0 <= -abs(self.config.trailing_stop_pct)
        ):
            return OrderIntent.sell("trailing_stop")
        if (
            self.config.max_holding_bars is not None
            and position.entry_bar is not None
            and bar_index - position.entry_bar >= self.config.max_holding_bars
        ):
            return OrderIntent.sell("max_holding_bars")
        return None
