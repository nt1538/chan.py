from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Sequence

from .Signal import ChanSignal, OrderIntent
from .Strategy import BaseStrategy, StrategyContext


@dataclass(frozen=True)
class SegBspStrategyConfig:
    """Parameters for the first rule-based segment/BSP strategy."""

    entry_segment_directions: FrozenSet[str] = field(default_factory=lambda: frozenset({"up"}))
    entry_bsp_types: FrozenSet[str] = field(default_factory=lambda: frozenset({"1", "1p", "2", "2s", "3a", "3b"}))
    required_sell_signals: int = 3
    sell_lookback_bars: int = 8
    sell_bsp_types: FrozenSet[str] = field(default_factory=lambda: frozenset({"1", "1p", "2", "2s", "3a", "3b"}))
    exit_on_down_segment: bool = True
    allow_unconfirmed_entry: bool = False
    reset_sell_count_on_buy: bool = True
    position_fraction: float = 1.0


class SegBspStrategy(BaseStrategy):
    """Long-only strategy driven by first-seen segment direction and BSP events."""

    def __init__(self, config: SegBspStrategyConfig | None = None):
        self.config = config or SegBspStrategyConfig()

    def on_bsp(self, signal: ChanSignal, context: StrategyContext) -> Sequence[OrderIntent]:
        cfg = self.config
        if cfg.reset_sell_count_on_buy and signal.direction == "buy":
            context.signals.clear_direction("sell")

        if context.position.is_flat:
            allowed_segment = signal.segment_direction in cfg.entry_segment_directions
            if cfg.allow_unconfirmed_entry and signal.segment_direction is None:
                allowed_segment = True
            if (
                signal.direction == "buy"
                and signal.bsp_type in cfg.entry_bsp_types
                and allowed_segment
            ):
                return (OrderIntent.buy(
                    reason=f"buy BSP {signal.bsp_type} in {signal.segment_direction or 'unconfirmed'} segment",
                    signal=signal,
                    position_fraction=cfg.position_fraction,
                ),)
            return ()

        if not context.position.is_long:
            return ()

        if cfg.exit_on_down_segment and signal.segment_direction == "down":
            return (OrderIntent.sell("segment direction down", signal),)

        if signal.direction == "sell":
            sell_count = context.signals.distinct_signal_bars(
                direction="sell",
                current_bar=context.bar_index,
                lookback_bars=cfg.sell_lookback_bars,
                bsp_types=cfg.sell_bsp_types,
            )
            if sell_count >= cfg.required_sell_signals:
                return (OrderIntent.sell(
                    f"{sell_count} sell-signal bars in {cfg.sell_lookback_bars}-bar window",
                    signal,
                ),)

        return ()
