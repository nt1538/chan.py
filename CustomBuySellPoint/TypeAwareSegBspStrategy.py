from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Mapping, Sequence

from .Signal import ChanSignal, OrderIntent
from .Strategy import BaseStrategy, StrategyContext


@dataclass(frozen=True)
class EntryPolicy:
    allowed_segment_directions: FrozenSet[str] = field(default_factory=lambda: frozenset({"up"}))
    position_fraction: float = 1.0
    allow_unconfirmed_segment: bool = False


@dataclass(frozen=True)
class ExitPolicy:
    sell_score_required: float = 2.0
    sell_lookback_bars: int = 8
    max_holding_bars: int | None = None
    exit_on_down_segment: bool = True


def _default_entry_policies() -> dict[str, EntryPolicy]:
    return {
        "1": EntryPolicy(position_fraction=1.0),
        "1p": EntryPolicy(position_fraction=1.0),
        "2": EntryPolicy(position_fraction=1.0),
        "2s": EntryPolicy(position_fraction=1.0),
        "3a": EntryPolicy(position_fraction=1.0),
        "3b": EntryPolicy(position_fraction=1.0),
    }


def _default_exit_policies() -> dict[str, ExitPolicy]:
    return {
        # Type-1 entries react to concentrated counter-signals in five bars.
        "1": ExitPolicy(sell_score_required=2.0, sell_lookback_bars=5, max_holding_bars=78 * 2),
        "1p": ExitPolicy(sell_score_required=2.0, sell_lookback_bars=5, max_holding_bars=78 * 2),
        # Type-2 entries are allowed a wider eight-bar confirmation window.
        "2": ExitPolicy(sell_score_required=2.5, sell_lookback_bars=8, max_holding_bars=78 * 3),
        "2s": ExitPolicy(sell_score_required=2.5, sell_lookback_bars=8, max_holding_bars=78 * 3),
        "3a": ExitPolicy(sell_score_required=1.5, sell_lookback_bars=5, max_holding_bars=78),
        "3b": ExitPolicy(sell_score_required=1.5, sell_lookback_bars=5, max_holding_bars=78),
    }


@dataclass(frozen=True)
class TypeAwareSegBspStrategyConfig:
    entry_policies: Mapping[str, EntryPolicy] = field(default_factory=_default_entry_policies)
    exit_policies: Mapping[str, ExitPolicy] = field(default_factory=_default_exit_policies)
    sell_signal_weights: Mapping[str, float] = field(default_factory=lambda: {
        "1": 2.0, "1p": 1.5, "2": 1.25, "2s": 1.0, "3a": 0.75, "3b": 0.75,
    })
    reset_sell_history_on_buy: bool = True


class TypeAwareSegBspStrategy(BaseStrategy):
    """Use the filled entry BSP type to select position size and exit behavior."""

    def __init__(self, config: TypeAwareSegBspStrategyConfig | None = None):
        self.config = config or TypeAwareSegBspStrategyConfig()
        self._pending_entry_type: str | None = None
        self._active_entry_type: str | None = None

    @property
    def active_entry_type(self) -> str | None:
        return self._active_entry_type

    def reset(self) -> None:
        self._pending_entry_type = None
        self._active_entry_type = None

    def on_fill(self, fill, context: StrategyContext) -> None:
        if fill.get("side") == "buy":
            self._active_entry_type = self._pending_entry_type
            self._pending_entry_type = None
        elif fill.get("side") == "sell":
            self._active_entry_type = None

    def _exit_policy(self) -> ExitPolicy:
        return self.config.exit_policies.get(self._active_entry_type or "", ExitPolicy())

    def _sell_score(self, context: StrategyContext, lookback: int) -> float:
        start = context.bar_index - lookback + 1
        # Multiple BSP labels on the same bar count once, using that bar's strongest label.
        score_by_bar: dict[int, float] = {}
        for event in context.signals.events:
            if event.direction != "sell" or not (start <= event.bar_index <= context.bar_index):
                continue
            weight = float(self.config.sell_signal_weights.get(event.bsp_type, 0.0))
            score_by_bar[event.bar_index] = max(score_by_bar.get(event.bar_index, 0.0), weight)
        return sum(score_by_bar.values())

    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]:
        if not context.position.is_long:
            return ()
        policy = self._exit_policy()
        if (
            policy.max_holding_bars is not None
            and context.position.entry_bar is not None
            and context.bar_index - context.position.entry_bar >= policy.max_holding_bars
        ):
            return (OrderIntent.sell(
                f"type {self._active_entry_type} max holding {policy.max_holding_bars} bars"
            ),)
        return ()

    def on_bsp(self, signal: ChanSignal, context: StrategyContext) -> Sequence[OrderIntent]:
        if context.position.is_flat:
            if signal.direction != "buy":
                return ()
            policy = self.config.entry_policies.get(signal.bsp_type)
            if policy is None:
                return ()
            allowed = signal.segment_direction in policy.allowed_segment_directions
            if signal.segment_direction is None and policy.allow_unconfirmed_segment:
                allowed = True
            if not allowed:
                return ()
            if self.config.reset_sell_history_on_buy:
                context.signals.clear_direction("sell")
            self._pending_entry_type = signal.bsp_type
            return (OrderIntent.buy(
                reason=f"type-aware entry: BSP {signal.bsp_type}, segment {signal.segment_direction}",
                signal=signal,
                position_fraction=policy.position_fraction,
            ),)

        if not context.position.is_long:
            return ()
        policy = self._exit_policy()
        if policy.exit_on_down_segment and signal.segment_direction == "down":
            return (OrderIntent.sell(
                f"type {self._active_entry_type} exit: segment direction down", signal
            ),)
        if signal.direction == "sell":
            score = self._sell_score(context, policy.sell_lookback_bars)
            if score >= policy.sell_score_required:
                return (OrderIntent.sell(
                    f"type {self._active_entry_type} exit: sell score {score:.2f} >= "
                    f"{policy.sell_score_required:.2f} in {policy.sell_lookback_bars} bars",
                    signal,
                ),)
        return ()
