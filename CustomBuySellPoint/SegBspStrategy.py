from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Mapping, Sequence

from .Signal import ChanSignal, OrderIntent
from .Strategy import BaseStrategy, StrategyContext


@dataclass(frozen=True)
class SegBspStrategyConfig:
    """Parameters for the first rule-based segment/BSP strategy."""

    entry_segment_directions: FrozenSet[str] = field(default_factory=lambda: frozenset({"up"}))
    entry_bsp_types: FrozenSet[str] = field(default_factory=lambda: frozenset({"1", "1p", "2", "2s", "3a", "3b"}))
    required_buy_signals: int = 1
    required_buy_signals_by_type: Mapping[str, int] = field(default_factory=dict)
    required_buy_signals_by_segment: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    buy_lookback_bars: int = 8
    buy_lookback_bars_by_segment: Mapping[str, int] = field(default_factory=dict)
    required_sell_signals: int = 3
    required_sell_signals_by_type: Mapping[str, int] = field(default_factory=dict)
    required_sell_signals_by_segment: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    required_sell_signals_by_entry_segment: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    sell_lookback_bars: int = 8
    sell_lookback_bars_by_segment: Mapping[str, int] = field(default_factory=dict)
    sell_lookback_bars_by_entry_segment: Mapping[str, int] = field(default_factory=dict)
    sell_bsp_types: FrozenSet[str] = field(default_factory=lambda: frozenset({"1", "1p", "2", "2s", "3a", "3b"}))
    exit_on_down_segment: bool = True
    exit_segment_directions_by_entry_segment: Mapping[str, FrozenSet[str]] = field(default_factory=dict)
    allow_unconfirmed_entry: bool = False
    reset_sell_count_on_buy: bool = True
    reset_buy_count_on_sell: bool = True
    position_fraction: float = 1.0

    def __post_init__(self):
        for name in ("required_buy_signals", "buy_lookback_bars", "required_sell_signals", "sell_lookback_bars"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.required_buy_signals > self.buy_lookback_bars:
            raise ValueError(
                "required_buy_signals cannot exceed buy_lookback_bars because signals are counted by distinct bars"
            )
        if self.required_sell_signals > self.sell_lookback_bars:
            raise ValueError(
                "required_sell_signals cannot exceed sell_lookback_bars because signals are counted by distinct bars"
            )
        for mapping_name, mapping, lookback in (
            ("required_buy_signals_by_type", self.required_buy_signals_by_type, self.buy_lookback_bars),
            ("required_sell_signals_by_type", self.required_sell_signals_by_type, self.sell_lookback_bars),
        ):
            for bsp_type, required in mapping.items():
                if int(required) <= 0:
                    raise ValueError(f"{mapping_name}[{bsp_type!r}] must be greater than zero")
                if int(required) > lookback:
                    raise ValueError(
                        f"{mapping_name}[{bsp_type!r}] cannot exceed its lookback window "
                        "because signals are counted by distinct bars"
                    )
        for segment, lookback in self.buy_lookback_bars_by_segment.items():
            if int(lookback) <= 0:
                raise ValueError(f"buy_lookback_bars_by_segment[{segment!r}] must be greater than zero")
        for segment, lookback in self.sell_lookback_bars_by_entry_segment.items():
            if int(lookback) <= 0:
                raise ValueError(f"sell_lookback_bars_by_entry_segment[{segment!r}] must be greater than zero")
        for segment, lookback in self.sell_lookback_bars_by_segment.items():
            if int(lookback) <= 0:
                raise ValueError(f"sell_lookback_bars_by_segment[{segment!r}] must be greater than zero")
        for mapping_name, nested, lookbacks, fallback in (
            ("required_buy_signals_by_segment", self.required_buy_signals_by_segment,
             self.buy_lookback_bars_by_segment, self.buy_lookback_bars),
            ("required_sell_signals_by_segment", self.required_sell_signals_by_segment,
             self.sell_lookback_bars_by_segment, self.sell_lookback_bars),
            ("required_sell_signals_by_entry_segment", self.required_sell_signals_by_entry_segment,
             self.sell_lookback_bars_by_entry_segment, self.sell_lookback_bars),
        ):
            for segment, requirements in nested.items():
                lookback = int(lookbacks.get(segment, fallback))
                for bsp_type, required in requirements.items():
                    if int(required) <= 0 or int(required) > lookback:
                        raise ValueError(
                            f"{mapping_name}[{segment!r}][{bsp_type!r}] must be between 1 and {lookback}"
                        )


class SegBspStrategy(BaseStrategy):
    """Long-only strategy driven by first-seen segment direction and BSP events."""

    def __init__(self, config: SegBspStrategyConfig | None = None):
        self.config = config or SegBspStrategyConfig()
        self._pending_entry_segment: str | None = None
        self._pending_entry_type: str | None = None
        self._active_entry_segment: str | None = None
        self._active_entry_type: str | None = None

    def reset(self) -> None:
        self._pending_entry_segment = None
        self._pending_entry_type = None
        self._active_entry_segment = None
        self._active_entry_type = None

    def on_fill(self, fill, context: StrategyContext) -> None:
        if fill.get("side") == "buy":
            self._active_entry_segment = self._pending_entry_segment
            self._active_entry_type = self._pending_entry_type
            self._pending_entry_segment = None
            self._pending_entry_type = None
        elif fill.get("side") == "sell":
            self._active_entry_segment = None
            self._active_entry_type = None

    @staticmethod
    def _distinct_signal_bars(context, direction, lookback, bsp_types, segment_direction=None) -> int:
        start = context.bar_index - int(lookback) + 1
        return len({
            event.bar_index
            for event in context.signals.events
            if start <= event.bar_index <= context.bar_index
            and event.direction == direction
            and event.bsp_type in bsp_types
            and (segment_direction is None or event.segment_direction == segment_direction)
        })

    def on_bsp(self, signal: ChanSignal, context: StrategyContext) -> Sequence[OrderIntent]:
        cfg = self.config
        if cfg.reset_sell_count_on_buy and signal.direction == "buy":
            context.signals.clear_direction("sell")
        if cfg.reset_buy_count_on_sell and signal.direction == "sell":
            context.signals.clear_direction("buy")

        if context.position.is_flat:
            allowed_segment = signal.segment_direction in cfg.entry_segment_directions
            if cfg.allow_unconfirmed_entry and signal.segment_direction is None:
                allowed_segment = True
            segment_buy_requirements = cfg.required_buy_signals_by_segment.get(
                signal.segment_direction or "", {}
            )
            buy_required = int(segment_buy_requirements.get(
                signal.bsp_type,
                cfg.required_buy_signals_by_type.get(signal.bsp_type, cfg.required_buy_signals),
            ))
            buy_lookback = int(cfg.buy_lookback_bars_by_segment.get(
                signal.segment_direction or "", cfg.buy_lookback_bars
            ))
            buy_types_to_count = (
                frozenset({signal.bsp_type})
                if signal.bsp_type in cfg.required_buy_signals_by_type
                or signal.bsp_type in segment_buy_requirements
                else cfg.entry_bsp_types
            )
            buy_count = self._distinct_signal_bars(
                context=context,
                direction="buy",
                lookback=buy_lookback,
                bsp_types=buy_types_to_count,
                segment_direction=(signal.segment_direction if segment_buy_requirements else None),
            )
            if (
                signal.direction == "buy"
                and signal.bsp_type in cfg.entry_bsp_types
                and allowed_segment
                and buy_count >= buy_required
            ):
                self._pending_entry_segment = signal.segment_direction
                self._pending_entry_type = signal.bsp_type
                return (OrderIntent.buy(
                    reason=(
                        f"{buy_count}/{buy_required} type-{signal.bsp_type} buy-signal bars "
                        f"in {buy_lookback}-bar window; "
                        f"BSP {signal.bsp_type} in {signal.segment_direction or 'unconfirmed'} segment"
                    ),
                    signal=signal,
                    position_fraction=cfg.position_fraction,
                ),)
            return ()

        if not context.position.is_long:
            return ()

        explicit_exit_directions = cfg.exit_segment_directions_by_entry_segment.get(
            self._active_entry_segment or ""
        )
        if explicit_exit_directions is not None:
            if signal.segment_direction in explicit_exit_directions:
                return (OrderIntent.sell(
                    f"entry segment {self._active_entry_segment}; exit segment {signal.segment_direction}", signal
                ),)
        elif cfg.exit_on_down_segment and signal.segment_direction == "down":
            return (OrderIntent.sell("segment direction down", signal),)

        if signal.direction == "sell":
            current_segment = signal.segment_direction or ""
            current_segment_requirements = cfg.required_sell_signals_by_segment.get(
                current_segment, {}
            )
            legacy_entry_segment_requirements = cfg.required_sell_signals_by_entry_segment.get(
                self._active_entry_segment or "", {}
            )
            if signal.bsp_type in current_segment_requirements:
                sell_required = int(current_segment_requirements[signal.bsp_type])
            elif signal.bsp_type in legacy_entry_segment_requirements:
                sell_required = int(legacy_entry_segment_requirements[signal.bsp_type])
            else:
                sell_required = int(cfg.required_sell_signals_by_type.get(
                    signal.bsp_type, cfg.required_sell_signals
                ))
            if current_segment in cfg.sell_lookback_bars_by_segment:
                sell_lookback = int(cfg.sell_lookback_bars_by_segment[current_segment])
            else:
                sell_lookback = int(cfg.sell_lookback_bars_by_entry_segment.get(
                    self._active_entry_segment or "", cfg.sell_lookback_bars
                ))
            sell_types_to_count = (
                frozenset({signal.bsp_type})
                if signal.bsp_type in cfg.required_sell_signals_by_type
                or signal.bsp_type in current_segment_requirements
                or signal.bsp_type in legacy_entry_segment_requirements
                else cfg.sell_bsp_types
            )
            sell_count = self._distinct_signal_bars(
                context=context,
                direction="sell",
                lookback=sell_lookback,
                bsp_types=sell_types_to_count,
                segment_direction=(current_segment if current_segment_requirements else None),
            )
            if sell_count >= sell_required:
                return (OrderIntent.sell(
                    f"{sell_count}/{sell_required} type-{signal.bsp_type} sell-signal bars "
                    f"in {sell_lookback}-bar window in current {current_segment or 'unknown'} segment",
                    signal,
                ),)

        return ()
