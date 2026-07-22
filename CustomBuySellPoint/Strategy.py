from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .Signal import ChanSignal, OrderIntent, SignalTracker


@dataclass(frozen=True)
class PositionView:
    """Read-only position state exposed to strategies."""

    side: str = "flat"
    quantity: float = 0.0
    entry_price: float | None = None
    entry_bar: int | None = None
    peak_price: float | None = None

    @property
    def is_flat(self) -> bool:
        return self.side == "flat" or self.quantity <= 0

    @property
    def is_long(self) -> bool:
        return self.side == "long" and self.quantity > 0


@dataclass(frozen=True)
class StrategyContext:
    """All causal state available when a strategy evaluates one event."""

    timestamp: pd.Timestamp
    bar_index: int
    bar: Mapping[str, Any]
    position: PositionView
    signals: SignalTracker


class BaseStrategy(ABC):
    """Abstract deterministic strategy independent of execution and accounting."""

    def on_bar(self, context: StrategyContext) -> Sequence[OrderIntent]:
        return ()

    @abstractmethod
    def on_bsp(self, signal: ChanSignal, context: StrategyContext) -> Sequence[OrderIntent]:
        raise NotImplementedError

    def on_fill(self, fill: Mapping[str, Any], context: StrategyContext) -> None:
        return None

    def reset(self) -> None:
        return None
