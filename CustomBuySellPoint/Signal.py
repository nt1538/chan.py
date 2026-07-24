from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable, Mapping, Optional

import pandas as pd


def _optional_text(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


@dataclass(frozen=True)
class ChanSignal:
    """Point-in-time Chan signal created from a BSP first-seen snapshot."""

    timestamp: pd.Timestamp
    bar_index: int
    direction: str
    bsp_type: str
    price: float
    bi_direction: Optional[str] = None
    segment_direction: Optional[str] = None
    segment_confirmed: bool = False
    snapshot_first_seen: Optional[int] = None
    features: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, str, str]:
        return self.bar_index, self.direction, self.bsp_type

    @classmethod
    def from_bsp_row(cls, row: Mapping[str, Any]) -> "ChanSignal":
        direction = _optional_text(row.get("direction"))
        if direction is None:
            direction = "buy" if bool(row.get("is_buy", True)) else "sell"
        segment_direction = _optional_text(row.get("segment_direction"))
        price = row.get("klu_close", row.get("price", row.get("_close")))
        if price is None or pd.isna(price):
            raise ValueError("BSP row needs klu_close, price, or _close")
        bar_index = row.get("klu_idx", row.get("bar_index", -1))
        return cls(
            timestamp=pd.to_datetime(row.get("timestamp")),
            bar_index=int(bar_index),
            direction=direction,
            bsp_type=str(row.get("bsp_type", "?")).lower(),
            price=float(price),
            bi_direction=_optional_text(row.get("bi_direction")),
            segment_direction=segment_direction,
            segment_confirmed=(
                bool(row.get("segment_confirmed"))
                if row.get("segment_confirmed") is not None and not pd.isna(row.get("segment_confirmed"))
                else segment_direction in {"up", "down"}
            ),
            snapshot_first_seen=(
                None if pd.isna(row.get("snapshot_first_seen"))
                else int(row.get("snapshot_first_seen"))
            ),
            features={k: v for k, v in row.items() if str(k).startswith("feat_")},
        )


@dataclass(frozen=True)
class OrderIntent:
    """A strategy request; the trade engine decides whether and how it fills."""

    side: str
    reason: str
    signal: Optional[ChanSignal] = None
    position_fraction: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def buy(cls, reason: str, signal: Optional[ChanSignal] = None, position_fraction: float = 1.0):
        return cls("buy", reason, signal, float(position_fraction))

    @classmethod
    def sell(cls, reason: str, signal: Optional[ChanSignal] = None):
        return cls("sell", reason, signal, 1.0)


class SignalTracker:
    """Deduplicated recent BSP event history used by deterministic strategies."""

    def __init__(self, max_events: int = 1000):
        self._events: Deque[ChanSignal] = deque(maxlen=int(max_events))
        self._keys: set[tuple[int, str, str]] = set()

    @property
    def events(self) -> tuple[ChanSignal, ...]:
        return tuple(self._events)

    def add(self, signal: ChanSignal) -> bool:
        if signal.key in self._keys:
            return False
        if self._events.maxlen and len(self._events) == self._events.maxlen:
            dropped = self._events[0]
            self._keys.discard(dropped.key)
        self._events.append(signal)
        self._keys.add(signal.key)
        return True

    def clear_direction(self, direction: str) -> None:
        direction = str(direction).lower()
        kept = [event for event in self._events if event.direction != direction]
        self._events = deque(kept, maxlen=self._events.maxlen)
        self._keys = {event.key for event in kept}

    def distinct_signal_bars(
        self,
        direction: str,
        current_bar: int,
        lookback_bars: int,
        bsp_types: Optional[Iterable[str]] = None,
    ) -> int:
        direction = str(direction).lower()
        allowed = None if bsp_types is None else {str(x).lower() for x in bsp_types}
        start = int(current_bar) - int(lookback_bars) + 1
        return len({
            event.bar_index
            for event in self._events
            if start <= event.bar_index <= current_bar
            and event.direction == direction
            and (allowed is None or event.bsp_type in allowed)
        })

    def consecutive_direction_count(self, direction: str) -> int:
        direction = str(direction).lower()
        count = 0
        seen_bars = set()
        for event in reversed(self._events):
            if event.bar_index in seen_bars:
                continue
            seen_bars.add(event.bar_index)
            if event.direction != direction:
                break
            count += 1
        return count
