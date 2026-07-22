from __future__ import annotations

from typing import Any
import copy

from CustomBuySellPoint.Strategy import PositionView


class ExecutionEngine:
    """Long-only next-bar-open simulator with fees and optional slippage."""

    def __init__(self, initial_capital: float = 100_000.0, fee_pct: float = 0.0005, slippage_pct: float = 0.0):
        self.initial_capital = float(initial_capital)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.cash = float(initial_capital)
        self.pos = 0
        self.qty = 0.0
        self.entry_px: float | None = None
        self.entry_idx: int | None = None
        self.peak_px: float | None = None
        self.pending_order: dict[str, Any] | None = None
        self.trades: list[dict[str, Any]] = []

    def place_order_for_next_bar(
        self,
        side: str,
        seen_idx: int,
        next_open_idx: int,
        next_open_px: float,
        reason: str = "",
        position_fraction: float = 1.0,
        overwrite: bool = True,
    ) -> bool:
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order side: {side}")
        if self.pending_order is not None and not overwrite:
            return False
        self.pending_order = {
            "side": side,
            "seen_idx": int(seen_idx),
            "exec_idx": int(next_open_idx),
            "exec_px": float(next_open_px),
            "reason": reason,
            "position_fraction": min(1.0, max(0.0, float(position_fraction))),
        }
        return True

    def maybe_execute_pending(self, idx: int) -> dict[str, Any] | None:
        order = self.pending_order
        if order is None or idx != order["exec_idx"]:
            return None
        self.pending_order = None
        side = order["side"]
        raw_px = float(order["exec_px"])
        px = raw_px * (1.0 + self.slippage_pct if side == "buy" else 1.0 - self.slippage_pct)
        fill: dict[str, Any] | None = None
        if side == "buy" and self.pos == 0 and order["position_fraction"] > 0:
            budget = self.cash * order["position_fraction"]
            qty = budget / (px * (1.0 + self.fee_pct))
            notional = qty * px
            fee = notional * self.fee_pct
            self.cash -= notional + fee
            self.pos, self.qty = 1, qty
            self.entry_px, self.entry_idx, self.peak_px = px, idx, px
            fill = {**order, "idx": idx, "px": px, "raw_px": raw_px, "qty": qty, "fee": fee}
        elif side == "sell" and self.pos == 1:
            notional = self.qty * px
            fee = notional * self.fee_pct
            entry_px = self.entry_px or px
            pnl = self.qty * (px - entry_px) - fee
            self.cash += notional - fee
            fill = {
                **order, "idx": idx, "px": px, "raw_px": raw_px, "qty": self.qty,
                "fee": fee, "pnl": pnl, "return_pct": px / entry_px - 1.0,
            }
            self.pos, self.qty = 0, 0.0
            self.entry_px = self.entry_idx = self.peak_px = None
        if fill is not None:
            self.trades.append(fill)
        return fill

    def update_market_price(self, price: float) -> None:
        if self.pos == 1:
            self.peak_px = max(float(price), self.peak_px or float(price))

    def position_view(self) -> PositionView:
        return PositionView(
            side="long" if self.pos == 1 else "flat",
            quantity=self.qty,
            entry_price=self.entry_px,
            entry_bar=self.entry_idx,
            peak_price=self.peak_px,
        )

    def mark_to_market(self, price: float) -> float:
        return self.cash + (self.qty * float(price) if self.pos == 1 else 0.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital, "fee_pct": self.fee_pct,
            "slippage_pct": self.slippage_pct, "cash": self.cash, "pos": self.pos,
            "qty": self.qty, "entry_px": self.entry_px, "entry_idx": self.entry_idx,
            "peak_px": self.peak_px, "pending_order": self.pending_order, "trades": self.trades,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("initial_capital", "fee_pct", "slippage_pct", "cash", "pos", "qty", "entry_px", "entry_idx", "peak_px", "pending_order", "trades"):
            if name in state:
                setattr(self, name, state[name])


class LegacyPipelineExecutionEngine:
    """Compatibility adapter for pipelineCurrent's historical array-based API."""

    def __init__(self, initial_capital: float, fee_pct: float):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.fee_pct = float(fee_pct)
        self.pos, self.qty = 0, 0.0
        self.entry_px = self.entry_idx = None
        self.pending_order = None
        self.trades = []

    def _exec_px(self, seen_idx, next_open_by_idx):
        if not (0 <= seen_idx < len(next_open_by_idx)):
            return None
        px = next_open_by_idx[seen_idx]
        try:
            if px is None or px != px:
                return None
        except TypeError:
            return None
        return float(px)

    def place_order_for_next_bar(self, side, seen_idx, reason, meta=None, overwrite=True):
        if self.pending_order is not None and not overwrite:
            return False
        self.pending_order = {"side": side, "seen_idx": int(seen_idx), "reason": reason, "meta": dict(meta) if meta else {}}
        return True

    def maybe_execute_pending(self, next_open_by_idx):
        if self.pending_order is None:
            return None
        order = self.pending_order
        px = self._exec_px(order["seen_idx"], next_open_by_idx)
        if px is None:
            return None
        side, idx, reason, meta = order["side"], order["seen_idx"], order["reason"], order.get("meta", {}) or {}
        fill = None
        if side == "buy" and self.pos == 0:
            notional = self.cash / (1.0 + self.fee_pct)
            fee = notional * self.fee_pct
            qty = notional / px
            self.cash -= notional + fee
            self.pos, self.qty, self.entry_px, self.entry_idx = 1, qty, px, idx
            fill = {"side": "buy", "seen_idx": idx, "exec_px": px, "qty": qty, "fee": fee, "reason": reason, **meta}
        elif side == "sell" and self.pos == 1:
            notional = self.qty * px
            fee = notional * self.fee_pct
            self.cash += notional - fee
            fill = {"side": "sell", "seen_idx": idx, "exec_px": px, "qty": self.qty, "fee": fee, "reason": reason,
                    "pnl": (px - self.entry_px) * self.qty - fee, "entry_px": self.entry_px, "entry_idx": self.entry_idx, **meta}
            self.pos, self.qty, self.entry_px, self.entry_idx = 0, 0.0, None, None
        if fill is not None:
            self.trades.append(fill)
        self.pending_order = None
        return fill

    def mark_to_market(self, last_close):
        return float(self.cash if self.pos == 0 else self.cash + self.qty * float(last_close))

    def state_dict(self):
        return {"cash": self.cash, "fee_pct": self.fee_pct, "pos": self.pos, "qty": self.qty,
                "entry_px": self.entry_px, "entry_idx": self.entry_idx,
                "pending_order": copy.deepcopy(self.pending_order), "trades": copy.deepcopy(self.trades)}

    def load_state_dict(self, state):
        self.cash = float(state.get("cash", self.cash))
        self.fee_pct = float(state.get("fee_pct", self.fee_pct))
        self.pos, self.qty = int(state.get("pos", 0)), float(state.get("qty", 0.0))
        self.entry_px, self.entry_idx = state.get("entry_px"), state.get("entry_idx")
        self.pending_order, self.trades = copy.deepcopy(state.get("pending_order")), copy.deepcopy(state.get("trades", []))
