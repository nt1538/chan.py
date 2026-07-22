from dataclasses import dataclass


@dataclass(frozen=True)
class OpenQuotaConfig:
    cash_fraction: float = 1.0
    max_notional: float | None = None
    cash_reserve: float = 0.0


class OpenQuotaGen:
    """Convert a strategy allocation request into an executable cash fraction."""

    def __init__(self, config: OpenQuotaConfig | None = None):
        self.config = config or OpenQuotaConfig()

    def fraction(self, cash: float, requested_fraction: float = 1.0) -> float:
        available = max(0.0, float(cash) - self.config.cash_reserve)
        if cash <= 0 or available <= 0:
            return 0.0
        fraction = min(1.0, max(0.0, requested_fraction), self.config.cash_fraction)
        notional = cash * fraction
        if self.config.max_notional is not None:
            notional = min(notional, max(0.0, self.config.max_notional))
        return min(1.0, max(0.0, notional / cash), available / cash)
