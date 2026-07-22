"""Compatibility entry point for user-defined strategies."""

from .SegBspStrategy import SegBspStrategy, SegBspStrategyConfig


class CustomStrategy(SegBspStrategy):
    """Default custom strategy; subclass or replace its config for experiments."""

    def __init__(self, config: SegBspStrategyConfig | None = None):
        super().__init__(config=config)
