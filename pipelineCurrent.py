"""Backward-compatible facade for the modular pipeline package.

New code should import focused modules or ``Pipeline``. Existing notebooks can
continue importing any historical name from ``pipelineCurrent``.
"""

from Pipeline import DailyBandit5mPipeline as _implementation

for _name in dir(_implementation):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_implementation, _name)


class ExecutionEngine(_implementation.ExecutionEngine):
    """Keep the historical pickle/import path available."""


__all__ = [name for name in globals() if not name.startswith("_")]
