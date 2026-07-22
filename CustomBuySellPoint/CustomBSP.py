"""Helpers for converting exported BSP rows into strategy signals."""

from typing import Any, Mapping

from .Signal import ChanSignal


def build_custom_bsp(row: Mapping[str, Any]) -> ChanSignal:
    return ChanSignal.from_bsp_row(row)
