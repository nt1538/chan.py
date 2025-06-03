from typing import Optional, Tuple
import numpy as np

class KeltnerChannel:
    def __init__(self, cycle: int = 20, multiplier: float = 2.0):
        self.cycle = cycle
        self.multiplier = multiplier
        self.close_vals = []
        self.high_vals = []
        self.low_vals = []
        self.ema = None
        self.atr_vals = []

    def _calc_ema(self, close):
        if self.ema is None:
            self.ema = close
        else:
            k = 2 / (self.cycle + 1)
            self.ema = close * k + self.ema * (1 - k)
        return self.ema

    def _calc_tr(self, high, low, prev_close):
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

    def add(self, high: float, low: float, close: float, prev_close: Optional[float]) -> Optional[Tuple[float, float, float]]:
        self.high_vals.append(high)
        self.low_vals.append(low)
        self.close_vals.append(close)

        if prev_close is None:
            return None  # 第一条没法计算 TR/ATR

        tr = self._calc_tr(high, low, prev_close)
        self.atr_vals.append(tr)

        if len(self.atr_vals) < self.cycle:
            return None

        atr = np.mean(self.atr_vals[-self.cycle:])
        ema = self._calc_ema(close)

        upper = ema + self.multiplier * atr
        lower = ema - self.multiplier * atr
        return upper, ema, lower
