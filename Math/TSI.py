# Math/TSI.py
from Math.EMA import EMA


class TSI:
    def __init__(self, first_smooth: int = 25, second_smooth: int = 13):
        self.first_smooth = first_smooth
        self.second_smooth = second_smooth
        self.prev_close = None
        self.momentum_ema1 = EMA(first_smooth)
        self.momentum_ema2 = EMA(second_smooth)
        self.abs_momentum_ema1 = EMA(first_smooth)
        self.abs_momentum_ema2 = EMA(second_smooth)

    def add(self, close):
        if self.prev_close is None:
            self.prev_close = close
            return 0.0
        
        momentum = close - self.prev_close
        abs_momentum = abs(momentum)
        
        # First smoothing
        first_smooth_momentum = self.momentum_ema1.add(momentum)
        first_smooth_abs_momentum = self.abs_momentum_ema1.add(abs_momentum)
        
        # Second smoothing
        second_smooth_momentum = self.momentum_ema2.add(first_smooth_momentum)
        second_smooth_abs_momentum = self.abs_momentum_ema2.add(first_smooth_abs_momentum)
        
        self.prev_close = close
        
        if second_smooth_abs_momentum == 0:
            return 0.0
        
        return 100 * (second_smooth_momentum / second_smooth_abs_momentum)