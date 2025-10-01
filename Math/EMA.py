# Math/EMA.py
class EMA:
    def __init__(self, period: int = 20):
        self.period = period
        self.multiplier = 2 / (period + 1)
        self.ema = None

    def add(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (value * self.multiplier) + (self.ema * (1 - self.multiplier))
        return self.ema