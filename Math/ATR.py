# Math/ATR.py
class ATR:
    def __init__(self, period: int = 14):
        self.period = period
        self.tr_values = []
        self.prev_close = None

    def add(self, high, low, close):
        if self.prev_close is not None:
            tr1 = high - low
            tr2 = abs(high - self.prev_close)
            tr3 = abs(low - self.prev_close)
            tr = max(tr1, tr2, tr3)
            
            self.tr_values.append(tr)
            if len(self.tr_values) > self.period:
                self.tr_values.pop(0)
        
        self.prev_close = close
        
        if len(self.tr_values) == 0:
            return 0.0
        return sum(self.tr_values) / len(self.tr_values)