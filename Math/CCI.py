# Math/CCI.py
class CCI:
    def __init__(self, period: int = 20):
        self.period = period
        self.tp_values = []  # Typical Price values

    def add(self, high, low, close):
        tp = (high + low + close) / 3
        self.tp_values.append(tp)
        
        if len(self.tp_values) > self.period:
            self.tp_values.pop(0)
        
        if len(self.tp_values) < self.period:
            return 0.0
        
        sma_tp = sum(self.tp_values) / len(self.tp_values)
        mean_deviation = sum(abs(tp - sma_tp) for tp in self.tp_values) / len(self.tp_values)
        
        if mean_deviation == 0:
            return 0.0
        
        return (tp - sma_tp) / (0.015 * mean_deviation)