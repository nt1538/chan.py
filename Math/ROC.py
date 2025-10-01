# Math/ROC.py
class ROC:
    def __init__(self, period: int = 10):
        self.period = period
        self.values = []

    def add(self, value):
        self.values.append(value)
        if len(self.values) <= self.period:
            return 0.0
        
        old_value = self.values[-self.period-1]
        if old_value == 0:
            return 0.0
        
        return ((value - old_value) / old_value) * 100