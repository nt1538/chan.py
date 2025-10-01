# Math/SMA.py
class SMA:
    def __init__(self, period: int = 20):
        self.period = period
        self.values = []

    def add(self, value):
        self.values.append(value)
        if len(self.values) > self.period:
            self.values.pop(0)
        return sum(self.values) / len(self.values)