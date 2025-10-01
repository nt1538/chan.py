# Math/Williams.py
class WilliamsR:
    def __init__(self, period: int = 14):
        self.period = period
        self.highs = []
        self.lows = []

    def add(self, high, low, close):
        self.highs.append(high)
        self.lows.append(low)
        
        if len(self.highs) > self.period:
            self.highs.pop(0)
            self.lows.pop(0)
        
        if len(self.highs) < self.period:
            return -50.0
        
        highest_high = max(self.highs)
        lowest_low = min(self.lows)
        
        if highest_high == lowest_low:
            return -50.0
        
        return ((highest_high - close) / (highest_high - lowest_low)) * -100