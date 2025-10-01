# Math/Stochastic.py
class Stochastic:
    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.highs = []
        self.lows = []
        self.closes = []
        self.k_values = []

    def add(self, high, low, close):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        if len(self.highs) > self.k_period:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
        
        if len(self.highs) < self.k_period:
            return {'k': 50.0, 'd': 50.0}
        
        highest_high = max(self.highs)
        lowest_low = min(self.lows)
        
        if highest_high == lowest_low:
            k = 50.0
        else:
            k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        self.k_values.append(k)
        if len(self.k_values) > self.d_period:
            self.k_values.pop(0)
        
        d = sum(self.k_values) / len(self.k_values)
        
        return {'k': k, 'd': d}