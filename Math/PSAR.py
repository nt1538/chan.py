# Math/PSAR.py - Parabolic SAR
class PSAR:
    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2):
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
        self.trend = None  # 1 for uptrend, -1 for downtrend
        self.af = af_start
        self.sar = None
        self.ep = None  # Extreme Point
        self.prev_high = None
        self.prev_low = None

    def add(self, high, low):
        if self.sar is None:
            # Initialize
            self.sar = low
            self.trend = 1
            self.ep = high
            self.prev_high = high
            self.prev_low = low
            return self.sar
        
        # Calculate new SAR
        new_sar = self.sar + self.af * (self.ep - self.sar)
        
        # Check for trend reversal
        if self.trend == 1:  # Uptrend
            if low <= new_sar:
                # Trend reversal to downtrend
                self.trend = -1
                self.sar = self.ep
                self.ep = low
                self.af = self.af_start
            else:
                self.sar = new_sar
                if high > self.ep:
                    self.ep = high
                    self.af = min(self.af + self.af_increment, self.af_max)
                # SAR cannot be above previous two lows in uptrend
                self.sar = min(self.sar, self.prev_low, low)
        else:  # Downtrend
            if high >= new_sar:
                # Trend reversal to uptrend
                self.trend = 1
                self.sar = self.ep
                self.ep = high
                self.af = self.af_start
            else:
                self.sar = new_sar
                if low < self.ep:
                    self.ep = low
                    self.af = min(self.af + self.af_increment, self.af_max)
                # SAR cannot be below previous two highs in downtrend
                self.sar = max(self.sar, self.prev_high, high)
        
        self.prev_high = high
        self.prev_low = low
        return self.sar