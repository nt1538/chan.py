# Math/UO.py - Ultimate Oscillator
class UO:
    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.buying_pressures = []
        self.true_ranges = []
        self.prev_close = None

    def add(self, high, low, close):
        if self.prev_close is not None:
            # Calculate Buying Pressure and True Range
            bp = close - min(low, self.prev_close)
            tr = max(high, self.prev_close) - min(low, self.prev_close)
            
            self.buying_pressures.append(bp)
            self.true_ranges.append(tr)
            
            # Keep only needed periods
            max_period = max(self.period1, self.period2, self.period3)
            if len(self.buying_pressures) > max_period:
                self.buying_pressures.pop(0)
                self.true_ranges.pop(0)
        
        self.prev_close = close
        
        if len(self.buying_pressures) < self.period3:
            return 50.0
        
        # Calculate averages for each period
        def calculate_avg(period):
            bp_sum = sum(self.buying_pressures[-period:])
            tr_sum = sum(self.true_ranges[-period:])
            return bp_sum / tr_sum if tr_sum != 0 else 0
        
        avg1 = calculate_avg(self.period1)
        avg2 = calculate_avg(self.period2)
        avg3 = calculate_avg(self.period3)
        
        # Ultimate Oscillator formula
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return uo