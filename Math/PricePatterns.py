# Math/PricePatterns.py
class PricePatterns:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.prices = []
        self.highs = []
        self.lows = []

    def add(self, high, low, close):
        self.prices.append(close)
        self.highs.append(high)
        self.lows.append(low)
        
        # Keep only needed history
        if len(self.prices) > self.lookback_period:
            self.prices.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)
        
        return self._detect_patterns()

    def _detect_patterns(self):
        if len(self.prices) < 10:
            return {}
        
        patterns = {}
        
        # Support and Resistance
        patterns['near_support'] = self._near_support()
        patterns['near_resistance'] = self._near_resistance()
        patterns['breakout_up'] = self._breakout_up()
        patterns['breakout_down'] = self._breakout_down()
        
        # Trend patterns
        patterns['higher_highs'] = self._higher_highs()
        patterns['lower_lows'] = self._lower_lows()
        patterns['double_top'] = self._double_top()
        patterns['double_bottom'] = self._double_bottom()
        
        # Consolidation patterns
        patterns['consolidation'] = self._consolidation()
        patterns['triangle'] = self._triangle_pattern()
        patterns['flag'] = self._flag_pattern()
        
        return patterns

    def _near_support(self):
        if len(self.prices) < 10:
            return False
        recent_low = min(self.lows[-10:])
        support_level = min(self.lows[:-5])
        return abs(recent_low - support_level) / support_level < 0.02

    def _near_resistance(self):
        if len(self.prices) < 10:
            return False
        recent_high = max(self.highs[-10:])
        resistance_level = max(self.highs[:-5])
        return abs(recent_high - resistance_level) / resistance_level < 0.02

    def _breakout_up(self):
        if len(self.prices) < 10:
            return False
        recent_high = max(self.highs[-5:])
        resistance_level = max(self.highs[-20:-5])
        return recent_high > resistance_level * 1.02

    def _breakout_down(self):
        if len(self.prices) < 10:
            return False
        recent_low = min(self.lows[-5:])
        support_level = min(self.lows[-20:-5])
        return recent_low < support_level * 0.98

    def _higher_highs(self):
        if len(self.highs) < 6:
            return False
        recent_highs = self.highs[-3:]
        return all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))

    def _lower_lows(self):
        if len(self.lows) < 6:
            return False
        recent_lows = self.lows[-3:]
        return all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))

    def _double_top(self):
        if len(self.highs) < 10:
            return False
        # Simplified double top detection
        max_high = max(self.highs)
        peaks = [i for i, h in enumerate(self.highs) if h > max_high * 0.98]
        return len(peaks) >= 2 and peaks[-1] - peaks[-2] > 3

    def _double_bottom(self):
        if len(self.lows) < 10:
            return False
        # Simplified double bottom detection
        min_low = min(self.lows)
        troughs = [i for i, l in enumerate(self.lows) if l < min_low * 1.02]
        return len(troughs) >= 2 and troughs[-1] - troughs[-2] > 3

    def _consolidation(self):
        if len(self.prices) < 10:
            return False
        price_range = max(self.prices) - min(self.prices)
        avg_price = sum(self.prices) / len(self.prices)
        return price_range / avg_price < 0.05

    def _triangle_pattern(self):
        if len(self.prices) < 15:
            return False
        # Simplified triangle detection - convergent highs and lows
        first_half_range = max(self.highs[:len(self.highs)//2]) - min(self.lows[:len(self.lows)//2])
        second_half_range = max(self.highs[len(self.highs)//2:]) - min(self.lows[len(self.lows)//2:])
        return second_half_range < first_half_range * 0.7

    def _flag_pattern(self):
        if len(self.prices) < 10:
            return False
        # Simplified flag pattern - consolidation after strong move
        first_half = self.prices[:len(self.prices)//2]
        second_half = self.prices[len(self.prices)//2:]
        
        first_range = max(first_half) - min(first_half)
        second_range = max(second_half) - min(second_half)
        
        # Strong initial move followed by consolidation
        return first_range > second_range * 2
