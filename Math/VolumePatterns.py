# Math/VolumePatterns.py
class VolumePatterns:
    def __init__(self, period: int = 20):
        self.period = period
        self.volumes = []
        self.prices = []

    def add(self, price, volume):
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.volumes) > self.period:
            self.volumes.pop(0)
            self.prices.pop(0)
        
        return self._detect_patterns()

    def _detect_patterns(self):
        if len(self.volumes) < 10:
            return {}
        
        patterns = {}
        avg_volume = sum(self.volumes) / len(self.volumes)
        current_volume = self.volumes[-1]
        
        patterns['volume_spike'] = current_volume > avg_volume * 2
        patterns['volume_dry_up'] = current_volume < avg_volume * 0.5
        patterns['accumulation'] = self._accumulation_pattern()
        patterns['distribution'] = self._distribution_pattern()
        patterns['climax_volume'] = self._climax_volume()
        
        return patterns

    def _accumulation_pattern(self):
        if len(self.volumes) < 10:
            return False
        # Rising prices with increasing volume
        recent_prices = self.prices[-5:]
        recent_volumes = self.volumes[-5:]
        
        price_trend = recent_prices[-1] > recent_prices[0]
        volume_trend = sum(recent_volumes[-3:]) > sum(recent_volumes[:3])
        
        return price_trend and volume_trend

    def _distribution_pattern(self):
        if len(self.volumes) < 10:
            return False
        # Rising prices with decreasing volume (distribution)
        recent_prices = self.prices[-5:]
        recent_volumes = self.volumes[-5:]
        
        price_trend = recent_prices[-1] > recent_prices[0]
        volume_trend = sum(recent_volumes[-3:]) < sum(recent_volumes[:3])
        
        return price_trend and volume_trend

    def _climax_volume(self):
        if len(self.volumes) < 5:
            return False
        current_volume = self.volumes[-1]
        avg_recent_volume = sum(self.volumes[-5:]) / 5
        return current_volume > avg_recent_volume * 3