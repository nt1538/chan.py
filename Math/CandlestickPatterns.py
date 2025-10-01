# Math/CandlestickPatterns.py
class CandlestickPatterns:
    def __init__(self):
        self.candles = []
        self.max_history = 5  # Keep last 5 candles for pattern detection

    def add(self, open_price, high, low, close):
        candle = {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'body': abs(close - open_price),
            'upper_shadow': high - max(open_price, close),
            'lower_shadow': min(open_price, close) - low,
            'range': high - low,
            'is_bullish': close > open_price,
            'is_bearish': close < open_price
        }
        
        self.candles.append(candle)
        if len(self.candles) > self.max_history:
            self.candles.pop(0)
        
        return self._detect_patterns()

    def _detect_patterns(self):
        if len(self.candles) == 0:
            return {}
        
        current = self.candles[-1]
        patterns = {}
        
        # Single candle patterns
        patterns['doji'] = self._is_doji(current)
        patterns['hammer'] = self._is_hammer(current)
        patterns['shooting_star'] = self._is_shooting_star(current)
        patterns['spinning_top'] = self._is_spinning_top(current)
        patterns['marubozu'] = self._is_marubozu(current)
        patterns['hanging_man'] = self._is_hanging_man(current)
        patterns['inverted_hammer'] = self._is_inverted_hammer(current)
        
        # Multi-candle patterns (need at least 2 candles)
        if len(self.candles) >= 2:
            patterns['bullish_engulfing'] = self._is_bullish_engulfing()
            patterns['bearish_engulfing'] = self._is_bearish_engulfing()
            patterns['piercing_line'] = self._is_piercing_line()
            patterns['dark_cloud_cover'] = self._is_dark_cloud_cover()
            patterns['harami'] = self._is_harami()
            patterns['harami_cross'] = self._is_harami_cross()
        
        # Three-candle patterns
        if len(self.candles) >= 3:
            patterns['morning_star'] = self._is_morning_star()
            patterns['evening_star'] = self._is_evening_star()
            patterns['three_white_soldiers'] = self._is_three_white_soldiers()
            patterns['three_black_crows'] = self._is_three_black_crows()
            patterns['three_inside_up'] = self._is_three_inside_up()
            patterns['three_inside_down'] = self._is_three_inside_down()
        
        return patterns

    def _is_doji(self, candle):
        return candle['body'] <= candle['range'] * 0.1

    def _is_hammer(self, candle):
        return (candle['lower_shadow'] >= candle['body'] * 2 and 
                candle['upper_shadow'] <= candle['body'] * 0.1 and
                candle['range'] > 0)

    def _is_shooting_star(self, candle):
        return (candle['upper_shadow'] >= candle['body'] * 2 and 
                candle['lower_shadow'] <= candle['body'] * 0.1 and
                candle['range'] > 0)

    def _is_spinning_top(self, candle):
        return (candle['body'] < candle['range'] * 0.3 and
                candle['upper_shadow'] > candle['body'] * 0.5 and
                candle['lower_shadow'] > candle['body'] * 0.5)

    def _is_marubozu(self, candle):
        return candle['body'] >= candle['range'] * 0.95

    def _is_hanging_man(self, candle):
        return self._is_hammer(candle) and candle['is_bearish']

    def _is_inverted_hammer(self, candle):
        return self._is_shooting_star(candle) and candle['is_bullish']

    def _is_bullish_engulfing(self):
        if len(self.candles) < 2:
            return False
        prev, curr = self.candles[-2], self.candles[-1]
        return (prev['is_bearish'] and curr['is_bullish'] and
                curr['open'] < prev['close'] and curr['close'] > prev['open'])

    def _is_bearish_engulfing(self):
        if len(self.candles) < 2:
            return False
        prev, curr = self.candles[-2], self.candles[-1]
        return (prev['is_bullish'] and curr['is_bearish'] and
                curr['open'] > prev['close'] and curr['close'] < prev['open'])

    def _is_piercing_line(self):
        if len(self.candles) < 2:
            return False
        prev, curr = self.candles[-2], self.candles[-1]
        midpoint = (prev['open'] + prev['close']) / 2
        return (prev['is_bearish'] and curr['is_bullish'] and
                curr['open'] < prev['low'] and curr['close'] > midpoint and
                curr['close'] < prev['open'])

    def _is_dark_cloud_cover(self):
        if len(self.candles) < 2:
            return False
        prev, curr = self.candles[-2], self.candles[-1]
        midpoint = (prev['open'] + prev['close']) / 2
        return (prev['is_bullish'] and curr['is_bearish'] and
                curr['open'] > prev['high'] and curr['close'] < midpoint and
                curr['close'] > prev['open'])

    def _is_harami(self):
        if len(self.candles) < 2:
            return False
        prev, curr = self.candles[-2], self.candles[-1]
        return (curr['high'] < prev['high'] and curr['low'] > prev['low'] and
                prev['body'] > curr['body'] * 2)

    def _is_harami_cross(self):
        if len(self.candles) < 2:
            return False
        return self._is_harami() and self._is_doji(self.candles[-1])

    def _is_morning_star(self):
        if len(self.candles) < 3:
            return False
        first, second, third = self.candles[-3], self.candles[-2], self.candles[-1]
        return (first['is_bearish'] and 
                second['body'] < first['body'] * 0.3 and
                third['is_bullish'] and
                third['close'] > (first['open'] + first['close']) / 2)

    def _is_evening_star(self):
        if len(self.candles) < 3:
            return False
        first, second, third = self.candles[-3], self.candles[-2], self.candles[-1]
        return (first['is_bullish'] and 
                second['body'] < first['body'] * 0.3 and
                third['is_bearish'] and
                third['close'] < (first['open'] + first['close']) / 2)

    def _is_three_white_soldiers(self):
        if len(self.candles) < 3:
            return False
        candles = self.candles[-3:]
        return (all(c['is_bullish'] for c in candles) and
                candles[1]['close'] > candles[0]['close'] and
                candles[2]['close'] > candles[1]['close'] and
                all(c['body'] > c['range'] * 0.6 for c in candles))

    def _is_three_black_crows(self):
        if len(self.candles) < 3:
            return False
        candles = self.candles[-3:]
        return (all(c['is_bearish'] for c in candles) and
                candles[1]['close'] < candles[0]['close'] and
                candles[2]['close'] < candles[1]['close'] and
                all(c['body'] > c['range'] * 0.6 for c in candles))

    def _is_three_inside_up(self):
        if len(self.candles) < 3:
            return False
        return (self._is_harami() and 
                self.candles[-3]['is_bearish'] and
                self.candles[-2]['is_bullish'] and
                self.candles[-1]['is_bullish'] and
                self.candles[-1]['close'] > self.candles[-3]['open'])

    def _is_three_inside_down(self):
        if len(self.candles) < 3:
            return False
        return (self._is_harami() and 
                self.candles[-3]['is_bullish'] and
                self.candles[-2]['is_bearish'] and
                self.candles[-1]['is_bearish'] and
                self.candles[-1]['close'] < self.candles[-3]['open'])

