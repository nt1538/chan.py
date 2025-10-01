from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

from BuySellPoint.BSPointList import CBSPointList
from .BS_Point import CBS_Point

@dataclass
class BSPointCandidate:
    """Potential BSPoint that doesn't meet strict criteria yet"""
    klu_idx: int
    bi_idx: int
    is_buy: bool
    confidence: float
    features: Dict
    timestamp: int
    reason: str  # Why it's a candidate

class CMultiLevelBSPointGenerator:
    def __init__(self):
        # Track candidates at different confidence levels
        self.high_confidence_bsp: List[CBS_Point] = []
        self.medium_confidence_candidates: List[BSPointCandidate] = []
        self.low_confidence_candidates: List[BSPointCandidate] = []
        
        # Historical data for training
        self.training_samples: List[Dict] = []
        
        # Configuration
        self.high_threshold = 0.8
        self.medium_threshold = 0.5
        self.low_threshold = 0.2
        
        # Feature extractors
        self.feature_extractors = []

    def generate_candidates_from_chan(self, chan, lv, lookback_window=50):
        """Generate BSPoint candidates using relaxed criteria"""
        kl_data = chan.kl_datas[lv]
        bi_list = kl_data.bi_list
        seg_list = kl_data.seg_list
        
        candidates = []
        
        # Get recent data window
        recent_bis = bi_list[-lookback_window:] if len(bi_list) > lookback_window else bi_list
        
        for bi in recent_bis:
            # Generate multiple types of candidates
            candidates.extend(self._generate_divergence_candidates(bi, seg_list))
            candidates.extend(self._generate_pattern_candidates(bi, bi_list))
            candidates.extend(self._generate_momentum_candidates(bi, bi_list))
            candidates.extend(self._generate_support_resistance_candidates(bi, bi_list))
        
        return candidates

    def _generate_divergence_candidates(self, bi, seg_list) -> List[BSPointCandidate]:
        """Generate candidates based on relaxed divergence criteria"""
        candidates = []
        
        # Find relevant segment
        bi_seg = None
        for seg in seg_list:
            if seg.start_bi.idx <= bi.idx <= seg.end_bi.idx:
                bi_seg = seg
                break
        
        if not bi_seg or len(bi_seg.zs_lst) == 0:
            return candidates
        
        for zs in bi_seg.zs_lst:
            if zs.is_one_bi_zs():
                continue
            
            # Relaxed divergence check
            try:
                in_metric = zs.get_bi_in().cal_macd_metric("peak", is_reverse=False)
                out_metric = bi.cal_macd_metric("peak", is_reverse=True)
                
                # Multiple divergence thresholds
                for threshold, confidence in [(1.5, 0.3), (1.2, 0.5), (1.0, 0.7), (0.8, 0.9)]:
                    if out_metric <= threshold * in_metric:
                        features = self._extract_divergence_features(bi, zs, in_metric, out_metric)
                        
                        candidate = BSPointCandidate(
                            klu_idx=bi.get_end_klu().idx,
                            bi_idx=bi.idx,
                            is_buy=bi.is_down(),
                            confidence=confidence,
                            features=features,
                            timestamp=bi.get_end_klu().idx,
                            reason=f"divergence_threshold_{threshold}"
                        )
                        candidates.append(candidate)
                        break
            except:
                continue
        
        return candidates

    def _generate_pattern_candidates(self, bi, bi_list) -> List[BSPointCandidate]:
        """Generate candidates based on pattern recognition"""
        candidates = []
        
        if bi.idx < 4:
            return candidates
        
        # Get surrounding bis for pattern analysis
        start_idx = max(0, bi.idx - 4)
        end_idx = min(len(bi_list), bi.idx + 3)
        pattern_bis = bi_list[start_idx:end_idx]
        
        # Double bottom/top patterns
        if self._is_double_bottom_pattern(bi, pattern_bis):
            features = self._extract_pattern_features(bi, pattern_bis, "double_bottom")
            candidates.append(BSPointCandidate(
                klu_idx=bi.get_end_klu().idx,
                bi_idx=bi.idx,
                is_buy=True,
                confidence=0.6,
                features=features,
                timestamp=bi.get_end_klu().idx,
                reason="double_bottom"
            ))
        
        if self._is_double_top_pattern(bi, pattern_bis):
            features = self._extract_pattern_features(bi, pattern_bis, "double_top")
            candidates.append(BSPointCandidate(
                klu_idx=bi.get_end_klu().idx,
                bi_idx=bi.idx,
                is_buy=False,
                confidence=0.6,
                features=features,
                timestamp=bi.get_end_klu().idx,
                reason="double_top"
            ))
        
        # Head and shoulders
        if self._is_head_shoulders_pattern(bi, pattern_bis):
            features = self._extract_pattern_features(bi, pattern_bis, "head_shoulders")
            candidates.append(BSPointCandidate(
                klu_idx=bi.get_end_klu().idx,
                bi_idx=bi.idx,
                is_buy=bi.is_down(),
                confidence=0.7,
                features=features,
                timestamp=bi.get_end_klu().idx,
                reason="head_shoulders"
            ))
        
        return candidates

    def _generate_momentum_candidates(self, bi, bi_list) -> List[BSPointCandidate]:
        """Generate candidates based on momentum indicators"""
        candidates = []
        
        if bi.idx < 3:
            return candidates
        
        # Calculate momentum features
        recent_bis = bi_list[max(0, bi.idx-3):bi.idx+1]
        
        # Price momentum
        price_momentum = self._calculate_price_momentum(recent_bis)
        
        # Volume momentum (if available)
        volume_momentum = self._calculate_volume_momentum(recent_bis)
        
        # MACD momentum
        macd_momentum = self._calculate_macd_momentum(bi)
        
        # RSI levels
        rsi_value = getattr(bi.get_end_klu(), 'rsi', None)
        
        # Determine candidate based on momentum
        confidence = 0.0
        is_buy = None
        reason = ""
        
        # Oversold conditions
        if rsi_value and rsi_value < 30 and price_momentum < -0.05:
            confidence = 0.6
            is_buy = True
            reason = "oversold_momentum"
        
        # Overbought conditions  
        elif rsi_value and rsi_value > 70 and price_momentum > 0.05:
            confidence = 0.6
            is_buy = False
            reason = "overbought_momentum"
        
        # Momentum reversal
        elif abs(price_momentum) > 0.03 and macd_momentum * price_momentum < 0:
            confidence = 0.4
            is_buy = price_momentum < 0
            reason = "momentum_reversal"
        
        if confidence > 0:
            features = {
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum,
                'macd_momentum': macd_momentum,
                'rsi': rsi_value,
                'bi_amp': bi.amp(),
                'bi_amp_rate': bi.amp() / bi.get_begin_val(),
            }
            
            candidates.append(BSPointCandidate(
                klu_idx=bi.get_end_klu().idx,
                bi_idx=bi.idx,
                is_buy=is_buy,
                confidence=confidence,
                features=features,
                timestamp=bi.get_end_klu().idx,
                reason=reason
            ))
        
        return candidates

    def _generate_support_resistance_candidates(self, bi, bi_list) -> List[BSPointCandidate]:
        """Generate candidates based on support/resistance levels"""
        candidates = []
        
        if bi.idx < 10:
            return candidates
        
        # Look for historical support/resistance levels
        lookback_bis = bi_list[max(0, bi.idx-20):bi.idx]
        
        current_price = bi.get_end_val()
        
        # Find nearby historical levels
        for hist_bi in lookback_bis:
            hist_price = hist_bi.get_end_val()
            price_diff = abs(current_price - hist_price) / current_price
            
            # Strong support/resistance if price is within 2%
            if price_diff < 0.02:
                # Count how many times this level was tested
                test_count = sum(1 for hb in lookback_bis 
                               if abs(hb.get_end_val() - hist_price) / hist_price < 0.01)
                
                if test_count >= 2:
                    confidence = min(0.8, 0.3 + test_count * 0.1)
                    
                    features = {
                        'support_resistance_level': hist_price,
                        'price_diff_pct': price_diff,
                        'test_count': test_count,
                        'bi_amp': bi.amp(),
                        'current_price': current_price,
                    }
                    
                    # Determine direction based on approach
                    is_buy = current_price < hist_price and bi.is_down()
                    
                    candidates.append(BSPointCandidate(
                        klu_idx=bi.get_end_klu().idx,
                        bi_idx=bi.idx,
                        is_buy=is_buy,
                        confidence=confidence,
                        features=features,
                        timestamp=bi.get_end_klu().idx,
                        reason=f"support_resistance_test_{test_count}"
                    ))
        
        return candidates

    def _extract_divergence_features(self, bi, zs, in_metric, out_metric) -> Dict:
        """Extract features for divergence candidates"""
        return {
            'divergence_ratio': out_metric / (in_metric + 1e-7),
            'zs_height': (zs.high - zs.low) / zs.low,
            'zs_width': zs.end.idx - zs.begin.idx,
            'bi_amp': bi.amp(),
            'bi_amp_rate': bi.amp() / bi.get_begin_val(),
            'in_metric': in_metric,
            'out_metric': out_metric,
        }

    def _extract_pattern_features(self, bi, pattern_bis, pattern_type) -> Dict:
        """Extract features for pattern candidates"""
        features = {
            'pattern_type': pattern_type,
            'pattern_length': len(pattern_bis),
            'bi_amp': bi.amp(),
            'bi_amp_rate': bi.amp() / bi.get_begin_val(),
        }
        
        # Add pattern-specific features
        if pattern_type in ["double_bottom", "double_top"]:
            # Find the two peaks/troughs
            extremes = []
            for pb in pattern_bis:
                if pattern_type == "double_bottom" and pb.is_down():
                    extremes.append(pb.get_end_val())
                elif pattern_type == "double_top" and pb.is_up():
                    extremes.append(pb.get_end_val())
            
            if len(extremes) >= 2:
                features['extreme_similarity'] = 1 - abs(extremes[-1] - extremes[-2]) / max(extremes[-1], extremes[-2])
        
        return features

    def _is_double_bottom_pattern(self, bi, pattern_bis) -> bool:
        """Check if current bi completes a double bottom pattern"""
        if not bi.is_down() or len(pattern_bis) < 4:
            return False
        
        # Find the previous low
        prev_lows = [pb for pb in pattern_bis[:-1] if pb.is_down()]
        if len(prev_lows) < 1:
            return False
        
        current_low = bi.get_end_val()
        prev_low = prev_lows[-1].get_end_val()
        
        # Check if lows are similar (within 3%)
        similarity = abs(current_low - prev_low) / max(current_low, prev_low)
        return similarity < 0.03

    def _is_double_top_pattern(self, bi, pattern_bis) -> bool:
        """Check if current bi completes a double top pattern"""
        if not bi.is_up() or len(pattern_bis) < 4:
            return False
        
        # Find the previous high
        prev_highs = [pb for pb in pattern_bis[:-1] if pb.is_up()]
        if len(prev_highs) < 1:
            return False
        
        current_high = bi.get_end_val()
        prev_high = prev_highs[-1].get_end_val()
        
        # Check if highs are similar (within 3%)
        similarity = abs(current_high - prev_high) / max(current_high, prev_high)
        return similarity < 0.03

    def _is_head_shoulders_pattern(self, bi, pattern_bis) -> bool:
        """Check if current bi completes a head and shoulders pattern"""
        if len(pattern_bis) < 5:
            return False
        
        # Simplified head and shoulders detection
        # Look for high-low-higher_high-low-high pattern
        if bi.is_down():
            ups = [pb for pb in pattern_bis if pb.is_up()]
            if len(ups) >= 3:
                # Check if middle high is higher than shoulders
                return ups[-2].get_end_val() > max(ups[-3].get_end_val(), ups[-1].get_end_val())
        
        return False

    def _calculate_price_momentum(self, bis) -> float:
        """Calculate price momentum over recent bis"""
        if len(bis) < 2:
            return 0.0
        
        start_price = bis[0].get_begin_val()
        end_price = bis[-1].get_end_val()
        
        return (end_price - start_price) / start_price

    def _calculate_volume_momentum(self, bis) -> float:
        """Calculate volume momentum if available"""
        volumes = []
        for bi in bis:
            klu = bi.get_end_klu()
            if hasattr(klu, 'volume') and klu.volume:
                volumes.append(klu.volume)
        
        if len(volumes) < 2:
            return 0.0
        
        return (volumes[-1] - volumes[0]) / max(volumes[0], 1)

    def _calculate_macd_momentum(self, bi) -> float:
        """Calculate MACD momentum"""
        klu = bi.get_end_klu()
        if hasattr(klu, 'macd') and klu.macd:
            return klu.macd.macd
        return 0.0

    def update_with_new_data(self, chan, lv):
        """Update candidates with new data and validate existing ones"""
        # Generate new candidates
        new_candidates = self.generate_candidates_from_chan(chan, lv)
        
        # Classify candidates by confidence
        for candidate in new_candidates:
            if candidate.confidence >= self.high_threshold:
                # Convert to actual BSPoint or store as high confidence
                pass
            elif candidate.confidence >= self.medium_threshold:
                self.medium_confidence_candidates.append(candidate)
            elif candidate.confidence >= self.low_threshold:
                self.low_confidence_candidates.append(candidate)
        
        # Validate and update existing candidates
        self._validate_existing_candidates(chan, lv)
        
        # Generate training samples
        self._generate_training_samples_from_candidates()

    def _validate_existing_candidates(self, chan, lv):
        """Validate existing candidates against new data"""
        current_klu_idx = chan.kl_datas[lv].lst[-1].lst[-1].idx if chan.kl_datas[lv].lst else 0
        
        # Update confidence based on subsequent price action
        for candidate_list in [self.medium_confidence_candidates, self.low_confidence_candidates]:
            for candidate in candidate_list:
                # Check if candidate was validated by subsequent price action
                time_since = current_klu_idx - candidate.klu_idx
                if time_since > 0:
                    validation_score = self._calculate_validation_score(candidate, chan, lv, time_since)
                    candidate.confidence = min(1.0, candidate.confidence + validation_score)

    def _calculate_validation_score(self, candidate: BSPointCandidate, chan, lv, time_since: int) -> float:
        """Calculate validation score based on subsequent price action"""
        if time_since < 1:
            return 0.0
        
        # Get price action since candidate
        bi_list = chan.kl_datas[lv].bi_list
        candidate_bi_idx = candidate.bi_idx
        
        if candidate_bi_idx >= len(bi_list):
            return 0.0
        
        candidate_bi = bi_list[candidate_bi_idx]
        candidate_price = candidate_bi.get_end_val()
        
        # Look at subsequent price action
        subsequent_bis = bi_list[candidate_bi_idx + 1:candidate_bi_idx + 1 + min(5, time_since)]
        
        if not subsequent_bis:
            return 0.0
        
        validation_score = 0.0
        
        # For buy candidates, reward upward movement
        if candidate.is_buy:
            for bi in subsequent_bis:
                if bi.is_up() and bi.get_end_val() > candidate_price:
                    gain = (bi.get_end_val() - candidate_price) / candidate_price
                    validation_score += min(0.2, gain * 10)  # Cap at 0.2 per move
        
        # For sell candidates, reward downward movement
        else:
            for bi in subsequent_bis:
                if bi.is_down() and bi.get_end_val() < candidate_price:
                    gain = (candidate_price - bi.get_end_val()) / candidate_price
                    validation_score += min(0.2, gain * 10)
        
        return validation_score

    def _generate_training_samples_from_candidates(self):
        """Generate training samples from all confidence levels"""
        all_candidates = (self.medium_confidence_candidates + 
                         self.low_confidence_candidates)
        
        for candidate in all_candidates:
            sample = candidate.features.copy()
            sample.update({
                'klu_idx': candidate.klu_idx,
                'bi_idx': candidate.bi_idx,
                'is_buy': candidate.is_buy,
                'confidence': candidate.confidence,
                'timestamp': candidate.timestamp,
                'reason': candidate.reason,
                'target_is_bsp': 1.0 if candidate.confidence > self.high_threshold else 0.0,
                'confidence_level': self._get_confidence_level(candidate.confidence)
            })
            
            self.training_samples.append(sample)

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level string"""
        if confidence >= self.high_threshold:
            return "high"
        elif confidence >= self.medium_threshold:
            return "medium"
        else:
            return "low"

    def get_training_data(self) -> List[Dict]:
        """Get all training samples"""
        return self.training_samples

    def get_statistics(self) -> Dict:
        """Get generator statistics"""
        return {
            'high_confidence_count': len(self.high_confidence_bsp),
            'medium_confidence_count': len(self.medium_confidence_candidates),
            'low_confidence_count': len(self.low_confidence_candidates),
            'total_training_samples': len(self.training_samples),
            'confidence_distribution': {
                'high': len([c for c in self.training_samples if c.get('confidence_level') == 'high']),
                'medium': len([c for c in self.training_samples if c.get('confidence_level') == 'medium']),
                'low': len([c for c in self.training_samples if c.get('confidence_level') == 'low']),
            }
        }