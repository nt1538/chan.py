# normalized_sliding_window_chan.py
"""
Normalized Sliding Window Chan
Extends SlidingWindowChan with built-in normalization capabilities
Makes pattern recognition more effective by normalizing features on-the-fly
"""

import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from collections import defaultdict

from sliding_window_chan import SlidingWindowChan
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE, DATA_SRC
from KLine.KLine_Unit import CKLine_Unit


class NormalizedSlidingWindowChan(SlidingWindowChan):
    """
    Extends SlidingWindowChan with built-in normalization.
    Normalizes price data, technical indicators, and creates pattern-specific features
    to make pattern recognition more effective.
    """
    
    def __init__(
        self,
        code: str,
        begin_time=None,
        end_time=None,
        data_src=None,
        lv_list=None,
        config=None,
        autype=None,
        max_klines: int = 500,
        normalization_type: str = 'z-score',
        window_normalization: bool = True,
        window_size: int = 50,
        normalize_features: bool = True
    ):
        """
        Initialize normalized sliding window Chan system.
        
        Args:
            normalization_type: Type of normalization ('z-score', 'minmax', 'percentage')
            window_normalization: Whether to normalize based on current window
            window_size: Size of normalization window (if window_normalization is True)
            normalize_features: Whether to normalize features
            All other args: Same as SlidingWindowChan
        """
        # Initialize the parent class
        super().__init__(
            code=code,
            begin_time=begin_time,
            end_time=end_time,
            data_src=data_src,
            lv_list=lv_list,
            config=config,
            autype=autype,
            max_klines=max_klines
        )
        
        # Normalization settings
        self.normalization_type = normalization_type
        self.window_normalization = window_normalization
        self.normalization_window_size = min(window_size, max_klines)
        self.normalize_features = normalize_features
        
        # Feature groups for normalization
        self.price_features = ['open', 'high', 'low', 'close']
        self.volume_features = ['volume']
        self.tech_indicator_features = [
            'macd', 'macd_dif', 'macd_dea', 'rsi', 'kdj_k', 'kdj_d', 'kdj_j',
            'dmi_plus', 'dmi_minus', 'dmi_adx'
        ]
        
        # Historical feature stats for normalization
        self.feature_stats = defaultdict(dict)
        
        # Store normalized versions of all BSP
        self.all_normalized_bsp: Dict[Tuple, Dict] = {}
        
        print(f"[🧮] Normalization: {self.normalization_type} ({'window' if window_normalization else 'global'})")
        if window_normalization:
            print(f"[🪟] Normalization window size: {self.normalization_window_size}")
    
    def step_load(self):
        """
        Override step_load to add normalization before extracting BSP.
        """
        # Load all K-lines once
        self.all_klines = self._load_all_klines()
        
        if not self.all_klines:
            print("[⚠️] No K-lines loaded")
            return
        
        print(f"[🪟] Starting sliding window processing (window size: {self.max_klines})")
        print(f"[📊] Total K-lines to process: {len(self.all_klines)}")
        
        # Calculate global stats if not using window normalization
        if not self.window_normalization:
            self._calculate_global_feature_stats()
        
        # Process each K-line
        for klu_idx, klu in enumerate(self.all_klines):
            self.snapshot_count += 1
            
            # Add K-line to window
            self.kline_window.append(klu)
            
            # Update window start position
            if len(self.kline_window) >= self.max_klines:
                self.current_window_start = klu_idx - self.max_klines + 1
            else:
                self.current_window_start = 0
            
            # Create Chan instance for current window
            window_chan = self._create_window_chan()
            
            # Feed window data to Chan
            window_klines = list(self.kline_window)
            window_chan.trigger_load({self.lv_list[0]: window_klines})
            
            # Normalize the window data
            self._normalize_window(window_klines)
            
            # Extract BSP from this window
            self._extract_window_bsp(window_chan, self.snapshot_count)
            
            # Track window changes
            if len(self.kline_window) == self.max_klines:
                self.total_windows_processed += 1
            
            # Store last chan for compatibility
            self.last_chan = window_chan
            
            # Yield snapshot
            yield window_chan
    
    def _calculate_global_feature_stats(self):
        """
        Calculate global feature statistics for normalization.
        Used when window_normalization=False.
        """
        print("[📊] Calculating global feature statistics for normalization...")
        
        # Create a DataFrame from all K-lines for efficient calculation
        klines_data = {}
        
        # Extract basic price and volume features
        for feat in self.price_features + self.volume_features:
            if feat in ['open', 'high', 'low', 'close', 'volume']:
                klines_data[feat] = [getattr(kl, feat, 0) for kl in self.all_klines]
        
        # Convert to DataFrame
        df = pd.DataFrame(klines_data)
        
        # Calculate statistics for each feature
        for feature in df.columns:
            values = df[feature].values
            values = values[~np.isnan(values)]  # Remove NaN values
            
            if len(values) > 0:
                # Store statistics based on normalization type
                if self.normalization_type == 'z-score':
                    self.feature_stats[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values) or 1.0  # Avoid division by zero
                    }
                elif self.normalization_type == 'minmax':
                    self.feature_stats[feature] = {
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                elif self.normalization_type == 'percentage':
                    # For percentage change, we don't need global stats
                    pass
        
        print(f"[✅] Calculated global statistics for {len(self.feature_stats)} features")
    
    def _normalize_window(self, klines: List[CKLine_Unit]):
        """
        Apply normalization to all K-lines in the current window.
        """
        if not klines or len(klines) < 2:
            return
        
        # Calculate window statistics if using window normalization
        if self.window_normalization:
            window_stats = self._calculate_window_stats(klines)
        
        # Apply normalization to each K-line
        for i, klu in enumerate(klines):
            # Normalize price features
            for feature in self.price_features:
                if hasattr(klu, feature):
                    value = getattr(klu, feature)
                    
                    if value is not None:
                        if self.window_normalization:
                            stats = window_stats.get(feature, {})
                        else:
                            stats = self.feature_stats.get(feature, {})
                            
                        normalized_value = self._apply_normalization(feature, value, stats, klines, i)
                        setattr(klu, f"norm_{feature}", normalized_value)
            
            # Normalize volume features
            for feature in self.volume_features:
                if hasattr(klu, feature):
                    value = getattr(klu, feature)
                    
                    if value is not None:
                        if self.window_normalization:
                            stats = window_stats.get(feature, {})
                        else:
                            stats = self.feature_stats.get(feature, {})
                            
                        normalized_value = self._apply_normalization(feature, value, stats, klines, i)
                        setattr(klu, f"norm_{feature}", normalized_value)
            
            # Normalize technical indicators
            self._normalize_technical_indicators(klu, window_stats if self.window_normalization else self.feature_stats)
            
            # Add additional normalized features
            self._add_normalized_pattern_features(klu, klines, i)
    
    def _calculate_window_stats(self, klines: List[CKLine_Unit]) -> Dict[str, Dict]:
        """
        Calculate statistics for the current window for normalization.
        """
        window_stats = {}
        
        # Calculate for price features
        for feature in self.price_features:
            values = [getattr(kl, feature, None) for kl in klines]
            values = [v for v in values if v is not None]
            
            if values:
                if self.normalization_type == 'z-score':
                    window_stats[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values) or 1.0  # Avoid division by zero
                    }
                elif self.normalization_type == 'minmax':
                    window_stats[feature] = {
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        # Calculate for volume features
        for feature in self.volume_features:
            values = [getattr(kl, feature, None) for kl in klines]
            values = [v for v in values if v is not None]
            
            if values:
                if self.normalization_type == 'z-score':
                    window_stats[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values) or 1.0  # Avoid division by zero
                    }
                elif self.normalization_type == 'minmax':
                    window_stats[feature] = {
                        'min': np.min(values),
                        'max': np.max(values) if np.max(values) > np.min(values) else np.min(values) + 1.0
                    }
        
        # Technical indicators - some may need specialized handling
        for feature in self.tech_indicator_features:
            values = []
            
            # For MACD, KDJ, and other complex indicators that are objects
            if feature in ['macd', 'macd_dif', 'macd_dea']:
                # MACD and components
                for kl in klines:
                    if hasattr(kl, 'macd') and kl.macd is not None:
                        if feature == 'macd':
                            values.append(getattr(kl.macd, 'macd', None))
                        elif feature == 'macd_dif':
                            values.append(getattr(kl.macd, 'DIF', None))
                        elif feature == 'macd_dea':
                            values.append(getattr(kl.macd, 'DEA', None))
            
            elif feature in ['kdj_k', 'kdj_d', 'kdj_j']:
                # KDJ components
                for kl in klines:
                    if hasattr(kl, 'kdj') and kl.kdj is not None:
                        if feature == 'kdj_k':
                            values.append(getattr(kl.kdj, 'k', None))
                        elif feature == 'kdj_d':
                            values.append(getattr(kl.kdj, 'd', None))
                        elif feature == 'kdj_j':
                            values.append(getattr(kl.kdj, 'j', None))
            
            elif feature in ['dmi_plus', 'dmi_minus', 'dmi_adx']:
                # DMI components
                for kl in klines:
                    if hasattr(kl, 'dmi') and kl.dmi is not None:
                        if feature == 'dmi_plus':
                            values.append(getattr(kl.dmi, 'plus_di', None))
                        elif feature == 'dmi_minus':
                            values.append(getattr(kl.dmi, 'minus_di', None))
                        elif feature == 'dmi_adx':
                            values.append(getattr(kl.dmi, 'adx', None))
            
            else:
                # Simple indicators
                for kl in klines:
                    if hasattr(kl, feature):
                        values.append(getattr(kl, feature, None))
            
            # Filter out None values
            values = [v for v in values if v is not None]
            
            if values:
                if self.normalization_type == 'z-score':
                    window_stats[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values) or 1.0  # Avoid division by zero
                    }
                elif self.normalization_type == 'minmax':
                    window_stats[feature] = {
                        'min': np.min(values),
                        'max': np.max(values) if np.max(values) > np.min(values) else np.min(values) + 1.0
                    }
        
        return window_stats
    
    def _apply_normalization(self, feature: str, value: float, stats: Dict, klines: List[CKLine_Unit], idx: int) -> float:
        """
        Apply normalization to a value based on the normalization type.
        """
        if self.normalization_type == 'z-score':
            mean = stats.get('mean', 0.0)
            std = stats.get('std', 1.0)
            return (value - mean) / std if std != 0 else 0.0
        
        elif self.normalization_type == 'minmax':
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 1.0)
            range_val = max_val - min_val
            return (value - min_val) / range_val if range_val != 0 else 0.5
        
        elif self.normalization_type == 'percentage':
            # Calculate percentage change from previous value
            if idx > 0 and feature in self.price_features + self.volume_features:
                prev_klu = klines[idx - 1]
                prev_value = getattr(prev_klu, feature, None)
                
                if prev_value is not None and prev_value != 0:
                    return (value - prev_value) / prev_value
            
            return 0.0
        
        # Default (no normalization)
        return value
    
    def _normalize_technical_indicators(self, klu: CKLine_Unit, stats: Dict[str, Dict]):
        """
        Normalize technical indicators from K-line.
        """
        # Normalize MACD
        if hasattr(klu, 'macd') and klu.macd is not None:
            for component, attr in [('macd', 'macd'), ('macd_dif', 'DIF'), ('macd_dea', 'DEA')]:
                if hasattr(klu.macd, attr):
                    value = getattr(klu.macd, attr)
                    if value is not None and component in stats:
                        if self.normalization_type == 'z-score':
                            mean = stats[component].get('mean', 0.0)
                            std = stats[component].get('std', 1.0)
                            setattr(klu, f"norm_{component}", (value - mean) / std if std != 0 else 0.0)
                        elif self.normalization_type == 'minmax':
                            min_val = stats[component].get('min', 0.0)
                            max_val = stats[component].get('max', 1.0)
                            range_val = max_val - min_val
                            setattr(klu, f"norm_{component}", (value - min_val) / range_val if range_val != 0 else 0.5)
        
        # Normalize RSI
        if hasattr(klu, 'rsi') and klu.rsi is not None:
            # RSI is already 0-100, can normalize to -1 to 1
            rsi_value = klu.rsi
            klu.norm_rsi = (rsi_value - 50) / 50  # -1 to 1 scale
        
        # Normalize KDJ
        if hasattr(klu, 'kdj') and klu.kdj is not None:
            for component, attr in [('kdj_k', 'k'), ('kdj_d', 'd'), ('kdj_j', 'j')]:
                if hasattr(klu.kdj, attr):
                    value = getattr(klu.kdj, attr)
                    if value is not None:
                        if component in stats:
                            if self.normalization_type == 'z-score':
                                mean = stats[component].get('mean', 0.0)
                                std = stats[component].get('std', 1.0)
                                setattr(klu, f"norm_{component}", (value - mean) / std if std != 0 else 0.0)
                            elif self.normalization_type == 'minmax':
                                min_val = stats[component].get('min', 0.0)
                                max_val = stats[component].get('max', 1.0)
                                range_val = max_val - min_val
                                setattr(klu, f"norm_{component}", (value - min_val) / range_val if range_val != 0 else 0.5)
                        else:
                            # KDJ components are typically 0-100
                            setattr(klu, f"norm_{component}", (value - 50) / 50)  # -1 to 1 scale
        
        # Normalize DMI
        if hasattr(klu, 'dmi') and klu.dmi is not None:
            for component, attr in [('dmi_plus', 'plus_di'), ('dmi_minus', 'minus_di'), ('dmi_adx', 'adx')]:
                if hasattr(klu.dmi, attr):
                    value = getattr(klu.dmi, attr)
                    if value is not None and component in stats:
                        if self.normalization_type == 'z-score':
                            mean = stats[component].get('mean', 0.0)
                            std = stats[component].get('std', 1.0)
                            setattr(klu, f"norm_{component}", (value - mean) / std if std != 0 else 0.0)
                        elif self.normalization_type == 'minmax':
                            min_val = stats[component].get('min', 0.0)
                            max_val = stats[component].get('max', 1.0)
                            range_val = max_val - min_val
                            setattr(klu, f"norm_{component}", (value - min_val) / range_val if range_val != 0 else 0.5)
    
    def _add_normalized_pattern_features(self, klu: CKLine_Unit, klines: List[CKLine_Unit], idx: int):
        """
        Add pattern-specific normalized features to the K-line.
        """
        # Price movement pattern (up/down)
        if idx > 0 and hasattr(klu, 'close') and hasattr(klines[idx-1], 'close'):
            current_close = klu.close
            prev_close = klines[idx-1].close
            
            if current_close > prev_close:
                klu.price_direction = 1  # Up
            elif current_close < prev_close:
                klu.price_direction = -1  # Down
            else:
                klu.price_direction = 0  # Unchanged
        else:
            klu.price_direction = 0
        
        # Calculate percentage change from previous K-line
        if idx > 0 and hasattr(klu, 'close') and hasattr(klines[idx-1], 'close'):
            prev_close = klines[idx-1].close
            if prev_close != 0:
                klu.pct_change = (klu.close - prev_close) / prev_close
            else:
                klu.pct_change = 0.0
        else:
            klu.pct_change = 0.0
        
        # Calculate log return (more symmetric around zero)
        if idx > 0 and hasattr(klu, 'close') and hasattr(klines[idx-1], 'close'):
            prev_close = klines[idx-1].close
            if prev_close > 0 and klu.close > 0:
                klu.log_return = np.log(klu.close / prev_close)
            else:
                klu.log_return = 0.0
        else:
            klu.log_return = 0.0
        
        # Calculate relative price features
        if hasattr(klu, 'open') and hasattr(klu, 'high') and hasattr(klu, 'low') and hasattr(klu, 'close'):
            # Upper shadow
            klu.upper_shadow = (klu.high - max(klu.open, klu.close)) / klu.close if klu.close > 0 else 0.0
            
            # Lower shadow
            klu.lower_shadow = (min(klu.open, klu.close) - klu.low) / klu.close if klu.close > 0 else 0.0
            
            # Body size
            klu.body_size = abs(klu.close - klu.open) / klu.close if klu.close > 0 else 0.0
            
            # High-low range
            klu.hl_range = (klu.high - klu.low) / klu.close if klu.close > 0 else 0.0
        
        # Calculate moving averages and relative distance from them
        for window in [5, 10, 20]:
            if idx >= window-1 and all(hasattr(k, 'close') for k in klines[idx-window+1:idx+1]):
                ma_values = [k.close for k in klines[idx-window+1:idx+1]]
                ma_avg = sum(ma_values) / window
                
                setattr(klu, f"ma_{window}", ma_avg)
                
                # Distance from MA
                if ma_avg > 0:
                    setattr(klu, f"rel_ma_{window}", (klu.close - ma_avg) / ma_avg)
                else:
                    setattr(klu, f"rel_ma_{window}", 0.0)
        
        # Generate multi-day price patterns (last 3 days)
        if idx >= 2:
            pattern = []
            for i in range(idx-2, idx+1):
                if i > 0 and hasattr(klines[i], 'close') and hasattr(klines[i-1], 'close'):
                    if klines[i].close > klines[i-1].close:
                        pattern.append('U')  # Up
                    elif klines[i].close < klines[i-1].close:
                        pattern.append('D')  # Down
                    else:
                        pattern.append('S')  # Same
                else:
                    pattern.append('S')
                    
            klu.price_pattern = ''.join(pattern)
            
            # Trend strength (consecutive up/down days)
            up_days = pattern.count('U')
            down_days = pattern.count('D')
            
            # Trend consistency
            klu.trend_strength = max(up_days, down_days) / len(pattern) if len(pattern) > 0 else 0.0
            
            # Trend direction
            klu.trend_direction = 1 if up_days > down_days else (-1 if down_days > up_days else 0)
    
    def _create_bsp_snapshot(self, bsp: Any, bs_type_str: str, snapshot_idx: int) -> Dict:
        """
        Override _create_bsp_snapshot to include normalized features.
        """
        # Get original snapshot from parent class
        snapshot = super()._create_bsp_snapshot(bsp, bs_type_str, snapshot_idx)
        
        # Add normalized features from KLU
        klu = bsp.klu
        
        # Add normalized price features
        for feature in self.price_features:
            norm_feature = f"norm_{feature}"
            if hasattr(klu, norm_feature):
                snapshot[norm_feature] = getattr(klu, norm_feature)
        
        # Add normalized volume features
        for feature in self.volume_features:
            norm_feature = f"norm_{feature}"
            if hasattr(klu, norm_feature):
                snapshot[norm_feature] = getattr(klu, norm_feature)
        
        # Add normalized technical indicators
        for component in ['macd', 'macd_dif', 'macd_dea', 'rsi', 'kdj_k', 'kdj_d', 'kdj_j', 
                         'dmi_plus', 'dmi_minus', 'dmi_adx']:
            norm_feature = f"norm_{component}"
            if hasattr(klu, norm_feature):
                snapshot[norm_feature] = getattr(klu, norm_feature)
        
        # Add pattern features
        pattern_features = ['price_direction', 'pct_change', 'log_return', 
                          'upper_shadow', 'lower_shadow', 'body_size', 'hl_range',
                          'price_pattern', 'trend_strength', 'trend_direction']
        
        for feature in pattern_features:
            if hasattr(klu, feature):
                snapshot[feature] = getattr(klu, feature)
        
        # Add moving averages and relative distances
        for window in [5, 10, 20]:
            ma_feature = f"ma_{window}"
            rel_ma_feature = f"rel_ma_{window}"
            
            if hasattr(klu, ma_feature):
                snapshot[ma_feature] = getattr(klu, ma_feature)
            
            if hasattr(klu, rel_ma_feature):
                snapshot[rel_ma_feature] = getattr(klu, rel_ma_feature)
        
        return snapshot
    
    def get_stats(self) -> Dict:
        """Get statistics about the normalized sliding window system."""
        stats = super().get_stats()
        
        # Add normalization stats
        stats.update({
            'normalization_type': self.normalization_type,
            'window_normalization': self.window_normalization,
            'normalization_window_size': self.normalization_window_size,
        })
        
        return stats


# Example usage
if __name__ == "__main__":
    from Common.CEnum import DATA_SRC, AUTYPE, KL_TYPE
    from ChanConfig import CChanConfig
    
    config = CChanConfig({
        "cal_kdj": True,
        "cal_dmi": True,
        "cal_rsi": True,
        "cal_macd": True,
        "bi_strict": True,
        "bs_type": '1,2,3a,1p,2s,3b',
    })
    
    # Create normalized sliding window Chan
    chan = NormalizedSlidingWindowChan(
        code="^GSPC",
        begin_time="2021-01-01",
        end_time="2021-03-31",
        data_src=DATA_SRC.CSV,
        lv_list=[KL_TYPE.K_DAY],
        config=config,
        autype=AUTYPE.QFQ,
        max_klines=500,
        normalization_type='z-score',  # 'z-score', 'minmax', or 'percentage'
        window_normalization=True,
        window_size=50
    )
    
    # Process with sliding window
    for snapshot_idx, snapshot in enumerate(chan.step_load()):
        if snapshot_idx % 100 == 0:
            stats = chan.get_stats()
            print(f"[📊] Snapshot {snapshot_idx}: {stats['unique_bsp_count']} unique BSP")
    
    # Get all normalized BSP
    all_bsp = chan.get_all_historical_bsp()
    print(f"\n[✅] Processing complete!")
    print(f"[📊] Total unique BSP: {len(all_bsp)}")
    
    # Check normalized features
    if all_bsp:
        norm_features = [k for k in all_bsp[0].keys() if k.startswith('norm_') or k in 
                       ['pct_change', 'log_return', 'rel_ma_5', 'price_pattern']]
        print(f"[🧮] Normalized features: {len(norm_features)}")
        print(f"[📝] Example normalized features: {norm_features[:5]}...")