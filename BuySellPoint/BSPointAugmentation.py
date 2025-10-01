from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import random

@dataclass
class SyntheticBSPoint:
    features: Dict
    label: float
    confidence: float
    generation_method: str
    source_indices: List[int]

class CBSPointDataAugmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(n_neighbors=5)
        self.feature_importance = {}
        self.noise_levels = {
            'low': 0.05,
            'medium': 0.1,
            'high': 0.2
        }
        
    def augment_training_data(self, original_data: List[Dict], 
                            target_size: int = 1000) -> List[Dict]:
        """
        Augment BSPoint training data using multiple techniques
        """
        if not original_data:
            return []
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(original_data)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in 
                       ['target_mature', 'target_is_bsp', 'chain_key', 'state', 'timestamp']]
        
        augmented_data = original_data.copy()
        current_size = len(augmented_data)
        
        while len(augmented_data) < target_size:
            remaining = target_size - len(augmented_data)
            batch_size = min(200, remaining)
            
            # Apply different augmentation techniques
            new_samples = []
            
            # 1. SMOTE-like generation (40% of new samples)
            smote_count = int(batch_size * 0.4)
            new_samples.extend(self._generate_smote_samples(df, feature_cols, smote_count))
            
            # 2. Noise injection (30% of new samples)
            noise_count = int(batch_size * 0.3)
            new_samples.extend(self._generate_noise_samples(df, feature_cols, noise_count))
            
            # 3. Feature mixing (20% of new samples)  
            mix_count = int(batch_size * 0.2)
            new_samples.extend(self._generate_mixed_samples(df, feature_cols, mix_count))
            
            # 4. Temporal perturbation (10% of new samples)
            temporal_count = batch_size - smote_count - noise_count - mix_count
            new_samples.extend(self._generate_temporal_samples(df, feature_cols, temporal_count))
            
            augmented_data.extend(new_samples)
        
        return augmented_data[:target_size]

    def _generate_smote_samples(self, df: pd.DataFrame, feature_cols: List[str], 
                               count: int) -> List[Dict]:
        """Generate synthetic samples using SMOTE-like interpolation"""
        samples = []
        
        if len(df) < 2:
            return samples
        
        # Fit nearest neighbors on feature data
        feature_data = df[feature_cols].fillna(0).values
        self.scaler.fit(feature_data)
        scaled_features = self.scaler.transform(feature_data)
        self.nn_model.fit(scaled_features)
        
        for _ in range(count):
            # Random sample selection
            idx1 = random.randint(0, len(df) - 1)
            sample1 = df.iloc[idx1]
            
            # Find nearest neighbors
            point = scaled_features[idx1].reshape(1, -1)
            distances, indices = self.nn_model.kneighbors(point)
            
            # Select random neighbor (excluding self)
            neighbor_idx = random.choice(indices[0][1:])
            sample2 = df.iloc[neighbor_idx]
            
            # Interpolate between samples
            alpha = random.uniform(0.1, 0.9)
            new_sample = self._interpolate_samples(sample1, sample2, feature_cols, alpha)
            new_sample['generation_method'] = 'smote'
            new_sample['source_indices'] = [idx1, neighbor_idx]
            
            samples.append(new_sample)
        
        return samples

    def _generate_noise_samples(self, df: pd.DataFrame, feature_cols: List[str],
                               count: int) -> List[Dict]:
        """Generate samples by adding controlled noise"""
        samples = []
        
        for _ in range(count):
            # Random source sample
            source_idx = random.randint(0, len(df) - 1)
            source_sample = df.iloc[source_idx].to_dict()
            
            # Add noise to numerical features
            new_sample = source_sample.copy()
            noise_level = random.choice(list(self.noise_levels.keys()))
            noise_std = self.noise_levels[noise_level]
            
            for col in feature_cols:
                if col in source_sample and pd.notnull(source_sample[col]):
                    if isinstance(source_sample[col], (int, float)):
                        # Add proportional noise
                        base_value = source_sample[col]
                        noise = np.random.normal(0, abs(base_value) * noise_std)
                        new_sample[col] = base_value + noise
            
            new_sample['generation_method'] = f'noise_{noise_level}'
            new_sample['source_indices'] = [source_idx]
            
            samples.append(new_sample)
        
        return samples

    def _generate_mixed_samples(self, df: pd.DataFrame, feature_cols: List[str],
                               count: int) -> List[Dict]:
        """Generate samples by mixing features from different samples"""
        samples = []
        
        for _ in range(count):
            # Select 2-3 source samples
            num_sources = random.randint(2, 3)
            source_indices = random.sample(range(len(df)), num_sources)
            source_samples = [df.iloc[idx] for idx in source_indices]
            
            # Create mixed sample
            new_sample = source_samples[0].to_dict()
            
            # Randomly mix features
            mix_ratio = 0.3  # Probability of taking feature from other sample
            for col in feature_cols:
                if random.random() < mix_ratio and len(source_samples) > 1:
                    # Take feature from random other source
                    other_sample = random.choice(source_samples[1:])
                    if col in other_sample and pd.notnull(other_sample[col]):
                        new_sample[col] = other_sample[col]
            
            new_sample['generation_method'] = 'feature_mixing'
            new_sample['source_indices'] = source_indices
            
            samples.append(new_sample)
        
        return samples

    def _generate_temporal_samples(self, df: pd.DataFrame, feature_cols: List[str],
                                  count: int) -> List[Dict]:
        """Generate samples with temporal perturbations"""
        samples = []
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
        else:
            df_sorted = df
        
        for _ in range(count):
            # Select a sequence of samples
            if len(df_sorted) < 3:
                continue
                
            start_idx = random.randint(0, len(df_sorted) - 3)
            sequence = df_sorted.iloc[start_idx:start_idx + 3]
            
            # Create sample with temporal trend features
            base_sample = sequence.iloc[1].to_dict()  # Middle sample
            
            # Add trend features
            for col in feature_cols:
                if col in sequence.columns:
                    values = sequence[col].dropna()
                    if len(values) >= 2:
                        # Add trend information
                        trend = (values.iloc[-1] - values.iloc[0]) / max(abs(values.iloc[0]), 1e-6)
                        base_sample[f'{col}_trend'] = trend
                        
                        # Add volatility information
                        if len(values) >= 3:
                            volatility = values.std() / max(abs(values.mean()), 1e-6)
                            base_sample[f'{col}_volatility'] = volatility
            
            base_sample['generation_method'] = 'temporal_perturbation'
            base_sample['source_indices'] = [start_idx, start_idx + 1, start_idx + 2]
            
            samples.append(base_sample)
        
        return samples

    def _interpolate_samples(self, sample1: pd.Series, sample2: pd.Series,
                           feature_cols: List[str], alpha: float) -> Dict:
        """Interpolate between two samples"""
        new_sample = {}
        
        for col in sample1.index:
            val1 = sample1[col]
            val2 = sample2[col] if col in sample2.index else val1
            
            if col in feature_cols and pd.notnull(val1) and pd.notnull(val2):
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Linear interpolation for numerical features
                    new_sample[col] = val1 * (1 - alpha) + val2 * alpha
                else:
                    # Random selection for categorical features
                    new_sample[col] = val1 if random.random() < 0.5 else val2
            else:
                new_sample[col] = val1
        
        # Interpolate labels
        if 'target_mature' in sample1.index and 'target_mature' in sample2.index:
            if pd.notnull(sample1['target_mature']) and pd.notnull(sample2['target_mature']):
                new_sample['target_mature'] = sample1['target_mature'] * (1 - alpha) + sample2['target_mature'] * alpha
        
        return new_sample

    def generate_market_regime_samples(self, original_data: List[Dict],
                                     market_regimes: List[str] = None) -> List[Dict]:
        """Generate samples for different market regimes"""
        if market_regimes is None:
            market_regimes = ['bull', 'bear', 'sideways', 'volatile']
        
        regime_samples = []
        
        for regime in market_regimes:
            regime_modifier = self._get_regime_modifier(regime)
            
            for sample in original_data:
                # Create regime-specific sample
                new_sample = sample.copy()
                
                # Apply regime-specific modifications
                for feature, modifier in regime_modifier.items():
                    if feature in new_sample and pd.notnull(new_sample[feature]):
                        if isinstance(new_sample[feature], (int, float)):
                            new_sample[feature] *= modifier
                
                new_sample['market_regime'] = regime
                new_sample['generation_method'] = f'regime_{regime}'
                
                regime_samples.append(new_sample)
        
        return regime_samples

    def _get_regime_modifier(self, regime: str) -> Dict[str, float]:
        """Get feature modifiers for different market regimes"""
        modifiers = {
            'bull': {
                'divergence_rate': 0.8,  # Less divergence needed
                'rsi': 1.1,  # Higher RSI levels
                'macd_value': 1.2,  # Stronger MACD
                'volume': 1.1,  # Higher volume
            },
            'bear': {
                'divergence_rate': 1.3,  # More divergence needed
                'rsi': 0.9,  # Lower RSI levels
                'macd_value': 0.8,  # Weaker MACD
                'volume': 0.9,  # Lower volume
            },
            'sideways': {
                'bi_amp': 0.7,  # Smaller moves
                'divergence_rate': 1.1,
                'volume': 0.8,
            },
            'volatile': {
                'bi_amp': 1.4,  # Larger moves
                'volume': 1.3,  # Higher volume
                'rsi': 1.0,
            }
        }
        
        return modifiers.get(regime, {})

    def create_balanced_dataset(self, original_data: List[Dict],
                              balance_ratio: float = 0.5) -> List[Dict]:
        """Create balanced dataset for binary classification"""
        df = pd.DataFrame(original_data)
        
        if 'target_mature' not in df.columns:
            return original_data
        
        # Separate positive and negative samples
        positive_samples = df[df['target_mature'] > 0.5].to_dict('records')
        negative_samples = df[df['target_mature'] <= 0.5].to_dict('records')
        
        target_positive = int(len(df) * balance_ratio)
        target_negative = len(df) - target_positive
        
        balanced_data = []
        
        # Augment positive samples if needed
        if len(positive_samples) < target_positive:
            augmented_positive = self.augment_training_data(positive_samples, target_positive)
            balanced_data.extend(augmented_positive)
        else:
            balanced_data.extend(random.sample(positive_samples, target_positive))
        
        # Augment negative samples if needed
        if len(negative_samples) < target_negative:
            augmented_negative = self.augment_training_data(negative_samples, target_negative)
            balanced_data.extend(augmented_negative)
        else:
            balanced_data.extend(random.sample(negative_samples, target_negative))
        
        return balanced_data

    def add_ensemble_features(self, data: List[Dict]) -> List[Dict]:
        """Add ensemble-based features"""
        df = pd.DataFrame(data)
        
        # Add rolling statistics
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            
            numerical_cols = df_sorted.select_dtypes(include=[np.number]).columns
            
            for window in [3, 5, 10]:
                for col in numerical_cols:
                    if col in ['timestamp', 'klu_idx']:
                        continue
                    
                    df_sorted[f'{col}_rolling_mean_{window}'] = df_sorted[col].rolling(window).mean()
                    df_sorted[f'{col}_rolling_std_{window}'] = df_sorted[col].rolling(window).std()
                    df_sorted[f'{col}_rolling_max_{window}'] = df_sorted[col].rolling(window).max()
                    df_sorted[f'{col}_rolling_min_{window}'] = df_sorted[col].rolling(window).min()
            
            return df_sorted.fillna(method='bfill').to_dict('records')
        
        return data