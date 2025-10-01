from typing import Dict, List, Optional, Tuple
import copy
from collections import defaultdict

from BuySellPoint.BSPointAugmentation import CBSPointDataAugmentation
from BuySellPoint.BSPointEnhanced import CEnhancedBSPointChainTracker
from BuySellPoint.BSPointGenerator import CMultiLevelBSPointGenerator

class CIntegratedBSPointSystem:
    """
    Integrated system combining enhanced chain tracking, multi-level generation,
    and data augmentation for comprehensive BSPoint training data generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        
        # Initialize components
        self.enhanced_tracker = CEnhancedBSPointChainTracker()
        self.multilevel_generator = CMultiLevelBSPointGenerator()
        self.data_augmentor = CBSPointDataAugmentation()
        
        # Configuration
        self.config = {
            'min_data_points': config.get('min_data_points', 500),
            'target_data_points': config.get('target_data_points', 2000),
            'augmentation_ratio': config.get('augmentation_ratio', 0.7),
            'candidate_confidence_threshold': config.get('candidate_confidence_threshold', 0.3),
            'enable_market_regime_samples': config.get('enable_market_regime_samples', True),
            'balance_dataset': config.get('balance_dataset', True),
            'lookback_window': config.get('lookback_window', 100),
        }
        
        # Data storage
        self.all_training_data = []
        self.metadata = {
            'total_samples': 0,
            'original_samples': 0,
            'generated_samples': 0,
            'augmented_samples': 0,
            'eliminated_chains': 0,
            'candidate_samples': 0,
        }

    def process_chan_update(self, prev_bsp_list, curr_bsp_list, chan, lv):
        """
        Main processing function called during Chan updates
        """
        # 1. Enhanced chain tracking
        self.enhanced_tracker.update_with_bspoint_diff(prev_bsp_list, curr_bsp_list, lv)
        
        # 2. Multi-level candidate generation
        self.multilevel_generator.update_with_new_data(chan, lv)
        
        # 3. Collect training data
        self._collect_training_data()
        
        # 4. Augment data if needed
        if len(self.all_training_data) >= self.config['min_data_points']:
            self._augment_training_data()

    def _collect_training_data(self):
        """Collect training data from all sources"""
        new_data = []
        
        # 1. From enhanced chain tracker
        chain_data = self.enhanced_tracker.get_training_data()
        for sample in chain_data:
            sample['data_source'] = 'chain_tracker'
        new_data.extend(chain_data)
        
        # 2. From multilevel generator
        candidate_data = self.multilevel_generator.get_training_data()
        filtered_candidates = [
            sample for sample in candidate_data 
            if sample.get('confidence', 0) >= self.config['candidate_confidence_threshold']
        ]
        for sample in filtered_candidates:
            sample['data_source'] = 'candidate_generator'
        new_data.extend(filtered_candidates)
        
        # 3. Update metadata
        self.metadata['original_samples'] += len(new_data)
        
        # 4. Add to main dataset (avoiding duplicates)
        existing_keys = {self._generate_sample_key(sample) for sample in self.all_training_data}
        
        for sample in new_data:
            sample_key = self._generate_sample_key(sample)
            if sample_key not in existing_keys:
                self.all_training_data.append(sample)
                existing_keys.add(sample_key)

    def _generate_sample_key(self, sample: Dict) -> str:
        """Generate unique key for sample to avoid duplicates"""
        key_fields = ['klu_idx', 'is_buy', 'chain_type', 'data_source']
        key_parts = []
        
        for field in key_fields:
            if field in sample:
                key_parts.append(f"{field}:{sample[field]}")
        
        return "|".join(key_parts)

    def _augment_training_data(self):
        """Augment training data using various techniques"""
        current_size = len(self.all_training_data)
        target_size = self.config['target_data_points']
        
        if current_size >= target_size:
            return
        
        # Calculate how much augmentation is needed
        augmentation_target = min(
            target_size,
            int(current_size * (1 + self.config['augmentation_ratio']))
        )
        
        print(f"Augmenting data from {current_size} to {augmentation_target} samples")
        
        # 1. Basic augmentation
        augmented_data = self.data_augmentor.augment_training_data(
            self.all_training_data, 
            augmentation_target
        )
        
        # 2. Market regime samples
        if self.config['enable_market_regime_samples']:
            regime_samples = self.data_augmentor.generate_market_regime_samples(
                self.all_training_data[:min(100, len(self.all_training_data))]
            )
            augmented_data.extend(regime_samples)
        
        # 3. Balance dataset
        if self.config['balance_dataset']:
            augmented_data = self.data_augmentor.create_balanced_dataset(
                augmented_data, balance_ratio=0.6
            )
        
        # 4. Add ensemble features
        augmented_data = self.data_augmentor.add_ensemble_features(augmented_data)
        
        # 5. Update main dataset
        original_count = len(self.all_training_data)
        self.all_training_data = augmented_data
        
        # 6. Update metadata
        self.metadata['augmented_samples'] = len(augmented_data) - original_count
        self.metadata['total_samples'] = len(augmented_data)

    def get_training_dataset(self, include_metadata: bool = True) -> Dict:
        """Get complete training dataset"""
        if not self.all_training_data:
            return {'data': [], 'metadata': self.metadata}
        
        # Prepare final dataset
        dataset = {
            'data': self.all_training_data,
            'feature_columns': self._get_feature_columns(),
            'target_columns': self._get_target_columns(),
            'size': len(self.all_training_data),
        }
        
        if include_metadata:
            dataset['metadata'] = self.metadata
            dataset['statistics'] = self._get_dataset_statistics()
        
        return dataset

    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        if not self.all_training_data:
            return []
        
        # Exclude target and metadata columns
        exclude_cols = {
            'target_mature', 'target_is_bsp', 'mature_probability',
            'chain_key', 'state', 'timestamp', 'data_source',
            'generation_method', 'source_indices', 'confidence_level'
        }
        
        all_columns = set()
        for sample in self.all_training_data[:100]:  # Sample first 100 for efficiency
            all_columns.update(sample.keys())
        
        feature_columns = [col for col in all_columns if col not in exclude_cols]
        return sorted(feature_columns)

    def _get_target_columns(self) -> List[str]:
        """Get list of target columns"""
        target_cols = ['target_mature', 'target_is_bsp', 'mature_probability']
        available_targets = []
        
        if self.all_training_data:
            sample_keys = set(self.all_training_data[0].keys())
            available_targets = [col for col in target_cols if col in sample_keys]
        
        return available_targets

    def _get_dataset_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.all_training_data:
            return {}
        
        stats = {
            'data_source_distribution': defaultdict(int),
            'confidence_distribution': defaultdict(int),
            'target_distribution': defaultdict(int),
            'generation_method_distribution': defaultdict(int),
        }
        
        for sample in self.all_training_data:
            # Data source distribution
            source = sample.get('data_source', 'unknown')
            stats['data_source_distribution'][source] += 1
            
            # Confidence distribution
            confidence = sample.get('confidence', 0)
            confidence_bucket = f"{int(confidence * 10) * 10}-{int(confidence * 10) * 10 + 10}%"
            stats['confidence_distribution'][confidence_bucket] += 1
            
            # Target distribution
            target = sample.get('target_mature', -1)
            if target >= 0.5:
                stats['target_distribution']['positive'] += 1
            elif target >= 0:
                stats['target_distribution']['negative'] += 1
            else:
                stats['target_distribution']['unknown'] += 1
            
            # Generation method distribution
            method = sample.get('generation_method', 'original')
            stats['generation_method_distribution'][method] += 1
        
        return dict(stats)

    def export_training_data(self, filepath: str, format: str = 'csv'):
        """Export training data to file"""
        import pandas as pd
        
        if not self.all_training_data:
            print("No training data available to export")
            return
        
        df = pd.DataFrame(self.all_training_data)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Exported {len(df)} samples to {filepath}")

    def get_statistics_summary(self) -> str:
        """Get formatted statistics summary"""
        stats = self._get_dataset_statistics()
        
        summary = f"""
BSPoint Training Data Statistics:
=================================
Total Samples: {self.metadata['total_samples']}
Original Samples: {self.metadata['original_samples']}
Augmented Samples: {self.metadata['augmented_samples']}

Data Source Distribution:
{self._format_distribution(stats.get('data_source_distribution', {}))}

Target Distribution:
{self._format_distribution(stats.get('target_distribution', {}))}

Generation Method Distribution:
{self._format_distribution(stats.get('generation_method_distribution', {}))}

Feature Columns: {len(self._get_feature_columns())}
Target Columns: {len(self._get_target_columns())}
        """
        
        return summary.strip()

    def _format_distribution(self, distribution: Dict) -> str:
        """Format distribution dictionary for display"""
        if not distribution:
            return "  No data available"
        
        total = sum(distribution.values())
        lines = []
        
        for key, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            lines.append(f"  {key}: {count} ({percentage:.1f}%)")
        
        return "\n".join(lines)

    def reset_data(self):
        """Reset all collected data"""
        self.all_training_data = []
        self.metadata = {
            'total_samples': 0,
            'original_samples': 0,
            'generated_samples': 0,
            'augmented_samples': 0,
            'eliminated_chains': 0,
            'candidate_samples': 0,
        }
        
        # Reset component states
        self.enhanced_tracker = CEnhancedBSPointChainTracker()
        self.multilevel_generator = CMultiLevelBSPointGenerator()

    def configure(self, new_config: Dict):
        """Update configuration"""
        self.config.update(new_config)

# Integration with existing Chan system
def integrate_with_chan(chan, integrated_system: CIntegratedBSPointSystem):
    """
    Integration function to add the enhanced BSPoint system to existing Chan
    """
    # Replace existing chain tracker
    chan.bs_chain_tracker = integrated_system
    
    # Modify step_load to use integrated system
    original_step_load = chan.step_load
    
    def enhanced_step_load():
        prev_bsp_list = None
        
        for idx, snapshot in enumerate(original_step_load()):
            if idx < chan.conf.skip_step:
                continue
            
            curr_bsp_list = snapshot.kl_datas[chan.lv_list[0]].bs_point_lst
            
            if prev_bsp_list is not None:
                integrated_system.process_chan_update(
                    prev_bsp_list, 
                    curr_bsp_list, 
                    snapshot, 
                    chan.lv_list[0]
                )
            
            prev_bsp_list = copy.deepcopy(curr_bsp_list)
            yield snapshot
    
    chan.enhanced_step_load = enhanced_step_load
    return chan