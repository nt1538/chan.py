from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import copy
from dataclasses import dataclass
from enum import Enum

from BuySellPoint.BSPointList import CBSPointList
from .BS_Point import CBS_Point

class BSPointState(Enum):
    ACTIVE = "active"
    MATURE = "mature" 
    ELIMINATED = "eliminated"
    EXTENDED = "extended"

@dataclass
class BSPointSnapshot:
    bsp: CBS_Point
    state: BSPointState
    timestamp: int
    klu_idx: int
    confidence: float = 0.0
    parent_idx: Optional[int] = None
    extension_count: int = 0
    features: Optional[Dict] = None

class CEnhancedBSPointChainTracker:
    def __init__(self):
        # Track all historical states of BSPoints
        self.bsp_history: Dict[str, List[BSPointSnapshot]] = defaultdict(list)
        
        # Active BSPoints currently being tracked
        self.active_bspoints: Dict[Tuple[bool, int], CBS_Point] = {}
        
        # Completed chains ready for training
        self.completed_chains: Dict[str, List[List[BSPointSnapshot]]] = defaultdict(list)
        
        # Tracking metrics
        self.total_snapshots = 0
        self.eliminated_count = 0
        self.mature_count = 0
        
        # Configuration
        self.min_chain_length = 2
        self.max_extension_lookback = 5
        self.confidence_threshold = 0.3

    def update_with_bspoint_diff(self, prev_bsp_list, curr_bsp_list, lv):
        """Enhanced update method with multiple tracking strategies"""
        old_bspoints = prev_bsp_list.getSortedBspList() if prev_bsp_list else []
        new_bspoints = curr_bsp_list.getSortedBspList()
        
        current_timestamp = len(new_bspoints)  # Use as proxy for time
        
        # Create mapping for quick lookup
        new_idx_map = {(bsp.is_buy, bsp.klu.idx): bsp for bsp in new_bspoints}
        old_idx_map = {(bsp.is_buy, bsp.klu.idx): bsp for bsp in old_bspoints}
        
        # 1. Track eliminated BSPoints
        self._track_eliminated_points(old_idx_map, new_idx_map, current_timestamp)
        
        # 2. Track extended/modified BSPoints  
        self._track_extended_points(old_idx_map, new_idx_map, current_timestamp)
        
        # 3. Track new BSPoints
        self._track_new_points(old_idx_map, new_idx_map, current_timestamp)
        
        # 4. Update confidence scores
        self._update_confidence_scores(new_bspoints, current_timestamp)
        
        # 5. Generate training samples from mature chains
        self._generate_training_samples()

    def _track_eliminated_points(self, old_idx_map, new_idx_map, timestamp):
        """Track BSPoints that were eliminated"""
        for old_key, old_bsp in old_idx_map.items():
            if old_key not in new_idx_map:
                # BSPoint was eliminated
                snapshot = self._create_snapshot(old_bsp, BSPointState.ELIMINATED, timestamp)
                
                # Try to find what it evolved into
                snapshot.parent_idx = self._find_evolution_target(old_bsp, new_idx_map)
                
                chain_key = self._get_chain_key(old_bsp)
                self.bsp_history[chain_key].append(snapshot)
                self.eliminated_count += 1

    def _track_extended_points(self, old_idx_map, new_idx_map, timestamp):
        """Track BSPoints that were extended or modified"""
        for new_key, new_bsp in new_idx_map.items():
            if new_key in old_idx_map:
                old_bsp = old_idx_map[new_key]
                
                # Check if BSPoint was modified (different features, etc.)
                if self._has_significant_change(old_bsp, new_bsp):
                    snapshot = self._create_snapshot(new_bsp, BSPointState.EXTENDED, timestamp)
                    snapshot.extension_count = getattr(old_bsp, 'extension_count', 0) + 1
                    
                    chain_key = self._get_chain_key(new_bsp)
                    self.bsp_history[chain_key].append(snapshot)

    def _track_new_points(self, old_idx_map, new_idx_map, timestamp):
        """Track newly appeared BSPoints"""
        for new_key, new_bsp in new_idx_map.items():
            if new_key not in old_idx_map:
                # New BSPoint appeared
                snapshot = self._create_snapshot(new_bsp, BSPointState.ACTIVE, timestamp)
                
                chain_key = self._get_chain_key(new_bsp)
                self.bsp_history[chain_key].append(snapshot)

    def _update_confidence_scores(self, current_bspoints, timestamp):
        """Update confidence scores based on persistence and validation"""
        for bsp in current_bspoints:
            chain_key = self._get_chain_key(bsp)
            history = self.bsp_history[chain_key]
            
            if history:
                # Calculate confidence based on persistence
                persistence_score = min(len(history) / 10.0, 1.0)
                
                # Calculate confidence based on feature strength
                feature_score = self._calculate_feature_strength(bsp)
                
                # Combined confidence
                confidence = (persistence_score + feature_score) / 2.0
                
                # Update latest snapshot
                if history and history[-1].state == BSPointState.ACTIVE:
                    history[-1].confidence = confidence
                    
                    # Mark as mature if confidence is high enough
                    if confidence > self.confidence_threshold:
                        history[-1].state = BSPointState.MATURE
                        self.mature_count += 1

    def _generate_training_samples(self):
        """Generate training samples from completed chains"""
        for chain_key, snapshots in self.bsp_history.items():
            if len(snapshots) >= self.min_chain_length:
                # Check if chain has mature points
                mature_snapshots = [s for s in snapshots if s.state == BSPointState.MATURE]
                
                if mature_snapshots:
                    # Create training chain
                    training_chain = self._create_training_chain(snapshots, mature_snapshots)
                    if training_chain:
                        self.completed_chains[chain_key].append(training_chain)

    def _create_snapshot(self, bsp: CBS_Point, state: BSPointState, timestamp: int) -> BSPointSnapshot:
        """Create a snapshot of BSPoint state"""
        features = {}
        if bsp.features:
            features = bsp.features.to_dict()
        
        # Add technical features
        features.update({
            'bi_amp': bsp.bi.amp(),
            'bi_klu_cnt': bsp.bi.get_klu_cnt(),
            'klu_close': bsp.klu.close,
            'klu_high': bsp.klu.high,
            'klu_low': bsp.klu.low,
            'type_str': bsp.type2str(),
        })
        
        return BSPointSnapshot(
            bsp=bsp,
            state=state,
            timestamp=timestamp,
            klu_idx=bsp.klu.idx,
            features=features
        )

    def _find_evolution_target(self, old_bsp: CBS_Point, new_idx_map: Dict) -> Optional[int]:
        """Find what the eliminated BSPoint evolved into"""
        # Look for nearby BSPoints of same type and direction
        for offset in range(1, self.max_extension_lookback):
            for direction in [-1, 1]:
                candidate_key = (old_bsp.is_buy, old_bsp.klu.idx + offset * direction)
                if candidate_key in new_idx_map:
                    candidate_bsp = new_idx_map[candidate_key]
                    
                    # Check if it's a likely evolution
                    if (candidate_bsp.bi.idx == old_bsp.bi.idx and
                        any(t in old_bsp.type for t in candidate_bsp.type)):
                        return candidate_bsp.klu.idx
        return None

    def _has_significant_change(self, old_bsp: CBS_Point, new_bsp: CBS_Point) -> bool:
        """Check if BSPoint has changed significantly"""
        # Compare features if available
        if old_bsp.features and new_bsp.features:
            old_features = old_bsp.features.to_dict()
            new_features = new_bsp.features.to_dict()
            
            # Check for significant feature changes
            for key in old_features:
                if key in new_features:
                    old_val = old_features[key]
                    new_val = new_features[key]
                    if old_val is not None and new_val is not None:
                        if abs(old_val - new_val) / max(abs(old_val), 1e-6) > 0.1:
                            return True
        
        # Check type changes
        return old_bsp.type2str() != new_bsp.type2str()

    def _calculate_feature_strength(self, bsp: CBS_Point) -> float:
        """Calculate feature strength score"""
        if not bsp.features:
            return 0.0
        
        features = bsp.features.to_dict()
        score = 0.0
        count = 0
        
        # Score based on divergence rate
        if 'divergence_rate' in features and features['divergence_rate'] is not None:
            score += min(features['divergence_rate'], 1.0)
            count += 1
        
        # Score based on amplitude
        if 'bsp1_bi_amp' in features and features['bsp1_bi_amp'] is not None:
            amp_score = min(features['bsp1_bi_amp'] / 100.0, 1.0)  # Normalize
            score += amp_score
            count += 1
        
        return score / max(count, 1)

    def _get_chain_key(self, bsp: CBS_Point) -> str:
        """Generate chain key for BSPoint"""
        direction = 'buy' if bsp.is_buy else 'sell'
        bs_type = bsp.type[0].value if bsp.type else 'unknown'
        return f"{bs_type}_{direction}"

    def _create_training_chain(self, all_snapshots: List[BSPointSnapshot], 
                             mature_snapshots: List[BSPointSnapshot]) -> Optional[List[BSPointSnapshot]]:
        """Create a training chain from snapshots"""
        if not mature_snapshots:
            return None
        
        # Find the best mature point (highest confidence)
        best_mature = max(mature_snapshots, key=lambda s: s.confidence)
        
        # Get all snapshots up to the mature point
        training_snapshots = []
        for snapshot in all_snapshots:
            training_snapshots.append(snapshot)
            if snapshot.timestamp >= best_mature.timestamp:
                break
        
        # Add labels for ML training
        for i, snapshot in enumerate(training_snapshots):
            if snapshot == best_mature:
                snapshot.features['target_mature'] = 1.0
                snapshot.features['mature_probability'] = 1.0
            else:
                snapshot.features['target_mature'] = 0.0
                # Calculate progress towards maturity
                progress = i / len(training_snapshots)
                snapshot.features['mature_probability'] = progress
        
        return training_snapshots

    def get_training_data(self) -> List[Dict]:
        """Extract training data for ML model"""
        training_data = []
        
        for chain_key, chains in self.completed_chains.items():
            for chain in chains:
                for snapshot in chain:
                    if snapshot.features:
                        training_sample = snapshot.features.copy()
                        training_sample.update({
                            'chain_key': chain_key,
                            'state': snapshot.state.value,
                            'confidence': snapshot.confidence,
                            'timestamp': snapshot.timestamp,
                            'klu_idx': snapshot.klu_idx,
                        })
                        training_data.append(training_sample)
        
        return training_data

    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        total_chains = sum(len(chains) for chains in self.completed_chains.values())
        return {
            'total_snapshots': self.total_snapshots,
            'eliminated_count': self.eliminated_count,
            'mature_count': self.mature_count,
            'completed_chains': total_chains,
            'active_chains': len(self.bsp_history),
            'chain_breakdown': {k: len(v) for k, v in self.completed_chains.items()}
        }