import os
import csv
import math
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict
from Common.CEnum import KL_TYPE
from Chan import CChan


def extract_all_bs_points_to_single_file(chan: CChan, lv: KL_TYPE) -> pd.DataFrame:
    """
    Extract all BS points from all types and chains into a single DataFrame.
    Similar to how PlotDriver gets all BS points for visualization.
    """
    # Get all BS points using the same logic as the visualization
    bs_points = chan.kl_datas[lv].bs_point_lst.getSortedBspList()
    
    all_rows = []
    
    for bsp in bs_points:
        # Base features that every BS point has
        row = {
            'klu_idx': bsp.klu.idx,
            'timestamp': str(bsp.klu.time),
            'klu_open': bsp.klu.open,
            'klu_high': bsp.klu.high,
            'klu_low': bsp.klu.low,
            'klu_close': bsp.klu.close,
            'klu_volume': getattr(bsp.klu, 'volume', None),
            'bsp_types': bsp.type2str(),  # e.g., "1,2s" if it has multiple types
            'is_buy': int(bsp.is_buy),
            'is_segbsp': int(getattr(bsp, 'is_segbsp', False)),
            'direction': 'buy' if bsp.is_buy else 'sell',
        }
        
        # Add all features from the features dictionary
        if getattr(bsp, 'features', None):
            feature_dict = bsp.features.to_dict()
            for key, value in feature_dict.items():
                # Prefix feature names to avoid confusion with base features
                row[f'feat_{key}'] = value
        
        # Add mature_rate and parent info if available (from chain tracking)
        row['mature_rate'] = getattr(bsp, 'mature_rate', None)
        row['is_mature'] = int(getattr(bsp, 'is_mature', False))
        row['parent_idx'] = getattr(bsp, 'parent_idx', None)
        
        all_rows.append(row)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_rows)
    
    # Sort by timestamp/index for chronological order
    if not df.empty:
        df = df.sort_values('klu_idx')
    
    return df


def enhance_with_chain_info(df: pd.DataFrame, chan: CChan) -> pd.DataFrame:
    """
    Enhance the DataFrame with chain information if available.
    This adds chain_id and position_in_chain columns.
    """
    if df.empty:
        return df

    if not hasattr(chan, "bs_chain_tracker") or not hasattr(chan.bs_chain_tracker, "chains_by_type"):
        df['chain_id'] = None
        df['position_in_chain'] = None
        df['chain_type'] = None
        return df
    
    chains_by_type = chan.bs_chain_tracker.chains_by_type
    
    # Create a mapping from klu_idx to chain info
    idx_to_chain = {}
    chain_counter = 0
    
    for chain_key, chain_list in chains_by_type.items():
        if not chain_list:
            continue
            
        # Handle nested lists if present
        if isinstance(chain_list[0], list):
            chains = chain_list
        else:
            chains = [chain_list]
            
        for chain in chains:
            for pos, bsp in enumerate(chain):
                klu_idx = bsp.klu.idx
                if klu_idx not in idx_to_chain:
                    idx_to_chain[klu_idx] = {
                        'chain_id': f"chain_{chain_counter}",
                        'position_in_chain': pos,
                        'chain_type': chain_key
                    }
            chain_counter += 1
    
    # Add chain info to DataFrame
    df['chain_id'] = df['klu_idx'].map(lambda x: idx_to_chain.get(x, {}).get('chain_id'))
    df['position_in_chain'] = df['klu_idx'].map(lambda x: idx_to_chain.get(x, {}).get('position_in_chain'))
    df['chain_type'] = df['klu_idx'].map(lambda x: idx_to_chain.get(x, {}).get('chain_type'))
    
    return df


def add_target_labels(df: pd.DataFrame, forward_periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """
    Add target labels for training based on future price movements.
    This creates binary labels (1 for profitable, 0 for not) and return percentages.
    """
    if df.empty:
        return df

    for period in forward_periods:
        # Calculate future returns
        df[f'return_{period}'] = df['klu_close'].shift(-period) / df['klu_close'] - 1
        
        # Create binary labels based on direction and return
        # For buy signals: 1 if positive return, 0 otherwise
        # For sell signals: 1 if negative return, 0 otherwise
        df[f'label_{period}'] = df.apply(
            lambda row: (
                1 if row['is_buy'] and row[f'return_{period}'] > 0
                else 1 if not row['is_buy'] and row[f'return_{period}'] < 0
                else 0
            ) if pd.notna(row[f'return_{period}']) else None,
            axis=1
        )
        
        # Add return magnitude (absolute value)
        df[f'return_abs_{period}'] = df[f'return_{period}'].abs()
    
    return df


# =========================
# 新增：映射器（long 形输出）
# =========================
class BSPFeatureMapper:
    def __init__(self):
        # 标准映射
        self.feature_mappings: Dict[str, Dict[str, Optional[str]]] = {
            '1':  {'bi_amp': 'feat_bsp1_bi_amp',  'bi_klu_cnt': 'feat_bsp1_bi_klu_cnt',  'bi_amp_rate': 'feat_bsp1_bi_amp_rate',  'retrace_rate': None},
            '1p': {'bi_amp': 'feat_bsp1_bi_amp',  'bi_klu_cnt': 'feat_bsp1_bi_klu_cnt',  'bi_amp_rate': 'feat_bsp1_bi_amp_rate',  'retrace_rate': None},
            '2':  {'bi_amp': 'feat_bsp2_bi_amp',  'bi_klu_cnt': 'feat_bsp2_bi_klu_cnt',  'bi_amp_rate': 'feat_bsp2_bi_amp_rate',  'retrace_rate': 'feat_bsp2_retrace_rate'},
            '2s': {'bi_amp': 'feat_bsp2s_bi_amp', 'bi_klu_cnt': 'feat_bsp2s_bi_klu_cnt', 'bi_amp_rate': 'feat_bsp2s_bi_amp_rate', 'retrace_rate': 'feat_bsp2s_retrace_rate'},
            '3a': {'bi_amp': 'feat_bsp3_bi_amp',  'bi_klu_cnt': 'feat_bsp3_bi_klu_cnt',  'bi_amp_rate': 'feat_bsp3_bi_amp_rate',  'retrace_rate': None},
            '3b': {'bi_amp': 'feat_bsp3_bi_amp',  'bi_klu_cnt': 'feat_bsp3_bi_klu_cnt',  'bi_amp_rate': 'feat_bsp3_bi_amp_rate',  'retrace_rate': None},
        }

        # 类型特有字段
        self.additional_features: Dict[str, List[str]] = {
            '2':  ['feat_bsp2_break_bi_amp',  'feat_bsp2_break_bi_bi_klu_cnt', 'feat_bsp2_break_bi_amp_rate'],
            '2s': ['feat_bsp2s_break_bi_amp', 'feat_bsp2s_break_bi_klu_cnt',  'feat_bsp2s_break_bi_amp_rate', 'feat_bsp2s_lv'],
            '3a': ['feat_bsp3_zs_height'],
            '3b': ['feat_bsp3_zs_height'],
        }

    def _safe_get(self, row: pd.Series, col: Optional[str]):
        if col is None:
            return None
        v = row.get(col, None)
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    def _base_cols(self) -> List[str]:
        # 与上游导出的基础列保持一致，便于训练集携带上下文
        return [
            'klu_idx','timestamp','klu_open','klu_high','klu_low','klu_close','klu_volume',
            'is_buy','is_segbsp','direction','mature_rate','is_mature','parent_idx',
            'chain_id','position_in_chain','chain_type'
        ]

    def to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输出 long 形表：每个类型一行。统一字段：
        bsp_type, bi_amp, bi_klu_cnt, bi_amp_rate, retrace_rate (+ 类型特有字段)
        同时透传 label_* / return_* / return_abs_* 列。
        """
        if df.empty:
            return df

        rows = []
        passthrough_cols = [c for c in df.columns if c.startswith('label_') or c.startswith('return_')]

        for _, row in df.iterrows():
            types_str = row.get('bsp_types')
            if not isinstance(types_str, str) or not types_str.strip():
                continue
            types = [t.strip() for t in types_str.split(',') if t.strip() in self.feature_mappings]

            for t in types:
                mapped = self.feature_mappings[t]
                out = {bc: row.get(bc, None) for bc in self._base_cols()}
                out['bsp_type']      = t
                out['bi_amp']        = self._safe_get(row, mapped['bi_amp'])
                out['bi_klu_cnt']    = self._safe_get(row, mapped['bi_klu_cnt'])
                out['bi_amp_rate']   = self._safe_get(row, mapped['bi_amp_rate'])
                out['retrace_rate']  = self._safe_get(row, mapped['retrace_rate'])
                # 类型特有字段
                for f in self.additional_features.get(t, []):
                    out[f] = self._safe_get(row, f)
                # 透传标签与收益
                for c in passthrough_cols:
                    out[c] = row.get(c, None)
                # 便捷列（与你之前保持一致的命名）
                out['bi_amplitude'] = out['bi_amp']
                rows.append(out)

        return pd.DataFrame(rows)


def export_consolidated_bs_features(
    chan: CChan, 
    lv: KL_TYPE, 
    output_path: str,
    add_targets: bool = True,
    forward_periods: List[int] = [1, 5, 10]
):
    """
    直接导出【long 形】的新特征表（不再保留原 df，也不另存原始 CSV）。
    每行代表一个 BS 点的某个类型（如 2, 2s, 3a...），
    统一字段：bsp_type, bi_amp, bi_klu_cnt, bi_amp_rate, retrace_rate，
    并保留各类型特有字段与标签列。
    """
    print(f"[📊] Extracting all BS points from level {lv}...")
    
    # Extract all BS points
    base_df = extract_all_bs_points_to_single_file(chan, lv)
    if base_df.empty:
        print("[⚠️] No BS points found!")
        return None
    
    print(f"[✅] Found {len(base_df)} BS points")
    
    # Enhance with chain information
    base_df = enhance_with_chain_info(base_df, chan)
    
    # Add target labels if requested
    if add_targets:
        base_df = add_target_labels(base_df, forward_periods)
        print(f"[✅] Added target labels for periods: {forward_periods}")
    
    # === 关键变更：映射为 long 形，并直接导出 ===
    mapper = BSPFeatureMapper()
    out_df = mapper.to_long(base_df)

    if out_df.empty:
        print("[⚠️] No rows after type mapping (check bsp_types).")
        return None
    
    # Save to CSV (仅保存新的 long 形)
    out_df.to_csv(output_path, index=False)
    print(f"[💾] Saved NEW long-form features to: {output_path}")
    
    # 简要统计（long 形）
    print("\n[📈] Summary (LONG-form):")
    print(f"  Total rows (point-type pairs): {len(out_df)}")
    uniq_points = out_df['klu_idx'].nunique() if 'klu_idx' in out_df.columns else None
    if uniq_points is not None:
        print(f"  Unique BS points covered: {uniq_points}")
    if 'bsp_type' in out_df.columns:
        type_counts = out_df['bsp_type'].value_counts().sort_index()
        print("  Rows by bsp_type:")
        for t, cnt in type_counts.items():
            print(f"    {t}: {cnt}")
    
    return out_df


# Additional utility function for feature engineering
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional technical indicators as features.
    """
    if df.empty:
        return df

    # Price-based features
    df['price_change'] = df['klu_close'] - df['klu_open']
    df['price_change_pct'] = df['price_change'] / df['klu_open']
    df['high_low_spread'] = df['klu_high'] - df['klu_low']
    df['high_low_spread_pct'] = df['high_low_spread'] / df['klu_low']
    
    # Volume features (if available)
    if 'klu_volume' in df.columns:
        df['volume_ma5'] = df['klu_volume'].rolling(5).mean()
        df['volume_ratio'] = df['klu_volume'] / df['volume_ma5']
    
    # MACD features (if available)
    if 'feat_macd_value' in df.columns:
        df['macd_signal'] = df['feat_macd_value'] > 0
        df['macd_cross'] = df['macd_signal'].diff()
    
    # RSI features (if available)
    if 'feat_rsi' in df.columns:
        df['rsi_oversold'] = df['feat_rsi'] < 30
        df['rsi_overbought'] = df['feat_rsi'] > 70
    
    return df
