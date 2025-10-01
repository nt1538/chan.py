import os
import csv
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from Common.CEnum import KL_TYPE
from Chan import CChan


def extract_features_from_cbs_points(chan: CChan, lv: KL_TYPE) -> Dict[str, List[dict]]:
    """
    Extract features from all CBS_Points in the given level of the chan system.
    Returns a dictionary of feature lists, grouped by BS type (e.g., "1", "2", "2s").
    """
    bs_points = chan.kl_datas[lv].bs_point_lst.getSortedBspList()
    grouped_features = defaultdict(list)

    for bsp in bs_points:
        base_feat = {
            'index': bsp.klu.idx,
            'type': bsp.type2str(),
            'is_buy': bsp.is_buy,
            'divergence_rate': getattr(bsp, 'divergence_rate', None),
            'is_segbsp': getattr(bsp, 'is_segbsp', False),
            'timestamp': bsp.klu.time,
        }

        if bsp.features:
            extra_feat = bsp.features.to_dict()
            all_feat = {**base_feat, **extra_feat}
        else:
            all_feat = base_feat

        for t in bsp.type:
            grouped_features[t.value].append(all_feat)

    return grouped_features


def export_bs_feature_files_by_type(chan: CChan, lv: KL_TYPE, output_dir: str):
    """
    Export BS point features to multiple CSV files, grouped by BS type.
    Each file will be named like: `bs_features_type_<type>.csv`.
    """
    os.makedirs(output_dir, exist_ok=True)

    grouped = extract_features_from_cbs_points(chan, lv)

    if not grouped:
        print("No BS points found.")
        return

    for bs_type, features in grouped.items():
        if not features:
            continue

        all_fieldnames = set()
        for feat in features:
            all_fieldnames.update(feat.keys())
        fieldnames = sorted(all_fieldnames)

        filename = os.path.join(output_dir, f"bs_features_type_{bs_type}.csv")
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features)

        print(f"[✓] Exported {len(features)} '{bs_type}' BS points to: {filename}")


def sanitize_row_key(chain_type: str, direction: str) -> str:
    """
    清洗 chain_type 和 direction，确保没有非法字符。
    比如 '1p'>' -> '1p'
    """
    def clean(s):
        return str(s).strip(" '>\n\t\"")  # 去除空格、引号、尖括号等

    return f"{clean(chain_type)}_{clean(direction)}"

def extract_bsp_feature_row(bsp, bs_type, direction, chain_id, point_index):
    """单独抽出一个 BSP 点的特征提取逻辑"""
    row = {
        'chain_type': bs_type,
        'direction': direction,
        'chain_id': chain_id,
        'point_index': point_index,
        'klu_idx': bsp.klu.idx,
        'timestamp': bsp.klu.time,
        'klu_close': bsp.klu.close,
        'bsp_type': bsp.type2str(),
        'is_buy': bsp.is_buy,
        'is_segbsp': getattr(bsp, 'is_segbsp', False),
        'mature_rate': None,
        'is_mature': 0,
        'parent_idx': None,
    }

    if bsp.features:
        row.update(bsp.features.to_dict())

    return row

def extract_all_chain_features(chains_by_type: dict) -> list:
    all_rows = []
    chain_counter = 0

    type_to_amp_field = {
        '1': 'bsp1_bi_amp',
        '1p': 'bsp1_bi_amp',
        '2': 'bsp2_bi_amp',
        '2s': 'bsp2s_bi_amp',
        '3a': 'bsp3_bi_amp',
        '3b': 'bsp3_bi_amp',
    }

    for raw_key, bsp_list in chains_by_type.items():
        if not bsp_list:
            continue

        # Flatten if nested
        if isinstance(bsp_list[0], list):
            bsp_list = [bsp for sublist in bsp_list for bsp in sublist]

        try:
            direction_raw = raw_key.split("_")[-1].strip()
            bs_type_raw = raw_key.split(":")[-1].split(">")[0].strip(" '")
            key = sanitize_row_key(bs_type_raw, direction_raw)
        except Exception as e:
            print(f"[⚠] Cannot parse raw_key: {raw_key}, skipping. Error: {e}")
            continue

        if "_" not in key:
            continue

        bs_type, direction = key.split("_")

        # 按时间戳排序，避免乱序
        chain = sorted(bsp_list, key=lambda b: getattr(b.klu, "ts", float("inf")))

        # 找所有成熟点索引
        mature_indices = []
        for i, bsp in enumerate(chain):
            features = bsp.features.to_dict() if bsp.features else {}
            if pd.notna(features.get("next_bi_return")):
                mature_indices.append(i)

        # 插入最后一个哨兵点（方便统一逻辑）
        segment_ends = mature_indices + [len(chain)]

        used_indices = set()

        for seg_idx in range(len(mature_indices)):
            start = 0 if seg_idx == 0 else mature_indices[seg_idx - 1] + 1
            end = segment_ends[seg_idx]

            mature_index = mature_indices[seg_idx]
            mature_bsp = chain[mature_index]
            mature_features = mature_bsp.features.to_dict() if mature_bsp.features else {}
            bsp_type = mature_bsp.type2str()
            amp_field = type_to_amp_field.get(bsp_type)

            if not amp_field:
                print(f"[⚠] Unknown bsp_type '{bsp_type}', skipping chain {key}")
                continue

            mature_amp = mature_features.get(amp_field)
            mature_klu_idx = getattr(mature_bsp.klu, "idx", None)

            if not mature_amp or mature_amp == 0:
                continue

            # 为 start ~ mature_index 的点加标注
            for i in range(start, mature_index + 1):
                bsp = chain[i]
                features = bsp.features.to_dict() if bsp.features else {}
                row = extract_bsp_feature_row(
                    bsp, bs_type, direction, chain_id=chain_counter, point_index=i
                )

                if i < mature_index:
                    current_amp = features.get(amp_field)
                    if current_amp is not None:
                        row['mature_rate'] = round(min(current_amp / mature_amp, 1), 6)
                    row['parent_idx'] = mature_klu_idx
                    row['is_mature'] = 0
                elif i == mature_index:
                    row['mature_rate'] = 1
                    row['is_mature'] = 1
                    row['parent_idx'] = None

                used_indices.add(i)
                all_rows.append(row)

            chain_counter += 1

        # 剩下的末尾未归属区域
        for i, bsp in enumerate(chain):
            if i in used_indices:
                continue
            row = extract_bsp_feature_row(
                bsp, bs_type, direction, chain_id=None, point_index=i
            )
            row['mature_rate'] = None
            row['is_mature'] = 0
            row['parent_idx'] = None
            all_rows.append(row)

    print(f"[✓] Extracted {len(all_rows)} BS point features from all reconstructed chains.")
    return all_rows



def export_chain_features_by_type(chan: CChan, lv: KL_TYPE, output_dir: str):
    """
    导出 chan.bs_chain_tracker 中所有链条的特征。
    每条链一份 CSV 文件，文件名包含链类型、方向、编号。
    """
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(chan, "bs_chain_tracker"):
        print("[❌] No bs_chain_tracker found on chan.")
        return

    chains_by_type = chan.bs_chain_tracker.chains_by_type
    rows = extract_all_chain_features(chains_by_type)
    #rows = postprocess_maturity_features(rows)

    export_chains_from_flat_rows(rows, output_dir)

def export_chains_from_flat_rows(rows: list, output_dir: str):
    """
    从扁平化的 rows 中，根据 chain_type 和 direction 分类导出 CSV。
    每种类型（如 1p_buy）导出一个文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)

    for row in rows:
        key = f"{row['chain_type']}_{row['direction']}"
        grouped[key].append(row)

    total_files = 0
    for key, group_rows in grouped.items():
        if not group_rows:
            continue
        path = os.path.join(output_dir, f"bs_chain_group_{key}.csv")
        all_fields = set()
        for row in group_rows:
            all_fields.update(row.keys())
        fieldnames = sorted(all_fields)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_rows)
        print(f"[✓] Exported {len(group_rows)} rows to {path}")
        total_files += 1

    if total_files == 0:
        print("[⚠] No data exported. Check if input rows are empty.")


# Updated optimized feature extractor to use built-in features
def export_bs_features_with_builtin_indicators(
    chan: CChan, 
    lv: KL_TYPE, 
    output_path: str,
    add_targets: bool = True,
    forward_periods: list = [1, 5, 10]
) -> pd.DataFrame:
    """
    Enhanced BS feature extraction using built-in technical indicators with profit targets
    """
    print(f"[📊] Extracting BS points with built-in indicators from level {lv}...")
    
    bs_points = chan.kl_datas[lv].bs_point_lst.getSortedBspList()
    
    if not bs_points:
        print("[⚠️] No BS points found!")
        return pd.DataFrame()
    
    print(f"[✅] Found {len(bs_points)} BS points")
    
    # Calculate profit taking targets
    profit_targets = calculate_profit_taking_targets(bs_points)
    same_type_targets = calculate_same_type_profit_targets(bs_points)
    
    all_rows = []
    
    for bsp in bs_points:
        # Extract base features
        base_features = bsp.features.to_dict() if bsp.features else {}
        klu = bsp.klu
        
        # Get profit target info
        profit_info = profit_targets.get(klu.idx, {})
        same_type_info = same_type_targets.get(klu.idx, {})
        
        # Process each type in the BSP
        for bs_type in bsp.type:
            bs_type_str = bs_type.value
            
            # Create consolidated feature dict
            row = {
                'klu_idx': klu.idx,
                'timestamp': str(klu.time),
                'klu_open': klu.open,
                'klu_high': klu.high,
                'klu_low': klu.low,
                'klu_close': klu.close,
                'klu_volume': getattr(klu, 'volume', None),
                'bsp_type': bs_type_str,
                'bsp_types': bsp.type2str(),
                'is_buy': int(bsp.is_buy),
                'direction': 'buy' if bsp.is_buy else 'sell',
                
                # PROFIT TAKING TARGETS (Reverse BSP)
                'profit_target_price': profit_info.get('profit_target_price'),
                'profit_target_pct': profit_info.get('profit_target_pct'),
                'profit_target_klu_idx': profit_info.get('profit_target_klu_idx'),
                'profit_target_type': profit_info.get('profit_target_type'),
                'profit_target_distance': profit_info.get('profit_target_distance'),
                
                # SAME TYPE CONTINUATION TARGETS
                'same_type_target_price': same_type_info.get('same_type_target_price'),
                'same_type_target_pct': same_type_info.get('same_type_target_pct'),
                'same_type_target_klu_idx': same_type_info.get('same_type_target_klu_idx'),
                'same_type_target_distance': same_type_info.get('same_type_target_distance'),
                
                # PROFIT TARGET ANALYSIS
                'has_profit_target': 1 if profit_info.get('profit_target_pct') is not None else 0,
                'profit_target_positive': 1 if (profit_info.get('profit_target_pct') or 0) > 0 else 0,
                'profit_target_abs': abs(profit_info.get('profit_target_pct') or 0),
            }
            
            # Add profit target categories
            if profit_info.get('profit_target_pct') is not None:
                pct = profit_info['profit_target_pct']
                row['profit_target_small'] = 1 if 0 < abs(pct) <= 2 else 0
                row['profit_target_medium'] = 1 if 2 < abs(pct) <= 5 else 0
                row['profit_target_large'] = 1 if abs(pct) > 5 else 0
                row['profit_target_loss'] = 1 if pct < 0 else 0
            else:
                row['profit_target_small'] = 0
                row['profit_target_medium'] = 0
                row['profit_target_large'] = 0
                row['profit_target_loss'] = 0
            
            # Add all original BS features
            for key, value in base_features.items():
                row[f'feat_{key}'] = value
            
            # BUILT-IN TECHNICAL INDICATORS
            
            # MACD (existing)
            if hasattr(klu, 'macd') and klu.macd:
                row['macd_value'] = klu.macd.macd
                row['macd_dif'] = klu.macd.DIF
                row['macd_dea'] = klu.macd.DEA
                row['macd_signal'] = 1 if klu.macd.macd > 0 else 0
            
            # RSI (existing)
            if hasattr(klu, 'rsi'):
                row['rsi'] = klu.rsi
                row['rsi_oversold'] = 1 if klu.rsi < 30 else 0
                row['rsi_overbought'] = 1 if klu.rsi > 70 else 0
            
            # KDJ (existing)
            if hasattr(klu, 'kdj') and klu.kdj:
                row['kdj_k'] = klu.kdj.k
                row['kdj_d'] = klu.kdj.d
                row['kdj_j'] = klu.kdj.j
                row['kdj_oversold'] = 1 if klu.kdj.k < 20 else 0
                row['kdj_overbought'] = 1 if klu.kdj.k > 80 else 0
            
            # DMI (existing)
            if hasattr(klu, 'dmi') and klu.dmi:
                row['dmi_plus'] = klu.dmi.plus_di
                row['dmi_minus'] = klu.dmi.minus_di
                row['dmi_adx'] = klu.dmi.adx
                row['dmi_trend_up'] = 1 if klu.dmi.plus_di > klu.dmi.minus_di else 0
            
            # NEW INDICATORS
            
            # Simple Moving Averages
            if hasattr(klu, 'sma'):
                for period, value in klu.sma.items():
                    row[f'sma_{period}'] = value
                    row[f'price_above_sma_{period}'] = 1 if klu.close > value else 0
            
            # Exponential Moving Averages
            if hasattr(klu, 'ema'):
                for period, value in klu.ema.items():
                    row[f'ema_{period}'] = value
                    row[f'price_above_ema_{period}'] = 1 if klu.close > value else 0
            
            # Average True Range
            if hasattr(klu, 'atr'):
                row['atr'] = klu.atr
                row['atr_ratio'] = (klu.high - klu.low) / klu.atr if klu.atr > 0 else 0
            
            # Stochastic
            if hasattr(klu, 'stochastic') and klu.stochastic:
                row['stoch_k'] = klu.stochastic['k']
                row['stoch_d'] = klu.stochastic['d']
                row['stoch_oversold'] = 1 if klu.stochastic['k'] < 20 else 0
                row['stoch_overbought'] = 1 if klu.stochastic['k'] > 80 else 0
            
            # Rate of Change
            if hasattr(klu, 'roc'):
                for period, value in klu.roc.items():
                    row[f'roc_{period}'] = value
                    row[f'roc_{period}_positive'] = 1 if value > 0 else 0
            
            # Williams %R
            if hasattr(klu, 'williams_r'):
                row['williams_r'] = klu.williams_r
                row['williams_oversold'] = 1 if klu.williams_r < -80 else 0
                row['williams_overbought'] = 1 if klu.williams_r > -20 else 0
            
            # Commodity Channel Index
            if hasattr(klu, 'cci'):
                row['cci'] = klu.cci
                row['cci_oversold'] = 1 if klu.cci < -100 else 0
                row['cci_overbought'] = 1 if klu.cci > 100 else 0
            
            # Money Flow Index
            if hasattr(klu, 'mfi'):
                row['mfi'] = klu.mfi
                row['mfi_oversold'] = 1 if klu.mfi < 20 else 0
                row['mfi_overbought'] = 1 if klu.mfi > 80 else 0
            
            # True Strength Index
            if hasattr(klu, 'tsi'):
                row['tsi'] = klu.tsi
                row['tsi_positive'] = 1 if klu.tsi > 0 else 0
            
            # Ultimate Oscillator
            if hasattr(klu, 'uo'):
                row['uo'] = klu.uo
                row['uo_oversold'] = 1 if klu.uo < 30 else 0
                row['uo_overbought'] = 1 if klu.uo > 70 else 0
            
            # Parabolic SAR
            if hasattr(klu, 'psar'):
                row['psar'] = klu.psar
                row['price_above_psar'] = 1 if klu.close > klu.psar else 0
            
            # PATTERN RECOGNITION
            
            # Candlestick Patterns
            if hasattr(klu, 'candlestick_patterns'):
                for pattern_name, detected in klu.candlestick_patterns.items():
                    row[f'candle_{pattern_name}'] = 1 if detected else 0
            
            # Price Patterns
            if hasattr(klu, 'price_patterns'):
                for pattern_name, detected in klu.price_patterns.items():
                    row[f'price_{pattern_name}'] = 1 if detected else 0
            
            # Volume Patterns
            if hasattr(klu, 'volume_patterns'):
                for pattern_name, detected in klu.volume_patterns.items():
                    row[f'volume_{pattern_name}'] = 1 if detected else 0
            
            all_rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows)
    
    if df.empty:
        print("[⚠️] No data to process")
        return df
    
    # Sort by timestamp/index
    df = df.sort_values('klu_idx').reset_index(drop=True)
    
    # Add target labels if requested
    if add_targets:
        print(f"[✅] Adding target labels for periods: {forward_periods}")
        
        for period in forward_periods:
            df[f'return_{period}'] = df['klu_close'].shift(-period) / df['klu_close'] - 1
            df[f'label_{period}'] = df.apply(
                lambda row: (
                    1 if row['is_buy'] and row[f'return_{period}'] > 0
                    else 1 if not row['is_buy'] and row[f'return_{period}'] < 0
                    else 0
                ) if pd.notna(row[f'return_{period}']) else None,
                axis=1
            )
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"[💾] Saved enhanced features to: {output_path}")
    
    # Print profit target statistics
    if len(df) > 0:
        print(f"\n[📈] Profit Target Analysis:")
        profit_available = df['has_profit_target'].sum()
        profit_positive = df['profit_target_positive'].sum()
        
        print(f"  BSP points with profit targets: {profit_available}/{len(df)} ({profit_available/len(df)*100:.1f}%)")
        
        if profit_available > 0:
            print(f"  Profitable targets: {profit_positive}/{profit_available} ({profit_positive/profit_available*100:.1f}%)")
            
            # Profit distribution
            small_profits = df['profit_target_small'].sum()
            medium_profits = df['profit_target_medium'].sum()
            large_profits = df['profit_target_large'].sum()
            losses = df['profit_target_loss'].sum()
            
            print(f"  Small profits (0-2%): {small_profits}")
            print(f"  Medium profits (2-5%): {medium_profits}")
            print(f"  Large profits (>5%): {large_profits}")
            print(f"  Losses (<0%): {losses}")
            
            # Average statistics
            if 'profit_target_pct' in df.columns:
                valid_profits = df['profit_target_pct'].dropna()
                if len(valid_profits) > 0:
                    print(f"  Average profit/loss: {valid_profits.mean():.2f}%")
                    print(f"  Median profit/loss: {valid_profits.median():.2f}%")
                    print(f"  Max profit: {valid_profits.max():.2f}%")
                    print(f"  Max loss: {valid_profits.min():.2f}%")
                    print(f"  Standard deviation: {valid_profits.std():.2f}%")
                    
                    # Distance statistics
                    valid_distances = df['profit_target_distance'].dropna()
                    if len(valid_distances) > 0:
                        print(f"  Average target distance: {valid_distances.mean():.1f} periods")
                        print(f"  Median target distance: {valid_distances.median():.1f} periods")
                        print(f"  Min distance: {valid_distances.min():.0f} periods")
                        print(f"  Max distance: {valid_distances.max():.0f} periods")
                    
                    # Profit by BSP type
                    print(f"\n  📊 Profit by BSP Type:")
                    for bsp_type in sorted(df['bsp_type'].unique()):
                        type_data = df[df['bsp_type'] == bsp_type]
                        type_profits = type_data['profit_target_pct'].dropna()
                        if len(type_profits) > 0:
                            win_rate = (type_profits > 0).mean() * 100
                            avg_profit = type_profits.mean()
                            print(f"    {bsp_type}: {len(type_profits)} samples, "
                                  f"{win_rate:.1f}% win rate, {avg_profit:.2f}% avg")
                    
                    # Profit by direction
                    print(f"\n  💹 Profit by Direction:")
                    for direction in ['buy', 'sell']:
                        dir_data = df[df['direction'] == direction]
                        dir_profits = dir_data['profit_target_pct'].dropna()
                        if len(dir_profits) > 0:
                            win_rate = (dir_profits > 0).mean() * 100
                            avg_profit = dir_profits.mean()
                            print(f"    {direction.capitalize()}: {len(dir_profits)} samples, "
                                  f"{win_rate:.1f}% win rate, {avg_profit:.2f}% avg")
    
    return df


def add_risk_reward_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk-reward and trading performance features"""
    if df.empty or 'profit_target_pct' not in df.columns:
        return df
    
    print("[⚖️] Adding risk-reward analysis features...")
    
    # Risk-Reward Ratios
    df['profit_target_risk_reward'] = df['profit_target_abs'] / 2.0  # Assuming 2% stop loss
    
    # Performance Categories
    df['excellent_trade'] = ((df['profit_target_pct'] > 5) & (df['profit_target_distance'] <= 20)).astype(int)
    df['good_trade'] = ((df['profit_target_pct'] > 2) & (df['profit_target_pct'] <= 5)).astype(int)
    df['break_even_trade'] = ((df['profit_target_pct'] >= -0.5) & (df['profit_target_pct'] <= 2)).astype(int)
    df['poor_trade'] = (df['profit_target_pct'] < -2).astype(int)
    
    # Speed to Target
    df['quick_target'] = (df['profit_target_distance'] <= 5).astype(int)
    df['medium_target'] = ((df['profit_target_distance'] > 5) & (df['profit_target_distance'] <= 20)).astype(int)
    df['slow_target'] = (df['profit_target_distance'] > 20).astype(int)
    
    # Target Efficiency (Profit per period)
    df['target_efficiency'] = df.apply(
        lambda row: row['profit_target_abs'] / row['profit_target_distance'] 
        if pd.notna(row['profit_target_distance']) and row['profit_target_distance'] > 0 
        else 0, axis=1
    )
    
    # High efficiency trades
    df['high_efficiency'] = (df['target_efficiency'] > 0.5).astype(int)
    
    return df


def calculate_advanced_profit_metrics(bs_points):
    """Calculate additional profit-related metrics"""
    print("[🔬] Calculating advanced profit metrics...")
    
    advanced_metrics = {}
    
    for i, bsp in enumerate(bs_points):
        klu_idx = bsp.klu.idx
        advanced_metrics[klu_idx] = {
            'max_favorable_move': None,
            'max_adverse_move': None,
            'drawdown_to_target': None,
            'peak_to_target_ratio': None
        }
        
        current_direction = bsp.is_buy
        entry_price = bsp.klu.close
        
        # Find the next reverse BSP first
        target_idx = None
        target_price = None
        
        for j in range(i + 1, len(bs_points)):
            next_bsp = bs_points[j]
            if next_bsp.is_buy != current_direction:
                target_idx = j
                target_price = next_bsp.klu.close
                break
        
        if target_idx is None:
            continue
        
        # Analyze price movement between entry and target
        max_favorable = entry_price
        max_adverse = entry_price
        
        # Look at all BSPs between entry and target
        for k in range(i + 1, target_idx):
            intermediate_bsp = bs_points[k]
            intermediate_price = intermediate_bsp.klu.close
            
            if current_direction:  # Buy position
                max_favorable = max(max_favorable, intermediate_price)
                max_adverse = min(max_adverse, intermediate_price)
            else:  # Sell position
                max_favorable = min(max_favorable, intermediate_price)
                max_adverse = max(max_adverse, intermediate_price)
        
        # Calculate metrics
        if current_direction:  # Buy position
            favorable_move = (max_favorable - entry_price) / entry_price * 100
            adverse_move = (entry_price - max_adverse) / entry_price * 100
            final_profit = (target_price - entry_price) / entry_price * 100
        else:  # Sell position
            favorable_move = (entry_price - max_favorable) / entry_price * 100
            adverse_move = (max_adverse - entry_price) / entry_price * 100
            final_profit = (entry_price - target_price) / entry_price * 100
        
        # Store advanced metrics
        advanced_metrics[klu_idx] = {
            'max_favorable_move': favorable_move,
            'max_adverse_move': adverse_move,
            'drawdown_to_target': adverse_move,
            'peak_to_target_ratio': favorable_move / abs(final_profit) if final_profit != 0 else 0
        }
    
    return advanced_metrics


# Updated main function to use new features
def export_bs_features_with_profit_analysis(
    chan: CChan, 
    lv: KL_TYPE, 
    output_path: str,
    add_targets: bool = True,
    forward_periods: list = [1, 5, 10],
    include_advanced_metrics: bool = True
) -> pd.DataFrame:
    """
    Complete BS feature extraction with comprehensive profit analysis
    """
    print(f"[📊] Extracting BS points with profit analysis from level {lv}...")
    
    # First extract basic features with profit targets
    df = export_bs_features_with_builtin_indicators(
        chan, lv, output_path, add_targets, forward_periods
    )
    
    if df.empty:
        return df
    
    # Add risk-reward features
    df = add_risk_reward_features(df)
    
    # Add advanced metrics if requested
    if include_advanced_metrics:
        bs_points = chan.kl_datas[lv].bs_point_lst.getSortedBspList()
        advanced_metrics = calculate_advanced_profit_metrics(bs_points)
        
        # Merge advanced metrics
        for idx, row in df.iterrows():
            klu_idx = row['klu_idx']
            if klu_idx in advanced_metrics:
                metrics = advanced_metrics[klu_idx]
                df.at[idx, 'max_favorable_move'] = metrics['max_favorable_move']
                df.at[idx, 'max_adverse_move'] = metrics['max_adverse_move']
                df.at[idx, 'drawdown_to_target'] = metrics['drawdown_to_target']
                df.at[idx, 'peak_to_target_ratio'] = metrics['peak_to_target_ratio']
    
    # Save enhanced version
    enhanced_output = output_path.replace('.csv', '_enhanced.csv')
    df.to_csv(enhanced_output, index=False)
    print(f"[💾] Saved enhanced profit analysis to: {enhanced_output}")
    
    # Additional statistics
    if include_advanced_metrics and 'max_favorable_move' in df.columns:
        print(f"\n[🔬] Advanced Profit Metrics:")
        favorable = df['max_favorable_move'].dropna()
        adverse = df['max_adverse_move'].dropna()
        
        if len(favorable) > 0 and len(adverse) > 0:
            print(f"  Average max favorable move: {favorable.mean():.2f}%")
            print(f"  Average max adverse move: {adverse.mean():.2f}%")
            print(f"  Favorable/Adverse ratio: {favorable.mean()/adverse.mean():.2f}")
            
            # Efficiency metrics
            efficiency = df['target_efficiency'].dropna()
            if len(efficiency) > 0:
                print(f"  Average target efficiency: {efficiency.mean():.3f}% per period")
                print(f"  High efficiency trades: {df['high_efficiency'].sum()}")
    
    return df