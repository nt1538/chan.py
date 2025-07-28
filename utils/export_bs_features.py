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

# def postprocess_maturity_features(rows: list) -> list:
#     """
#     根据每种 BS 点类型的专属幅度字段（如 bsp2_bi_amp）来计算 mature_rate。
#     判断条件是 next_bi_return 非空的点即为成熟点。
#     """
#     from collections import defaultdict
#     import pandas as pd

#     # 不同类型使用不同的幅度字段
#     type_to_amp_field = {
#         '1': 'bsp1_bi_amp',
#         '1p': 'bsp1_bi_amp',
#         '2': 'bsp2_bi_amp',
#         '2s': 'bsp2s_bi_amp',
#         '3a': 'bsp3_bi_amp',
#         '3b': 'bsp3_bi_amp',
#     }

#     grouped = defaultdict(list)
#     for row in rows:
#         key = (row['chain_type'], row['direction'], row['chain_id'])
#         grouped[key].append(row)


#     for key, chain_rows in grouped.items():
#         chain_rows.sort(key=lambda x: x['klu_idx'])

#         # 找第一个成熟点
#         mature_index = None
#         for i, row in enumerate(chain_rows):
#             if pd.notna(row.get('next_bi_return')):
#                 mature_index = i
#                 break

#         if mature_index is None:
#             continue

#         mature_row = chain_rows[mature_index]
#         bsp_type = mature_row.get('bsp_type')

#         amp_field = type_to_amp_field.get(bsp_type)
#         if not amp_field:
#             print(f"[⚠] Unknown bsp_type '{bsp_type}', skipping chain {key}")
#             continue

#         mature_amp = mature_row.get(amp_field)
#         #print(mature_amp)
#         if mature_amp is None or mature_amp == 0:
#             continue
#         print(f"\n[🔍] Chain {key} — checking next_bi_return:")
#         for i, row in enumerate(chain_rows):
#             print(f"  Index {i}, klu_idx={row['klu_idx']}, next_bi_return={row.get('next_bi_return')}")
#             current_amp = row.get(amp_field)
#             #print(current_amp)
#             if i < mature_index:
#                 if current_amp is not None:
#                     row['mature_rate'] = round(min(current_amp / mature_amp, 1), 6)
#                 else:
#                     row['mature_rate'] = None
#                 #print(row['mature_rate'])
#                 row['parent_idx'] = mature_row['klu_idx']
#                 row['is_mature'] = 0

#             elif i == mature_index:
#                 row['mature_rate'] = 10
#                 row['is_mature'] = 10
#                 row['parent_idx'] = None

#             else:
#                 row['mature_rate'] = None
#                 row['parent_idx'] = None
#                 row['is_mature'] = 0

#     print("[✓] Post-processed mature_rate using type-specific amplitude fields.")
#     return rows
