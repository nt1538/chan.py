import os
import csv
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


def extract_all_chain_features(chains_by_type: dict) -> list:
    """
    遍历所有类型的链，提取出每一条链的特征，返回统一列表。
    每个返回元素为 dict，包含所属类型、方向、链编号等信息。
    """
    all_rows = []

    for raw_key, chain_list in chains_by_type.items():
        try:
            if "]_" in raw_key:
                suffix = raw_key.split("]_")[-1]
                type_part = raw_key.split("[<")[-1].split(":")[1].strip().strip("'\"> ")
                bs_type_raw, direction_raw = type_part, suffix
            else:
                bs_type_raw, direction_raw = raw_key.split("_")
        except Exception as e:
            print(f"[⚠] Cannot parse raw_key: {raw_key}, skipping. Error: {e}")
            continue
        # 清洗后构造标准 key
        key = sanitize_row_key(bs_type_raw, direction_raw)

        if "_" not in key:
            print(f"[⚠] Invalid key format after sanitize: {key}")
            continue

        bs_type, direction = key.split("_")

        for chain_id, chain in enumerate(chain_list):
            if not chain or not hasattr(chain[0], "features"):
                print(f"[⚠] Skipping chain with no BSPoint: {key} #{chain_id}")
                continue

            for i, bsp in enumerate(chain):
                if bsp.features is None:
                    continue

                row = {
                    'chain_type': bs_type,
                    'direction': direction,
                    'chain_id': chain_id,
                    'point_index': i,
                    'klu_idx': bsp.klu.idx,
                    'timestamp': bsp.klu.time,
                    'bsp_type': bsp.type2str(),
                    'is_buy': bsp.is_buy,
                    'is_segbsp': getattr(bsp, 'is_segbsp', False),
                    'mature_rate': getattr(bsp, 'mature_rate', None),
                    'is_mature': int(getattr(bsp, 'is_mature_point', False)),
                    'parent_idx': getattr(bsp, 'parent_idx', None),
                }

                if bsp.features:
                    row.update(bsp.features.to_dict())

                all_rows.append(row)

    print(f"[✓] Extracted {len(all_rows)} BS point features from all chains.")
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
        fieldnames = list(group_rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_rows)
        print(f"[✓] Exported {len(group_rows)} rows to {path}")
        total_files += 1

    if total_files == 0:
        print("[⚠] No data exported. Check if input rows are empty.")

