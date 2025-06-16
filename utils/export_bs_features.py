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

        # Add all features from the CFeatures object
        if bsp.features:
            extra_feat = bsp.features.to_dict()
            all_feat = {**base_feat, **extra_feat}
        else:
            all_feat = base_feat

        # group by each bs_type this point has
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
