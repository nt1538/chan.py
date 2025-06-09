import csv
from typing import List
from Common.CEnum import KL_TYPE
from Chan import CChan


def extract_features_from_cbs_points(chan: CChan, lv: KL_TYPE) -> List[dict]:
    """
    Extract features from all CBS_Points in the given level of the chan system.
    """
    bs_points = chan.kl_datas[lv].bs_point_lst.getSortedBspList()
    features = []

    for bsp in bs_points:
        base_feat = {
            'index': bsp.klu.idx,
            'timestamp': bsp.klu.time,
            'type': bsp.type2str(),
            'is_buy': bsp.is_buy,
            'divergence_rate': getattr(bsp, 'divergence_rate', None),
            'is_segbsp': getattr(bsp, 'is_segbsp', False)
        }

        # Add all features from the CFeatures object
        if bsp.features:
            extra_feat = bsp.features.to_dict()
            all_feat = {**base_feat, **extra_feat}
        else:
            all_feat = base_feat

        features.append(all_feat)

    return features


def export_bs_feature_file(chan: CChan, lv: KL_TYPE, filename: str, window: int = 5):
    """
    Export BS features to a CSV file.
    """
    print(f"Exporting BS features for level {lv} to: {filename}")
    features = extract_features_from_cbs_points(chan, lv)

    if not features:
        print("No features extracted.")
        return

    # ✅ 收集所有 feature keys
    all_fieldnames = set()
    for feat in features:
        all_fieldnames.update(feat.keys())

    fieldnames = sorted(all_fieldnames)

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)

    print(f"Exported {len(features)} BS points to: {filename}")
