# bsp_chain_tracker.py
from typing import List, Dict, Optional
from collections import defaultdict
import copy

from BuySellPoint.BSPointList import CBSPointList
#from Chan import CChan
from .BS_Point import CBS_Point



class CBSPointNode:
    def __init__(self, bsp: CBS_Point, parent: Optional[CBS_Point] = None):
        self.bsp = bsp
        self.parent = parent


class CBSPointChainTracker:
    def __init__(self):
        # { "1_buy": List[List[CBS_Point]], "1_sell": ... }
        self.chains_by_type: Dict[str, List[List]] = defaultdict(list)
        self.last_snapshots: Dict[str, Dict[int, any]] = {}  # type: Dict[str, Dict[bi_idx, CBS_Point]]

    def update_with_bspoint_diff(self, prev_bsp_list, curr_bsp_list, lv):
        old_bspoints = prev_bsp_list.getSortedBspList()
        new_bspoints = curr_bsp_list.getSortedBspList()

        new_idx_map = {(bsp.is_buy, bsp.klu.idx): bsp for bsp in new_bspoints}

        # 跟踪当前轮次新发现的链（或点）
        new_chains = defaultdict(list)

        for old_bsp in old_bspoints:
            key = (old_bsp.is_buy, old_bsp.klu.idx)
            if key not in new_idx_map:
                bs_type = old_bsp.type
                direction = 'buy' if old_bsp.is_buy else 'sell'
                chain_key = f"{bs_type}_{direction}"

               # 设置 parent_idx（基础延伸逻辑）
                old_bsp.parent_idx = None
                for offset in range(1, 4):
                    candidate_key = (old_bsp.is_buy, old_bsp.klu.idx + offset)
                    if candidate_key in new_idx_map:
                        candidate_bsp = new_idx_map[candidate_key]
                        if candidate_bsp.bi == old_bsp.bi:
                            old_bsp.parent_idx = candidate_bsp.klu.idx
                            break
            
                new_chains[chain_key].append(old_bsp)

    # 将每一组 new_chains[chain_key] 添加为一个新的链
        for key, bsp_list in new_chains.items():
            if bsp_list:
                self.chains_by_type[key].append(bsp_list)

        #print(self.chains_by_type)
        # print(self.chains_by_type)
        #print("[✓] Snapshot diff chain update complete.")
    def get_chains_by_type(self, bs_type_str: str) -> List[List]:
        return self.chains_by_type.get(f"{bs_type_str}_buy", []) + self.chains_by_type.get(f"{bs_type_str}_sell", [])
