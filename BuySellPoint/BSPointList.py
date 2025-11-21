from typing import Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

from Bi.Bi import CBi
from Bi.BiList import CBiList
from Common.CEnum import BSP_TYPE
from Common.func_util import has_overlap
from Seg.Seg import CSeg
from Seg.SegListComm import CSegListComm
from ZS.ZS import CZS

from .BS_Point import CBS_Point
from .BSPointConfig import CBSPointConfig, CPointConfig

LINE_TYPE = TypeVar('LINE_TYPE', CBi, CSeg)
LINE_LIST_TYPE = TypeVar('LINE_LIST_TYPE', CBiList, CSegListComm)


class CBSPointList(Generic[LINE_TYPE, LINE_LIST_TYPE]):
    def __init__(self, bs_point_config: CBSPointConfig):
        self.bsp_store_dict: Dict[BSP_TYPE, Tuple[List[CBS_Point[LINE_TYPE]], List[CBS_Point[LINE_TYPE]]]] = {}
        self.bsp_store_flat_dict: Dict[int, CBS_Point[LINE_TYPE]] = {}

        self.bsp1_list: List[CBS_Point[LINE_TYPE]] = []
        self.bsp1_dict: Dict[int, CBS_Point[LINE_TYPE]] = {}

        self.config = bs_point_config
        self.last_sure_pos = -1
        self.last_sure_seg_idx = 0

    def store_add_bsp(self, bsp_type: BSP_TYPE, bsp: CBS_Point[LINE_TYPE]):
        if bsp_type not in self.bsp_store_dict:
            self.bsp_store_dict[bsp_type] = ([], [])
        if len(self.bsp_store_dict[bsp_type][bsp.is_buy]) > 0:
            assert self.bsp_store_dict[bsp_type][bsp.is_buy][-1].bi.idx < bsp.bi.idx, f"{bsp_type}, {bsp.is_buy} {self.bsp_store_dict[bsp_type][bsp.is_buy][-1].bi.idx} {bsp.bi.idx}"
        self.bsp_store_dict[bsp_type][bsp.is_buy].append(bsp)
        self.bsp_store_flat_dict[bsp.bi.idx] = bsp
        # Standardize features for this BSP
        self.standardize_features_for_bsp(bsp)

    def add_bsp1(self, bsp: CBS_Point[LINE_TYPE]):
        if len(self.bsp1_list) > 0:
            assert self.bsp1_list[-1].bi.idx < bsp.bi.idx
        self.bsp1_list.append(bsp)
        self.bsp1_dict[bsp.bi.idx] = bsp

    def clear_store_end(self):
        for bsp_list in self.bsp_store_dict.values():
            for is_buy in [True, False]:
                while len(bsp_list[is_buy]) > 0:
                    if bsp_list[is_buy][-1].bi.get_end_klu().idx <= self.last_sure_pos:
                        break
                    del self.bsp_store_flat_dict[bsp_list[is_buy][-1].bi.idx]
                    bsp_list[is_buy].pop()

    def clear_bsp1_end(self):
        while len(self.bsp1_list) > 0:
            if self.bsp1_list[-1].bi.get_end_klu().idx <= self.last_sure_pos:
                break
            del self.bsp1_dict[self.bsp1_list[-1].bi.idx]
            self.bsp1_list.pop()

    def bsp_iter(self) -> Iterable[CBS_Point[LINE_TYPE]]:
        for bsp_list in self.bsp_store_dict.values():
            yield from bsp_list[True]
            yield from bsp_list[False]

    def __len__(self):
        return len(self.bsp_store_flat_dict)

    def cal(self, bi_list: LINE_LIST_TYPE, seg_list: CSegListComm[LINE_TYPE]):
        self.clear_store_end()
        self.clear_bsp1_end()
        self.cal_seg_bs1point(seg_list, bi_list)
        self.cal_seg_bs2point(seg_list, bi_list)
        self.cal_seg_bs3point(seg_list, bi_list)
        self.standardize_all_features()  # Standardize features for all BSPoints
        self.update_last_pos(seg_list)

    def update_last_pos(self, seg_list: CSegListComm):
        self.last_sure_pos = -1
        self.last_sure_seg_idx = 0
        seg_idx = len(seg_list)-1
        while seg_idx >= 0:
            seg = seg_list[seg_idx]
            if seg.is_sure:
                self.last_sure_pos = seg.end_bi.get_begin_klu().idx
                self.last_sure_seg_idx = seg.idx
                return
            seg_idx -= 1

    def seg_need_cal(self, seg: CSeg):
        return seg.end_bi.get_end_klu().idx > self.last_sure_pos

    def add_bs(
        self,
        bs_type: BSP_TYPE,
        bi: LINE_TYPE,
        relate_bsp1: Optional[CBS_Point],
        is_target_bsp: bool = True,
        feature_dict=None,
    ):
        is_buy = bi.is_down()
        if exist_bsp := self.bsp_store_flat_dict.get(bi.idx):
            assert exist_bsp.is_buy == is_buy
            exist_bsp.add_another_bsp_prop(bs_type, relate_bsp1)
            if feature_dict is not None:
                exist_bsp.add_feat(feature_dict)
                # Standardize features again when adding new features
                self.standardize_features_for_bsp(exist_bsp)
            return
        if bs_type not in self.config.GetBSConfig(is_buy).target_types:
            is_target_bsp = False

        if is_target_bsp or bs_type in [BSP_TYPE.T1, BSP_TYPE.T1P]:
            bsp = CBS_Point[LINE_TYPE](
                bi=bi,
                is_buy=is_buy,
                bs_type=bs_type,
                relate_bsp1=relate_bsp1,
                feature_dict=feature_dict,
            )
            # Standardize features for new BSP
            self.standardize_features_for_bsp(bsp)
        else:
            return
        if is_target_bsp:
            self.store_add_bsp(bs_type, bsp)
        if bs_type in [BSP_TYPE.T1, BSP_TYPE.T1P]:
            self.add_bsp1(bsp)

    def standardize_all_features(self):
        """Standardize features for all BSPoints in the list"""
        for bsp in self.bsp_iter():
            self.standardize_features_for_bsp(bsp)

    # New method to standardize features for a single BSPoint
    def standardize_features_for_bsp(self, bsp: CBS_Point):
        """
        Standardize features for a single BSPoint to ensure all types have the same feature set
        """
        if not hasattr(bsp, 'features') or not bsp.features:
            return
            
        feature_dict = bsp.features.to_dict() if bsp.features else {}
        if not feature_dict:
            return
            
        bs_type_str = bsp.type[0].value if bsp.type else ''
        
        # 1. Standardize divergence_rate
        if 'divergence_rate' in feature_dict:
            # For BSP1, keep existing divergence_rate
            bsp.add_feat({'divergence_rate': feature_dict['divergence_rate']})
        elif bs_type_str in ['2', '2s'] and f'bsp{bs_type_str}_retrace_rate' in feature_dict:
            # For BSP2/2S, use retrace_rate as divergence_rate
            bsp.add_feat({'divergence_rate': feature_dict[f'bsp{bs_type_str}_retrace_rate']})
        elif bs_type_str in ['3', '3a', '3b'] and 'bsp3_zs_height' in feature_dict:
            # For BSP3, use zs_height as divergence_rate
            bsp.add_feat({'divergence_rate': feature_dict['bsp3_zs_height']})
        
        # 2. Standardize ZS count
        if 'zs_cnt' in feature_dict:
            bsp.add_feat({'zs_cnt': feature_dict['zs_cnt']})
        else:
            # Add default values for types without zs_cnt
            if bs_type_str in ['2', '2s']:
                bsp.add_feat({'zs_cnt': 1.0})
            elif bs_type_str in ['3', '3a', '3b']:
                bsp.add_feat({'zs_cnt': 2.0})
            else:
                bsp.add_feat({'zs_cnt': 0.0})
        
        # 3. Standardize break_bi features
        # These features describe the breakout pattern and are crucial for BSP2/2S
        # but need to be derived for BSP1 and BSP3
        
        if bs_type_str == '2' and 'bsp2_break_bi_amp' in feature_dict:
            # For BSP2, use existing break_bi features
            bsp.add_feat({
                'break_bi_amp': feature_dict['bsp2_break_bi_amp'],
                'break_bi_klu_cnt': feature_dict.get('bsp2_break_bi_bi_klu_cnt', 0),
                'break_bi_amp_rate': feature_dict.get('bsp2_break_bi_amp_rate', 0)
            })
        elif bs_type_str == '2s' and 'bsp2s_break_bi_amp' in feature_dict:
            # For BSP2S, use existing break_bi features
            bsp.add_feat({
                'break_bi_amp': feature_dict['bsp2s_break_bi_amp'],
                'break_bi_klu_cnt': feature_dict.get('bsp2s_break_bi_klu_cnt', 0),
                'break_bi_amp_rate': feature_dict.get('bsp2s_break_bi_amp_rate', 0)
            })
        else:
            # For BSP1 and BSP3, derive break_bi features
            # This is a placeholder implementation - in a real system, 
            # you would calculate these based on actual bi data
            if bs_type_str in ['1', '1p'] and 'bsp1_bi_amp' in feature_dict:
                bsp.add_feat({
                    'break_bi_amp': feature_dict['bsp1_bi_amp'],
                    'break_bi_klu_cnt': feature_dict.get('bsp1_bi_klu_cnt', 0),
                    'break_bi_amp_rate': feature_dict.get('bsp1_bi_amp_rate', 0)
                })
            elif bs_type_str in ['3', '3a', '3b'] and 'bsp3_bi_amp' in feature_dict:
                bsp.add_feat({
                    'break_bi_amp': feature_dict['bsp3_bi_amp'],
                    'break_bi_klu_cnt': feature_dict.get('bsp3_bi_klu_cnt', 0),
                    'break_bi_amp_rate': feature_dict.get('bsp3_bi_amp_rate', 0)
                })
        
        # 4. Standardize bi features (remove type prefix)
        if bs_type_str == '1' and 'bsp1_bi_amp' in feature_dict:
            bsp.add_feat({
                'bi_amp': feature_dict['bsp1_bi_amp'],
                'bi_klu_cnt': feature_dict.get('bsp1_bi_klu_cnt', 0),
                'bi_amp_rate': feature_dict.get('bsp1_bi_amp_rate', 0)
            })
        elif bs_type_str == '2' and 'bsp2_bi_amp' in feature_dict:
            bsp.add_feat({
                'bi_amp': feature_dict['bsp2_bi_amp'],
                'bi_klu_cnt': feature_dict.get('bsp2_bi_klu_cnt', 0),
                'bi_amp_rate': feature_dict.get('bsp2_bi_amp_rate', 0)
            })
        elif bs_type_str == '2s' and 'bsp2s_bi_amp' in feature_dict:
            bsp.add_feat({
                'bi_amp': feature_dict['bsp2s_bi_amp'],
                'bi_klu_cnt': feature_dict.get('bsp2s_bi_klu_cnt', 0),
                'bi_amp_rate': feature_dict.get('bsp2s_bi_amp_rate', 0)
            })
        elif bs_type_str in ['3', '3a', '3b'] and 'bsp3_bi_amp' in feature_dict:
            bsp.add_feat({
                'bi_amp': feature_dict['bsp3_bi_amp'],
                'bi_klu_cnt': feature_dict.get('bsp3_bi_klu_cnt', 0),
                'bi_amp_rate': feature_dict.get('bsp3_bi_amp_rate', 0)
            })
        elif 'bsp_bi_amp' in feature_dict:
            # Use the common bi_amp if type-specific ones are not available
            bsp.add_feat({
                'bi_amp': feature_dict['bsp_bi_amp'],
                'bi_klu_cnt': 0,  # Default value as we don't have klu_cnt
                'bi_amp_rate': 0  # Default value as we don't have amp_rate
            })
        
        # 5. Add level feature
        if bs_type_str == '1' or bs_type_str == '1p':
            bsp.add_feat({'level': 1.0})
        elif bs_type_str == '2':
            bsp.add_feat({'level': 2.0})
        elif bs_type_str == '2s' and 'bsp2s_lv' in feature_dict:
            bsp.add_feat({'level': feature_dict['bsp2s_lv']})
        elif bs_type_str == '2s':
            bsp.add_feat({'level': 2.0})  # Default level for BSP2S
        elif bs_type_str in ['3', '3a', '3b']:
            bsp.add_feat({'level': 3.0})
            
        # Add BSP type as a feature
        if bs_type_str == '1' or bs_type_str == '1p':
            bsp.add_feat({'bsp_type': 1.0})
        elif bs_type_str == '2':
            bsp.add_feat({'bsp_type': 2.0})
        elif bs_type_str == '2s':
            bsp.add_feat({'bsp_type': 2.5})  # 2.5 to differentiate from BSP2
        elif bs_type_str in ['3', '3a']:
            bsp.add_feat({'bsp_type': 3.0})
        elif bs_type_str == '3b':
            bsp.add_feat({'bsp_type': 3.5})  # 3.5 to differentiate from BSP3A

    def cal_seg_bs1point(self, seg_list: CSegListComm[LINE_TYPE], bi_list: LINE_LIST_TYPE):
        for seg in seg_list[self.last_sure_seg_idx:]:
            if not self.seg_need_cal(seg):
                continue
            self.cal_single_bs1point(seg, bi_list)

    def cal_single_bs1point(self, seg: CSeg[LINE_TYPE], bi_list: LINE_LIST_TYPE):
        BSP_CONF = self.config.GetBSConfig(seg.is_down())
        zs_cnt = seg.get_multi_bi_zs_cnt() if BSP_CONF.bsp1_only_multibi_zs else len(seg.zs_lst)
        is_target_bsp = (BSP_CONF.min_zs_cnt <= 0 or zs_cnt >= BSP_CONF.min_zs_cnt)
        if len(seg.zs_lst) > 0 and \
           not seg.zs_lst[-1].is_one_bi_zs() and \
           ((seg.zs_lst[-1].bi_out and seg.zs_lst[-1].bi_out.idx >= seg.end_bi.idx) or seg.zs_lst[-1].bi_lst[-1].idx >= seg.end_bi.idx) \
           and seg.end_bi.idx - seg.zs_lst[-1].get_bi_in().idx > 2:
            self.treat_bsp1(seg, BSP_CONF, bi_list, is_target_bsp)
        else:
            self.treat_pz_bsp1(seg, BSP_CONF, bi_list, is_target_bsp)

    def treat_bsp1(self, seg: CSeg[LINE_TYPE], BSP_CONF: CPointConfig, bi_list: LINE_LIST_TYPE, is_target_bsp: bool):
        last_zs = seg.zs_lst[-1]
        break_peak, _ = last_zs.out_bi_is_peak(seg.end_bi.idx)
        if BSP_CONF.bs1_peak and not break_peak:
            is_target_bsp = False
        is_diver, divergence_rate = last_zs.is_divergence(BSP_CONF, out_bi=seg.end_bi)
        if not is_diver:
            is_target_bsp = False
        last_bi = seg.end_bi
        kl_data = seg.end_bi.get_end_klu()
        macd_item = getattr(kl_data, 'macd', None)
        macd_value = getattr(macd_item, 'macd', None)
        macd_diff = getattr(macd_item, 'DIF', None)
        macd_dea = getattr(macd_item, 'DEA', None)
        macd_fast = getattr(macd_item, 'fast_ema', None)
        macd_slow = getattr(macd_item, 'slow_ema', None)
        ppo = (macd_fast - macd_slow) / macd_slow

        kdj_item = getattr(kl_data, 'kdj', None)
        k_value = getattr(kdj_item, 'k', None)
        d_value = getattr(kdj_item, 'd', None)
        j_value = getattr(kdj_item, 'j', None)
        
        feature_dict = {
            'divergence_rate': divergence_rate,
            'bsp1_bi_amp': last_bi.amp(),
            'bsp1_bi_klu_cnt': last_bi.get_klu_cnt(),
            'bsp1_bi_amp_rate': last_bi.amp()/last_bi.get_begin_val(),
            'zs_cnt': len(seg.zs_lst),
            'macd_value': macd_value,
            'macd_dea': macd_dea,
            'macd_diff': macd_diff,
            'ppo': ppo,
            'rsi': getattr(kl_data, 'rsi', None),
            'kdj_k': k_value,
            'kdj_d': d_value,
            'kdj_j': j_value,
            'volume': getattr(kl_data, 'volume', None),
            'next_bi_return': self.safe_get_next_bi_return(bi_list, last_bi.idx),
        }
        self.add_bs(bs_type=BSP_TYPE.T1, bi=seg.end_bi, relate_bsp1=None, is_target_bsp=is_target_bsp, feature_dict=feature_dict)

    def treat_pz_bsp1(self, seg: CSeg[LINE_TYPE], BSP_CONF: CPointConfig, bi_list: LINE_LIST_TYPE, is_target_bsp):
        last_bi = seg.end_bi
        pre_bi = bi_list[last_bi.idx-2]
        if last_bi.seg_idx != pre_bi.seg_idx:
            return
        if last_bi.dir != seg.dir:
            return
        if last_bi.is_down() and last_bi._low() > pre_bi._low():  # åˆ›æ–°ä½Ž
            return
        if last_bi.is_up() and last_bi._high() < pre_bi._high():  # åˆ›æ–°é«˜
            return
        in_metric = pre_bi.cal_macd_metric(BSP_CONF.macd_algo, is_reverse=False)
        out_metric = last_bi.cal_macd_metric(BSP_CONF.macd_algo, is_reverse=True)
        is_diver, divergence_rate = out_metric <= BSP_CONF.divergence_rate*in_metric, out_metric/(in_metric+1e-7)
        if not is_diver:
            is_target_bsp = False
        if isinstance(bi_list, CBiList):
            assert isinstance(last_bi, CBi) and isinstance(pre_bi, CBi)
            kl_data = last_bi.get_end_klu()
            macd_item = getattr(kl_data, 'macd', None)
            macd_value = getattr(macd_item, 'macd', None)
            macd_diff = getattr(macd_item, 'DIF', None)
            macd_dea = getattr(macd_item, 'DEA', None)
            macd_fast = getattr(macd_item, 'fast_ema', None)
            macd_slow = getattr(macd_item, 'slow_ema', None)
            ppo = (macd_fast - macd_slow) / macd_slow

            kdj_item = getattr(kl_data, 'kdj', None)
            k_value = getattr(kdj_item, 'k', None)
            d_value = getattr(kdj_item, 'd', None)
            j_value = getattr(kdj_item, 'j', None)

            feature_dict = {
                'divergence_rate': divergence_rate,
                'bsp1_bi_amp': last_bi.amp(),
                'bsp1_bi_klu_cnt': last_bi.get_klu_cnt(),
                'bsp1_bi_amp_rate': last_bi.amp()/last_bi.get_begin_val(),
                'macd_value': macd_value,
                'macd_dea': macd_dea,
                'macd_diff': macd_diff,
                'ppo': ppo,
                'rsi': getattr(kl_data, 'rsi', None),
                'kdj_k': k_value,
                'kdj_d': d_value,
                'kdj_j': j_value,
                'volume': getattr(kl_data, 'volume', None),
                'next_bi_return': self.safe_get_next_bi_return(bi_list, last_bi.idx),
            }
        elif isinstance(last_bi, CSeg) and len(last_bi.bi_list) > 0 and isinstance(last_bi.bi_list[-1], CBi):
            bi_tail = last_bi.bi_list[-1]
            kl_data = bi_tail.get_end_klu()
            macd_item = getattr(kl_data, 'macd', None)
            macd_value = getattr(macd_item, 'macd', None)
            macd_diff = getattr(macd_item, 'DIF', None)
            macd_dea = getattr(macd_item, 'DEA', None)
            macd_fast = getattr(macd_item, 'fast_ema', None)
            macd_slow = getattr(macd_item, 'slow_ema', None)
            ppo = (macd_fast - macd_slow) / macd_slow

            kdj_item = getattr(kl_data, 'kdj', None)
            k_value = getattr(kdj_item, 'k', None)
            d_value = getattr(kdj_item, 'd', None)
            j_value = getattr(kdj_item, 'j', None)

            feature_dict = {
                'divergence_rate': divergence_rate,
                'bsp1_bi_amp': last_bi.amp(),
                'bsp1_bi_klu_cnt': last_bi.get_klu_cnt(),
                'bsp1_bi_amp_rate': last_bi.amp()/last_bi.get_begin_val(),
                'macd_value': macd_value,
                'macd_dea': macd_dea,
                'macd_diff': macd_diff,
                'ppo': ppo,
                'rsi': getattr(kl_data, 'rsi', None),
                'kdj_k': k_value,
                'kdj_d': d_value,
                'kdj_j': j_value,
                'volume': getattr(kl_data, 'volume', None),
                'next_bi_return': self.safe_get_next_bi_return(bi_list, last_bi.idx),
            }
        self.add_bs(bs_type=BSP_TYPE.T1P, bi=last_bi, relate_bsp1=None, is_target_bsp=is_target_bsp, feature_dict=feature_dict)

    def cal_seg_bs2point(self, seg_list: CSegListComm[LINE_TYPE], bi_list: LINE_LIST_TYPE):
        for seg in seg_list[self.last_sure_seg_idx:]:
            config = self.config.GetBSConfig(seg.is_down())
            if BSP_TYPE.T2 not in config.target_types and BSP_TYPE.T2S not in config.target_types:
                continue
            if not self.seg_need_cal(seg):
                continue
            self.treat_bsp2(seg, seg_list, bi_list)

    def treat_bsp2(self, seg: CSeg, seg_list: CSegListComm[LINE_TYPE], bi_list: LINE_LIST_TYPE):
        if len(seg_list) > 1:
            BSP_CONF = self.config.GetBSConfig(seg.is_down())
            bsp1_bi = seg.end_bi
            real_bsp1 = self.bsp1_dict.get(bsp1_bi.idx)
            if bsp1_bi.idx + 2 >= len(bi_list):
                return
            break_bi = bi_list[bsp1_bi.idx + 1]
            bsp2_bi = bi_list[bsp1_bi.idx + 2]
        else:
            BSP_CONF = self.config.GetBSConfig(seg.is_up())
            bsp1_bi, real_bsp1 = None, None
            if len(bi_list) == 1:
                return
            bsp2_bi = bi_list[1]
            break_bi = bi_list[0]
        if BSP_CONF.bsp2_follow_1 and (not bsp1_bi or bsp1_bi.idx not in self.bsp_store_flat_dict):
            return
        retrace_rate = bsp2_bi.amp()/break_bi.amp()
        bsp2_flag = retrace_rate <= BSP_CONF.max_bs2_rate
        if bsp2_flag:
            kl_data = bsp2_bi.get_end_klu()
            macd_item = getattr(kl_data, 'macd', None)
            macd_value = getattr(macd_item, 'macd', None)
            macd_diff = getattr(macd_item, 'DIF', None)
            macd_dea = getattr(macd_item, 'DEA', None)
            macd_fast = getattr(macd_item, 'fast_ema', None)
            macd_slow = getattr(macd_item, 'slow_ema', None)
            ppo = (macd_fast - macd_slow) / macd_slow

            kdj_item = getattr(kl_data, 'kdj', None)
            k_value = getattr(kdj_item, 'k', None)
            d_value = getattr(kdj_item, 'd', None)
            j_value = getattr(kdj_item, 'j', None)

            feature_dict = {
                'bsp2_retrace_rate': retrace_rate,
                'bsp2_break_bi_amp': break_bi.amp(),
                'bsp2_break_bi_bi_klu_cnt': break_bi.get_klu_cnt(),
                'bsp2_break_bi_amp_rate': break_bi.amp()/break_bi.get_begin_val(),
                'bsp2_bi_amp': bsp2_bi.amp(),
                'bsp2_bi_klu_cnt': bsp2_bi.get_klu_cnt(),
                'bsp2_bi_amp_rate': bsp2_bi.amp()/bsp2_bi.get_begin_val(),
                'macd_value': macd_value,
                'macd_dea': macd_dea,
                'macd_diff': macd_diff,
                'ppo': ppo,
                'rsi': getattr(kl_data, 'rsi', None),
                'kdj_k': k_value,
                'kdj_d': d_value,
                'kdj_j': j_value,
                'volume': getattr(kl_data, 'volume', None),
                'next_bi_return': self.safe_get_next_bi_return(bi_list, bsp2_bi.idx),
            }
            self.add_bs(bs_type=BSP_TYPE.T2, bi=bsp2_bi, relate_bsp1=real_bsp1, feature_dict=feature_dict)
        elif BSP_CONF.bsp2s_follow_2:
            return
        if BSP_TYPE.T2S not in self.config.GetBSConfig(seg.is_down()).target_types:
            return
        self.treat_bsp2s(seg_list, bi_list, bsp2_bi, break_bi, real_bsp1, BSP_CONF)  # type: ignore

    def treat_bsp2s(
        self,
        seg_list: CSegListComm,
        bi_list: LINE_LIST_TYPE,
        bsp2_bi: LINE_TYPE,
        break_bi: LINE_TYPE,
        real_bsp1: Optional[CBS_Point],
        BSP_CONF: CPointConfig,
    ):
        bias = 2
        _low, _high = None, None
        while bsp2_bi.idx + bias < len(bi_list):  # è®¡ç®—ç±»äºŒ
            bsp2s_bi = bi_list[bsp2_bi.idx + bias]
            assert bsp2s_bi.seg_idx is not None and bsp2_bi.seg_idx is not None
            if BSP_CONF.max_bsp2s_lv is not None and bias/2 > BSP_CONF.max_bsp2s_lv:
                break
            if bsp2s_bi.seg_idx != bsp2_bi.seg_idx and (bsp2s_bi.seg_idx < len(seg_list)-1 or bsp2s_bi.seg_idx - bsp2_bi.seg_idx >= 2 or seg_list[bsp2_bi.seg_idx].is_sure):
                break
            if bias == 2:
                if not has_overlap(bsp2_bi._low(), bsp2_bi._high(), bsp2s_bi._low(), bsp2s_bi._high()):
                    break
                _low = max([bsp2_bi._low(), bsp2s_bi._low()])
                _high = min([bsp2_bi._high(), bsp2s_bi._high()])
            elif not has_overlap(_low, _high, bsp2s_bi._low(), bsp2s_bi._high()):
                break

            if bsp2s_break_bsp1(bsp2s_bi, break_bi):
                break
            retrace_rate = abs(bsp2s_bi.get_end_val()-break_bi.get_end_val())/break_bi.amp()
            if retrace_rate > BSP_CONF.max_bs2_rate:
                break
            kl_data = bsp2s_bi.get_end_klu()
            macd_item = getattr(kl_data, 'macd', None)
            macd_value = getattr(macd_item, 'macd', None)
            macd_diff = getattr(macd_item, 'DIF', None)
            macd_dea = getattr(macd_item, 'DEA', None)
            macd_fast = getattr(macd_item, 'fast_ema', None)
            macd_slow = getattr(macd_item, 'slow_ema', None)
            ppo = (macd_fast - macd_slow) / macd_slow
            
            kdj_item = getattr(kl_data, 'kdj', None)
            k_value = getattr(kdj_item, 'k', None)
            d_value = getattr(kdj_item, 'd', None)
            j_value = getattr(kdj_item, 'j', None)

            feature_dict = {
                'bsp2s_retrace_rate': retrace_rate,
                'bsp2s_break_bi_amp': break_bi.amp(),
                'bsp2s_break_bi_klu_cnt': break_bi.get_klu_cnt(),
                'bsp2s_break_bi_amp_rate': break_bi.amp()/break_bi.get_begin_val(),
                'bsp2s_bi_amp': bsp2s_bi.amp(),
                'bsp2s_bi_klu_cnt': bsp2s_bi.get_klu_cnt(),
                'bsp2s_bi_amp_rate': bsp2s_bi.amp()/bsp2s_bi.get_begin_val(),
                'bsp2s_lv': bias / 2,
                'macd_value': macd_value,
                'macd_dea': macd_dea,
                'macd_diff': macd_diff,
                'ppo': ppo,
                'rsi': getattr(kl_data, 'rsi', None),
                'kdj_k': k_value,
                'kdj_d': d_value,
                'kdj_j': j_value,
                'volume': getattr(kl_data, 'volume', None),
                'next_bi_return': self.safe_get_next_bi_return(bi_list, bsp2s_bi.idx),
            }
            self.add_bs(bs_type=BSP_TYPE.T2S, bi=bsp2s_bi, relate_bsp1=real_bsp1, feature_dict=feature_dict)  # type: ignore
            bias += 2

    def cal_seg_bs3point(self, seg_list: CSegListComm[LINE_TYPE], bi_list: LINE_LIST_TYPE):
        for seg in seg_list[self.last_sure_seg_idx:]:
            if not self.seg_need_cal(seg):
                continue
            config = self.config.GetBSConfig(seg.is_down())
            if BSP_TYPE.T3A not in config.target_types and BSP_TYPE.T3B not in config.target_types:
                continue
            if len(seg_list) > 1:
                bsp1_bi = seg.end_bi
                bsp1_bi_idx = bsp1_bi.idx
                BSP_CONF = self.config.GetBSConfig(seg.is_down())
                real_bsp1 = self.bsp1_dict.get(bsp1_bi.idx)
                next_seg_idx = seg.idx+1
                next_seg = seg.next  # å¯èƒ½ä¸ºNone, æ‰€ä»¥å¹¶ä¸ä¸€å®šå¯ä»¥ä¿è¯next_seg_idx == next_seg.idx
            else:
                next_seg = seg
                next_seg_idx = seg.idx
                bsp1_bi, real_bsp1 = None, None
                bsp1_bi_idx = -1
                BSP_CONF = self.config.GetBSConfig(seg.is_up())
            if BSP_CONF.bsp3_follow_1 and (not bsp1_bi or bsp1_bi.idx not in self.bsp_store_flat_dict):
                continue
            if next_seg:
                self.treat_bsp3_after(seg_list, next_seg, BSP_CONF, bi_list, real_bsp1, bsp1_bi_idx, next_seg_idx)
            self.treat_bsp3_before(seg_list, seg, next_seg, bsp1_bi, BSP_CONF, bi_list, real_bsp1, next_seg_idx)

    def treat_bsp3_after(
        self,
        seg_list: CSegListComm[LINE_TYPE],
        next_seg: CSeg[LINE_TYPE],
        BSP_CONF: CPointConfig,
        bi_list: LINE_LIST_TYPE,
        real_bsp1,
        bsp1_bi_idx,
        next_seg_idx
    ):
        first_zs = next_seg.get_first_multi_bi_zs()
        if first_zs is None:
            return
        if BSP_CONF.strict_bsp3 and first_zs.get_bi_in().idx != bsp1_bi_idx+1:
            return
        if first_zs.bi_out is None or first_zs.bi_out.idx+1 >= len(bi_list):
            return
        bsp3_bi = bi_list[first_zs.bi_out.idx+1]
        if bsp3_bi.parent_seg is None:
            if next_seg.idx != len(seg_list)-1:
                return
        elif bsp3_bi.parent_seg.idx != next_seg.idx:
            if len(bsp3_bi.parent_seg.bi_list) >= 3:
                return
        if bsp3_bi.dir == next_seg.dir:
            return
        if bsp3_bi.seg_idx != next_seg_idx and next_seg_idx < len(seg_list)-2:
            return
        if bsp3_back2zs(bsp3_bi, first_zs):
            return
        bsp3_peak_zs = bsp3_break_zspeak(bsp3_bi, first_zs)
        if BSP_CONF.bsp3_peak and not bsp3_peak_zs:
            return
        kl_data = bsp3_bi.get_end_klu()
        macd_item = getattr(kl_data, 'macd', None)
        macd_value = getattr(macd_item, 'macd', None)
        macd_diff = getattr(macd_item, 'DIF', None)
        macd_dea = getattr(macd_item, 'DEA', None)
        macd_fast = getattr(macd_item, 'fast_ema', None)
        macd_slow = getattr(macd_item, 'slow_ema', None)

        kdj_item = getattr(kl_data, 'kdj', None)
        k_value = getattr(kdj_item, 'k', None)
        d_value = getattr(kdj_item, 'd', None)
        j_value = getattr(kdj_item, 'j', None)

        ppo = (macd_fast - macd_slow) / macd_slow
        feature_dict = {
            'bsp3_zs_height': (first_zs.high - first_zs.low) / first_zs.low,
            'bsp3_bi_amp': bsp3_bi.amp(),
            'bsp3_bi_klu_cnt': bsp3_bi.get_klu_cnt(),
            'bsp3_bi_amp_rate': bsp3_bi.amp()/bsp3_bi.get_begin_val(),
            'macd_value': macd_value,
            'macd_dea': macd_dea,
            'macd_diff': macd_diff,
            'ppo': ppo,
            'rsi': getattr(kl_data, 'rsi', None),
            'kdj_k': k_value,
            'kdj_d': d_value,
            'kdj_j': j_value,
            'volume': getattr(kl_data, 'volume', None),
            'next_bi_return': self.safe_get_next_bi_return(bi_list, bsp3_bi.idx),
        }
        self.add_bs(bs_type=BSP_TYPE.T3A, bi=bsp3_bi, relate_bsp1=real_bsp1, feature_dict=feature_dict)  # type: ignore

    def treat_bsp3_before(
        self,
        seg_list: CSegListComm[LINE_TYPE],
        seg: CSeg[LINE_TYPE],
        next_seg: Optional[CSeg[LINE_TYPE]],
        bsp1_bi: Optional[LINE_TYPE],
        BSP_CONF: CPointConfig,
        bi_list: LINE_LIST_TYPE,
        real_bsp1,
        next_seg_idx
    ):
        if next_seg is None or next_seg_idx >= len(seg_list):
            return
        if next_seg_idx+1 >= len(seg_list):
            # æžšåæ­‡
            return
        bi_end_idx = cal_bsp3_bi_end_idx(seg_list[next_seg_idx+1])
        if bi_end_idx == float("inf"):
            return
        # Fix: using start_bi instead of begin_bi
        if next_seg_idx+2 < len(seg_list) and (bi_end_idx >= seg_list[next_seg_idx+2].start_bi.idx):
            return  # ç»"æžšä½ç½®è·¨è¿‡ä¸‹ä¸€ä¸ªçº¿æ®µï¼Œåœæ­¢
        if next_seg.get_multi_bi_zs_cnt() == 0:
            return
        cmp_zs = None
        if next_seg.get_multi_bi_zs_cnt() == 0:
            return
        for zs in next_seg.zs_lst:
            if zs.is_one_bi_zs():
                continue
            cmp_zs = zs
            break
        if cmp_zs is None or cmp_zs.bi_out is None:
            return
        if bi_end_idx > cmp_zs.bi_out.idx or next_seg.bi_list[-1].idx < cmp_zs.bi_out.idx:
            return
        if bi_end_idx+1 >= len(bi_list):
            # æžšåæ­‡
            return
        bsp3_bi = bi_list[bi_end_idx+1]
        if bsp3_bi.dir == next_seg.dir:
            return
        if bsp3_bi.seg_idx != next_seg.idx+1 and next_seg.idx < len(seg_list)-2:
            return

        if bsp3_back2zs(bsp3_bi, cmp_zs):
            return
        bsp3_peak_zs = bsp3_break_zspeak(bsp3_bi, cmp_zs)
        if BSP_CONF.bsp3_peak and not bsp3_peak_zs:
            return
        kl_data = bsp3_bi.get_end_klu()
        macd_item = getattr(kl_data, 'macd', None)
        macd_value = getattr(macd_item, 'macd', None)
        macd_diff = getattr(macd_item, 'DIF', None)
        macd_dea = getattr(macd_item, 'DEA', None)
        macd_fast = getattr(macd_item, 'fast_ema', None)
        macd_slow = getattr(macd_item, 'slow_ema', None)

        kdj_item = getattr(kl_data, 'kdj', None)
        k_value = getattr(kdj_item, 'k', None)
        d_value = getattr(kdj_item, 'd', None)
        j_value = getattr(kdj_item, 'j', None)

        ppo = (macd_fast - macd_slow) / macd_slow
        feature_dict = {
            'bsp3_zs_height': (cmp_zs.high - cmp_zs.low) / cmp_zs.low,
            'bsp3_bi_amp': bsp3_bi.amp(),
            'bsp3_bi_klu_cnt': bsp3_bi.get_klu_cnt(),
            'bsp3_bi_amp_rate': bsp3_bi.amp()/bsp3_bi.get_begin_val(),
            'macd_value': macd_value,
            'macd_dea': macd_dea,
            'macd_diff': macd_diff,
            'ppo': ppo,
            'rsi': getattr(kl_data, 'rsi', None),
            'kdj_k': k_value,
            'kdj_d': d_value,
            'kdj_j': j_value,
            'volume': getattr(kl_data, 'volume', None),
            'next_bi_return': self.safe_get_next_bi_return(bi_list, bsp3_bi.idx),
        }
        self.add_bs(bs_type=BSP_TYPE.T3B, bi=bsp3_bi, relate_bsp1=real_bsp1, feature_dict=feature_dict)  # type: ignore
        
    def getSortedBspList(self) -> List[CBS_Point[LINE_TYPE]]:
        sorted_bsps = sorted(self.bsp_iter(), key=lambda bsp: bsp.bi.idx)
        # Make sure all BSPs have standardized features before returning
        for bsp in sorted_bsps:
            self.standardize_features_for_bsp(bsp)
        return sorted_bsps
    
    def safe_get_next_bi_return(self, bi_list, current_idx):
        if current_idx + 1 < len(bi_list):
            next_bi = bi_list[current_idx + 1]
            return next_bi.amp() / next_bi.get_begin_val()
        return None
    
    def get_bs_list(self, bs_type, is_buy=True):
        """
        Get the BS point list for a specific type and direction
        """
        if bs_type not in self.bsp_store_dict:
            return []
        bsp_list = self.bsp_store_dict[bs_type][0 if is_buy else 1]
        # Ensure all BSPs have standardized features before returning
        for bsp in bsp_list:
            self.standardize_features_for_bsp(bsp)
        return bsp_list


def bsp2s_break_bsp1(bsp2s_bi: LINE_TYPE, bsp2_break_bi: LINE_TYPE) -> bool:
    return (bsp2s_bi.is_down() and bsp2s_bi._low() < bsp2_break_bi._low()) or \
           (bsp2s_bi.is_up() and bsp2s_bi._high() > bsp2_break_bi._high())


def bsp3_back2zs(bsp3_bi: LINE_TYPE, zs: CZS) -> bool:
    return (bsp3_bi.is_down() and bsp3_bi._low() < zs.high) or (bsp3_bi.is_up() and bsp3_bi._high() > zs.low)


def bsp3_break_zspeak(bsp3_bi: LINE_TYPE, zs: CZS) -> bool:
    return (bsp3_bi.is_down() and bsp3_bi._high() >= zs.peak_high) or (bsp3_bi.is_up() and bsp3_bi._low() <= zs.peak_low)


def cal_bsp3_bi_end_idx(seg: Optional[CSeg[LINE_TYPE]]):
    if not seg:
        return float("inf")
    if seg.get_multi_bi_zs_cnt() == 0 and seg.next is None:
        return float("inf")
    end_bi_idx = seg.end_bi.idx-1
    for zs in seg.zs_lst:
        if zs.is_one_bi_zs():
            continue
        if zs.bi_out is not None:
            end_bi_idx = zs.bi_out.idx
            break
    return end_bi_idx