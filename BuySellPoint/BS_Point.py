from typing import Dict, Generic, List, Optional, TypeVar, Union

from Bi.Bi import CBi
from ChanModel.Features import CFeatures
from Common.CEnum import BSP_TYPE
from Seg.Seg import CSeg

LINE_TYPE = TypeVar('LINE_TYPE', CBi, CSeg)


class CBS_Point(Generic[LINE_TYPE]):
    def __init__(self, bi: LINE_TYPE, is_buy, bs_type: BSP_TYPE, relate_bsp1: Optional['CBS_Point'], feature_dict=None):
        self.bi: LINE_TYPE = bi
        self.klu = bi.get_end_klu()
        self.is_buy = is_buy
        self.type: List[BSP_TYPE] = [bs_type]
        self.relate_bsp1 = relate_bsp1

        self.bi.bsp = self  # type: ignore
        self.features = CFeatures(feature_dict)

        self.is_segbsp = False

        self.init_common_feature()
        self.mature_rate: Optional[float] = None           # 成熟率：0 ~ 1
        self.is_mature_point: bool = False                 # 是否是最终确认的成熟点
        self.is_post_mature: bool = False                  # 是否为成熟点后的观察点（延伸）

    def add_type(self, bs_type: BSP_TYPE):
        self.type.append(bs_type)

    def type2str(self):
        return ",".join([x.value for x in self.type])

    def add_another_bsp_prop(self, bs_type: BSP_TYPE, relate_bsp1):
        self.add_type(bs_type)
        if self.relate_bsp1 is None:
            self.relate_bsp1 = relate_bsp1
        elif relate_bsp1 is not None:
            assert self.relate_bsp1.klu.idx == relate_bsp1.klu.idx

    def add_feat(self, inp1: Union[str, Dict[str, float], Dict[str, Optional[float]], 'CFeatures'], inp2: Optional[float] = None):
        self.features.add_feat(inp1, inp2)

    def init_common_feature(self):
        # 用于配置适用所有买卖点的特征
        self.add_feat({
            'bsp_bi_amp': self.bi.amp(),
        })
