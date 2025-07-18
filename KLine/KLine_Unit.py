import copy
from typing import Dict, Optional

from Common.CEnum import DATA_FIELD, TRADE_INFO_LST, TREND_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Math.KeltnerChannel import KeltnerChannel
from Math.BOLL import BOLL_Metric, BollModel
from Math.Demark import CDemarkEngine, CDemarkIndex
from Math.KDJ import KDJ
from Math.MACD import CMACD, CMACD_item
from Math.DMI import CDMI  # Ensure DMI module is imported
from Math.RSI import RSI
from Math.BollingerBands import BollingerBands
from Math.RSL import RSL
from Math.DemandIndex import DemandIndex
from Math.ADLine import ADLine
from Math.STARC import STARC

from Math.TrendModel import CTrendModel

from .TradeInfo import CTradeInfo


class CKLine_Unit:
    def __init__(self, kl_dict, autofix=True):
        # _time, _close, _open, _high, _low, _extra_info={}
        self.kl_type = None
        self.time: CTime = kl_dict[DATA_FIELD.FIELD_TIME]
        self.close = kl_dict[DATA_FIELD.FIELD_CLOSE]
        self.open = kl_dict[DATA_FIELD.FIELD_OPEN]
        self.high = kl_dict[DATA_FIELD.FIELD_HIGH]
        self.low = kl_dict[DATA_FIELD.FIELD_LOW]
        self.volume = kl_dict[DATA_FIELD.FIELD_VOLUME]

        self.check(autofix)

        self.trade_info = CTradeInfo(kl_dict)

        self.demark: CDemarkIndex = CDemarkIndex()

        self.sub_kl_list = []  # 次级别KLU列表
        self.sup_kl: Optional[CKLine_Unit] = None  # 指向更高级别KLU

        from KLine.KLine import CKLine
        self.__klc: Optional[CKLine] = None  # 指向KLine

        # self.macd: Optional[CMACD_item] = None
        # self.boll: Optional[BOLL_Metric] = None
        self.trend: Dict[TREND_TYPE, Dict[int, float]] = {}  # int -> float

        self.limit_flag = 0  # 0:普通 -1:跌停，1:涨停
        self.pre: Optional[CKLine_Unit] = None
        self.next: Optional[CKLine_Unit] = None

        self.set_idx(-1)

    def __deepcopy__(self, memo):
        _dict = {
            DATA_FIELD.FIELD_TIME: self.time,
            DATA_FIELD.FIELD_CLOSE: self.close,
            DATA_FIELD.FIELD_OPEN: self.open,
            DATA_FIELD.FIELD_HIGH: self.high,
            DATA_FIELD.FIELD_LOW: self.low,
            DATA_FIELD.FIELD_VOLUME: self.volume,
        }
        for metric in TRADE_INFO_LST:
            if metric in self.trade_info.metric:
                _dict[metric] = self.trade_info.metric[metric]
        obj = CKLine_Unit(_dict)
        obj.demark = copy.deepcopy(self.demark, memo)
        obj.trend = copy.deepcopy(self.trend, memo)
        obj.limit_flag = self.limit_flag
        obj.macd = copy.deepcopy(self.macd, memo)
        obj.boll = copy.deepcopy(self.boll, memo)
        if hasattr(self, "rsi"):
            obj.rsi = copy.deepcopy(self.rsi, memo)
        if hasattr(self, "kdj"):
            obj.kdj = copy.deepcopy(self.kdj, memo)
        if hasattr(self, "rsl"):
            obj.rsl = copy.deepcopy(self.rsl, memo)
        if hasattr(self, "dmi"):
            obj.dmi = copy.deepcopy(self.dmi, memo)
        if hasattr(self, "demand_index"):
            obj.demand_index = copy.deepcopy(self.demand_index, memo)
        if hasattr(self, "adline"):
            obj.ad_line = copy.deepcopy(self.ad_line, memo)
        if hasattr(self, "bollinger_bands"):
            obj.bb_vals = copy.deepcopy(self.bb_vals, memo)
        if hasattr(self, "keltner_channel"):
            obj.kc_vals = copy.deepcopy(self.kc_vals, memo)
        if hasattr(self, "STARC"):
            obj.starc_vals = copy.deepcopy(self.starc_vals, memo)
        obj.set_idx(self.idx)
        memo[id(self)] = obj
        return obj

    @property
    def klc(self):
        assert self.__klc is not None
        return self.__klc

    def set_klc(self, klc):
        self.__klc = klc

    @property
    def idx(self):
        return self.__idx

    def set_idx(self, idx):
        self.__idx: int = idx

    def __str__(self):
        return f"{self.idx}:{self.time}/{self.kl_type} open={self.open} close={self.close} high={self.high} low={self.low} {self.trade_info}"

    def check(self, autofix=False):
        if self.low > min([self.low, self.open, self.high, self.close]):
            if autofix:
                self.low = min([self.low, self.open, self.high, self.close])
            else:
                raise CChanException(f"{self.time} low price={self.low} is not min of [low={self.low}, open={self.open}, high={self.high}, close={self.close}]", ErrCode.KL_DATA_INVALID)
        if self.high < max([self.low, self.open, self.high, self.close]):
            if autofix:
                self.high = max([self.low, self.open, self.high, self.close])
            else:
                raise CChanException(f"{self.time} high price={self.high} is not max of [low={self.low}, open={self.open}, high={self.high}, close={self.close}]", ErrCode.KL_DATA_INVALID)

    def add_children(self, child):
        self.sub_kl_list.append(child)

    def set_parent(self, parent: 'CKLine_Unit'):
        self.sup_kl = parent

    def get_children(self):
        yield from self.sub_kl_list

    def _low(self):
        return self.low

    def _high(self):
        return self.high

    def set_metric(self, metric_model_lst: list) -> None:
        for metric_model in metric_model_lst:
            if isinstance(metric_model, CMACD):
                self.macd: CMACD_item = metric_model.add(self.close)
            elif isinstance(metric_model, CTrendModel):
                if metric_model.type not in self.trend:
                    self.trend[metric_model.type] = {}
                self.trend[metric_model.type][metric_model.T] = metric_model.add(self.close)
            elif isinstance(metric_model, BollModel):
                self.boll: BOLL_Metric = metric_model.add(self.close)
            elif isinstance(metric_model, CDemarkEngine):
                self.demark = metric_model.update(idx=self.idx, close=self.close, high=self.high, low=self.low)
            elif isinstance(metric_model, RSI):
                self.rsi = metric_model.add(self.close)
            elif isinstance(metric_model, RSL):
                self.rsl = metric_model.add(self.close)
            elif isinstance(metric_model, KDJ):
                self.kdj = metric_model.add(self.high, self.low, self.close)
            elif isinstance(metric_model, CDMI):
                self.dmi = metric_model.add(self.high, self.low, self.close)
            elif isinstance(metric_model, DemandIndex):
                self.demand_index = metric_model.add(self.close, self.volume)
            elif isinstance(metric_model, ADLine):
                self.ad_line = metric_model.add(self.close, self.open, self.volume)
            elif isinstance(metric_model, BollingerBands):
                self.bb_vals = metric_model.add(self.close)
                if self.bb_vals is not None:
                    self.bb_upper, self.bb_middle, self.bb_lower = self.bb_vals
                else:
                    self.bb_upper = self.bb_middle = self.bb_lower = None
            elif isinstance(metric_model, KeltnerChannel):
                prev_close = self.pre.close if self.pre is not None else None
                self.kc_vals = metric_model.add(self.high, self.low, self.close, prev_close)
                if self.kc_vals is not None:
                    self.kc_upper, self.kc_middle, self.kc_lower = self.kc_vals
                else:
                    self.kc_upper = self.kc_middle = self.kc_lower = None
            elif isinstance(metric_model, STARC):
                self.starc_vals = metric_model.add(self.high, self.low, self.close)
                if self.starc_vals is not None:
                    self.starc_upper, self.starc_middle, self.starc_lower = self.starc_vals
                else:
                    self.starc_upper = self.starc_middle = self.starc_lower = None



    def get_parent_klc(self):
        assert self.sup_kl is not None
        return self.sup_kl.klc

    def include_sub_lv_time(self, sub_lv_t: str) -> bool:
        if self.time.to_str() == sub_lv_t:
            return True
        for sub_klu in self.sub_kl_list:
            if sub_klu.time.to_str() == sub_lv_t:
                return True
            if sub_klu.include_sub_lv_time(sub_lv_t):
                return True
        return False

    def set_pre_klu(self, pre_klu: Optional['CKLine_Unit']):
        if pre_klu is None:
            return
        pre_klu.next = self
        self.pre = pre_klu
