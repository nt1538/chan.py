import yfinance as yf
import pandas as pd
from typing import Iterable
from datetime import datetime

from Common.CEnum import AUTYPE, KL_TYPE, DATA_FIELD
from Common.CTime import CTime
from KLine.KLine_Unit import CKLine_Unit
from .CommonStockAPI import CCommonStockApi

def safe_float(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)

class CYahooFinance(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.NONE):
        super(CYahooFinance, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self) -> Iterable[CKLine_Unit]:
        interval = self._convert_interval()
        df = yf.download(self.code, start=self.begin_date, end=self.end_date, interval=interval, progress=False)

        if df.empty:
            raise Exception(f"yfinance returned no data for symbol: {self.code}")

        for index, row in df.iterrows():
            dt = index.to_pydatetime()
            item = {
                DATA_FIELD.FIELD_TIME: CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute),
                DATA_FIELD.FIELD_OPEN: safe_float(row["Open"]),
                DATA_FIELD.FIELD_HIGH: safe_float(row["High"]),
                DATA_FIELD.FIELD_LOW: safe_float(row["Low"]),
                DATA_FIELD.FIELD_CLOSE: safe_float(row["Close"]),
                DATA_FIELD.FIELD_VOLUME: safe_float(row.get("Volume", 0)),
            }

            yield CKLine_Unit(item)

    def SetBasciInfo(self):
        self.name = self.code
        self.is_stock = True

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass

    def _convert_interval(self):
        mapping = {
            KL_TYPE.K_1M: "1m",
            KL_TYPE.K_DAY: "1d",
            KL_TYPE.K_5M: "5m",
            KL_TYPE.K_15M: "15m",
            KL_TYPE.K_30M: "30m",
            KL_TYPE.K_60M: "60m",
            KL_TYPE.K_WEEK: "1wk",
            KL_TYPE.K_MON: "1mo",
        }
        if self.k_type not in mapping:
            raise Exception(f"k_type {self.k_type} not supported by yfinance.")
        return mapping[self.k_type]
    
