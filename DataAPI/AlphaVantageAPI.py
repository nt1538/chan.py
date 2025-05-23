import os
import requests
from typing import Iterable
from datetime import datetime

from Common.CEnum import AUTYPE, KL_TYPE, DATA_FIELD
from Common.CTime import CTime
from KLine.KLine_Unit import CKLine_Unit
from .CommonStockAPI import CCommonStockApi


class CAlphaVantage(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.NONE):
        super(CAlphaVantage, self).__init__(code, k_type, begin_date, end_date, autype)
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise Exception("Alpha Vantage API key not found in environment variables.")

    def get_kl_data(self) -> Iterable[CKLine_Unit]:
        function, interval = self._get_function_and_interval()
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": self.code,
            "apikey": self.api_key,
            "outputsize": "full",
        }
        if interval:
            params["interval"] = interval

        response = requests.get(url, params=params)
        data = response.json()
        # print("DEBUG AlphaVantage response:", data)
        # print("DEBUG API key:", self.api_key)
        # print("DEBUG URL:", url)
        # print("DEBUG PARAMS:", params)

        if not any("Time Series" in key for key in data.keys()):
            raise Exception(f"Alpha Vantage error: {data}")

        time_series_key = next(k for k in data if k.startswith("Time Series"))
        series = data[time_series_key]

        for timestamp in sorted(series):
            record = series[timestamp]
            t = self._parse_time(timestamp)
            open_p = float(record["1. open"])
            high = float(record["2. high"])
            low = float(record["3. low"])
            close = float(record["4. close"])
            volume = float(record.get("5. volume", 0))
            item = {
                DATA_FIELD.FIELD_TIME: t,
                DATA_FIELD.FIELD_OPEN: open_p,
                DATA_FIELD.FIELD_HIGH: high,
                DATA_FIELD.FIELD_LOW: low,
                DATA_FIELD.FIELD_CLOSE: close,
                DATA_FIELD.FIELD_VOLUME: volume,
            }
            yield CKLine_Unit(item)

    def SetBasciInfo(self):
        # Alpha Vantage does not provide symbol metadata in free API, stub:
        self.name = self.code
        self.is_stock = True

    @classmethod
    def do_init(cls):
        pass  # Not required

    @classmethod
    def do_close(cls):
        pass  # Not required

    def _get_function_and_interval(self):
        if self.k_type == KL_TYPE.K_DAY:
            return "TIME_SERIES_DAILY", None
        elif self.k_type == KL_TYPE.K_60M:
            return "TIME_SERIES_INTRADAY", "60min"
        elif self.k_type == KL_TYPE.K_30M:
            return "TIME_SERIES_INTRADAY", "30min"
        elif self.k_type == KL_TYPE.K_15M:
            return "TIME_SERIES_INTRADAY", "15min"
        elif self.k_type == KL_TYPE.K_5M:
            return "TIME_SERIES_INTRADAY", "5min"
        else:
            raise Exception(f"k_type {self.k_type} not supported for Alpha Vantage.")

    def _parse_time(self, time_str: str) -> CTime:
        # Supports both daily and intraday timestamps
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%d")
        except ValueError:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute)