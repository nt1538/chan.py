# chan_streamer.py

"""
ChanStreamer - streaming-style Chan wrapper built on sliding window logic.

- Loads all K-lines once from your data source.
- Maintains a sliding buffer of up to max_klines.
- For each new bar:
    - Builds a fresh CChan on the current window
    - Extracts BSPs from that window
    - Deduplicates them globally by (timestamp, bsp_type)
    - Returns ONLY the BSPs that are NEW at this bar
"""

from collections import deque
from typing import List, Dict, Tuple, Iterator, Optional

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE, DATA_SRC
from KLine.KLine_Unit import CKLine_Unit
from BuySellPoint.BS_Point import CBS_Point


class ChanStreamer:
    def __init__(
        self,
        code: str,
        begin_time: str,
        end_time: str,
        data_src: DATA_SRC,
        lv: KL_TYPE,
        config: CChanConfig,
        autype=None,
        max_klines: int = 500,
    ):
        """
        Args:
            code: symbol (e.g. 'SPY')
            begin_time: start date string
            end_time: end date string
            data_src: DATA_SRC enum (CSV / Yahoo / etc.)
            lv: KL_TYPE (e.g. KL_TYPE.K_5M)
            config: CChanConfig instance
            autype: AUTYPE if needed
            max_klines: sliding window size (e.g., 500)
        """
        self.code = code
        self.begin_time = begin_time
        self.end_time = end_time
        self.data_src = data_src
        self.lv = lv
        self.lv_list = [lv]
        self.config = config
        self.autype = autype

        self.max_klines = max_klines
        self.kline_window: deque = deque(maxlen=max_klines)

        # Full K-line history from source
        self.all_klines: List[CKLine_Unit] = []

        # Global BSP store: (timestamp, bsp_type) -> snapshot dict
        self.all_historical_bsp: Dict[Tuple[str, str], Dict] = {}

        # Progress stats
        self.snapshot_count = 0
        self.total_windows_processed = 0
        self.current_window_start = 0

        # last window chan (optional, for compatibility if you need kl_datas)
        self.last_chan: Optional[CChan] = None

    # ------------------------------------------------------------------
    # Data source helpers
    # ------------------------------------------------------------------
    def _get_stock_api(self):
        if self.data_src == DATA_SRC.YAHOO_FINANCE:
            from DataAPI.YahooFinanceAPI import CYahooFinance
            return CYahooFinance
        elif self.data_src == DATA_SRC.ALPHA_VANTAGE:
            from DataAPI.AlphaVantageAPI import CAlphaVantage
            return CAlphaVantage
        elif self.data_src == DATA_SRC.BAO_STOCK:
            from DataAPI.BaoStockAPI import CBaoStock
            return CBaoStock
        elif self.data_src == DATA_SRC.CSV:
            from DataAPI.csvAPI import CSV_API
            return CSV_API
        else:
            raise ValueError(f"Unsupported data source: {self.data_src}")

    def _load_all_klines(self) -> List[CKLine_Unit]:
        """
        Load ALL K-lines from the data source once, for the chosen lv.
        """
        print("[ChanStreamer] Loading K-lines from data source...")
        stockapi_cls = self._get_stock_api()
        stockapi_cls.do_init()

        try:
            all_klines: List[CKLine_Unit] = []
            stockapi = stockapi_cls(
                code=self.code,
                k_type=self.lv,
                begin_date=self.begin_time,
                end_date=self.end_time,
                autype=self.autype,
            )

            for idx, klu in enumerate(stockapi.get_kl_data()):
                klu.kl_type = self.lv
                klu.set_idx(idx)        # set index in full series
                all_klines.append(klu)

            # Ensure strictly sorted by time
            all_klines.sort(key=lambda x: x.time)

            print(f"[ChanStreamer] Loaded {len(all_klines)} K-lines.")
            return all_klines
        finally:
            stockapi_cls.do_close()

    # ------------------------------------------------------------------
    # Core streaming interface
    # ------------------------------------------------------------------
    def stream_from_source(self) -> Iterator[Tuple[int, CKLine_Unit, List[Dict]]]:
        """
        Main generator:

        Yields: (global_idx, klu, new_bsp_list)

        where:
        - global_idx : index of this K-line in the full series
        - klu        : CKLine_Unit for this bar
        - new_bsp_list: list of NEW BSP snapshot dicts discovered after this bar
        """
        # 1) Load all K-lines once
        self.all_klines = self._load_all_klines()
        if not self.all_klines:
            print("[ChanStreamer] No K-lines loaded, stream ends.")
            return

        print(
            f"[ChanStreamer] Starting streaming with sliding window = {self.max_klines}, "
            f"total K-lines = {len(self.all_klines)}"
        )

        # 2) Stream each bar
        for global_idx, klu in enumerate(self.all_klines):
            self.snapshot_count += 1

            # append to sliding window
            self.kline_window.append(klu)

            if len(self.kline_window) >= self.max_klines:
                self.current_window_start = global_idx - self.max_klines + 1
            else:
                self.current_window_start = 0

            # Build a fresh Chan instance on the current window
            window_chan = self._create_window_chan()
            window_list = list(self.kline_window)
            # This is safe: within this call, times are strictly increasing
            window_chan.trigger_load({self.lv: window_list})

            # Extract BSPs from this window, get only NEW ones
            new_bsp_list = self._extract_window_bsp(window_chan, self.snapshot_count)

            # update stats
            if len(self.kline_window) == self.max_klines:
                self.total_windows_processed += 1

            self.last_chan = window_chan

            # yield this bar + new BSPs
            yield global_idx, klu, new_bsp_list

    def _create_window_chan(self) -> CChan:
        """
        Create a fresh CChan instance for the window.
        """
        return CChan(
            code=self.code,
            begin_time=None,
            end_time=None,
            data_src=self.data_src,
            lv_list=self.lv_list,
            config=self.config,
            autype=self.autype,
        )

    # ------------------------------------------------------------------
    # BSP extraction & snapshot helpers
    # ------------------------------------------------------------------
    def _extract_window_bsp(self, chan: CChan, snapshot_idx: int) -> List[Dict]:
        """
        Extract BSPs from the window's Chan and update global history.

        Returns a list of NEW BSP snapshots discovered in this snapshot.
        """
        new_bsp_list: List[Dict] = []

        try:
            bsp_list = chan.kl_datas[self.lv].bs_point_lst.getSortedBspList()
        except Exception as e:
            print(f"[ChanStreamer] Warning: error getting BSP list at snapshot {snapshot_idx}: {e}")
            return new_bsp_list

        for bsp in bsp_list:
            klu = bsp.klu
            timestamp = str(klu.time)

            for bs_type in bsp.type:
                bs_type_str = bs_type.value
                key = (timestamp, bs_type_str)

                # Build snapshot
                snapshot = self._create_bsp_snapshot(
                    bsp=bsp,
                    bs_type_str=bs_type_str,
                    snapshot_idx=snapshot_idx,
                )

                # If already seen, just update last_seen; else new BSP
                if key in self.all_historical_bsp:
                    snapshot["snapshot_first_seen"] = self.all_historical_bsp[key]["snapshot_first_seen"]
                    snapshot["snapshot_last_seen"] = snapshot_idx
                else:
                    snapshot["snapshot_first_seen"] = snapshot_idx
                    snapshot["snapshot_last_seen"] = snapshot_idx
                    # this is a brand new BSP
                    new_bsp_list.append(snapshot)

                self.all_historical_bsp[key] = snapshot

        return new_bsp_list

    def _create_bsp_snapshot(self, bsp: CBS_Point, bs_type_str: str, snapshot_idx: int) -> Dict:
        klu = bsp.klu

        def safe_get(obj, attr, default=0.0):
            try:
                val = getattr(obj, attr, default)
                return default if val is None else val
            except Exception:
                return default

        # original index in full list
        original_idx = self._find_original_klu_idx(klu)

        snap: Dict = {
            "klu_idx": original_idx,
            "timestamp": str(klu.time),
            "klu_open": safe_get(klu, "open"),
            "klu_high": safe_get(klu, "high"),
            "klu_low": safe_get(klu, "low"),
            "klu_close": safe_get(klu, "close"),
            "klu_volume": safe_get(klu, "volume", 0),
            "bsp_type": bs_type_str,
            "bsp_types": bsp.type2str(),
            "is_buy": int(bsp.is_buy),
            "direction": "buy" if bsp.is_buy else "sell",
            "is_segbsp": safe_get(bsp, "is_segbsp", False),
            "snapshot_idx": snapshot_idx,
        }

        # BSP features
        if getattr(bsp, "features", None) is not None:
            try:
                feat_dict = bsp.features.to_dict()
                for k, v in feat_dict.items():
                    if k == "next_bi_return":
                        continue
                    snap[f"feat_{k}"] = 0.0 if v is None else v
            except Exception as e:
                print(f"[ChanStreamer] Warning: error extracting BSP features: {e}")

        # technical indicators from K-line
        self._add_klu_indicators(snap, klu)
        return snap

    def _find_original_klu_idx(self, klu: CKLine_Unit) -> int:
        ts = str(klu.time)
        for idx, orig in enumerate(self.all_klines):
            if str(orig.time) == ts:
                return idx
        # fallback: use whatever idx is on the object
        return getattr(klu, "idx", -1)

    def _add_klu_indicators(self, snap: Dict, klu: CKLine_Unit):
        def safe_get(obj, attr, default=0.0):
            try:
                val = getattr(obj, attr, default)
                return default if val is None else val
            except Exception:
                return default

        # MACD
        if hasattr(klu, "macd") and klu.macd:
            snap["macd_value"] = safe_get(klu.macd, "macd", 0.0)
            snap["macd_dif"] = safe_get(klu.macd, "DIF", 0.0)
            snap["macd_dea"] = safe_get(klu.macd, "DEA", 0.0)
        else:
            snap["macd_value"] = 0.0
            snap["macd_dif"] = 0.0
            snap["macd_dea"] = 0.0

        # RSI
        snap["rsi"] = safe_get(klu, "rsi", 50.0) if hasattr(klu, "rsi") else 50.0

        # KDJ
        if hasattr(klu, "kdj") and klu.kdj:
            snap["kdj_k"] = safe_get(klu.kdj, "k", 50.0)
            snap["kdj_d"] = safe_get(klu.kdj, "d", 50.0)
            snap["kdj_j"] = safe_get(klu.kdj, "j", 50.0)
        else:
            snap["kdj_k"] = 50.0
            snap["kdj_d"] = 50.0
            snap["kdj_j"] = 50.0

        # DMI
        if hasattr(klu, "dmi") and klu.dmi:
            snap["dmi_plus"] = safe_get(klu.dmi, "plus_di", 25.0)
            snap["dmi_minus"] = safe_get(klu.dmi, "minus_di", 25.0)
            snap["dmi_adx"] = safe_get(klu.dmi, "adx", 25.0)
        else:
            snap["dmi_plus"] = 25.0
            snap["dmi_minus"] = 25.0
            snap["dmi_adx"] = 25.0

        # Price action
        open_val = klu.open if klu.open else klu.close
        close_val = klu.close
        high_val = klu.high if klu.high else klu.close
        low_val = klu.low if klu.low else klu.close

        snap["price_change_pct"] = (
            (close_val - open_val) / open_val * 100 if open_val != 0 else 0.0
        )
        snap["high_low_spread_pct"] = (
            (high_val - low_val) / low_val * 100 if low_val != 0 else 0.0
        )
        snap["upper_shadow"] = high_val - max(open_val, close_val)
        snap["lower_shadow"] = min(open_val, close_val) - low_val
        snap["body_size"] = abs(close_val - open_val)
        snap["is_bullish_candle"] = 1 if close_val > open_val else 0

    # ------------------------------------------------------------------
    # Public helpers used by your ML code
    # ------------------------------------------------------------------
    def get_all_historical_bsp(self) -> List[Dict]:
        """
        Return all unique BSP snapshots, sorted by klu_idx.
        """
        lst = list(self.all_historical_bsp.values())
        lst.sort(key=lambda x: x["klu_idx"])
        return lst

    def get_stats(self) -> Dict:
        return {
            "total_klines_loaded": len(self.all_klines),
            "snapshots_generated": self.snapshot_count,
            "windows_processed": self.total_windows_processed,
            "unique_bsp_count": len(self.all_historical_bsp),
            "buffer_size": len(self.kline_window),
            "window_start_idx": self.current_window_start,
        }

    @property
    def kl_datas(self):
        """
        For compatibility if you want to access last_chan.kl_datas.
        """
        if self.last_chan:
            return self.last_chan.kl_datas
        return {}
