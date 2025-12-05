"""
Sliding Window Chan System - TRUE sliding window implementation
Limits Chan calculation to max_klines (e.g., 500) for faster processing
Preserves ALL BSP points from all windows
"""

import copy
from typing import List, Optional, Dict, Iterator, Set, Tuple
from collections import deque
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE
from KLine.KLine_Unit import CKLine_Unit
from BuySellPoint.BS_Point import CBS_Point


class SlidingWindowChan:
    """
    Sliding window Chan system that processes data in fixed-size windows.
    Each window runs Chan calculation on only max_klines K-lines.
    Much faster than processing entire dataset at once.

    ✅ NEW:
      - process_new_kline(klu) for realtime/streaming:
          - feed one K-line
          - get back new BSPs created by this snapshot
    """
    
    def __init__(
        self,
        code: str,
        begin_time=None,
        end_time=None,
        data_src=None,
        lv_list=None,
        config=None,
        autype=None,
        max_klines: int = 500,
    ):
        """
        Initialize sliding window Chan system.
        
        Args:
            max_klines: Window size (e.g., 500 K-lines per calculation)
            All other args: Same as CChan
        """
        self.code = code
        self.begin_time = begin_time
        self.end_time = end_time
        self.data_src = data_src
        self.lv_list = lv_list if lv_list else [KL_TYPE.K_DAY]
        self.config = config if config else CChanConfig()
        self.autype = autype
        
        self.max_klines = max_klines
        self.sliding_enabled = max_klines > 0
        
        # Historical storage
        # key: (timestamp, bsp_type) -> data
        self.all_historical_bsp: Dict[Tuple, Dict] = {}
        # All K-lines in time order (used both for batch + streaming)
        self.all_klines: List[CKLine_Unit] = []
        self.kline_window: deque = deque(maxlen=max_klines)  # Sliding window
        
        # Progress tracking
        self.total_windows_processed = 0
        self.snapshot_count = 0
        self.current_window_start = 0  # Track window position
        
        # Store last Chan instance for compatibility
        self.last_chan: Optional[CChan] = None

    # ----------------------------------------------------------------------
    #  Internal helpers
    # ----------------------------------------------------------------------
    
    def _load_all_klines(self) -> List[CKLine_Unit]:
        """
        Load all K-lines from data source once.
        Returns list of all K-lines.
        """
        print(f"[📥] Loading K-lines from data source...")
        
        stockapi_cls = self._get_stock_api()
        stockapi_cls.do_init()
        
        try:
            all_klines = []
            for lv in self.lv_list:
                stockapi = stockapi_cls(
                    code=self.code,
                    k_type=lv,
                    begin_date=self.begin_time,
                    end_date=self.end_time,
                    autype=self.autype
                )
                
                for idx, klu in enumerate(stockapi.get_kl_data()):
                    klu.kl_type = lv
                    klu.set_idx(idx)  # IMPORTANT: set original index here
                    all_klines.append(klu)
            
            print(f"[✅] Loaded {len(all_klines)} K-lines from source")
            return all_klines
            
        finally:
            stockapi_cls.do_close()
    
    def _get_stock_api(self):
        """Get appropriate stock API class."""
        from Common.CEnum import DATA_SRC
        
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

    def _create_window_chan(self) -> CChan:
        """
        Create a fresh Chan instance for the current window.
        Each window is independent, keeping calculations fast.
        """
        chan = CChan(
            code=self.code,
            begin_time=None,
            end_time=None,
            data_src=self.data_src,
            lv_list=self.lv_list,
            config=self.config,
            autype=self.autype
        )
        return chan

    def _find_original_klu_idx(self, klu: CKLine_Unit) -> int:
        """
        Find the original index of K-line in the full dataset.

        👉 NOW we simply rely on klu.idx, which we set consistently
        in both batch (step_load) and streaming (process_new_kline).
        """
        return getattr(klu, "idx", 0)

    def _add_klu_indicators(self, snapshot: Dict, klu: CKLine_Unit):
        """Add technical indicators from K-line to snapshot."""
        def safe_get(obj, attr, default=0.0):
            try:
                val = getattr(obj, attr, default)
                return default if val is None else val
            except:
                return default
        
        # MACD
        if hasattr(klu, 'macd') and klu.macd:
            snapshot['macd_value'] = safe_get(klu.macd, 'macd', 0.0)
            snapshot['macd_dif'] = safe_get(klu.macd, 'DIF', 0.0)
            snapshot['macd_dea'] = safe_get(klu.macd, 'DEA', 0.0)
        else:
            snapshot['macd_value'] = 0.0
            snapshot['macd_dif'] = 0.0
            snapshot['macd_dea'] = 0.0
        
        # RSI
        snapshot['rsi'] = safe_get(klu, 'rsi', 50.0) if hasattr(klu, 'rsi') else 50.0
        
        # KDJ
        if hasattr(klu, 'kdj') and klu.kdj:
            snapshot['kdj_k'] = safe_get(klu.kdj, 'k', 50.0)
            snapshot['kdj_d'] = safe_get(klu.kdj, 'd', 50.0)
            snapshot['kdj_j'] = safe_get(klu.kdj, 'j', 50.0)
        else:
            snapshot['kdj_k'] = 50.0
            snapshot['kdj_d'] = 50.0
            snapshot['kdj_j'] = 50.0
        
        # DMI
        if hasattr(klu, 'dmi') and klu.dmi:
            snapshot['dmi_plus'] = safe_get(klu.dmi, 'plus_di', 25.0)
            snapshot['dmi_minus'] = safe_get(klu.dmi, 'minus_di', 25.0)
            snapshot['dmi_adx'] = safe_get(klu.dmi, 'adx', 25.0)
        else:
            snapshot['dmi_plus'] = 25.0
            snapshot['dmi_minus'] = 25.0
            snapshot['dmi_adx'] = 25.0
        
        # Price action (no future info; only current bar)
        open_val = klu.open if klu.open else klu.close
        close_val = klu.close
        high_val = klu.high if klu.high else klu.close
        low_val = klu.low if klu.low else klu.close
        
        snapshot['price_change_pct'] = (close_val - open_val) / open_val * 100 if open_val != 0 else 0.0
        snapshot['high_low_spread_pct'] = (high_val - low_val) / low_val * 100 if low_val != 0 else 0.0
        snapshot['upper_shadow'] = high_val - max(open_val, close_val)
        snapshot['lower_shadow'] = min(open_val, close_val) - low_val
        snapshot['body_size'] = abs(close_val - open_val)
        snapshot['is_bullish_candle'] = 1 if close_val > open_val else 0

    def _create_bsp_snapshot(self, bsp: CBS_Point, bs_type_str: str, snapshot_idx: int) -> Dict:
        """Create a complete snapshot of BSP with all features."""
        klu = bsp.klu
        
        def safe_get(obj, attr, default=0.0):
            try:
                val = getattr(obj, attr, default)
                return default if val is None else val
            except:
                return default
        
        # Original index in full dataset (we now trust klu.idx)
        original_klu_idx = self._find_original_klu_idx(klu)
        
        snapshot = {
            'klu_idx': original_klu_idx,  # Original position in full dataset
            'timestamp': str(klu.time),
            'klu_open': safe_get(klu, 'open'),
            'klu_high': safe_get(klu, 'high'),
            'klu_low': safe_get(klu, 'low'),
            'klu_close': safe_get(klu, 'close'),
            'klu_volume': safe_get(klu, 'volume', 0),
            'bsp_type': bs_type_str,
            'bsp_types': bsp.type2str(),
            'is_buy': int(bsp.is_buy),
            'direction': 'buy' if bsp.is_buy else 'sell',
            'is_segbsp': safe_get(bsp, 'is_segbsp', False),
        }
        
        # Add features from BSP
        if bsp.features:
            try:
                features = bsp.features.to_dict()
                for key, value in features.items():
                    if key == 'next_bi_return':
                        continue
                    snapshot[f'feat_{key}'] = 0.0 if value is None else value
            except Exception as e:
                print(f"[Warning] Error extracting features: {e}")
        
        # Add technical indicators
        self._add_klu_indicators(snapshot, klu)
        
        return snapshot

    def _extract_window_bsp(self, chan: CChan, snapshot_idx: int) -> List[Dict]:
        """
        Extract BSP from current window and store in history.

        ✅ NOW returns a list of *new* BSP snapshots created in this call.
        """
        new_bsp_list: List[Dict] = []
        try:
            lv = self.lv_list[0]
            bsp_list = chan.kl_datas[lv].bs_point_lst.getSortedBspList()
            
            for bsp in bsp_list:
                klu = bsp.klu
                timestamp = str(klu.time)
                
                # Process each type in the BSP
                for bs_type in bsp.type:
                    bs_type_str = bs_type.value
                    
                    # Unique key: timestamp + bs_type
                    bsp_key = (timestamp, bs_type_str)
                    
                    # Create snapshot
                    bsp_snapshot = self._create_bsp_snapshot(
                        bsp=bsp,
                        bs_type_str=bs_type_str,
                        snapshot_idx=snapshot_idx
                    )
                    
                    if bsp_key in self.all_historical_bsp:
                        # Existing BSP -> update last_seen, keep first_seen
                        bsp_snapshot['snapshot_first_seen'] = \
                            self.all_historical_bsp[bsp_key]['snapshot_first_seen']
                        bsp_snapshot['snapshot_last_seen'] = snapshot_idx
                    else:
                        # NEW BSP
                        bsp_snapshot['snapshot_first_seen'] = snapshot_idx
                        bsp_snapshot['snapshot_last_seen'] = snapshot_idx
                        new_bsp_list.append(bsp_snapshot)
                    
                    # Store/update in global dict
                    self.all_historical_bsp[bsp_key] = bsp_snapshot
            
        except Exception as e:
            print(f"[Warning] Error extracting BSP from window {snapshot_idx}: {e}")
        
        return new_bsp_list

    def _process_single_kline(self, klu: CKLine_Unit, klu_idx: int):
        """
        Core logic for ONE new K-line:
          - update window
          - run Chan on current window
          - extract BSP
          - return (window_chan, new_bsp_list)
        """
        self.snapshot_count += 1

        # append to full history & window
        self.all_klines.append(klu)
        self.kline_window.append(klu)

        # ensure its idx is correct (for BSP snapshots)
        klu.set_idx(klu_idx)

        # Update window start position
        if len(self.kline_window) >= self.max_klines:
            self.current_window_start = klu_idx - self.max_klines + 1
        else:
            self.current_window_start = 0
        
        # Create Chan instance for current window
        window_chan = self._create_window_chan()
        window_klines = list(self.kline_window)
        window_chan.trigger_load({self.lv_list[0]: window_klines})
        
        # Extract BSP from this window
        new_bsp_list = self._extract_window_bsp(window_chan, self.snapshot_count)
        
        # Track window count
        if len(self.kline_window) == self.max_klines:
            self.total_windows_processed += 1
        
        # Store last chan for compatibility
        self.last_chan = window_chan
        
        return window_chan, new_bsp_list

    # ----------------------------------------------------------------------
    #  PUBLIC: batch mode (same as before)
    # ----------------------------------------------------------------------
    
    def step_load(self) -> Iterator['CChan']:
        """
        Sliding window step load.
        Processes data in windows of max_klines, yielding a snapshot after each K-line.
        Much faster than processing entire dataset.

        ✅ SAME external behavior as before.
        """
        # Load all K-lines once
        self.all_klines = self._load_all_klines()
        
        if not self.all_klines:
            print("[⚠️] No K-lines loaded")
            return
        
        print(f"[🪟] Starting sliding window processing (window size: {self.max_klines})")
        print(f"[📊] Total K-lines to process: {len(self.all_klines)}")
        
        # Process each K-line
        for klu_idx, klu in enumerate(self.all_klines):
            window_chan, _ = self._process_single_kline(klu, klu_idx)
            yield window_chan

    # ----------------------------------------------------------------------
    #  ✅ NEW: streaming mode (one kline at a time)
    # ----------------------------------------------------------------------
    
    def process_new_kline(self, klu: CKLine_Unit):
        """
        Streaming / realtime usage:
        ---------------------------------
        Feed ONE new K-line (continuous from previous data).

        Returns:
            (window_chan, new_bsp_list)

        - window_chan: CChan instance built on the current window
        - new_bsp_list: list of NEW BSP snapshots discovered at this step

        If len(new_bsp_list) == 0 → this bar did not create any new BSP
        (or only updated last_seen of old ones).
        """
        # next index in overall history
        klu_idx = len(self.all_klines)
        return self._process_single_kline(klu, klu_idx)
    
    # ----------------------------------------------------------------------
    #  OTHER HELPERS
    # ----------------------------------------------------------------------
    
    def get_all_historical_bsp(self) -> List[Dict]:
        """
        Get ALL historical BSP points collected during step_load or streaming.
        Each BSP appears only once.
        Sorted by klu_idx for temporal order.
        """
        bsp_list = list(self.all_historical_bsp.values())
        bsp_list.sort(key=lambda x: x['klu_idx'])
        return bsp_list
    
    def get_stats(self) -> Dict:
        """Get statistics about the sliding window system."""
        return {
            'sliding_enabled': self.sliding_enabled,
            'max_klines': self.max_klines,
            'window_start_idx': self.current_window_start,
            'total_klines_loaded': len(self.all_klines),
            'windows_processed': self.total_windows_processed,
            'snapshots_generated': self.snapshot_count,
            'unique_bsp_count': len(self.all_historical_bsp),
            'total_historical_bsp': len(self.all_historical_bsp),
            'current_window_size': len(self.kline_window),
        }
    
    def export_historical_bsp_to_list(self) -> List[Dict]:
        """
        Export all historical BSP as list of dicts.
        Compatible with your existing dataset generation code.
        """
        return self.get_all_historical_bsp()
    
    # Compatibility properties for existing code
    @property
    def kl_datas(self):
        """Provide access to last Chan's kl_datas for compatibility."""
        if self.last_chan:
            return self.last_chan.kl_datas
        return {}


# Example standalone test
if __name__ == "__main__":
    from Common.CEnum import DATA_SRC, AUTYPE
    
    config = CChanConfig({
        "cal_kdj": True,
        "cal_dmi": True,
        "cal_rsi": True,
        "bi_strict": True,
        "trigger_step": False,
        "bs_type": '1,2,3a,1p,2s,3b',
    })
    
    # Batch mode test (same as before)
    chan = SlidingWindowChan(
        code="^GSPC",
        begin_time="2024-01-01",
        end_time="2024-03-31",
        data_src=DATA_SRC.CSV,
        lv_list=[KL_TYPE.K_5M],
        config=config,
        autype=AUTYPE.QFQ,
        max_klines=500
    )
    
    for snapshot_idx, snapshot in enumerate(chan.step_load()):
        if snapshot_idx % 100 == 0:
            stats = chan.get_stats()
            print(f"[📊] Snapshot {snapshot_idx}: {stats['unique_bsp_count']} unique BSP")
    
    all_bsp = chan.get_all_historical_bsp()
    print(f"\n[✅] Batch processing complete!")
    print(f"[📊] Total unique BSP: {len(all_bsp)}")
    
    stats = chan.get_stats()
    print(f"[📈] Stats: {stats}")
