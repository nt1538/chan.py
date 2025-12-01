# streaming_labels.py

from typing import List, Dict
from collections import defaultdict

class StreamingBest24hLabeler:
    """
    Incremental version of "BEST reverse BSP within time_window_bars".

    - You append new BSP snapshots as they appear.
    - On each new bar (with klu_idx = current_klu_idx), call update_labels_until(current_klu_idx).
    - This will attempt to label any BSP whose 24h window is fully in the past:
        entry_idx + time_window_bars <= current_klu_idx

    Each BSP dict is expected to have:
        'klu_idx'  : int
        'is_buy'   : 1 for buy, 0 for sell
        'klu_close': float
    """

    def __init__(self, time_window_bars: int):
        self.time_window_bars = time_window_bars

        # All BSP dicts in time order
        self.bsp_records: List[Dict] = []

        # indices of BSPs that still need labels
        self.unlabeled_indices: List[int] = []

        # fast lookup: klu_idx -> list of bsp indices at that bar
        self.bsp_by_klu_idx = defaultdict(list)

        # the max klu_idx we’ve seen so far in the stream
        self.max_seen_klu_idx = -1

    def add_new_bsp_list(self, bsp_list: List[Dict]):
        """
        Append newly discovered BSPs (from the latest Chan window snapshot).
        """
        for bsp in bsp_list:
            # ensure label fields exist
            bsp.setdefault("profit_target_pct", None)
            bsp.setdefault("profit_target_distance", None)
            bsp.setdefault("has_profit_target", 0)
            bsp.setdefault("exit_type", None)
            bsp.setdefault("exit_klu_idx", None)
            bsp.setdefault("exit_price", None)

            idx = len(self.bsp_records)
            self.bsp_records.append(bsp)
            self.unlabeled_indices.append(idx)

            k_idx = int(bsp["klu_idx"])
            self.bsp_by_klu_idx[k_idx].append(idx)

            if k_idx > self.max_seen_klu_idx:
                self.max_seen_klu_idx = k_idx

    def update_labels_until(self, current_klu_idx: int):
        """
        Try to label any BSP whose 24h window is fully inside [entry_idx, current_klu_idx].
        That is, for entry_idx we need:
            entry_idx + time_window_bars <= current_klu_idx
        """
        if not self.unlabeled_indices:
            return

        new_unlabeled = []
        for idx in self.unlabeled_indices:
            bsp = self.bsp_records[idx]
            entry_idx = int(bsp["klu_idx"])
            direction = int(bsp["is_buy"])
            entry_price = float(bsp["klu_close"])

            # if we haven't reached the end of its 24h window, skip for now
            if entry_idx + self.time_window_bars > current_klu_idx:
                new_unlabeled.append(idx)
                continue

            # we have enough future bars to evaluate BEST reverse in [entry_idx+1, entry_idx+time_window]
            window_start = entry_idx + 1
            window_end = entry_idx + self.time_window_bars

            candidates = []
            for k in range(window_start, window_end + 1):
                for j in self.bsp_by_klu_idx.get(k, []):
                    future_bsp = self.bsp_records[j]
                    if int(future_bsp["is_buy"]) != direction:
                        candidates.append(future_bsp)

            if not candidates:
                # no reverse signal in the window -> keep unlabeled (or set has_profit_target = 0)
                # Here we decide to *not* use this BSP for training
                bsp["has_profit_target"] = 0
                bsp["profit_target_pct"] = None
                continue

            # pick BEST candidate in the window
            if direction == 1:  # BUY -> pick max close
                best_exit = max(candidates, key=lambda x: float(x["klu_close"]))
            else:               # SELL -> pick min close
                best_exit = min(candidates, key=lambda x: float(x["klu_close"]))

            exit_price = float(best_exit["klu_close"])
            exit_idx = int(best_exit["klu_idx"])

            if direction == 1:
                profit_pct = (exit_price - entry_price) / entry_price * 100.0
            else:
                profit_pct = (entry_price - exit_price) / entry_price * 100.0

            bsp["profit_target_pct"] = profit_pct
            bsp["profit_target_distance"] = exit_idx - entry_idx
            bsp["has_profit_target"] = 1
            bsp["exit_type"] = "best_24h"
            bsp["exit_klu_idx"] = exit_idx
            bsp["exit_price"] = exit_price

        self.unlabeled_indices = new_unlabeled

    def get_labeled_bsp_df(self):
        """
        Convenience: return a pandas DataFrame of all BSPs (labeled and unlabeled).
        """
        import pandas as pd

        if not self.bsp_records:
            import pandas as pd
            return pd.DataFrame()

        df = pd.DataFrame(self.bsp_records)
        # ensure timestamp->datetime and date
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date
        return df
