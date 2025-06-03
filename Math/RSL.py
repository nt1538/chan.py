from typing import List
import numpy as np

class RSL:
    def __init__(self, cycle=14):
        self.cycle = cycle
        self.values = []

    def add(self, close: List[float]) -> List[float]:
        """
        输入一个价格序列，返回对应的 RSL 值序列。
        """
        rsl_values = []
        for i in range(len(close)):
            if i < self.cycle:
                rsl_values.append(None)
            else:
                window = close[i - self.cycle + 1:i + 1]
                rsl = window[-1] / window[0] if window[0] != 0 else None
                rsl_values.append(rsl)
        self.values = rsl_values
        return rsl_values