from typing import Optional, List
import numpy as np

class RSL:
    def __init__(self, cycle=14):
        self.cycle = cycle
        self.cache = []

    def add(self, close_val: float) -> Optional[float]:
        self.cache.append(close_val)
        if len(self.cache) > self.cycle:
            self.cache.pop(0)

        if len(self.cache) < self.cycle:
            return None

        return self.cache[-1] / self.cache[0] if self.cache[0] != 0 else None