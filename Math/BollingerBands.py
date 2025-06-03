from typing import Optional, Tuple
import numpy as np

class BollingerBands:
    def __init__(self, cycle: int = 20, std_dev: float = 2.0):
        self.cycle = cycle
        self.std_dev = std_dev
        self.close_vals = []

    def add(self, close_val: float) -> Optional[Tuple[float, float, float]]:
        self.close_vals.append(close_val)

        if len(self.close_vals) < self.cycle:
            return None

        window = self.close_vals[-self.cycle:]
        mean = np.mean(window)
        std = np.std(window)

        upper = mean + self.std_dev * std
        lower = mean - self.std_dev * std
        middle = mean

        return upper, middle, lower