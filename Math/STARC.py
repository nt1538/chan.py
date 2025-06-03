class STARC:
    def __init__(self, cycle=20, k=2.0):
        self.cycle = cycle
        self.k = k
        self.high_list = []
        self.low_list = []
        self.close_list = []

    def add(self, high: float, low: float, close: float):
        self.high_list.append(high)
        self.low_list.append(low)
        self.close_list.append(close)

        if len(self.close_list) < self.cycle:
            return None

        import numpy as np
        ma = np.mean(self.close_list[-self.cycle:])
        atr = np.mean([h - l for h, l in zip(self.high_list[-self.cycle:], self.low_list[-self.cycle:])])
        upper = ma + self.k * atr
        lower = ma - self.k * atr
        return upper, ma, lower