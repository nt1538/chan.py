import math

class HPI:
    def __init__(self, cycle=21, multiplier=1.0):
        self.cycle = cycle
        self.multiplier = multiplier
        self.values = []

    def add(self, close: float, open_: float, volume: float, open_interest: float) -> float:
        """
        Herrick Payoff Index 计算公式（适用于期货或带持仓量的数据）：
        HPI = Volume × Multiplier × (2 × Close - Open) × ΔOpenInterest
        """
        if not hasattr(self, "last_open_interest"):
            self.last_open_interest = open_interest
            self.values.append(0)
            return 0

        delta_oi = open_interest - self.last_open_interest
        self.last_open_interest = open_interest

        payoff = volume * self.multiplier * (2 * close - open_) * delta_oi
        self.values.append(payoff)
        return payoff