from typing import List

class CDMIItem:
    def __init__(self, plus_di, minus_di, adx):
        self.plus_di = plus_di
        self.minus_di = minus_di
        self.adx = adx


class CDMI:
    def __init__(self, period=14):
        self.period = period
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.dmi_info: List[CDMIItem] = []

    def add(self, high, low, close) -> CDMIItem:
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        if len(self.closes) <= self.period:
            self.dmi_info.append(CDMIItem(0, 0, 0))
            return self.dmi_info[-1]

        plus_dm = []
        minus_dm = []
        tr = []

        for i in range(1, self.period + 1):
            up_move = self.highs[-i] - self.highs[-i - 1]
            down_move = self.lows[-i - 1] - self.lows[-i]
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

            high_low = self.highs[-i] - self.lows[-i]
            high_close = abs(self.highs[-i] - self.closes[-i - 1])
            low_close = abs(self.lows[-i] - self.closes[-i - 1])
            tr.append(max(high_low, high_close, low_close))

        sum_tr = sum(tr)
        sum_plus_dm = sum(plus_dm)
        sum_minus_dm = sum(minus_dm)

        plus_di = 100 * (sum_plus_dm / sum_tr) if sum_tr != 0 else 0
        minus_di = 100 * (sum_minus_dm / sum_tr) if sum_tr != 0 else 0

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0

        prev_adx = self.dmi_info[-1].adx if self.dmi_info else dx
        adx = (prev_adx * (self.period - 1) + dx) / self.period

        self.dmi_info.append(CDMIItem(plus_di, minus_di, adx))
        return self.dmi_info[-1]