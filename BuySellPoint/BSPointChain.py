from typing import List, Optional
from .BS_Point import CBS_Point

class CBSPointChain:
    def __init__(self, is_buy: bool):
        self.is_buy = is_buy
        self.points: List[CBS_Point] = []
        self.mature_bsp: Optional[CBS_Point] = None
        self.last_opposite_mature_price: Optional[float] = None
        self.post_mature_points: List[CBS_Point] = []

    def add_point(self, bsp: CBS_Point):
        self.points.append(bsp)

    def mark_mature(self, mature_bsp: CBS_Point, last_opposite_price: float):
        self.mature_bsp = mature_bsp
        self.last_opposite_mature_price = last_opposite_price
        mature_price = mature_bsp.klu.close()

        for p in self.points:
            price = p.klu.close()
            denom = abs(mature_price - last_opposite_price) + 1e-6
            rate = abs(price - last_opposite_price) / denom
            p.mature_rate = min(max(rate, 0.0), 1.0)
            p.is_mature_point = (p == mature_bsp)

        self.post_mature_points = []

    def add_post_mature_point(self, bsp: CBS_Point):
        if self.mature_bsp:
            self.post_mature_points.append(bsp)
            price = bsp.klu.close()
            last_opposite_price = self.last_opposite_mature_price
            mature_price = self.mature_bsp.klu.close()
            denom = abs(mature_price - last_opposite_price) + 1e-6
            rate = abs(price - last_opposite_price) / denom
            bsp.mature_rate = rate
            bsp.is_post_mature = True


class CBSPointChainManager:
    def __init__(self):
        self.buy_chain = CBSPointChain(is_buy=True)
        self.sell_chain = CBSPointChain(is_buy=False)
        self.last_mature_buy: Optional[CBS_Point] = None
        self.last_mature_sell: Optional[CBS_Point] = None

    def add_bsp(self, bsp: CBS_Point):
        if bsp.is_buy:
            self.buy_chain.add_point(bsp)
        else:
            self.sell_chain.add_point(bsp)

    def confirm_mature(self, bsp: CBS_Point):
        if bsp.is_buy:
            last_opposite = self.last_mature_sell
            last_price = last_opposite.klu.close() if last_opposite else bsp.klu.close() * 0.99
            self.buy_chain.mark_mature(bsp, last_price)
            self.last_mature_buy = bsp
        else:
            last_opposite = self.last_mature_buy
            last_price = last_opposite.klu.close() if last_opposite else bsp.klu.close() * 1.01
            self.sell_chain.mark_mature(bsp, last_price)
            self.last_mature_sell = bsp

    def handle_post_mature(self, bsp: CBS_Point):
        if bsp.is_buy and self.buy_chain.mature_bsp:
            self.buy_chain.add_post_mature_point(bsp)
        elif not bsp.is_buy and self.sell_chain.mature_bsp:
            self.sell_chain.add_post_mature_point(bsp)