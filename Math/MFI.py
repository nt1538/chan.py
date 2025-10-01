# Math/MFI.py
class MFI:
    def __init__(self, period: int = 14):
        self.period = period
        self.tp_values = []  # Typical Price
        self.volumes = []
        self.money_flows = []
        self.prev_tp = None

    def add(self, high, low, close, volume):
        tp = (high + low + close) / 3
        
        if self.prev_tp is not None:
            raw_money_flow = tp * volume
            if tp > self.prev_tp:
                positive_flow = raw_money_flow
                negative_flow = 0
            elif tp < self.prev_tp:
                positive_flow = 0
                negative_flow = raw_money_flow
            else:
                positive_flow = 0
                negative_flow = 0
            
            self.money_flows.append({'positive': positive_flow, 'negative': negative_flow})
            
            if len(self.money_flows) > self.period:
                self.money_flows.pop(0)
        
        self.prev_tp = tp
        
        if len(self.money_flows) < self.period:
            return 50.0
        
        positive_sum = sum(mf['positive'] for mf in self.money_flows)
        negative_sum = sum(mf['negative'] for mf in self.money_flows)
        
        if negative_sum == 0:
            return 100.0
        
        money_ratio = positive_sum / negative_sum
        return 100 - (100 / (1 + money_ratio))