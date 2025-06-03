from typing import Optional

class ADLine:
    def __init__(self):
        self.values = []
        self.running_total = 0

    def add(self, close_val: float, open_val: float, volume_val: float) -> Optional[float]:
        if None in (close_val, open_val, volume_val):
            self.values.append(None)
            return None

        money_flow = (close_val - open_val) * volume_val
        self.running_total += money_flow
        self.values.append(self.running_total)
        return self.running_total