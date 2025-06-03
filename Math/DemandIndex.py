from typing import List, Optional

class DemandIndex:
    def __init__(self, cycle=14):
        self.cycle = cycle
        self.close_buffer = []
        self.volume_buffer = []

    def add(self, close_val: float, volume_val: float) -> Optional[float]:
        self.close_buffer.append(close_val)
        self.volume_buffer.append(volume_val)

        if len(self.close_buffer) < self.cycle:
            return None
        elif len(self.close_buffer) > self.cycle:
            self.close_buffer.pop(0)
            self.volume_buffer.pop(0)

        # 计算 DI
        price_change = self.close_buffer[-1] - self.close_buffer[0]
        volume_sum = sum(self.volume_buffer)
        if volume_sum == 0:
            return 0

        return price_change * (self.volume_buffer[-1] / volume_sum)