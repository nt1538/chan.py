# utils/alpha158_calculator.py

import pandas as pd
import numpy as np
from qlib.qlib.contrib.data.handler import Alpha158
from qlib.qlib.data.dataset.handler import ExpressionParser


class Alpha158Calculator:
    def __init__(self):
        self.parser = ExpressionParser()

    def calculate_features(self, df: pd.DataFrame, window: int = 30) -> dict:
        """
        Calculate alpha158 features on the given DataFrame.
        :param df: DataFrame with ['open', 'high', 'low', 'close', 'volume'] columns
        :param window: how many rows to use from the tail
        :return: dict of alpha feature values
        """
        df = df.rename(columns=str.lower)
        df = df.tail(window).copy()
        result = {}

        for name, expr in alpha158.items():
            try:
                series = self.parser(expr).calc(df)
                result[name] = series.iloc[-1] if not series.empty else np.nan
            except Exception as e:
                print(f"[Alpha158 Warning] {name} failed: {e}")
                result[name] = np.nan

        return result
