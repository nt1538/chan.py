import pandas as pd


def load_5m_index(df_5m: pd.DataFrame, start_time: str, end_time: str):
    df = df_5m.copy()
    df = df[(df["timestamp"] >= pd.to_datetime(start_time)) & (df["timestamp"] <= pd.to_datetime(end_time))].reset_index(drop=True)
    df = df.rename(columns={"_open": "Open", "_high": "High", "_low": "Low", "_close": "Close", "_vol": "Volume"})
    df.columns = [str(column).strip() for column in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df["date"] = df["timestamp"].dt.date
    df["next_open"], df["next_close"] = df["Open"].shift(-1), df["Close"].shift(-1)
    day_close_map = df.groupby("date")["Close"].last().to_dict()
    return (df, df["next_open"].to_numpy(), df["next_close"].to_numpy(), df["Close"].to_numpy(float),
            df["High"].to_numpy(float), df["Low"].to_numpy(float), day_close_map, sorted(df["date"].unique()))


def compute_buy_hold_equity(day_close_map: dict, daily_dates: list, initial_capital: float) -> pd.Series:
    pairs = [(day, day_close_map.get(day)) for day in daily_dates]
    pairs = [(day, float(price)) for day, price in pairs if price is not None and not pd.isna(price)]
    if not pairs:
        return pd.Series(dtype=float)
    first = pairs[0][1]
    return pd.Series([initial_capital * price / first for _, price in pairs], index=pd.to_datetime([day for day, _ in pairs]))
