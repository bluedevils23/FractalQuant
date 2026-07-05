from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = ["NoiseArea"]


class NoiseArea:
    def __init__(self, ohlcv: pd.DataFrame) -> None:
        self.ohlcv: pd.DataFrame = ohlcv
        self.pivot_frame: pd.DataFrame = pd.pivot_table(
            ohlcv,
            index="trade_time",
            columns="code",
            values=["close", "open", "volume"],
        )
        self.close: pd.DataFrame = self.pivot_frame["close"]
        self.open: pd.DataFrame = self.pivot_frame.at_time("09:30:00")["open"]

    def calculate_intraday_vwap(self) -> pd.DataFrame:
        return self.pivot_frame.groupby(
            self.pivot_frame.index.date, group_keys=False
        ).apply(lambda df: (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum())

    def calculate_intraday_price_distance(self) -> pd.DataFrame:
        pct_chg: pd.DataFrame = self.close.div(self.open.reindex(self.close.index).ffill()) - 1
        return pct_chg.abs()

    def calculate_sigma(self, window: int = 14) -> pd.DataFrame:
        distance: pd.DataFrame = self.calculate_intraday_price_distance()
        return distance.groupby(distance.index.time, group_keys=False).apply(
            lambda ser: ser.rolling(window=window).mean()
        )

    def calculate_bound(self, window: int = 14, method: str = "U") -> pd.DataFrame:
        sigma: pd.DataFrame = self.calculate_sigma(window)
        idx: pd.DatetimeIndex = sigma.index
        sigma.index = sigma.index.normalize()

        daily_idx: pd.DatetimeIndex = self.open.index.normalize()
        cols = self.open.columns

        if method.upper() == "U":
            threshold: pd.DataFrame = pd.DataFrame(
                np.maximum(self.open.values, self.close.at_time("15:00:00").shift(1).values),
                index=daily_idx,
                columns=cols,
            )
            out: pd.DataFrame = threshold.mul(1 + sigma, axis=0)
        elif method.upper() == "L":
            threshold: pd.DataFrame = pd.DataFrame(
                np.minimum(self.open.values, self.close.at_time("15:00:00").shift(1).values),
                index=daily_idx,
                columns=cols,
            )
            out = threshold.mul(1 - sigma, axis=0)
        else:
            raise ValueError(f"Unsupported method: {method}")

        out.index = idx
        return out

    def concat_signal(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        upperbound: pd.DataFrame = self.calculate_bound(window, method="U")
        lowerbound: pd.DataFrame = self.calculate_bound(window, method="L")
        vwaps: pd.DataFrame = self.calculate_intraday_vwap()
        signal: pd.Series = self.ohlcv.set_index(["trade_time", "code"])["close"].to_frame(name="signal")
        return (
            pd.concat(
                [
                    data.set_index(["trade_time", "code"]),
                    upperbound.stack().to_frame(name="upperbound"),
                    signal,
                    lowerbound.stack().to_frame(name="lowerbound"),
                    vwaps.stack().to_frame(name="vwap"),
                ],
                axis=1,
            )
            .reset_index()
            .sort_values(["trade_time", "code"])
        )

    def fit(self, window: int = 14) -> pd.DataFrame:
        return self.concat_signal(self.ohlcv, window)
