"""Causal fixed-width fractional-difference price factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import PriceFactor


def fixed_width_fractional_weights(
    order: float,
    threshold: float = 1e-3,
) -> np.ndarray:
    """Build truncated fractional-difference weights from newest to oldest."""
    if not 0.0 < order < 1.0:
        raise ValueError("order must be between 0 and 1")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    weights = [1.0]
    for lag in range(1, 100_000):
        next_weight = -weights[-1] * (order - lag + 1.0) / lag
        if abs(next_weight) < threshold:
            break
        weights.append(next_weight)
    return np.asarray(weights, dtype=float)


class FractionalDiffLogCloseFactor(PriceFactor):
    """Fractionally difference log close after removing the input segment's starting level."""

    def __init__(self, order: float = 0.4, threshold: float = 1e-3, window: int = 50):
        if window <= 0:
            raise ValueError("window must be positive")
        self.order = order
        self.threshold = threshold
        self.weights = fixed_width_fractional_weights(order, threshold)[:window]
        order_label = f"{order:g}".replace(".", "")
        super().__init__(f"fractional_diff_log_close_d{order_label}", len(self.weights))

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        result = np.full(len(close), np.nan, dtype=float)
        valid = np.isfinite(close) & (close > 0)
        if not valid.any():
            return pd.Series(result, index=df.index)

        log_close = np.full(len(close), np.nan, dtype=float)
        log_close[valid] = np.log(close[valid])
        first_valid = np.flatnonzero(valid)[0]
        log_close = log_close - log_close[first_valid]
        clean_values = np.where(np.isfinite(log_close), log_close, 0.0)
        fractional_values = np.convolve(clean_values, self.weights, mode="full")[: len(close)]

        width = len(self.weights)
        complete_windows = (
            pd.Series(valid, index=df.index)
            .rolling(width, min_periods=width)
            .sum()
            .eq(width)
            .to_numpy()
        )
        result[complete_windows] = fractional_values[complete_windows]
        return pd.Series(result, index=df.index)
