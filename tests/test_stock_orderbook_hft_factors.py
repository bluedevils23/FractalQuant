from __future__ import annotations

import numpy as np
import pandas as pd

from factor.stock_orderbook import calculate_snapshot_factors


def _quotes(bid_sizes: list[list[float]], ask_sizes: list[list[float]]) -> pd.DataFrame:
    rows = len(bid_sizes)
    index = pd.date_range("2026-01-05 09:30:00", periods=rows, freq="s")
    data: dict[str, list[float]] = {}
    for level in range(1, 6):
        data[f"bid_price{level}"] = [100.0 - level * 0.01] * rows
        data[f"ask_price{level}"] = [100.0 + level * 0.01] * rows
        data[f"bid_qty{level}"] = [sizes[level - 1] for sizes in bid_sizes]
        data[f"ask_qty{level}"] = [sizes[level - 1] for sizes in ask_sizes]
    return pd.DataFrame(data, index=index)


def test_near_touch_weighting_detects_depth_placement() -> None:
    quotes = _quotes([[50, 40, 30, 20, 10]], [[10, 20, 30, 40, 50]])
    factors = calculate_snapshot_factors(quotes)

    assert factors["depth_imbalance_l5"].iloc[0] == 0.0
    assert np.isclose(factors["weighted_depth_imbalance_l5"].iloc[0], 37.0 / 137.0)
    assert 0.0 < factors["weighted_depth_pressure_l5"].iloc[0] <= 1.0


def test_refill_uses_only_preceding_snapshots() -> None:
    bid_sizes = [[10, 10, 10, 10, 10] for _ in range(6)] + [[20, 20, 20, 20, 20]]
    ask_sizes = [[10, 10, 10, 10, 10] for _ in range(7)]
    factors = calculate_snapshot_factors(_quotes(bid_sizes, ask_sizes))

    assert factors["bid_refill_intensity_l5"].iloc[:6].isna().all()
    assert np.isclose(factors["bid_refill_intensity_l5"].iloc[6], 1.0)
    assert factors["ask_refill_intensity_l5"].iloc[6] == 0.0


def test_weighted_velocity_is_five_snapshot_difference() -> None:
    bid_sizes = [[10 + i, 10, 10, 10, 10] for i in range(7)]
    ask_sizes = [[10, 10, 10, 10, 10] for _ in range(7)]
    factors = calculate_snapshot_factors(_quotes(bid_sizes, ask_sizes))

    assert factors["weighted_imbalance_velocity_l5"].iloc[:5].isna().all()
    expected = (
        factors["weighted_depth_imbalance_l5"].iloc[5]
        - factors["weighted_depth_imbalance_l5"].iloc[0]
    )
    assert np.isclose(factors["weighted_imbalance_velocity_l5"].iloc[5], expected)
