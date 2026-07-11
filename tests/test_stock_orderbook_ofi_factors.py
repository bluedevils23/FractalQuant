from __future__ import annotations

import numpy as np
import pandas as pd

from factor.stock_orderbook import calculate_snapshot_factors


def _quotes(
    index: pd.DatetimeIndex,
    bid_prices: list[float],
    bid_sizes: list[list[float]],
    ask_prices: list[float],
    ask_sizes: list[list[float]],
) -> pd.DataFrame:
    data: dict[str, list[float]] = {}
    for level in range(1, 6):
        data[f"bid_price{level}"] = [
            price - (level - 1) * 0.01 for price in bid_prices
        ]
        data[f"ask_price{level}"] = [
            price + (level - 1) * 0.01 for price in ask_prices
        ]
        data[f"bid_qty{level}"] = [sizes[level - 1] for sizes in bid_sizes]
        data[f"ask_qty{level}"] = [sizes[level - 1] for sizes in ask_sizes]
    return pd.DataFrame(data, index=index)


def test_l1_ofi_uses_previous_queue_for_price_worsening() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=4, freq="s")
    quantities = [[10.0] * 5, [13.0] + [10.0] * 4, [11.0] + [10.0] * 4, [11.0] + [10.0] * 4]
    ask_quantities = [[10.0] * 5, [8.0] + [10.0] * 4, [8.0] + [10.0] * 4, [9.0] + [10.0] * 4]
    factors = calculate_snapshot_factors(
        _quotes(index, [100.0, 100.0, 99.0, 99.0], quantities, [101.0, 101.0, 101.0, 102.0], ask_quantities)
    )

    assert factors["normalized_ofi_l1"].iloc[0] != factors["normalized_ofi_l1"].iloc[0]
    assert np.isclose(factors["normalized_ofi_l1"].iloc[1], 5.0 / 20.5)
    assert np.isclose(factors["normalized_ofi_l1"].iloc[2], -13.0 / 20.0)
    assert np.isclose(factors["normalized_ofi_l1"].iloc[3], 8.0 / 19.5)
    assert np.isclose(factors["normalized_ofi_l1_60s"].iloc[3], 0.0)


def test_mlofi_captures_deeper_queue_change() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=2, freq="s")
    bid_sizes = [[10.0] * 5, [10.0, 20.0, 10.0, 10.0, 10.0]]
    ask_sizes = [[10.0] * 5, [10.0] * 5]
    factors = calculate_snapshot_factors(
        _quotes(index, [100.0, 100.0], bid_sizes, [101.0, 101.0], ask_sizes)
    )

    weighted_depth = 10.0 * sum(1.0 / np.arange(1, 6))
    expected = (10.0 / 2.0) / (2.0 * weighted_depth + 2.5)
    assert factors["normalized_ofi_l1"].iloc[1] == 0.0
    assert np.isclose(factors["normalized_mlofi_l5"].iloc[1], expected)
    assert np.isclose(factors["normalized_mlofi_l5_60s"].iloc[1], expected)


def test_ofi_resets_at_the_start_of_each_trade_day() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-05 14:59:59",
            "2026-01-05 15:00:00",
            "2026-01-06 09:30:00",
            "2026-01-06 09:30:01",
        ]
    )
    quantities = [[10.0] * 5, [12.0] + [10.0] * 4, [30.0] + [10.0] * 4, [32.0] + [10.0] * 4]
    ask_quantities = [[10.0] * 5 for _ in range(4)]
    factors = calculate_snapshot_factors(
        _quotes(index, [100.0] * 4, quantities, [101.0] * 4, ask_quantities)
    )

    assert np.isclose(factors["normalized_ofi_l1"].iloc[1], 2.0 / 21.0)
    assert np.isnan(factors["normalized_ofi_l1"].iloc[2])
    assert np.isnan(factors["normalized_ofi_l1_60s"].iloc[2])
    assert np.isclose(factors["normalized_ofi_l1"].iloc[3], 2.0 / 41.0)
