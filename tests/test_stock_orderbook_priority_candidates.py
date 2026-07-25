from __future__ import annotations

import numpy as np
import pandas as pd
from factor.stock_orderbook import (
    _calculate_ofi_impact_nonlinearity,
    build_stock_orderbook_factor_frame,
    calculate_amihud_illiquidity_factor,
    calculate_snapshot_factors,
)

from scripts.generate_stock_orderbook_factors import BASE_FACTOR_COLUMNS


def _quotes(
    index: pd.DatetimeIndex,
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    mid_changes: np.ndarray | None = None,
) -> pd.DataFrame:
    mid = 100.5 + (np.zeros(len(index)) if mid_changes is None else mid_changes)
    data: dict[str, np.ndarray] = {}
    for level in range(1, 6):
        data[f"bid_price{level}"] = mid - 0.5 - (level - 1) * 0.01
        data[f"ask_price{level}"] = mid + 0.5 + (level - 1) * 0.01
        data[f"bid_qty{level}"] = bid_sizes[:, level - 1]
        data[f"ask_qty{level}"] = ask_sizes[:, level - 1]
    return pd.DataFrame(data, index=index)


def test_depth_level_ofi_slope_matches_cross_level_ols() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=2, freq="s")
    bid_sizes = np.array([[100.0] * 5, [110.0, 120.0, 130.0, 140.0, 150.0]])
    ask_sizes = np.array([[100.0] * 5, [100.0] * 5])

    factors = calculate_snapshot_factors(_quotes(index, bid_sizes, ask_sizes))

    normalized = np.arange(10.0, 51.0, 10.0) / np.arange(205.0, 226.0, 5.0)
    expected = np.polyfit(np.arange(1.0, 6.0), normalized, 1)[0]
    assert np.isnan(factors["depth_level_ofi_slope"].iloc[0])
    assert np.isclose(factors["depth_level_ofi_slope"].iloc[1], expected)


def test_ofi_impact_nonlinearity_is_lagged_and_recovers_coefficient() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=80, freq="s")
    ofi_state = np.sin(np.arange(len(index)) * 0.7) + 0.35 * np.cos(
        np.arange(len(index)) * 0.17
    )
    bid_sizes = np.full((len(index), 5), 1000.0)
    ask_sizes = np.full((len(index), 5), 1000.0)
    bid_sizes[:, 0] += np.cumsum(ofi_state * 20.0)
    mid_changes = np.zeros(len(index))
    for position in range(1, len(index)):
        x = ofi_state[position - 1]
        mid_changes[position] = mid_changes[position - 1] + (
            0.4 * x + 1.8 * x * abs(x)
        ) / 10000.0 * 100.5

    quotes = _quotes(index, bid_sizes, ask_sizes, mid_changes)
    base = calculate_snapshot_factors(quotes)
    changed_quotes = quotes.copy()
    changed_quotes.loc[index[-1], [f"bid_price{i}" for i in range(1, 6)]] += 5.0
    changed_quotes.loc[index[-1], [f"ask_price{i}" for i in range(1, 6)]] += 5.0
    changed = calculate_snapshot_factors(changed_quotes)

    assert np.isfinite(base["ofi_impact_nonlinearity"].iloc[-1])
    assert base["ofi_impact_nonlinearity"].iloc[-1] == changed[
        "ofi_impact_nonlinearity"
    ].iloc[-1]


def test_ofi_impact_nonlinearity_recovers_known_quadratic_coefficient() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=80, freq="s")
    normalized_ofi = np.sin(np.arange(len(index)) * 0.43) + 0.4 * np.cos(
        np.arange(len(index)) * 0.19
    )
    linear_coefficient = 0.6
    nonlinear_coefficient = 1.8
    mid = np.full(len(index), 100.0)
    for position in range(1, len(index)):
        x = normalized_ofi[position]
        return_bps = (
            linear_coefficient * x + nonlinear_coefficient * x * abs(x)
        )
        mid[position] = mid[position - 1] * (1.0 + return_bps / 10000.0)

    result = _calculate_ofi_impact_nonlinearity(normalized_ofi, mid, index)

    assert np.isclose(result[-1], nonlinear_coefficient, rtol=0.01)


def test_amihud_illiquidity_uses_only_trailing_five_minutes() -> None:
    base = pd.Timestamp("2026-01-05 09:30:00")
    trades = pd.DataFrame(
        {
            "event_time": [
                base,
                base + pd.Timedelta(minutes=1),
                base + pd.Timedelta(minutes=2),
                base + pd.Timedelta(minutes=8),
                base + pd.Timedelta(minutes=9),
                base + pd.Timedelta(minutes=10),
            ],
            "side": ["B", "S", "B", "S", "B", "S"],
            "price": [100.0, 101.0, 102.01, 200.0, 202.0, 204.02],
            "qty": [10.0] * 6,
            "notional": [1000.0] * 6,
        }
    )
    quote_index = pd.DatetimeIndex(
        [base + pd.Timedelta(minutes=2), base + pd.Timedelta(minutes=10)]
    )

    factor = calculate_amihud_illiquidity_factor(
        trades, quote_index, min_returns=2
    )

    assert np.isclose(factor.iloc[0], 0.01 / 1000.0)
    assert np.isclose(factor.iloc[1], 0.01 / 1000.0)


def test_priority_candidates_are_registered_end_to_end() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=40, freq="s")
    bid_sizes = np.full((len(index), 5), 100.0)
    ask_sizes = np.full((len(index), 5), 100.0)
    quotes = _quotes(index, bid_sizes, ask_sizes)
    trades = pd.DataFrame(
        {
            "event_time": index,
            "side": np.where(np.arange(len(index)) % 2 == 0, "B", "S"),
            "price": 100.0 + np.arange(len(index)) * 0.01,
            "qty": np.full(len(index), 100.0),
            "notional": np.full(len(index), 10000.0),
        }
    )

    result = build_stock_orderbook_factor_frame(quotes, pd.DataFrame(), trades)
    expected = {
        "amihud_illiquidity_5m",
        "depth_level_ofi_slope",
        "ofi_impact_nonlinearity",
    }
    assert expected <= set(BASE_FACTOR_COLUMNS)
    assert expected <= set(result.columns)
