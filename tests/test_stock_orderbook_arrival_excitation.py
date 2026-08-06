from __future__ import annotations

import numpy as np
import pandas as pd
from factor.stock_orderbook import (
    calculate_trade_arrival_excitation_factors,
    calculate_trade_flow_factors,
)

from scripts.generate_stock_orderbook_factors import BASE_FACTOR_COLUMNS

FACTOR_COLUMNS = [
    "trade_arrival_excitation_30s",
    "trade_arrival_excitation_imbalance_30s",
    "trade_cross_side_excitation_30s",
]


def _trades(times: list[pd.Timestamp], sides: list[str]) -> pd.DataFrame:
    qty = np.full(len(times), 100.0)
    price = np.full(len(times), 10.0)
    return pd.DataFrame(
        {
            "event_time": times,
            "side": sides,
            "price": price,
            "qty": qty,
            "notional": price * qty,
        }
    )


def _baseline(quote_time: pd.Timestamp) -> tuple[list[pd.Timestamp], list[str]]:
    times = [quote_time - pd.Timedelta(seconds=280 - 10 * i) for i in range(20)]
    sides = ["B", "S"] * 10
    return times, sides


def test_clustered_arrivals_have_stronger_excitation_than_uniform_arrivals() -> None:
    quote_time = pd.Timestamp("2026-01-05 10:00:00")
    baseline_times, baseline_sides = _baseline(quote_time)
    clustered_times = [quote_time - pd.Timedelta(milliseconds=100 * i) for i in range(10)]
    uniform_times = [quote_time - pd.Timedelta(seconds=3 * i) for i in range(10)]

    clustered = calculate_trade_arrival_excitation_factors(
        _trades(baseline_times + clustered_times, baseline_sides + ["B"] * 10),
        pd.DatetimeIndex([quote_time]),
    )
    uniform = calculate_trade_arrival_excitation_factors(
        _trades(baseline_times + uniform_times, baseline_sides + ["B"] * 10),
        pd.DatetimeIndex([quote_time]),
    )

    assert clustered["trade_arrival_excitation_30s"].iloc[0] > uniform[
        "trade_arrival_excitation_30s"
    ].iloc[0]


def test_arrival_excitation_imbalance_tracks_trade_side() -> None:
    quote_time = pd.Timestamp("2026-01-05 10:00:00")
    baseline_times, baseline_sides = _baseline(quote_time)
    recent_times = [quote_time - pd.Timedelta(seconds=1)] * 10

    def imbalance(sides: list[str]) -> float:
        factors = calculate_trade_arrival_excitation_factors(
            _trades(baseline_times + recent_times, baseline_sides + sides),
            pd.DatetimeIndex([quote_time]),
        )
        return factors["trade_arrival_excitation_imbalance_30s"].iloc[0]

    assert np.isclose(imbalance(["B"] * 10), 1.0)
    assert np.isclose(imbalance(["S"] * 10), -1.0)
    assert np.isclose(imbalance(["B", "S"] * 5), 0.0)


def test_alternating_sides_have_more_cross_side_excitation() -> None:
    quote_time = pd.Timestamp("2026-01-05 10:00:00")
    recent_times = [quote_time - pd.Timedelta(seconds=10 - i) for i in range(10)]

    persistent = calculate_trade_arrival_excitation_factors(
        _trades(recent_times, ["B"] * 10), pd.DatetimeIndex([quote_time])
    )
    alternating = calculate_trade_arrival_excitation_factors(
        _trades(recent_times, ["B", "S"] * 5), pd.DatetimeIndex([quote_time])
    )

    persistent_value = persistent["trade_cross_side_excitation_30s"].iloc[0]
    alternating_value = alternating["trade_cross_side_excitation_30s"].iloc[0]
    assert np.isclose(persistent_value, 0.0)
    assert alternating_value > persistent_value
    assert 0.0 <= alternating_value <= 1.0


def test_arrival_excitation_is_causal_and_includes_same_timestamp_events() -> None:
    first_quote = pd.Timestamp("2026-01-05 10:00:00")
    second_quote = first_quote + pd.Timedelta(minutes=1)
    baseline_times, baseline_sides = _baseline(first_quote)
    current = _trades(
        baseline_times + [first_quote], baseline_sides + ["B"]
    )
    changed_future = pd.concat(
        [current, _trades([first_quote + pd.Timedelta(seconds=1)], ["S"])],
        ignore_index=True,
    )
    quote_index = pd.DatetimeIndex([first_quote, second_quote])

    before = calculate_trade_arrival_excitation_factors(current, quote_index)
    after = calculate_trade_arrival_excitation_factors(changed_future, quote_index)

    pd.testing.assert_series_equal(before.iloc[0], after.iloc[0])
    assert np.isfinite(before["trade_arrival_excitation_imbalance_30s"].iloc[0])


def test_arrival_excitation_spans_lunch_but_resets_next_day() -> None:
    morning_quote = pd.Timestamp("2026-01-05 11:29:50")
    afternoon_quote = pd.Timestamp("2026-01-05 13:00:05")
    next_day_quote = pd.Timestamp("2026-01-06 09:30:05")
    baseline_times = [
        morning_quote - pd.Timedelta(seconds=280 - 10 * i) for i in range(20)
    ]
    trades = _trades(baseline_times + [morning_quote], ["B", "S"] * 10 + ["B"])

    factors = calculate_trade_arrival_excitation_factors(
        trades, pd.DatetimeIndex([afternoon_quote, next_day_quote])
    )

    assert np.isfinite(factors["trade_arrival_excitation_30s"].iloc[0])
    assert factors["trade_arrival_excitation_imbalance_30s"].iloc[0] == 1.0
    assert factors.iloc[1].isna().all()


def test_missing_history_and_empty_windows_are_reported_as_missing() -> None:
    quote_time = pd.Timestamp("2026-01-05 10:00:00")
    recent = _trades([quote_time - pd.Timedelta(seconds=1)], ["B"])
    insufficient = calculate_trade_arrival_excitation_factors(
        recent, pd.DatetimeIndex([quote_time])
    )
    empty_window = calculate_trade_arrival_excitation_factors(
        recent, pd.DatetimeIndex([quote_time + pd.Timedelta(minutes=1)])
    )
    empty = calculate_trade_arrival_excitation_factors(
        recent.iloc[0:0], pd.DatetimeIndex([quote_time])
    )

    assert np.isnan(insufficient["trade_arrival_excitation_30s"].iloc[0])
    assert insufficient["trade_arrival_excitation_imbalance_30s"].iloc[0] == 1.0
    assert empty_window.iloc[0].isna().all()
    assert empty.iloc[0].isna().all()


def test_arrival_windows_use_documented_boundaries_and_ignore_invalid_sides() -> None:
    quote_time = pd.Timestamp("2026-01-05 10:00:00")
    interior_baseline = [
        quote_time - pd.Timedelta(seconds=290 - 10 * i) for i in range(19)
    ]
    boundary_times = [
        quote_time - pd.Timedelta(seconds=300),
        quote_time - pd.Timedelta(seconds=30),
        quote_time,
    ]
    trades = _trades(
        interior_baseline + boundary_times,
        ["B", "S"] * 9 + ["B"] + ["S", "S", "B"],
    )
    invalid = pd.concat(
        [trades, _trades([quote_time], ["invalid"])], ignore_index=True
    )

    expected = calculate_trade_arrival_excitation_factors(
        trades, pd.DatetimeIndex([quote_time])
    )
    actual = calculate_trade_arrival_excitation_factors(
        invalid, pd.DatetimeIndex([quote_time])
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert np.isfinite(actual["trade_arrival_excitation_30s"].iloc[0])
    assert actual["trade_arrival_excitation_imbalance_30s"].iloc[0] == 1.0


def test_arrival_excitation_fields_are_available_end_to_end() -> None:
    quote_index = pd.DatetimeIndex(["2026-01-05 10:00:00"])
    quotes = pd.DataFrame({"mid_price": [10.0]}, index=quote_index)
    factors = calculate_trade_flow_factors(
        _trades([quote_index[0]], ["B"]), quotes
    )

    assert set(FACTOR_COLUMNS) <= set(BASE_FACTOR_COLUMNS)
    assert set(FACTOR_COLUMNS) <= set(factors.columns)
