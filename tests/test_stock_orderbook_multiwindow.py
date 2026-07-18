from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from factor.stock_orderbook import (
    build_stock_orderbook_factor_frame,
    calculate_order_flow_factors,
    calculate_snapshot_factors,
    calculate_trade_flow_factors,
)
from scripts.generate_stock_orderbook_factors import (
    BASE_FACTOR_COLUMNS,
    factor_columns_for_profile,
    output_columns_for_profile,
)


def _quotes(index: pd.DatetimeIndex) -> pd.DataFrame:
    data: dict[str, list[float]] = {}
    for level in range(1, 6):
        data[f"bid_price{level}"] = [10.0 - level * 0.01] * len(index)
        data[f"ask_price{level}"] = [10.0 + level * 0.01] * len(index)
        data[f"bid_qty{level}"] = [100.0] * len(index)
        data[f"ask_qty{level}"] = [100.0] * len(index)
    return pd.DataFrame(data, index=index)


def _events(times: pd.DatetimeIndex, sides: list[str]) -> pd.DataFrame:
    quantities = np.full(len(times), 100.0)
    prices = np.full(len(times), 10.0)
    return pd.DataFrame(
        {
            "event_time": times,
            "side": sides,
            "price": prices,
            "qty": quantities,
            "notional": prices * quantities,
        }
    )


def test_base_profile_preserves_default_factor_frame() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=40, freq="s")
    quotes = _quotes(index)
    events = _events(index[::2], ["B", "S"] * 10)

    default = build_stock_orderbook_factor_frame(quotes, events, events)
    explicit_base = build_stock_orderbook_factor_frame(
        quotes, events, events, window_profile="base"
    )

    assert_frame_equal(default, explicit_base)
    assert set(BASE_FACTOR_COLUMNS) <= set(default.columns)


def test_multi_profile_has_unique_schema_and_all_expected_columns() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=40, freq="s")
    quotes = _quotes(index)
    events = _events(index[::2], ["B", "S"] * 10)
    frame = build_stock_orderbook_factor_frame(
        quotes, events, events, window_profile="multi"
    )
    columns = factor_columns_for_profile("multi")

    assert len(columns) == 135
    assert len(columns) == len(set(columns))
    assert set(columns) <= set(frame.columns)
    assert output_columns_for_profile("multi")[-len(columns) :] == columns


def test_short_window_excludes_left_boundary_and_lunch_session() -> None:
    base = pd.Timestamp("2026-01-05 09:30:00")
    quote_index = pd.DatetimeIndex([base + pd.Timedelta(seconds=20)])
    events = _events(
        pd.DatetimeIndex([base, base + pd.Timedelta(seconds=20)]), ["B", "S"]
    )

    factors = calculate_order_flow_factors(events, quote_index, window_profile="multi")

    assert factors["order_qty_imbalance_10s"].iloc[0] == -1.0
    assert factors["order_qty_imbalance_30s"].iloc[0] == 0.0

    afternoon_quote = pd.DatetimeIndex([pd.Timestamp("2026-01-05 13:00:01")])
    morning_event = _events(pd.DatetimeIndex([pd.Timestamp("2026-01-05 11:29:59")]), ["B"])
    afternoon = calculate_order_flow_factors(
        morning_event, afternoon_quote, window_profile="multi"
    )

    assert afternoon["order_qty_imbalance_300s"].iloc[0] == 0.0


def test_ofi_resets_at_afternoon_open_both_profiles() -> None:
    """Both base and multi profiles must NaN the instantaneous OFI event at the
    afternoon-session open so that cross-lunch queue comparisons never appear."""
    index = pd.DatetimeIndex(["2026-01-05 11:29:59", "2026-01-05 13:00:00"])
    quotes = _quotes(index)
    quotes.loc[index[1], "bid_qty1"] = 300.0

    base_factors = calculate_snapshot_factors(quotes, window_profile="base")
    multi_factors = calculate_snapshot_factors(quotes, window_profile="multi")

    # Instantaneous OFI must be NaN at the afternoon open in BOTH profiles.
    assert np.isnan(base_factors["normalized_ofi_l1"].iloc[1])
    assert np.isnan(base_factors["normalized_mlofi_l5"].iloc[1])
    # Multi-window rolling columns must also be NaN at the afternoon open.
    assert np.isnan(multi_factors["normalized_ofi_l1_10s"].iloc[1])


def test_low_trade_count_keeps_simple_flow_and_marks_path_factors_missing() -> None:
    quote_index = pd.DatetimeIndex(["2026-01-05 09:30:10"])
    quotes = pd.DataFrame({"mid_price": [10.0]}, index=quote_index)
    trades = _events(pd.DatetimeIndex(["2026-01-05 09:30:05"]), ["B"])

    factors = calculate_trade_flow_factors(trades, quotes, window_profile="multi")

    assert factors["trade_qty_imbalance_10s"].iloc[0] == 1.0
    assert np.isnan(factors["market_impact_30s"].iloc[0])
    assert np.isnan(factors["market_impact_300s"].iloc[0])
