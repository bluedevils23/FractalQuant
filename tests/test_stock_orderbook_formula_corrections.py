from __future__ import annotations

import numpy as np
import pandas as pd

from factor.stock_orderbook import (
    build_stock_orderbook_factor_frame,
    calculate_order_flow_factors,
    calculate_snapshot_factors,
    calculate_trade_flow_factors,
)


def _quotes(index: pd.DatetimeIndex, bid_qty1: list[float] | None = None) -> pd.DataFrame:
    rows = len(index)
    bid_qty1 = bid_qty1 or [100.0] * rows
    data: dict[str, list[float]] = {}
    for level in range(1, 6):
        data[f"bid_price{level}"] = [10.0 - level * 0.01] * rows
        data[f"ask_price{level}"] = [10.0 + level * 0.01] * rows
        data[f"bid_qty{level}"] = bid_qty1 if level == 1 else [100.0] * rows
        data[f"ask_qty{level}"] = [100.0] * rows
    return pd.DataFrame(data, index=index)


def test_book_pressure_is_positive_for_buy_heavy_depth() -> None:
    index = pd.DatetimeIndex(["2026-01-05 09:30:00"])
    quotes = _quotes(index, bid_qty1=[1000.0])

    factors = calculate_snapshot_factors(quotes)

    assert factors["book_pressure_wap5"].iloc[0] > 0.0


def test_event_windows_are_anchored_to_each_quote() -> None:
    base = pd.Timestamp("2026-01-05 09:30:00")
    quote_index = pd.DatetimeIndex([base + pd.Timedelta(seconds=100)])
    events = pd.DataFrame(
        {
            "event_time": [base, base + pd.Timedelta(seconds=50)],
            "side": ["B", "S"],
            "price": [10.0, 11.0],
            "qty": [100.0, 100.0],
            "notional": [1000.0, 1100.0],
        }
    )

    order_factors = calculate_order_flow_factors(events, quote_index)
    quote_factors = pd.DataFrame({"mid_price": [10.0]}, index=quote_index)
    trade_factors = calculate_trade_flow_factors(events, quote_factors)

    assert order_factors["order_qty_imbalance_60s"].iloc[0] == -1.0
    assert trade_factors["trade_qty_imbalance_60s"].iloc[0] == -1.0
    assert np.isclose(trade_factors["trade_vwap_gap_60s"].iloc[0], 0.1)


def test_trade_impact_and_flow_use_trade_direction() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=20, freq="s")
    prices = 10.0 + np.arange(20) * 0.001
    qty = 1000.0 + np.arange(20) * 10.0
    trades = pd.DataFrame(
        {
            "event_time": index,
            "side": ["S"] * 20,
            "price": prices,
            "qty": qty,
            "notional": prices * qty,
        }
    )
    quotes = pd.DataFrame({"mid_price": [prices[-1]]}, index=index[-1:])

    factors = calculate_trade_flow_factors(trades, quotes)

    assert np.isfinite(factors["market_impact_60s"].iloc[0])
    assert abs(factors["market_impact_60s"].iloc[0]) < 10000.0
    assert factors["order_flow_imbalance_60s"].iloc[0] == -1.0
    assert factors["orderbook_pressure_60s"].iloc[0] == -1.0
    assert np.isclose(factors["market_efficiency_60s"].iloc[0], 1.0)
    assert 0.0 <= factors["price_volume_decoupling_60s"].iloc[0] <= 1.0
    assert factors["trade_size_distribution_60s"].iloc[0] >= 0.0


def test_orderbook_velocity_is_normalized_by_elapsed_time() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-05 09:30:00",
            "2026-01-05 09:30:01",
            "2026-01-05 09:30:02",
            "2026-01-05 09:30:03",
            "2026-01-05 09:30:04",
            "2026-01-05 09:30:10",
        ]
    )
    quotes = _quotes(index, bid_qty1=[100.0, 100.0, 100.0, 100.0, 100.0, 300.0])

    factors = build_stock_orderbook_factor_frame(
        quotes, pd.DataFrame(), pd.DataFrame()
    )

    expected = (
        factors["depth_imbalance_l5"].iloc[-1]
        - factors["depth_imbalance_l5"].iloc[0]
    ) / 10.0
    assert np.isclose(factors["orderbook_velocity_l5"].iloc[-1], expected)
