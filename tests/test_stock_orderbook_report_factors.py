from __future__ import annotations

import numpy as np
import pandas as pd

from factor.stock_orderbook import (
    build_stock_orderbook_factor_frame,
    calculate_snapshot_factors,
    calculate_trade_flow_factors,
)
from scripts.generate_stock_orderbook_factors import FACTOR_COLUMNS


def _quotes(index: pd.DatetimeIndex) -> pd.DataFrame:
    frame: dict[str, np.ndarray] = {}
    for level in range(1, 6):
        frame[f"ask_price{level}"] = 10.0 + 0.01 * level + np.arange(len(index)) * 0.01
        frame[f"bid_price{level}"] = 10.0 - 0.01 * level + np.arange(len(index)) * 0.01
        frame[f"ask_qty{level}"] = np.full(len(index), 50.0)
        frame[f"bid_qty{level}"] = np.full(len(index), 150.0)
    return pd.DataFrame(frame, index=index)


def _trades(times: pd.DatetimeIndex) -> pd.DataFrame:
    prices = np.where(np.arange(len(times)) % 2, 10.02, 9.98).astype(float)
    quantities = np.arange(1, len(times) + 1, dtype=float) * 100.0
    return pd.DataFrame(
        {
            "event_time": times,
            "side": np.where(np.arange(len(times)) % 2, "B", "S"),
            "price": prices,
            "qty": quantities,
            "notional": prices * quantities,
        }
    )


def test_report_snapshot_factors_cover_mci_soir_and_mpc() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=8, freq="min")
    factors = calculate_snapshot_factors(_quotes(index))

    assert np.isclose(factors["soir_l5_decay"].iloc[0], 0.5)
    assert factors["mci_bid_l5"].iloc[0] > 0.0
    assert factors["mci_ask_l5"].iloc[0] > 0.0
    assert factors["mpc_1m_mean_5m"].iloc[-1] > 0.0
    assert factors["mpc_5m_mean_5m"].iloc[-1] > 0.0


def test_mpc_spans_the_lunch_break_on_the_continuous_trading_clock() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-05 11:26:00",
            "2026-01-05 11:27:00",
            "2026-01-05 11:28:00",
            "2026-01-05 11:29:00",
            "2026-01-05 13:00:00",
        ]
    )
    factors = calculate_snapshot_factors(_quotes(index))

    assert np.isfinite(
        factors.loc[pd.Timestamp("2026-01-05 13:00:00"), "mpc_1m_mean_5m"]
    )


def test_report_trade_factors_are_causal_and_registered() -> None:
    quote_index = pd.DatetimeIndex(
        ["2026-01-05 09:30:00", "2026-01-05 09:30:30"]
    )
    raw_quotes = _quotes(quote_index)
    quotes = raw_quotes.join(calculate_snapshot_factors(raw_quotes))
    trades = _trades(pd.date_range("2026-01-05 09:30:01", periods=25, freq="s"))

    factors = calculate_trade_flow_factors(trades, quotes)
    final = factors.iloc[-1]
    expected = {
        "cautious_to_aggressive_buy_ratio_60s",
        "trade_notional_quantile_position_60s",
        "price_band_high_trade_count_share_60s",
        "price_band_low_trade_count_share_60s",
        "price_band_high_trade_size_rel_60s",
        "price_band_low_trade_size_rel_60s",
    }
    assert expected <= set(FACTOR_COLUMNS)
    assert final["cautious_to_aggressive_buy_ratio_60s"] > 0.0
    assert np.isfinite(final["trade_notional_quantile_position_60s"])
    assert 0.0 <= final["price_band_high_trade_count_share_60s"] <= 1.0
    assert 0.0 <= final["price_band_low_trade_count_share_60s"] <= 1.0

    future_trade = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2026-01-05 09:30:31")],
            "side": ["B"],
            "price": [99.0],
            "qty": [1_000_000.0],
            "notional": [99_000_000.0],
        }
    )
    changed = calculate_trade_flow_factors(
        pd.concat([trades, future_trade], ignore_index=True), quotes
    )
    pd.testing.assert_series_equal(
        factors.iloc[-1][list(expected)].sort_index(),
        changed.iloc[-1][list(expected)].sort_index(),
    )


def test_report_factors_are_present_in_the_end_to_end_frame() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=4, freq="min")
    frame = build_stock_orderbook_factor_frame(
        _quotes(index),
        pd.DataFrame(columns=["event_time", "side", "price", "qty", "notional"]),
        pd.DataFrame(columns=["event_time", "side", "price", "qty", "notional"]),
    )

    assert {
        "mci_bid_l5",
        "mci_ask_l5",
        "soir_l5_decay",
        "mpc_1m_mean_5m",
        "trade_notional_quantile_position_60s",
    } <= set(frame.columns)
