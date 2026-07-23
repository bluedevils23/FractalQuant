from __future__ import annotations

import numpy as np
import pandas as pd

from factor.stock_orderbook import (
    build_stock_orderbook_factor_frame,
    calculate_snapshot_factors,
    calculate_vpin_factor,
)


def _quotes(index: pd.DatetimeIndex) -> pd.DataFrame:
    rows: dict[str, np.ndarray] = {}
    for level in range(1, 6):
        rows[f"ask_price{level}"] = np.full(len(index), 10.01 + 0.01 * (level - 1))
        rows[f"bid_price{level}"] = np.full(len(index), 9.99 - 0.01 * (level - 1))
        rows[f"ask_qty{level}"] = np.full(len(index), 100.0)
        rows[f"bid_qty{level}"] = np.full(len(index), 100.0)
    return pd.DataFrame(rows, index=index)


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["event_time", "side", "price", "qty", "notional"])


def test_ofi_level_entropy_distinguishes_concentrated_and_distributed_events() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=3, freq="s")
    concentrated = _quotes(index)
    concentrated.loc[index[1], "bid_qty1"] = 200.0
    concentrated.loc[index[2], "bid_qty1"] = 400.0
    concentrated.loc[index[2], [f"bid_qty{i}" for i in range(2, 6)]] = 300.0

    entropy = calculate_snapshot_factors(concentrated)["ofi_level_entropy_l5"]

    assert np.isclose(entropy.iloc[1], 0.0)
    assert np.isclose(entropy.iloc[2], 1.0)


def test_vpin_uses_completed_equal_volume_buckets_across_lunch() -> None:
    quote_index = pd.DatetimeIndex(
        ["2026-01-05 09:30:03", "2026-01-05 13:00:03"]
    )
    trades = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-05 09:30:00",
                    "2026-01-05 09:30:01",
                    "2026-01-05 13:00:00",
                    "2026-01-05 13:00:01",
                ]
            ),
            "side": ["B", "B", "B", "S"],
            "price": [10.0] * 4,
            "qty": [5.0] * 4,
            "notional": [50.0] * 4,
        }
    )

    vpin = calculate_vpin_factor(trades, quote_index, bucket_volume=10.0, num_buckets=2)

    assert np.isnan(vpin.iloc[0])
    assert np.isclose(vpin.iloc[1], 0.5)


def test_adaptive_vpin_is_invariant_to_trade_quantity_scale() -> None:
    event_time = pd.date_range("2026-01-05 09:30:00", periods=1200, freq="100ms")
    quote_index = pd.DatetimeIndex([event_time[-1] + pd.Timedelta(seconds=1)])

    def calculate(scale: float) -> float:
        qty = np.full(len(event_time), 100.0 * scale)
        trades = pd.DataFrame(
            {
                "event_time": event_time,
                "side": np.tile(["B", "S"], len(event_time) // 2),
                "price": np.full(len(event_time), 10.0),
                "qty": qty,
                "notional": qty * 10.0,
            }
        )
        return calculate_vpin_factor(trades, quote_index).iloc[0]

    assert np.isclose(calculate(1.0), 0.0)
    assert np.isclose(calculate(100.0), 0.0)


def test_adaptive_vpin_does_not_resize_before_a_block_trade() -> None:
    regular_times = pd.date_range("2026-01-05 09:30:00", periods=1200, freq="100ms")
    block_time = regular_times[-1] + pd.Timedelta(milliseconds=100)
    event_time = regular_times.append(pd.DatetimeIndex([block_time]))
    qty = np.r_[np.full(len(regular_times), 100.0), 100000.0]
    trades = pd.DataFrame(
        {
            "event_time": event_time,
            "side": np.concatenate(
                [np.tile(["B", "S"], len(regular_times) // 2), ["B"]]
            ),
            "price": np.full(len(event_time), 10.0),
            "qty": qty,
            "notional": qty * 10.0,
        }
    )
    quote_index = pd.DatetimeIndex([block_time + pd.Timedelta(seconds=1)])

    vpin = calculate_vpin_factor(trades, quote_index).iloc[0]

    assert vpin > 0.9


def test_side_resilience_recovers_after_depth_shock() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=4, freq="10s")
    quotes = _quotes(index)
    quotes.loc[index[1], [f"bid_qty{i}" for i in range(1, 6)]] = 20.0
    quotes.loc[index[2], [f"bid_qty{i}" for i in range(1, 6)]] = 60.0
    quotes.loc[index[3], [f"bid_qty{i}" for i in range(1, 6)]] = 100.0

    factors = calculate_snapshot_factors(quotes)

    assert 0.0 < factors["bid_resilience_30s"].iloc[2] < 1.0
    assert np.isclose(factors["bid_resilience_30s"].iloc[3], 1.0)
    assert np.isclose(factors["ask_resilience_30s"].iloc[2], 1.0)
    assert factors["resilience_imbalance_30s"].iloc[2] < 0.0


def test_markout_is_emitted_only_after_thirty_seconds() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=11, freq="10s")
    quotes = _quotes(index)
    for level in range(1, 6):
        quotes.loc[index[3]:, f"ask_price{level}"] += 0.02
        quotes.loc[index[3]:, f"bid_price{level}"] += 0.02
    trades = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-01-05 09:30:00"]),
            "side": ["B"],
            "price": [10.0],
            "qty": [100.0],
            "notional": [1000.0],
        }
    )

    factors = build_stock_orderbook_factor_frame(quotes, _empty_events(), trades)

    assert factors["adverse_selection_markout_30s"].iloc[:3].isna().all()
    assert factors["adverse_selection_markout_30s"].iloc[3] > 0.0
    assert np.isnan(factors["adverse_selection_markout_30s"].iloc[-1])


def test_markout_matures_across_lunch_on_the_trading_clock() -> None:
    index = pd.DatetimeIndex(["2026-01-05 11:29:50", "2026-01-05 13:00:20"])
    quotes = _quotes(index)
    for level in range(1, 6):
        quotes.loc[index[1], f"ask_price{level}"] += 0.02
        quotes.loc[index[1], f"bid_price{level}"] += 0.02
    trades = pd.DataFrame(
        {
            "event_time": [index[0]],
            "side": ["B"],
            "price": [10.0],
            "qty": [100.0],
            "notional": [1000.0],
        }
    )

    factors = build_stock_orderbook_factor_frame(quotes, _empty_events(), trades)

    assert factors["adverse_selection_markout_30s"].iloc[1] > 0.0


def test_first_batch_columns_are_present_end_to_end() -> None:
    index = pd.date_range("2026-01-05 09:30:00", periods=4, freq="10s")
    frame = build_stock_orderbook_factor_frame(_quotes(index), _empty_events(), _empty_events())

    expected = {
        "vpin_50bucket",
        "ofi_spread_scaled_impact",
        "bid_resilience_30s",
        "ask_resilience_30s",
        "resilience_imbalance_30s",
        "adverse_selection_markout_30s",
        "ofi_level_entropy_l5",
    }
    assert expected <= set(frame.columns)
