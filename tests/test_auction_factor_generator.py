from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_auction_factors import (
    EVENT_FACTOR_COLUMNS,
    OUTPUT_COLUMNS,
    apply_historical_ratios,
    apply_external_context,
    build_asset_universe,
    build_historical_context,
    build_session_path_factor_frame,
    calculate_daily_auction_factors,
    group_symbol_paths,
    load_auction_event_frame,
    load_daily_amount_history,
    merge_symbol_output,
)


def _quote_row(
    timestamp: str,
    price: float,
    bid_total: float,
    ask_total: float,
    *,
    bid_qty2: float = 0.0,
    ask_qty2: float = 0.0,
    previous_close: float = 10.0,
    open_price: float = np.nan,
    trade_volume: float = 0.0,
    trade_amount: float = 0.0,
) -> dict[str, object]:
    bid_qty1 = max(0.0, bid_total - bid_qty2)
    ask_qty1 = max(0.0, ask_total - ask_qty2)
    return {
        "trade_time": pd.Timestamp(timestamp),
        "trade_price": open_price,
        "trade_volume": trade_volume,
        "trade_amount": trade_amount,
        "open_price": open_price,
        "previous_close": previous_close,
        "ask_price1": price,
        "ask_price2": np.nan,
        "ask_price3": np.nan,
        "ask_qty1": ask_qty1,
        "ask_qty2": ask_qty2,
        "ask_qty3": 0.0,
        "bid_price1": price,
        "bid_price2": np.nan,
        "bid_price3": np.nan,
        "bid_qty1": bid_qty1,
        "bid_qty2": bid_qty2,
        "bid_qty3": 0.0,
    }


def _auction_quotes(
    *,
    include_match: bool = True,
    constant_stage2: bool = False,
) -> pd.DataFrame:
    stage2_prices = [10.02, 10.02, 10.02] if constant_stage2 else [10.02, 10.04, 10.06]
    rows = [
        _quote_row("2026-03-31 09:15:00", 10.00, 600, 400),
        _quote_row("2026-03-31 09:18:00", 10.01, 700, 300),
        _quote_row("2026-03-31 09:20:00", stage2_prices[0], 800, 200),
        _quote_row("2026-03-31 09:22:00", stage2_prices[1], 600, 400),
        _quote_row(
            "2026-03-31 09:24:59",
            stage2_prices[2],
            900,
            100,
            bid_qty2=100,
        ),
    ]
    if include_match:
        rows.append(
            _quote_row(
                "2026-03-31 09:25:03",
                10.05,
                500,
                500,
                open_price=10.05,
                trade_volume=1000,
                trade_amount=10050,
            )
        )
    return pd.DataFrame(rows)


def _event(
    timestamp: str,
    event_type: str,
    side: str,
    order_id: int,
    price: float,
    quantity: float,
    *,
    original_quantity: float | None = None,
) -> dict[str, object]:
    original_quantity = quantity if original_quantity is None else original_quantity
    return {
        "trade_time": pd.Timestamp(timestamp),
        "event_type": event_type,
        "side": side,
        "order_id": order_id,
        "price": price,
        "quantity": quantity,
        "notional": price * quantity,
        "original_notional": price * original_quantity,
    }


def _auction_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _event("2026-03-31 09:15:00", "A", "B", 1, 10.0, 100),
            _event("2026-03-31 09:15:01", "A", "S", 2, 10.0, 200),
            _event(
                "2026-03-31 09:18:00",
                "C",
                "B",
                1,
                10.0,
                20,
                original_quantity=100,
            ),
            _event(
                "2026-03-31 09:19:30",
                "C",
                "S",
                2,
                10.0,
                50,
                original_quantity=200,
            ),
            _event("2026-03-31 09:20:00", "A", "B", 3, 10.0, 300),
            _event("2026-03-31 09:23:00", "A", "S", 4, 10.0, 100),
            _event("2026-03-31 09:24:00", "A", "B", 5, 10.0, 100),
            _event("2026-03-31 09:25:00", "A", "B", 6, 10.0, 999),
        ]
    )


def test_daily_factor_formulas_and_output_contract() -> None:
    quotes = _auction_quotes()
    row = calculate_daily_auction_factors(quotes, "000001.SZ")

    assert list(row) == OUTPUT_COLUMNS
    assert row["trade_date"] == "2026-03-31"
    assert row["available_time"] == pd.Timestamp("2026-03-31 09:25:03")
    assert row["auction_has_match"] is True
    assert row["snapshot_count_stage1"] == 2
    assert row["snapshot_count_stage2"] == 3
    assert np.isclose(row["auction_overnight_return"], 0.005)
    assert np.isclose(row["auction_return_stage1"], 10.01 / 10.00 - 1.0)
    assert np.isclose(row["auction_return_stage2"], 10.06 / 10.02 - 1.0)
    assert np.isclose(row["auction_imbalance_change_stage1"], 0.2)
    assert np.isclose(row["auction_imbalance_change_stage2"], 0.2)
    assert np.isclose(row["auction_commitment_shift"], 0.3)
    assert np.isclose(row["auction_stage2_range_bps"], 40.0)
    assert np.isclose(row["auction_stage2_efficiency_ratio"], 1.0)
    assert np.isclose(row["auction_unmatched_imbalance"], 1.0)

    expected_twap = (10.02 * 120 + 10.04 * 179 + 10.06) / 300
    expected_imbalance_twap = (0.6 * 120 + 0.2 * 179 + 0.8) / 300
    assert row["auction_stage2_twap_coverage_ratio"] == 1.0
    assert np.isclose(row["auction_stage2_twap_price"], expected_twap)
    assert np.isclose(row["auction_final_vs_stage2_twap"], 10.05 / expected_twap - 1)
    assert np.isclose(
        row["auction_l3_imbalance_twap_stage2"], expected_imbalance_twap
    )
    assert row["auction_relative_spread_twap_stage2"] == 0.0

    expected_slope = (
        np.polyfit(
            np.array([0.0, 2.0, 4.0 + 59.0 / 60.0]),
            np.log(np.array([10.02, 10.04, 10.06]) / 10.0),
            1,
        )[0]
        * 10000.0
    )
    assert np.isclose(row["auction_stage2_slope_bps_per_min"], expected_slope)


def test_stage2_twap_does_not_carry_values_across_invalid_snapshots() -> None:
    quotes = _auction_quotes()
    invalid = quotes["trade_time"].eq(pd.Timestamp("2026-03-31 09:22:00"))
    quotes.loc[invalid, ["ask_price1", "bid_price1"]] = [9.99, 10.01]
    quotes.loc[invalid, ["bid_qty1", "ask_qty1", "bid_qty2", "ask_qty2"]] = 0.0

    row = calculate_daily_auction_factors(quotes, "000001.SZ")

    assert np.isclose(row["auction_stage2_twap_coverage_ratio"], 121 / 300)
    assert np.isnan(row["auction_l3_imbalance_twap_stage2"])
    assert np.isnan(row["auction_relative_spread_twap_stage2"])


def test_quote_at_0925_is_not_part_of_the_auction_path() -> None:
    quotes = _auction_quotes()
    after_stage2 = _quote_row("2026-03-31 09:25:00", 99.0, 1, 999)
    quotes = pd.concat([quotes, pd.DataFrame([after_stage2])], ignore_index=True)
    quotes = quotes.sort_values("trade_time", kind="mergesort").reset_index(drop=True)

    row = calculate_daily_auction_factors(quotes, "000001.SZ")

    assert row["snapshot_count_stage2"] == 3
    assert np.isclose(row["auction_final_indicative_price"], 10.06)


def test_second_batch_event_factor_formulas_and_boundaries() -> None:
    row = calculate_daily_auction_factors(
        _auction_quotes(), "000001.SZ", _auction_events(), True
    )

    assert row["auction_event_reconstruction_ok"] is True
    assert row["auction_add_count_stage1"] == 2
    assert row["auction_cancel_count_stage1"] == 2
    assert row["auction_add_count_stage2"] == 3
    assert np.isclose(row["auction_stage1_add_notional"], 3000.0)
    assert np.isclose(row["auction_stage1_cancel_notional"], 700.0)
    assert np.isclose(row["auction_stage2_add_notional"], 5000.0)
    assert np.isclose(row["auction_bid_cancel_qty_ratio_stage1"], 0.2)
    assert np.isclose(row["auction_ask_cancel_qty_ratio_stage1"], 0.25)
    assert np.isclose(row["auction_cancel_notional_ratio_stage1"], 700 / 3000)
    assert np.isclose(row["auction_cancel_imbalance_stage1"], 300 / 700)
    assert np.isclose(row["auction_late_cancel_notional_share"], 500 / 700)
    assert np.isclose(row["auction_stage2_add_imbalance"], 0.6)
    assert np.isclose(row["auction_stage2_commitment_ratio"], 5000 / 7300)
    assert np.isclose(row["auction_stage2_last60s_add_share"], 0.2)
    assert row["auction_submitted_volume"] == 800.0
    assert np.isclose(row["auction_matched_volume_to_submitted_ratio"], 1.25)
    expected_fake_pressure = (-1000 / 3000) - ((800 - 1500) / 2300)
    assert np.isclose(row["auction_fake_pressure_proxy"], expected_fake_pressure)
    assert row["auction_stage_reversal_strength_bps"] == 0.0


def test_third_batch_price_path_and_robust_imbalance_formulas() -> None:
    row = calculate_daily_auction_factors(_auction_quotes(), "000001.SZ")

    assert np.isclose(row["auction_stage2_mid_mean_return"], 10.04 / 10.0 - 1.0)
    assert np.isclose(row["auction_stage2_mid_max_return"], 10.06 / 10.0 - 1.0)
    assert np.isclose(row["auction_stage2_mid_min_return"], 10.02 / 10.0 - 1.0)
    assert np.isclose(row["auction_stage2_total_variation_bps"], 40.0)
    assert row["auction_stage2_up_step_ratio"] == 1.0
    assert row["auction_stage2_reversal_count"] == 0
    assert np.isclose(row["auction_imbalance_relative_change_stage1"], 1.0)
    assert np.isclose(row["auction_imbalance_relative_change_stage2"], 1 / 3)
    assert np.isclose(
        row["auction_imbalance_fisher_change_stage1"],
        np.arctanh(0.4) - np.arctanh(0.2),
    )
    assert np.isclose(
        row["auction_imbalance_fisher_change_stage2"],
        np.arctanh(0.8) - np.arctanh(0.6),
    )


def test_flat_and_reversing_stage2_paths_have_stable_direction_statistics() -> None:
    flat = calculate_daily_auction_factors(
        _auction_quotes(constant_stage2=True), "000001.SZ"
    )
    assert flat["auction_stage2_total_variation_bps"] == 0.0
    assert flat["auction_stage2_up_step_ratio"] == 0.0
    assert flat["auction_stage2_reversal_count"] == 0

    quotes = _auction_quotes()
    stage2_indexes = quotes.index[
        quotes["trade_time"].between("2026-03-31 09:20", "2026-03-31 09:24:59")
    ]
    quotes.loc[stage2_indexes, ["ask_price1", "bid_price1"]] = np.array(
        [[10.02, 10.02], [10.06, 10.06], [10.01, 10.01]]
    )
    reversing = calculate_daily_auction_factors(quotes, "000001.SZ")
    assert reversing["auction_stage2_up_step_ratio"] == 0.5
    assert reversing["auction_stage2_reversal_count"] == 1


def test_stage_reversal_is_signed_and_missing_events_do_not_remove_quote_factor() -> (
    None
):
    quotes = _auction_quotes()
    quotes.loc[quotes["trade_time"].ge("2026-03-31 09:20"), "ask_price1"] = [
        10.02,
        10.00,
        9.98,
        10.05,
    ]
    quotes.loc[quotes["trade_time"].ge("2026-03-31 09:20"), "bid_price1"] = [
        10.02,
        10.00,
        9.98,
        10.05,
    ]
    row = calculate_daily_auction_factors(quotes, "000001.SZ")

    stage1_return = 10.01 / 10.00 - 1.0
    stage2_return = 9.98 / 10.02 - 1.0
    expected = -min(abs(stage1_return), abs(stage2_return)) * 10000
    assert row["auction_event_reconstruction_ok"] is False
    assert np.isclose(row["auction_stage_reversal_strength_bps"], expected)
    assert np.isnan(row["auction_bid_cancel_qty_ratio_stage1"])


def test_zero_cancellations_are_zero_not_missing() -> None:
    events = _auction_events().loc[lambda frame: ~frame["event_type"].eq("C")]
    row = calculate_daily_auction_factors(_auction_quotes(), "000001.SZ", events, True)

    assert row["auction_bid_cancel_qty_ratio_stage1"] == 0.0
    assert row["auction_ask_cancel_qty_ratio_stage1"] == 0.0
    assert row["auction_cancel_notional_ratio_stage1"] == 0.0
    assert row["auction_cancel_imbalance_stage1"] == 0.0
    assert row["auction_late_cancel_notional_share"] == 0.0


def test_one_sided_orders_and_fully_cancelled_book_keep_nan_semantics() -> None:
    one_sided = (
        _auction_events()
        .loc[lambda frame: ~frame["side"].eq("S")]
        .reset_index(drop=True)
    )
    row = calculate_daily_auction_factors(
        _auction_quotes(), "000001.SZ", one_sided, True
    )
    assert np.isnan(row["auction_ask_cancel_qty_ratio_stage1"])
    assert np.isfinite(row["auction_bid_cancel_qty_ratio_stage1"])

    fully_cancelled = pd.DataFrame(
        [
            _event("2026-03-31 09:15:00", "A", "B", 1, 10.0, 100),
            _event("2026-03-31 09:15:01", "A", "S", 2, 10.0, 100),
            _event(
                "2026-03-31 09:18:00",
                "C",
                "B",
                1,
                10.0,
                100,
                original_quantity=100,
            ),
            _event(
                "2026-03-31 09:18:01",
                "C",
                "S",
                2,
                10.0,
                100,
                original_quantity=100,
            ),
        ]
    )
    cancelled_row = calculate_daily_auction_factors(
        _auction_quotes(), "000001.SZ", fully_cancelled, True
    )
    assert np.isnan(cancelled_row["auction_fake_pressure_proxy"])
    assert np.isnan(cancelled_row["auction_stage2_commitment_ratio"])
    values = pd.to_numeric(
        pd.Series([cancelled_row[column] for column in EVENT_FACTOR_COLUMNS]),
        errors="coerce",
    )
    assert not np.isinf(values.to_numpy(dtype=float)).any()


def test_no_match_keeps_causal_path_factors_and_match_fields_missing() -> None:
    row = calculate_daily_auction_factors(
        _auction_quotes(include_match=False), "000001.SZ"
    )

    assert row["auction_has_match"] is False
    assert row["available_time"] == pd.Timestamp("2026-03-31 09:25:00")
    assert np.isnan(row["auction_open_price"])
    assert np.isnan(row["auction_overnight_return"])
    assert np.isfinite(row["auction_return_stage1"])
    assert np.isfinite(row["auction_return_stage2"])


def test_constant_stage2_has_zero_efficiency_and_insufficient_slope_is_nan() -> None:
    constant = calculate_daily_auction_factors(
        _auction_quotes(constant_stage2=True), "000001.SZ"
    )
    assert constant["auction_stage2_efficiency_ratio"] == 0.0
    assert np.isclose(constant["auction_stage2_slope_bps_per_min"], 0.0)

    quotes = _auction_quotes().drop(index=[3, 4]).reset_index(drop=True)
    too_short = calculate_daily_auction_factors(quotes, "000001.SZ")
    assert np.isnan(too_short["auction_stage2_slope_bps_per_min"])


def test_zero_depth_and_zero_previous_close_do_not_create_infinity() -> None:
    quotes = _auction_quotes()
    quotes.loc[0, ["bid_qty1", "ask_qty1"]] = 0.0
    quotes["previous_close"] = 0.0
    row = calculate_daily_auction_factors(quotes, "000001.SZ")

    numeric_values = pd.to_numeric(
        pd.Series([row[column] for column in OUTPUT_COLUMNS[6:]]), errors="coerce"
    )
    assert not np.isinf(numeric_values.to_numpy(dtype=float)).any()
    assert np.isnan(row["auction_overnight_return"])
    assert np.isnan(row["auction_stage2_range_bps"])


def test_zero_initial_imbalance_uses_relative_floor_without_infinity() -> None:
    quotes = _auction_quotes()
    quotes.loc[0, ["bid_qty1", "ask_qty1"]] = [500.0, 500.0]
    row = calculate_daily_auction_factors(quotes, "000001.SZ")

    assert np.isclose(row["auction_imbalance_relative_change_stage1"], 8.0)
    assert np.isfinite(row["auction_imbalance_fisher_change_stage1"])


def _historical_frame(amounts: list[float]) -> pd.DataFrame:
    rows = []
    for offset, amount in enumerate(amounts):
        row = {column: np.nan for column in OUTPUT_COLUMNS}
        row.update(
            {
                "trade_date": (
                    pd.Timestamp("2026-01-01") + pd.Timedelta(days=offset)
                ).strftime("%Y-%m-%d"),
                "available_time": pd.Timestamp("2026-01-01 09:25")
                + pd.Timedelta(days=offset),
                "ts_code": "000001.SZ",
                "auction_has_match": True,
                "snapshot_count_stage1": 3,
                "snapshot_count_stage2": 3,
                "auction_amount": amount,
                "auction_matched_volume": amount / 10.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_history_ratios_use_only_previous_five_valid_days() -> None:
    base = apply_historical_ratios(_historical_frame([10, 20, 30, 40, 50, 60, 70]))
    assert base["auction_amount_ratio_5d"].iloc[:5].isna().all()
    assert np.isclose(base.loc[5, "auction_amount_ratio_5d"], 60 / 30)
    assert np.isclose(base.loc[6, "auction_amount_ratio_5d"], 70 / 40)

    changed_future = apply_historical_ratios(
        _historical_frame([10, 20, 30, 40, 50, 60, 700000])
    )
    assert np.isclose(
        base.loc[5, "auction_amount_ratio_5d"],
        changed_future.loc[5, "auction_amount_ratio_5d"],
    )


def test_history_skips_invalid_days_when_selecting_five_observations() -> None:
    frame = _historical_frame([10, 20, np.nan, 30, 40, 50, 60])
    result = apply_historical_ratios(frame)
    assert np.isnan(result.loc[5, "auction_amount_ratio_5d"])
    assert np.isclose(result.loc[6, "auction_amount_ratio_5d"], 60 / 30)


def test_twenty_day_auction_zscore_and_prior_adv_are_strictly_historical() -> None:
    auction_amounts = [float(value) for value in range(10, 230, 10)]
    frame = _historical_frame(auction_amounts)
    daily_index = pd.date_range("2026-01-01", periods=22, freq="D")
    daily_amounts = pd.Series(np.arange(1.0, 23.0) * 1000.0, index=daily_index)

    result = apply_historical_ratios(frame, daily_amount_history=daily_amounts)
    recent_auction = np.arange(10.0, 201.0, 10.0)
    expected_adv = np.arange(1000.0, 21000.0, 1000.0).mean()
    assert np.isnan(result.loc[19, "auction_amount_zscore_20d"])
    assert np.isclose(
        result.loc[20, "auction_amount_zscore_20d"],
        (210.0 - recent_auction.mean()) / recent_auction.std(ddof=0),
    )
    assert np.isclose(result.loc[20, "previous_20d_average_daily_amount"], expected_adv)
    assert np.isclose(
        result.loc[20, "auction_amount_to_prev20d_adv"], 210.0 / expected_adv
    )
    expected_5d_adv = np.arange(16000.0, 21000.0, 1000.0).mean()
    assert np.isclose(
        result.loc[20, "previous_5d_average_daily_amount"], expected_5d_adv
    )
    assert np.isclose(
        result.loc[20, "auction_amount_to_prev5d_adv_240"],
        210.0 / (expected_5d_adv / 240.0),
    )

    changed_future = daily_amounts.copy()
    changed_future.iloc[21] = 999_999_999.0
    changed = apply_historical_ratios(frame, daily_amount_history=changed_future)
    assert np.isclose(
        result.loc[20, "auction_amount_to_prev20d_adv"],
        changed.loc[20, "auction_amount_to_prev20d_adv"],
    )

    changed_auction_frame = frame.copy()
    changed_auction_frame.loc[21, "auction_amount"] = 999_999_999.0
    changed_auction = apply_historical_ratios(
        changed_auction_frame, daily_amount_history=daily_amounts
    )
    assert np.isclose(
        result.loc[20, "auction_amount_zscore_20d"],
        changed_auction.loc[20, "auction_amount_zscore_20d"],
    )


def test_prev5d_adv_requires_five_consecutive_valid_observations() -> None:
    frame = _historical_frame([10.0] * 7)
    daily_index = pd.date_range("2026-01-01", periods=7, freq="D")
    daily_amounts = pd.Series([1000, 2000, np.nan, 4000, 5000, 6000, 7000], index=daily_index)

    result = apply_historical_ratios(frame, daily_amount_history=daily_amounts)

    assert np.isnan(result.loc[5, "previous_5d_average_daily_amount"])
    assert np.isnan(result.loc[5, "auction_amount_to_prev5d_adv_240"])
    assert np.isnan(result.loc[6, "auction_amount_to_prev5d_adv_240"])


def test_daily_amount_history_sums_minute_bars_by_trade_date(tmp_path) -> None:
    index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            pd.to_datetime(
                [
                    "2026-01-01 09:30",
                    "2026-01-01 09:31",
                    "2026-01-02 09:30",
                ]
            ),
        ],
        names=["trade_date", "trade_time"],
    )
    minute_path = tmp_path / "000001.SZ.parquet"
    pd.DataFrame({"amount": [100.0, 200.0, 400.0]}, index=index).to_parquet(minute_path)

    daily = load_daily_amount_history(minute_path)

    assert daily.to_dict() == {
        pd.Timestamp("2026-01-01"): 300.0,
        pd.Timestamp("2026-01-02"): 400.0,
    }


def test_historical_context_uses_prior_day_and_full_universe_ranks(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=23, freq="D")
    closes_by_code = {
        "000001.SZ": 100.0 + np.arange(23),
        "000002.SZ": 100.0 + np.arange(23) * 2.0,
        "000003.SZ": np.full(23, 100.0),
    }
    rows = []
    for ts_code, closes in closes_by_code.items():
        for trade_date, close in zip(dates, closes, strict=True):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "close": close,
                    "high": close + 10.0,
                    "low": close - 10.0,
                    "pre_close": close - 1.0,
                    "adj_factor": 1.0,
                }
            )
    daily_path = tmp_path / "daily.parquet"
    pd.DataFrame(rows).set_index(["trade_date", "ts_code"]).to_parquet(daily_path)

    contexts = build_historical_context(
        daily_path, ["2026-01-23"], {"000001.SZ"}
    )
    context = contexts["000001.SZ"].iloc[0]

    assert context["trade_date"] == "2026-01-23"
    assert np.isclose(
        context["prevday_intraday_drawdown_from_session_high"], 121 / 131 - 1
    )
    assert np.isclose(
        context["prevday_intraday_rebound_from_session_low"], 121 / 111 - 1
    )
    assert np.isclose(
        context["prevday_intraday_return_from_prev_close"], 121 / 120 - 1
    )
    assert np.isclose(context["prev_2d_return_rank_cs"], 2 / 3)
    assert np.isclose(context["prev_20d_return_rank_cs"], 2 / 3)
    assert context["_market_above_ma20"] == 1.0
    assert np.isclose(context["_prev_2d_return"], 121 / 119 - 1)

    suspended_path = tmp_path / "daily_with_suspension.parquet"
    suspended = pd.DataFrame(rows).loc[
        lambda frame: ~(
            frame["ts_code"].eq("000001.SZ")
            & frame["trade_date"].eq(pd.Timestamp("2026-01-20"))
        )
    ]
    suspended.set_index(["trade_date", "ts_code"]).to_parquet(suspended_path)
    suspended_context = build_historical_context(
        suspended_path, ["2026-01-23"], {"000001.SZ"}
    )["000001.SZ"].iloc[0]
    assert np.isnan(suspended_context["_prev_2d_return"])
    assert np.isnan(suspended_context["prev_2d_return_rank_cs"])
    assert np.isnan(suspended_context["_market_above_ma20"])


def test_external_context_applies_benchmark_time_and_excess_returns() -> None:
    frame = _historical_frame([100.0])
    frame.loc[0, "auction_overnight_return"] = 0.02
    frame.loc[0, "auction_return_stage2"] = 0.01
    frame.loc[0, "available_time"] = pd.Timestamp("2026-01-01 09:25:02")
    symbol_context = pd.DataFrame(
        [
            {
                "trade_date": "2026-01-01",
                "prevday_intraday_drawdown_from_session_high": -0.03,
                "prevday_intraday_rebound_from_session_low": 0.04,
                "prevday_intraday_return_from_prev_close": 0.01,
                "prev_2d_return_rank_cs": 0.75,
                "prev_20d_return_rank_cs": 0.6,
            }
        ]
    )
    benchmark_context = pd.DataFrame(
        [
            {
                "trade_date": "2026-01-01",
                "benchmark_ts_code": "510300.SH",
                "benchmark_available_time": pd.Timestamp("2026-01-01 09:25:03"),
                "benchmark_auction_has_match": True,
                "market_return_from_prev_close": 0.005,
                "_benchmark_auction_return_stage2": 0.004,
                "market_above_ma20_prevclose": 1.0,
                "market_momentum_2d_prevclose": 0.02,
            }
        ]
    )

    result = apply_external_context(frame, symbol_context, benchmark_context).iloc[0]

    assert result["available_time"] == pd.Timestamp("2026-01-01 09:25:03")
    assert result["benchmark_ts_code"] == "510300.SH"
    assert result["prev_2d_return_rank_cs"] == 0.75
    assert result["market_above_ma20_prevclose"] == 1.0
    assert np.isclose(result["auction_gap_excess_benchmark"], 0.015)
    assert np.isclose(result["auction_stage2_excess_return_benchmark"], 0.006)


def test_session_path_companion_is_minute_causal_and_resets_at_lunch(tmp_path) -> None:
    trade_dates = pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-02"]
    )
    trade_times = pd.to_datetime(
        [
            "2026-01-01 14:59",
            "2026-01-02 09:30",
            "2026-01-02 09:31",
            "2026-01-02 13:00",
        ]
    )
    index = pd.MultiIndex.from_arrays(
        [trade_dates, trade_times], names=["trade_date", "trade_time"]
    )
    minute_path = tmp_path / "000001.SZ.parquet"
    pd.DataFrame(
        {
            "high": [10.0, 10.5, 11.0, 10.3],
            "low": [9.8, 10.0, 10.2, 10.1],
            "close": [10.0, 10.4, 10.3, 10.2],
        },
        index=index,
    ).to_parquet(minute_path)

    result = build_session_path_factor_frame(
        minute_path, "000001.SZ", {"2026-01-02"}
    )

    assert len(result) == 3
    assert result.loc[0, "available_time"] == pd.Timestamp("2026-01-02 09:31")
    assert np.isclose(
        result.loc[1, "intraday_drawdown_from_session_high"], 10.3 / 11.0 - 1
    )
    assert np.isclose(
        result.loc[1, "intraday_rebound_from_session_low"], 10.3 / 10.0 - 1
    )
    assert np.isclose(
        result.loc[2, "intraday_drawdown_from_session_high"], 10.2 / 10.3 - 1
    )
    assert np.isclose(result.loc[2, "intraday_rebound_from_session_low"], 10.2 / 10.1 - 1)
    assert np.allclose(result["intraday_return_from_prev_close"], [0.04, 0.03, 0.02])


def _dated_event(
    trade_date: pd.Timestamp,
    event_type: str,
    quantity: float,
    *,
    original_quantity: float | None = None,
) -> pd.DataFrame:
    timestamp = trade_date + pd.Timedelta(hours=9, minutes=15)
    return pd.DataFrame(
        [
            _event(
                str(timestamp),
                event_type,
                "B",
                1,
                1.0,
                quantity,
                original_quantity=original_quantity,
            )
        ]
    )


def test_large_order_threshold_uses_only_prior_twenty_valid_days() -> None:
    frame = _historical_frame([100.0] * 22)
    frame["auction_event_reconstruction_ok"] = True
    event_frames: dict[str, pd.DataFrame] = {}
    for index in range(20):
        trade_day = pd.Timestamp(frame.loc[index, "trade_date"])
        event_frames[frame.loc[index, "trade_date"]] = _dated_event(
            trade_day, "A", (index + 1) * 10.0
        )

    current_day = pd.Timestamp(frame.loc[20, "trade_date"])
    current_add = _dated_event(current_day, "A", 500.0)
    current_cancel = _dated_event(current_day, "C", 250.0, original_quantity=500.0)
    event_frames[frame.loc[20, "trade_date"]] = pd.concat(
        [current_add, current_cancel], ignore_index=True
    )
    future_day = pd.Timestamp(frame.loc[21, "trade_date"])
    event_frames[frame.loc[21, "trade_date"]] = _dated_event(
        future_day, "A", 10_000_000.0
    )

    result = apply_historical_ratios(frame, event_frames)

    assert result.loc[19, "auction_large_order_history_days"] == 19
    assert np.isnan(result.loc[19, "auction_large_order_threshold"])
    assert result.loc[20, "auction_large_order_history_days"] == 20
    assert np.isclose(
        result.loc[20, "auction_large_order_threshold"],
        np.quantile(np.arange(10.0, 201.0, 10.0), 0.9),
    )
    assert np.isclose(result.loc[20, "auction_large_order_cancel_ratio_stage1"], 0.5)
    assert result.loc[20, "auction_large_cancel_imbalance_stage1"] == -1.0

    changed_future = event_frames.copy()
    changed_future[frame.loc[21, "trade_date"]] = _dated_event(future_day, "A", 1.0)
    changed = apply_historical_ratios(frame, changed_future)
    assert np.isclose(
        result.loc[20, "auction_large_order_threshold"],
        changed.loc[20, "auction_large_order_threshold"],
    )


def test_large_order_factors_distinguish_no_large_order_and_no_large_cancel() -> None:
    frame = _historical_frame([100.0] * 22)
    frame["auction_event_reconstruction_ok"] = True
    event_frames: dict[str, pd.DataFrame] = {}
    for index in range(20):
        trade_day = pd.Timestamp(frame.loc[index, "trade_date"])
        event_frames[frame.loc[index, "trade_date"]] = _dated_event(
            trade_day, "A", (index + 1) * 10.0
        )
    no_large_day = pd.Timestamp(frame.loc[20, "trade_date"])
    event_frames[frame.loc[20, "trade_date"]] = _dated_event(no_large_day, "A", 10.0)
    large_no_cancel_day = pd.Timestamp(frame.loc[21, "trade_date"])
    event_frames[frame.loc[21, "trade_date"]] = _dated_event(
        large_no_cancel_day, "A", 500.0
    )

    result = apply_historical_ratios(frame, event_frames)

    assert np.isnan(result.loc[20, "auction_large_order_cancel_ratio_stage1"])
    assert np.isnan(result.loc[20, "auction_large_cancel_imbalance_stage1"])
    assert result.loc[21, "auction_large_order_cancel_ratio_stage1"] == 0.0
    assert result.loc[21, "auction_large_cancel_imbalance_stage1"] == 0.0


def _write_raw_order_file(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path / "逐笔委托.csv", index=False, encoding="gbk")


def _raw_order(
    raw_time: int,
    order_id: int,
    order_type: str,
    side: str,
    quantity: int,
) -> dict[str, object]:
    return {
        "自然日": 20260331,
        "时间": raw_time,
        "交易所委托号": order_id,
        "委托类型": order_type,
        "委托代码": side,
        "委托价格": 100000,
        "委托数量": quantity,
    }


def _raw_cancel(
    raw_time: int,
    quantity: int,
    *,
    ask_order_id: int = 0,
    bid_order_id: int = 0,
) -> dict[str, object]:
    return {
        "自然日": 20260331,
        "时间": raw_time,
        "成交代码": "C",
        "成交数量": quantity,
        "叫卖序号": ask_order_id,
        "叫买序号": bid_order_id,
    }


def test_sh_and_sz_raw_cancellation_encodings_reconstruct_equivalent_events(
    tmp_path,
) -> None:
    sh_dir = tmp_path / "510300.SZ"
    sz_dir = tmp_path / "000001.SZ"
    sh_dir.mkdir()
    sz_dir.mkdir()
    adds = [
        _raw_order(91500000, 1, "A", "B", 100),
        _raw_order(91500010, 2, "A", "S", 200),
    ]
    _write_raw_order_file(
        sh_dir,
        adds
        + [
            _raw_order(91800000, 1, "D", "B", 20),
            _raw_order(91930000, 2, "D", "S", 50),
        ],
    )
    _write_raw_order_file(
        sz_dir,
        [dict(row, 委托类型="0") for row in adds],
    )
    pd.DataFrame(
        [
            _raw_cancel(91800000, 20, bid_order_id=1),
            _raw_cancel(91930000, 50, ask_order_id=2),
        ]
    ).to_csv(sz_dir / "逐笔成交.csv", index=False, encoding="gbk")

    sh_events, sh_ok = load_auction_event_frame(sh_dir, "510300.SH")
    sz_events, sz_ok = load_auction_event_frame(sz_dir, "000001.SZ")

    assert sh_ok is True
    assert sz_ok is True
    columns = ["event_type", "side", "order_id", "price", "quantity", "notional"]
    pd.testing.assert_frame_equal(
        sh_events[columns].reset_index(drop=True),
        sz_events[columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_invalid_or_stage2_cancellation_marks_reconstruction_failed(tmp_path) -> None:
    raw_dir = tmp_path / "600000.SH"
    raw_dir.mkdir()
    _write_raw_order_file(
        raw_dir,
        [
            _raw_order(91500000, 1, "A", "B", 100),
            _raw_order(92000000, 1, "D", "B", 20),
        ],
    )
    events, ok = load_auction_event_frame(raw_dir, "600000.SH")
    row = calculate_daily_auction_factors(_auction_quotes(), "600000.SH", events, ok)

    assert ok is False
    assert row["auction_event_reconstruction_ok"] is False
    assert all(np.isnan(row[column]) for column in EVENT_FACTOR_COLUMNS[:-1])


def test_duplicate_and_over_cancelled_orders_fail_reconstruction(tmp_path) -> None:
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    _write_raw_order_file(
        duplicate_dir,
        [
            _raw_order(91500000, 1, "A", "B", 100),
            _raw_order(91500010, 1, "A", "B", 100),
        ],
    )
    _, duplicate_ok = load_auction_event_frame(duplicate_dir, "600000.SH")

    over_cancel_dir = tmp_path / "over_cancel"
    over_cancel_dir.mkdir()
    _write_raw_order_file(
        over_cancel_dir,
        [
            _raw_order(91500000, 1, "A", "B", 100),
            _raw_order(91800000, 1, "D", "B", 101),
        ],
    )
    _, over_cancel_ok = load_auction_event_frame(over_cancel_dir, "600000.SH")

    assert duplicate_ok is False
    assert over_cancel_ok is False


def test_unmatched_sz_cancellation_fails_reconstruction(tmp_path) -> None:
    raw_dir = tmp_path / "000001.SZ"
    raw_dir.mkdir()
    _write_raw_order_file(raw_dir, [_raw_order(91500000, 1, "0", "B", 100)])
    pd.DataFrame([_raw_cancel(91800000, 20, bid_order_id=999)]).to_csv(
        raw_dir / "逐笔成交.csv", index=False, encoding="gbk"
    )

    events, ok = load_auction_event_frame(raw_dir, "000001.SZ")

    assert not events.empty
    assert ok is False


def test_universe_uses_canonical_minute_suffix_and_detects_raw_code() -> None:
    # Raw tick suffixes are deliberately not involved in universe ownership.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        stock_root = root / "stocks"
        etf_root = root / "etfs"
        stock_root.mkdir()
        etf_root.mkdir()
        (stock_root / "000001.SZ.parquet").touch()
        (etf_root / "510300.SH.parquet").touch()

        assets = build_asset_universe("both", stock_root, etf_root, None)

    assert assets == [
        ("etf", "510300", "510300.SH"),
        ("stock", "000001", "000001.SZ"),
    ]


def test_group_symbol_paths_ignores_wrong_raw_suffix(tmp_path) -> None:
    date_dir = tmp_path / "2026" / "202603" / "20260331"
    wrong_suffix = date_dir / "510300.SZ"
    wrong_suffix.mkdir(parents=True)
    (date_dir / "000001.SZ").mkdir()

    grouped = group_symbol_paths([date_dir], {"510300"})

    assert grouped == {"510300": [wrong_suffix]}


def test_incremental_merge_preserves_or_replaces_requested_dates(tmp_path) -> None:
    output_path = tmp_path / "000001.SZ.parquet"
    existing = apply_historical_ratios(_historical_frame([10, 20]))
    existing.to_parquet(output_path, index=False)

    replacement = existing.iloc[[1]].copy()
    replacement.loc[:, "auction_amount"] = 999.0

    preserved = merge_symbol_output(output_path, replacement, overwrite=False)
    assert preserved.loc[1, "auction_amount"] == 20.0

    overwritten = merge_symbol_output(output_path, replacement, overwrite=True)
    assert overwritten.loc[1, "auction_amount"] == 999.0
    assert overwritten["trade_date"].tolist() == ["2026-01-01", "2026-01-02"]
