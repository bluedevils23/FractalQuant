from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.intraday_strategy import (  # noqa: E402
    COMBINED_FACTOR_COLUMNS,
    COMBINED_OUTPUT_COLUMNS,
    FACTOR_COLUMNS,
    GFTD_FACTOR_COLUMNS,
    OPENING_PATH_FACTOR_COLUMNS,
    OPENING_POLY_FACTOR_COLUMNS,
    OUTPUT_COLUMNS,
    PRIORITY_PROFILE_P0,
    PRIORITY_PROFILE_P0_P1,
    build_daily_context,
    build_intraday_strategy_factor_frame,
    calculate_gftd_state,
    factor_columns_for_profile,
    output_columns_for_profile,
)
from factor.intraday_strategy_p1 import (  # noqa: E402
    EARLY_RANGE_FACTOR_COLUMNS,
    MARKET_FLOW_FACTOR_COLUMNS,
    PATH_KNN_FACTOR_COLUMNS,
    P1_FACTOR_COLUMNS,
    VOLATILITY_CONVERGENCE_FACTOR_COLUMNS,
    active_notional_imbalance,
    aggregate_directional_notional,
)
from scripts.generate_intraday_strategy_factors import (  # noqa: E402
    DEFAULT_ETF_P0_P1_OUTPUT_ROOT,
    DEFAULT_STOCK_P0_P1_OUTPUT_ROOT,
    build_active_flow_context,
    load_pool_membership,
    load_target_pool_mapping,
    merge_output,
    process_symbol_file,
    resolve_output_root,
)


TS_CODE = "000001.SZ"


def _flow_context(minute: pd.DataFrame, value: float = 0.0) -> pd.DataFrame:
    keys = minute.reset_index()[["trade_date", "trade_time"]]
    for column in MARKET_FLOW_FACTOR_COLUMNS:
        keys[column] = value
    return keys


def _daily_history(periods: int = 40, start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    index = np.arange(periods, dtype=float)
    open_price = 100.0 + 0.2 * index
    intraday_return = 0.01 * np.sin(index / 2.0) + 0.0002 * index
    close = open_price * (1.0 + intraday_return)
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": TS_CODE,
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0 + 0.05 * index,
            "low": np.minimum(open_price, close) - 1.0 - 0.03 * index,
            "close": close,
            "adj_factor": np.ones(periods),
        }
    )
    return frame


def _minute_day(
    trade_date: str,
    close: np.ndarray,
    *,
    times: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    if times is None:
        times = pd.date_range(f"{trade_date} 09:30", periods=len(close), freq="min")
    trade_dates = pd.DatetimeIndex([pd.Timestamp(trade_date)] * len(close))
    index = pd.MultiIndex.from_arrays(
        [trade_dates, times], names=["trade_date", "trade_time"]
    )
    return pd.DataFrame(
        {
            "ts_code": TS_CODE,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "vol": np.full(len(close), 1000.0),
            "amount": close * 1000.0,
            "adj_factor": np.ones(len(close)),
        },
        index=index,
    )


def test_registry_has_six_groups_and_twenty_one_unique_factors() -> None:
    assert len(FACTOR_COLUMNS) == 21
    assert len(FACTOR_COLUMNS) == len(set(FACTOR_COLUMNS))
    assert len(OPENING_PATH_FACTOR_COLUMNS) == 3
    assert len(OPENING_POLY_FACTOR_COLUMNS) == 3
    assert len(GFTD_FACTOR_COLUMNS) == 5


def test_opening_path_and_polynomial_factors_use_fixed_completed_prefix() -> None:
    x = np.linspace(-1.0, 1.0, 60)
    log_price = 0.01 * x**2 + 0.02 * x
    prices = 100.0 * np.exp(log_price - log_price[0])
    minute = _minute_day("2025-03-03", np.r_[prices, 200.0])
    result = build_intraday_strategy_factor_frame(minute, _daily_history(), TS_CODE)

    assert result.loc[:48, OPENING_PATH_FACTOR_COLUMNS].isna().all().all()
    assert np.isclose(result.loc[49, "opening_path_mean_drawdown_50bar"], 0.0)
    assert np.isclose(result.loc[49, "opening_path_smoothness_50bar"], 0.0)
    assert result.loc[:58, OPENING_POLY_FACTOR_COLUMNS].isna().all().all()
    assert np.isclose(result.loc[59, "opening_poly_slope_60bar"], 0.02)
    assert np.isclose(result.loc[59, "opening_poly_curvature_60bar"], 0.02)
    assert np.isclose(result.loc[59, "opening_poly_trend_acceleration_60bar"], 0.0004)

    changed = minute.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] = 9999.0
    changed_result = build_intraday_strategy_factor_frame(
        changed, _daily_history(), TS_CODE
    )
    pd.testing.assert_frame_equal(
        result[OPENING_PATH_FACTOR_COLUMNS + OPENING_POLY_FACTOR_COLUMNS],
        changed_result[OPENING_PATH_FACTOR_COLUMNS + OPENING_POLY_FACTOR_COLUMNS],
    )


def test_gftd_buy_and_sell_state_machine_counts_after_setup() -> None:
    buy_close = np.array([10.0, 9.0, 8.0, 9.0, 10.0])
    buy = calculate_gftd_state(buy_close, buy_close, buy_close, n1=1, n2=2, n3=2)
    assert buy.loc[2, "gftd_setup_direction_5_3"] == 1.0
    assert buy.loc[3, "gftd_buy_count_6"] == 1.0
    assert buy.loc[4, "gftd_buy_count_6"] == 2.0
    assert buy.loc[4, "gftd_signal_state_5_3_6"] == 1.0

    sell_close = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
    sell = calculate_gftd_state(sell_close, sell_close, sell_close, n1=1, n2=2, n3=2)
    assert sell.loc[2, "gftd_setup_direction_5_3"] == -1.0
    assert sell.loc[3, "gftd_sell_count_6"] == 1.0
    assert sell.loc[4, "gftd_sell_count_6"] == 2.0
    assert sell.loc[4, "gftd_signal_state_5_3_6"] == -1.0


def test_gftd_resets_at_lunch_but_delayed_extreme_keeps_morning_path() -> None:
    morning_times = pd.date_range("2025-03-03 11:15", periods=16, freq="min")
    afternoon_times = pd.date_range("2025-03-03 13:01", periods=8, freq="min")
    times = morning_times.append(afternoon_times)
    prices = np.r_[np.linspace(100.0, 110.0, 16), np.linspace(105.0, 108.0, 8)]
    minute = _minute_day("2025-03-03", prices, times=times)
    result = build_intraday_strategy_factor_frame(minute, _daily_history(), TS_CODE)

    first_pm = 16
    assert result.loc[first_pm : first_pm + 4, "gftd_setup_streak_5"].eq(0).all()
    expected = prices[first_pm] / (prices[14] + 0.05) - 1.0
    assert np.isclose(
        result.loc[first_pm, "distance_to_delayed_session_high_lag2"], expected
    )


def test_daily_context_is_strictly_historical_and_matches_formulas() -> None:
    daily = _daily_history()
    target = daily["trade_date"].max() + pd.offsets.BDay(1)
    context = build_daily_context(daily, [target], TS_CODE).iloc[0]

    adjusted_high = daily["high"].to_numpy()
    adjusted_low = daily["low"].to_numpy()
    adjusted_close = daily["close"].to_numpy()
    previous_close = np.r_[np.nan, adjusted_close[:-1]]
    true_range = np.maximum(adjusted_high, previous_close) - np.minimum(
        adjusted_low, previous_close
    )
    expected_atr = np.mean(true_range[-10:])
    expected_drange = max(
        adjusted_high[-5:].max() - adjusted_close[-5:].min(),
        adjusted_close[-5:].max() - adjusted_low[-5:].min(),
    )
    returns = daily["close"].to_numpy() / daily["open"].to_numpy() - 1.0
    expected_kurtosis = stats.kurtosis(returns[-30:], fisher=False, bias=False)
    assert np.isclose(context["avg_true_range_10"], expected_atr)
    assert np.isclose(context["dual_thrust_drange_5"], expected_drange)
    assert np.isclose(
        context["prev30d_open_close_return_pearson_kurtosis"],
        expected_kurtosis,
    )

    future = pd.DataFrame(
        {
            "trade_date": [target, target + pd.offsets.BDay(1)],
            "ts_code": [TS_CODE, TS_CODE],
            "open": [1.0, 1.0],
            "high": [1_000_000.0, 2_000_000.0],
            "low": [0.01, 0.01],
            "close": [500_000.0, 1_000_000.0],
            "adj_factor": [1.0, 1.0],
        }
    )
    changed = build_daily_context(pd.concat([daily, future]), [target], TS_CODE).iloc[0]
    pd.testing.assert_series_equal(context, changed)


def test_breakout_tail_and_available_time_are_populated_without_future_bars() -> None:
    daily = _daily_history()
    target = (daily["trade_date"].max() + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    minute = _minute_day(target, np.linspace(110.0, 112.0, 20))
    result = build_intraday_strategy_factor_frame(minute, daily, TS_CODE)

    assert result["available_time"].equals(result["trade_time"])
    assert result["atr10_orb_width_to_open"].notna().all()
    assert result["dual_thrust_drange5_to_open"].notna().all()
    assert result["prev30d_open_close_return_pearson_kurtosis"].notna().all()

    changed = minute.copy()
    changed.iloc[-1, changed.columns.get_loc("high")] = 10_000.0
    changed.iloc[-1, changed.columns.get_loc("close")] = 9_000.0
    changed_result = build_intraday_strategy_factor_frame(changed, daily, TS_CODE)
    pd.testing.assert_frame_equal(
        result.iloc[:-1][FACTOR_COLUMNS],
        changed_result.iloc[:-1][FACTOR_COLUMNS],
    )


def test_generator_writes_and_incrementally_replaces_requested_dates(tmp_path) -> None:
    daily = _daily_history()
    first_date = (daily["trade_date"].max() + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    second_date = (daily["trade_date"].max() + pd.offsets.BDay(2)).strftime("%Y-%m-%d")
    minute = pd.concat(
        [
            _minute_day(first_date, np.linspace(110.0, 111.0, 61)),
            _minute_day(second_date, np.linspace(111.0, 112.0, 61)),
        ]
    )
    input_path = tmp_path / f"{TS_CODE}.parquet"
    output_root = tmp_path / "output"
    minute.to_parquet(input_path)

    first = process_symbol_file(
        input_path, output_root, daily, first_date, first_date, True
    )
    assert first["status"] == "written"
    second = process_symbol_file(
        input_path, output_root, daily, second_date, second_date, True
    )
    assert second["status"] == "written"

    output = pd.read_parquet(output_root / f"{TS_CODE}.parquet")
    assert list(output.columns) == OUTPUT_COLUMNS
    assert len(output) == 122
    assert output["trade_date"].nunique() == 2


def test_p1_registry_and_profile_contract_preserve_p0_defaults() -> None:
    assert len(P1_FACTOR_COLUMNS) == 13
    assert len(P1_FACTOR_COLUMNS) == len(set(P1_FACTOR_COLUMNS))
    assert len(MARKET_FLOW_FACTOR_COLUMNS) == 3
    assert len(PATH_KNN_FACTOR_COLUMNS) == 4
    assert len(VOLATILITY_CONVERGENCE_FACTOR_COLUMNS) == 3
    assert len(EARLY_RANGE_FACTOR_COLUMNS) == 3
    assert len(COMBINED_FACTOR_COLUMNS) == 34
    assert len(COMBINED_OUTPUT_COLUMNS) == 45
    assert factor_columns_for_profile(PRIORITY_PROFILE_P0) == FACTOR_COLUMNS
    assert output_columns_for_profile(PRIORITY_PROFILE_P0) == OUTPUT_COLUMNS
    assert factor_columns_for_profile(PRIORITY_PROFILE_P0_P1) == COMBINED_FACTOR_COLUMNS
    assert output_columns_for_profile(PRIORITY_PROFILE_P0_P1) == COMBINED_OUTPUT_COLUMNS
    assert (
        resolve_output_root("stock", PRIORITY_PROFILE_P0_P1, None)
        == DEFAULT_STOCK_P0_P1_OUTPUT_ROOT
    )
    assert (
        resolve_output_root("etf", PRIORITY_PROFILE_P0_P1, None)
        == DEFAULT_ETF_P0_P1_OUTPUT_ROOT
    )

    minute = _minute_day("2025-03-03", np.linspace(100.0, 101.0, 61))
    default = build_intraday_strategy_factor_frame(minute, _daily_history(), TS_CODE)
    explicit = build_intraday_strategy_factor_frame(
        minute,
        _daily_history(),
        TS_CODE,
        priority_profile=PRIORITY_PROFILE_P0,
    )
    pd.testing.assert_frame_equal(default, explicit)


def test_directional_notional_aggregation_and_unknown_side_validation() -> None:
    bars = pd.DatetimeIndex(
        ["2025-03-03 11:30", "2025-03-03 13:01", "2025-03-03 13:02"]
    )
    trades = pd.DataFrame(
        {
            "event_time": [
                "2025-03-03 11:29:10",
                "2025-03-03 11:29:20",
                "2025-03-03 13:00:15",
            ],
            "side": ["B", "S", "B"],
            "price": [10.0, 10.0, 11.0],
            "qty": [100.0, 40.0, 20.0],
        }
    )
    aggregated = aggregate_directional_notional(trades, bars)
    assert aggregated["buy_notional_1m"].tolist() == [1000.0, 220.0, 0.0]
    assert aggregated["sell_notional_1m"].tolist() == [400.0, 0.0, 0.0]
    np.testing.assert_allclose(
        active_notional_imbalance(
            aggregated["buy_notional_1m"], aggregated["sell_notional_1m"]
        ),
        [(1000.0 - 400.0) / 1400.0, 1.0, 0.0],
    )

    invalid = trades.copy()
    invalid.loc[0, "side"] = "X"
    with np.testing.assert_raises_regex(ValueError, "unknown directions"):
        aggregate_directional_notional(invalid, bars)


def test_path_knn_uses_k10_prior_days_and_crosses_lunch_without_future_days() -> None:
    historical_dates = pd.bdate_range("2025-01-02", periods=10)
    target_date = historical_dates[-1] + pd.offsets.BDay(1)
    morning = pd.date_range("2025-01-02 11:21", periods=10, freq="min").time
    time_values = list(morning) + [pd.Timestamp("13:01").time()]
    prefix = np.array(
        [100.0, 100.8, 100.2, 101.1, 100.7, 101.5, 101.0, 102.0, 101.4, 102.3]
    )
    next_returns = np.array(
        [-0.01, 0.02, 0.03, -0.02, 0.01, 0.04, -0.03, 0.05, 0.02, -0.01]
    )

    history_days: list[pd.DataFrame] = []
    for date, next_return in zip(historical_dates, next_returns, strict=True):
        times = pd.DatetimeIndex(
            [pd.Timestamp.combine(date.date(), value) for value in time_values]
        )
        prices = np.r_[prefix, prefix[-1] * (1.0 + next_return)]
        history_days.append(_minute_day(date.strftime("%Y-%m-%d"), prices, times=times))
    target_times = pd.DatetimeIndex(
        [pd.Timestamp.combine(target_date.date(), value) for value in time_values]
    )
    target = _minute_day(
        target_date.strftime("%Y-%m-%d"),
        np.r_[prefix, prefix[-1] * 1.5],
        times=target_times,
    )
    minute_history = pd.concat([*history_days, target])
    result = build_intraday_strategy_factor_frame(
        target,
        _daily_history(),
        TS_CODE,
        priority_profile=PRIORITY_PROFILE_P0_P1,
        minute_history=minute_history,
        active_flow_context=_flow_context(target),
    )
    signal = result.loc[9]
    assert np.isclose(signal["path_knn_expected_next_return"], next_returns.mean())
    assert np.isclose(signal["path_knn_up_probability"], (next_returns > 0).mean())
    assert np.isclose(signal["path_knn_mean_distance"], 0.0, atol=1e-8)
    assert np.isclose(
        signal["path_knn_direction_agreement"],
        np.abs(np.sign(next_returns).mean()),
    )

    future_date = target_date + pd.offsets.BDay(1)
    future_times = pd.DatetimeIndex(
        [pd.Timestamp.combine(future_date.date(), value) for value in time_values]
    )
    future = _minute_day(
        future_date.strftime("%Y-%m-%d"),
        np.linspace(1.0, 1000.0, len(time_values)),
        times=future_times,
    )
    changed = build_intraday_strategy_factor_frame(
        target,
        _daily_history(),
        TS_CODE,
        priority_profile=PRIORITY_PROFILE_P0_P1,
        minute_history=pd.concat([minute_history, future]),
        active_flow_context=_flow_context(target),
    )
    pd.testing.assert_frame_equal(
        result[PATH_KNN_FACTOR_COLUMNS], changed[PATH_KNN_FACTOR_COLUMNS]
    )


def test_p1_envelope_min_volatility_and_early_range_formulas_are_causal() -> None:
    prices = 100.0 + 0.02 * np.arange(50) + 0.3 * np.sin(np.arange(50) / 3.0)
    minute = _minute_day("2025-03-03", prices)
    result = build_intraday_strategy_factor_frame(
        minute,
        _daily_history(),
        TS_CODE,
        priority_profile=PRIORITY_PROFILE_P0_P1,
        minute_history=minute,
        active_flow_context=_flow_context(minute),
    )

    expected_upper = (prices[:26] + 0.05).max()
    expected_lower = (prices[:26] - 0.05).min()
    assert np.isclose(
        result.loc[26, "distance_to_prior26_upper_envelope"],
        prices[26] / expected_upper - 1.0,
    )
    assert np.isclose(
        result.loc[26, "distance_to_prior26_lower_envelope"],
        prices[26] / expected_lower - 1.0,
    )
    sigma = pd.Series(np.log(prices)).diff().rolling(26, min_periods=26).std(ddof=0)
    expected_min_volatility = sigma.shift(1).rolling(7, min_periods=7).min()
    assert np.isclose(
        result.loc[33, "min_volatility_regime_7"], expected_min_volatility.iloc[33]
    )

    early_high = (prices[:41] + 0.05).max()
    early_low = (prices[:41] - 0.05).min()
    assert result.loc[:39, EARLY_RANGE_FACTOR_COLUMNS].isna().all().all()
    assert np.isclose(
        result.loc[40, "early_range_position_41m"],
        2.0 * (prices[40] - early_low) / (early_high - early_low) - 1.0,
    )
    assert np.isclose(
        result.loc[40, "distance_to_early_high_41m"], prices[40] / early_high - 1.0
    )

    changed_minute = minute.copy()
    changed_minute.iloc[-1, changed_minute.columns.get_loc("close")] = 10_000.0
    changed = build_intraday_strategy_factor_frame(
        changed_minute,
        _daily_history(),
        TS_CODE,
        priority_profile=PRIORITY_PROFILE_P0_P1,
        minute_history=changed_minute,
        active_flow_context=_flow_context(minute),
    )
    pd.testing.assert_frame_equal(
        result.iloc[:-1][P1_FACTOR_COLUMNS], changed.iloc[:-1][P1_FACTOR_COLUMNS]
    )


def _write_raw_trades(path: Path, rows: list[tuple[str, str, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: list[list[object]] = []
    for raw_time, side, price, qty in rows:
        row: list[object] = ["x"] * 10
        row[2] = "20250303"
        row[3] = raw_time
        row[7] = side
        row[8] = price
        row[9] = qty
        data.append(row)
    pd.DataFrame(data, columns=[f"c{i}" for i in range(10)]).to_csv(
        path, index=False, encoding="gbk"
    )


def test_point_in_time_market_flow_cache_is_full_day_and_strict(tmp_path) -> None:
    trade_date = pd.Timestamp("2025-03-03")
    times = pd.DatetimeIndex(
        ["2025-03-03 11:30", "2025-03-03 13:01", "2025-03-03 13:02"]
    )
    stock_minute_root = tmp_path / "stock_minute"
    tick_root = tmp_path / "ticks"
    cache_root = tmp_path / "cache"
    members = [TS_CODE, "600000.SH"]
    stock_minute_root.mkdir()
    for symbol in members:
        _minute_day("2025-03-03", np.array([10.0, 10.1, 10.2]), times=times).to_parquet(
            stock_minute_root / f"{symbol}.parquet"
        )
    date_root = tick_root / "2025" / "202503" / "20250303"
    _write_raw_trades(
        date_root / TS_CODE / "逐笔成交.csv",
        [("112959000", "B", 100000, 10), ("130059000", "S", 100000, 5)],
    )
    _write_raw_trades(
        date_root / "600000.SH" / "逐笔成交.csv",
        [("112959000", "S", 200000, 5), ("130059000", "B", 200000, 10)],
    )
    membership = pd.DataFrame(
        {
            "pool_id": ["pool", "pool"],
            "member_ts_code": members,
            "effective_from": [trade_date, trade_date],
            "effective_to": [pd.Timestamp.max.normalize()] * 2,
        }
    )
    target_mapping = pd.DataFrame(
        {
            "target_ts_code": [TS_CODE],
            "pool_id": ["pool"],
            "effective_from": [trade_date],
            "effective_to": [pd.Timestamp.max.normalize()],
        }
    )
    requested = _minute_day(
        "2025-03-03", np.array([10.0, 10.1, 10.2]), times=times
    ).reset_index()
    context = build_active_flow_context(
        TS_CODE,
        requested,
        membership,
        target_mapping,
        tick_root=tick_root,
        stock_minute_root=stock_minute_root,
        cache_root=cache_root,
    )
    assert context[MARKET_FLOW_FACTOR_COLUMNS].notna().all().all()
    assert np.isclose(
        context.loc[1, "market_active_notional_imbalance_cum_session"],
        (300.0 - 150.0) / (300.0 + 150.0),
    )
    assert len(list(cache_root.rglob("*.parquet"))) == 1

    missing_member = "300001.SZ"
    _minute_day("2025-03-03", np.array([9.0, 9.1, 9.2]), times=times).to_parquet(
        stock_minute_root / f"{missing_member}.parquet"
    )
    incomplete = pd.concat(
        [
            membership,
            pd.DataFrame(
                {
                    "pool_id": ["pool"],
                    "member_ts_code": [missing_member],
                    "effective_from": [trade_date],
                    "effective_to": [pd.Timestamp.max.normalize()],
                }
            ),
        ],
        ignore_index=True,
    )
    with np.testing.assert_raises_regex(FileNotFoundError, "Missing trade file"):
        build_active_flow_context(
            TS_CODE,
            requested,
            incomplete,
            target_mapping,
            tick_root=tick_root,
            stock_minute_root=stock_minute_root,
            cache_root=cache_root,
        )


def test_mapping_overlap_and_profile_schema_mixing_fail(tmp_path) -> None:
    membership_path = tmp_path / "membership.csv"
    pd.DataFrame(
        {
            "pool_id": ["pool"],
            "member_ts_code": [TS_CODE],
            "effective_from": ["2025-01-01"],
            "effective_to": [""],
        }
    ).to_csv(membership_path, index=False)
    assert len(load_pool_membership(membership_path)) == 1

    target_path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "target_ts_code": [TS_CODE, TS_CODE],
            "pool_id": ["pool", "pool2"],
            "effective_from": ["2025-01-01", "2025-06-01"],
            "effective_to": ["2025-12-31", ""],
        }
    ).to_csv(target_path, index=False)
    with np.testing.assert_raises_regex(ValueError, "Overlapping target-pool"):
        load_target_pool_mapping(target_path)

    output_path = tmp_path / "mixed.parquet"
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_parquet(output_path, index=False)
    requested = pd.DataFrame(columns=COMBINED_OUTPUT_COLUMNS)
    with np.testing.assert_raises_regex(ValueError, "schema"):
        merge_output(
            output_path,
            requested,
            replace_all=False,
            output_columns=COMBINED_OUTPUT_COLUMNS,
        )


def test_generator_writes_combined_profile_to_45_column_contract(tmp_path) -> None:
    daily = _daily_history()
    trade_date = (daily["trade_date"].max() + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    minute = _minute_day(trade_date, 100.0 + np.sin(np.arange(50) / 5.0))
    input_path = tmp_path / f"{TS_CODE}.parquet"
    output_root = tmp_path / "combined"
    minute.to_parquet(input_path)
    result = process_symbol_file(
        input_path,
        output_root,
        daily,
        trade_date,
        trade_date,
        True,
        PRIORITY_PROFILE_P0_P1,
        _flow_context(minute),
    )
    assert result["status"] == "written"
    assert result["factor_count"] == 34
    output = pd.read_parquet(output_root / f"{TS_CODE}.parquet")
    assert list(output.columns) == COMBINED_OUTPUT_COLUMNS
    assert output.shape == (50, 45)
