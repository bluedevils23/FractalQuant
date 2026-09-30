from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_auction_factors import OUTPUT_COLUMNS
from scripts.generate_previous_day_factors import (
    DAILY_FACTOR_COLUMNS,
    calculate_previous_day_factors,
    write_outputs,
)


def _daily_fixture(path: Path) -> None:
    rows = []
    for code, multiplier in (("000001.SZ", 1.0), ("000002.SZ", 2.0), ("510300.SH", 1.5)):
        for index, trade_date in enumerate(pd.date_range("2026-01-01", periods=23)):
            close = 100.0 + index * multiplier
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "close": close,
                    "high": close + 10.0,
                    "low": close - 10.0,
                    "pre_close": close - 1.0,
                    "adj_factor": 1.0,
                    "vol": 100.0,
                    "circ_mv": 1000.0,
                }
            )
    pd.DataFrame(rows).set_index(["trade_date", "ts_code"]).to_parquet(path)


def test_previous_day_factors_use_prior_daily_session_and_benchmark(tmp_path: Path) -> None:
    daily_path = tmp_path / "daily.parquet"
    _daily_fixture(daily_path)

    result = calculate_previous_day_factors(
        daily_path,
        requested_codes={"000001.SZ"},
        date_from="2026-01-23",
        benchmark_ts_code="510300.SH",
    )

    row = result.iloc[0]
    assert row["trade_date"] == "2026-01-23"
    assert row["source_trade_date"] == "2026-01-22"
    assert row["available_time"] == pd.Timestamp("2026-01-23 09:15:00")
    assert np.isclose(row["prevday_intraday_drawdown_from_session_high"], 121 / 131 - 1)
    assert np.isclose(row["prevday_intraday_rebound_from_session_low"], 121 / 111 - 1)
    assert np.isclose(row["prevday_intraday_return_from_prev_close"], 121 / 120 - 1)
    assert not np.isnan(row["prev_skew_long_60d"])
    assert not np.isnan(row["prev_skew_short_10d"])
    assert np.isclose(row["prev_rel_skew_60_10"], row["prev_skew_long_60d"] - row["prev_skew_short_10d"])
    assert 0.0 <= row["prev_rel_skew_rank_cs"] <= 1.0
    assert set(DAILY_FACTOR_COLUMNS).issubset(result.columns)
    assert not set(DAILY_FACTOR_COLUMNS) & set(OUTPUT_COLUMNS)


def test_previous_day_factors_directory_preserves_full_universe_ranks(tmp_path: Path) -> None:
    daily_path = tmp_path / "daily.parquet"
    daily_dir = tmp_path / "daily_by_symbol"
    daily_dir.mkdir()
    _daily_fixture(daily_path)
    for code, group in pd.read_parquet(daily_path).reset_index().groupby("ts_code"):
        group.drop(columns="ts_code").set_index("trade_date").to_parquet(
            daily_dir / f"{code}.parquet"
        )

    kwargs = {
        "requested_codes": {"000001.SZ"},
        "date_from": "2026-01-23",
        "date_to": "2026-01-23",
    }
    expected = calculate_previous_day_factors(daily_path, **kwargs)
    actual = calculate_previous_day_factors(daily_dir, **kwargs)

    assert len(actual) == 1
    assert actual.iloc[0]["source_trade_date"] == expected.iloc[0]["source_trade_date"]
    for column in (
        "prev_2d_return_rank_cs",
        "prev_20d_return_rank_cs",
        "prev_rel_skew_rank_cs",
    ):
        assert np.isclose(actual.iloc[0][column], expected.iloc[0][column])


def test_prevday_return_uses_adjusted_closes_across_split(tmp_path: Path) -> None:
    daily_path = tmp_path / "daily.parquet"
    pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-01-01", periods=4),
            "ts_code": "000001.SZ",
            "close": [100.0, 90.0, 45.0, 46.0],
            "high": [101.0, 91.0, 46.0, 47.0],
            "low": [99.0, 89.0, 44.0, 45.0],
            "pre_close": [100.0, 100.0, 44.99, 45.0],
            "adj_factor": [1.0, 1.0, 2.0, 2.0],
        }
    ).set_index(["trade_date", "ts_code"]).to_parquet(daily_path)

    result = calculate_previous_day_factors(
        daily_path,
        requested_codes={"000001.SZ"},
        date_from="2026-01-04",
        date_to="2026-01-04",
        etf_daily_path=tmp_path / "missing_etf.parquet",
    )

    assert result.iloc[0]["source_trade_date"] == "2026-01-03"
    assert result.iloc[0]["prevday_intraday_return_from_prev_close"] == 0.0


def test_previous_day_factor_outputs_merge_without_duplicate_dates(tmp_path: Path) -> None:
    daily_path = tmp_path / "daily.parquet"
    _daily_fixture(daily_path)
    result = calculate_previous_day_factors(
        daily_path, requested_codes={"000001.SZ"}, date_from="2026-01-23"
    )
    output_root = tmp_path / "output"

    assert write_outputs(result, output_root, overwrite=True) == 1
    assert write_outputs(result, output_root, overwrite=False) == 1
    written = pd.read_parquet(output_root / "000001.SZ.parquet")
    assert written[["trade_date", "ts_code"]].duplicated().sum() == 0
    assert len(written) == len(result)


def test_previous_day_factors_fallback_to_etf_daily_benchmark(tmp_path: Path) -> None:
    stock_path = tmp_path / "stock_daily.parquet"
    etf_path = tmp_path / "etf_daily.parquet"

    # Stock daily without benchmark 510300.SH
    stock_rows = []
    for code, multiplier in (("000001.SZ", 1.0), ("000002.SZ", 2.0)):
        for index, trade_date in enumerate(pd.date_range("2026-01-01", periods=23)):
            close = 100.0 + index * multiplier
            stock_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "close": close,
                    "high": close + 10.0,
                    "low": close - 10.0,
                    "pre_close": close - 1.0,
                    "adj_factor": 1.0,
                    "vol": 100.0,
                    "circ_mv": 1000.0,
                }
            )
    pd.DataFrame(stock_rows).set_index(["trade_date", "ts_code"]).to_parquet(stock_path)

    # ETF daily containing benchmark 510300.SH
    etf_rows = []
    for index, trade_date in enumerate(pd.date_range("2026-01-01", periods=23)):
        close = 100.0 + index * 1.5
        etf_rows.append(
            {
                "trade_date": trade_date,
                "ts_code": "510300.SH",
                "close": close,
                "high": close + 5.0,
                "low": close - 5.0,
                "pre_close": close - 1.0,
                "adj_factor": 1.0,
                "vol": 100.0,
                "circ_mv": 1000.0,
            }
        )
    pd.DataFrame(etf_rows).set_index(["trade_date", "ts_code"]).to_parquet(etf_path)

    result = calculate_previous_day_factors(
        stock_path,
        requested_codes={"000001.SZ"},
        date_from="2026-01-23",
        benchmark_ts_code="510300.SH",
        etf_daily_path=etf_path,
    )

    row = result.iloc[0]
    assert row["trade_date"] == "2026-01-23"
    assert row["ts_code"] == "000001.SZ"
    assert not np.isnan(row["market_above_ma20_prevclose"])
    assert not np.isnan(row["market_momentum_2d_prevclose"])
    assert row["market_above_ma20_prevclose"] == 1.0


def test_previous_day_factors_fallback_when_stock_benchmark_stops(tmp_path: Path) -> None:
    stock_path = tmp_path / "stock_daily.parquet"
    etf_path = tmp_path / "etf_daily.parquet"
    _daily_fixture(stock_path)
    stock = pd.read_parquet(stock_path).reset_index()
    stock = stock.loc[
        ~(stock["ts_code"].eq("510300.SH") & stock["trade_date"].ge("2026-01-22"))
    ]
    stock.set_index(["trade_date", "ts_code"]).to_parquet(stock_path)

    etf_rows = []
    for index, trade_date in enumerate(pd.date_range("2026-01-01", periods=23)):
        close = 200.0 - index * 2.0
        etf_rows.append(
            {
                "trade_date": trade_date,
                "ts_code": "510300.SH",
                "close": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "pre_close": close + 2.0,
                "adj_factor": 1.0,
            }
        )
    pd.DataFrame(etf_rows).set_index(["trade_date", "ts_code"]).to_parquet(etf_path)

    result = calculate_previous_day_factors(
        stock_path,
        requested_codes={"000001.SZ"},
        date_from="2026-01-22",
        date_to="2026-01-23",
        etf_daily_path=etf_path,
    ).set_index("trade_date")

    assert result.loc["2026-01-22", "market_above_ma20_prevclose"] == 1.0
    assert result.loc["2026-01-22", "market_momentum_2d_prevclose"] > 0
    assert result.loc["2026-01-23", "market_above_ma20_prevclose"] == 0.0
    assert result.loc["2026-01-23", "market_momentum_2d_prevclose"] < 0
