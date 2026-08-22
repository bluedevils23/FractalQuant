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
    assert set(DAILY_FACTOR_COLUMNS).issubset(result.columns)
    assert not set(DAILY_FACTOR_COLUMNS) & set(OUTPUT_COLUMNS)


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
