#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for copy_etf_daily_adj_factors_to_qmt."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.copy_etf_daily_adj_factors_to_qmt import (
    load_daily_adj_factors,
    process_single_file,
    sync_dataset,
)


@pytest.fixture
def mock_daily_path(tmp_path: Path) -> Path:
    p = tmp_path / "etf_daily.parquet"
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df1 = pd.DataFrame({
        "trade_date": dates,
        "ts_code": ["510300.SH"] * 5,
        "adj_factor": [1.0, 1.0, 1.1, 1.1, 1.2],
    })
    df2 = pd.DataFrame({
        "trade_date": dates,
        "ts_code": ["159901.SZ"] * 5,
        "adj_factor": [1.0] * 5,
    })
    df = pd.concat([df1, df2], ignore_index=True)
    df.set_index(["trade_date", "ts_code"]).to_parquet(p)
    return p


def test_load_daily_adj_factors(mock_daily_path: Path) -> None:
    factors = load_daily_adj_factors(mock_daily_path)
    assert "510300.SH" in factors
    assert "159901.SZ" in factors
    s = factors["510300.SH"]
    assert len(s) == 5
    assert s.iloc[-1] == 1.2


def test_process_single_file_1d(tmp_path: Path) -> None:
    file_path = tmp_path / "510300.SH.parquet"
    dates = pd.date_range("2026-01-01", periods=7, freq="D")  # 2 days beyond daily
    df = pd.DataFrame(
        {"ts_code": "510300.SH", "open": 3.0, "close": 3.1, "adj_factor": 1.0},
        index=pd.DatetimeIndex(dates, name="trade_date"),
    )
    df.to_parquet(file_path)

    daily_s = pd.Series([1.0, 1.0, 1.1, 1.1, 1.2], index=dates[:5])
    res = process_single_file("1d", str(file_path), "510300.SH", daily_s)
    assert res[3] is True

    reloaded = pd.read_parquet(file_path)
    assert reloaded.index.name == "trade_date"
    assert reloaded["adj_factor"].iloc[2] == 1.1
    assert reloaded["adj_factor"].iloc[4] == 1.2
    # ffill for subsequent dates
    assert reloaded["adj_factor"].iloc[5] == 1.2
    assert reloaded["adj_factor"].iloc[6] == 1.2


def test_process_single_file_tick(tmp_path: Path) -> None:
    file_path = tmp_path / "510300.SH.parquet"
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    idx = pd.MultiIndex.from_tuples(
        [
            (dates[0], pd.Timestamp("2026-01-01 09:15:00")),
            (dates[0], pd.Timestamp("2026-01-01 09:15:03")),
            (dates[1], pd.Timestamp("2026-01-02 09:15:00")),
        ],
        names=["trade_date", "trade_time"],
    )
    df = pd.DataFrame({"ts_code": "510300.SH", "last_price": 3.0}, index=idx)
    df.to_parquet(file_path)

    daily_s = pd.Series([1.05, 1.10], index=dates)
    res = process_single_file("tick", str(file_path), "510300.SH", daily_s)
    assert res[3] is True

    reloaded = pd.read_parquet(file_path)
    assert reloaded.index.names == ["trade_date", "trade_time"]
    assert "adj_factor" in reloaded.columns
    assert reloaded["adj_factor"].iloc[0] == 1.05
    assert reloaded["adj_factor"].iloc[1] == 1.05
    assert reloaded["adj_factor"].iloc[2] == 1.10


def test_sync_dataset_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "qmt_1d"
    root.mkdir()
    f = root / "510300.SH.parquet"
    pd.DataFrame({"adj_factor": [1.0]}, index=pd.date_range("2026-01-01", periods=1)).to_parquet(f)

    daily_s = pd.Series([1.5], index=pd.date_range("2026-01-01", periods=1))
    total, succeeded, failed = sync_dataset("1d", root, {"510300.SH": daily_s}, dry_run=True)
    assert total == 1 and succeeded == 1 and failed == 0

    # File should not be modified in dry-run
    reloaded = pd.read_parquet(f)
    assert reloaded["adj_factor"].iloc[0] == 1.0


def test_process_single_file_date_range(tmp_path: Path) -> None:
    file_path = tmp_path / "510300.SH.parquet"
    dates = pd.date_range("2026-09-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"ts_code": "510300.SH", "open": 3.0, "close": 3.1, "adj_factor": 1.0},
        index=pd.DatetimeIndex(dates, name="trade_date"),
    )
    df.to_parquet(file_path)

    # daily has new factor on 2026-09-05
    daily_s = pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], index=dates)

    date_from = pd.Timestamp("2026-09-02")
    date_to = pd.Timestamp("2026-09-07")

    res = process_single_file(
        "1d",
        str(file_path),
        "510300.SH",
        daily_s,
        date_from=date_from,
        date_to=date_to,
        ffill_subsequent=False,
    )
    assert res[3] is True

    reloaded = pd.read_parquet(file_path)
    # Before 2026-09-02: untouched (1.0)
    assert reloaded.loc["2026-09-01", "adj_factor"] == 1.0
    # In range [2026-09-02, 2026-09-07]: updated
    assert reloaded.loc["2026-09-02", "adj_factor"] == 1.0
    assert reloaded.loc["2026-09-05", "adj_factor"] == 2.0
    assert reloaded.loc["2026-09-07", "adj_factor"] == 2.0
    # After 2026-09-07: untouched without ffill_subsequent (still original 1.0)
    assert reloaded.loc["2026-09-08", "adj_factor"] == 1.0

