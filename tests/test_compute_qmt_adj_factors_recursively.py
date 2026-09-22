#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for compute_qmt_adj_factors_recursively."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compute_qmt_adj_factors_recursively import (
    parse_date,
    process_symbol_recursive,
)


def test_parse_date() -> None:
    assert parse_date("2026-09-02") == pd.Timestamp("2026-09-02")
    assert parse_date("20260902") == pd.Timestamp("2026-09-02")
    assert parse_date(None) is None
    assert parse_date("") is None


def test_process_symbol_recursive_range(tmp_path: Path) -> None:
    path_1d = tmp_path / "588810.SH_1d.parquet"
    path_1min = tmp_path / "588810.SH_1min.parquet"

    dates = pd.date_range("2026-09-01", periods=10, freq="D")
    df_1d = pd.DataFrame(
        {"ts_code": "588810.SH", "close": 2.4, "adj_factor": 1.0},
        index=pd.DatetimeIndex(dates, name="trade_date"),
    )
    df_1d.to_parquet(path_1d)

    # 1min frame with 2 bars per day
    idx_tuples = []
    for d in dates:
        idx_tuples.append((d, d + pd.Timedelta(hours=9, minutes=30)))
        idx_tuples.append((d, d + pd.Timedelta(hours=14, minutes=59)))
    idx_1m = pd.MultiIndex.from_tuples(idx_tuples, names=["trade_date", "trade_time"])
    df_1m = pd.DataFrame({"ts_code": "588810.SH", "close": 2.4, "adj_factor": 1.0}, index=idx_1m)
    df_1m.to_parquet(path_1min)

    # Split on 2026-09-05 (dr = 2.0)
    div_map = {pd.Timestamp("2026-09-05"): 2.0}

    date_from = pd.Timestamp("2026-09-02")
    date_to = pd.Timestamp("2026-09-07")

    res = process_symbol_recursive(
        "588810.SH",
        str(path_1d),
        str(path_1min),
        div_map,
        date_from=date_from,
        date_to=date_to,
        ffill_subsequent=True,
    )
    assert res[1] is True  # success

    # Verify 1d
    res_1d = pd.read_parquet(path_1d)
    assert res_1d.loc["2026-09-01", "adj_factor"] == 1.0  # before date_from
    assert res_1d.loc["2026-09-02", "adj_factor"] == 1.0
    assert res_1d.loc["2026-09-04", "adj_factor"] == 1.0
    assert res_1d.loc["2026-09-05", "adj_factor"] == 2.0  # split!
    assert res_1d.loc["2026-09-07", "adj_factor"] == 2.0
    assert res_1d.loc["2026-09-08", "adj_factor"] == 2.0  # ffill_subsequent
    assert res_1d.loc["2026-09-10", "adj_factor"] == 2.0

    # Verify 1min
    res_1m = pd.read_parquet(path_1min)
    sub_1m = res_1m.loc["2026-09-05"]
    assert (sub_1m["adj_factor"] == 2.0).all()
    sub_1m_post = res_1m.loc["2026-09-08"]
    assert (sub_1m_post["adj_factor"] == 2.0).all()


def test_process_symbol_recursive_full_history(tmp_path: Path) -> None:
    path_1d = tmp_path / "test_1d.parquet"
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df_1d = pd.DataFrame(
        {"ts_code": "TEST", "adj_factor": 1.0},
        index=pd.DatetimeIndex(dates, name="trade_date"),
    )
    df_1d.to_parquet(path_1d)

    div_map = {
        pd.Timestamp("2026-01-02"): 1.1,
        pd.Timestamp("2026-01-04"): 1.2,
    }

    res = process_symbol_recursive("TEST", str(path_1d), None, div_map)
    assert res[1] is True

    reloaded = pd.read_parquet(path_1d)
    assert reloaded.loc["2026-01-01", "adj_factor"] == 1.0
    assert pytest.approx(reloaded.loc["2026-01-02", "adj_factor"]) == 1.1
    assert pytest.approx(reloaded.loc["2026-01-03", "adj_factor"]) == 1.1
    assert pytest.approx(reloaded.loc["2026-01-04", "adj_factor"]) == 1.1 * 1.2
    assert pytest.approx(reloaded.loc["2026-01-05", "adj_factor"]) == 1.1 * 1.2
