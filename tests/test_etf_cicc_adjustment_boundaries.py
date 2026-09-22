from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from scripts.generate_etf_cicc_minute_factors import (
    _cal_corr_prvr,
    _cal_liq_amihud_1min,
    build_factor_registry,
)


def _split_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": ["510300.SH"] * 6,
            "date": [dt.date(2026, 1, 2)] * 3 + [dt.date(2026, 1, 5)] * 3,
            "time": [93000000, 93100000, 93200000] * 2,
            "close": [100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
            "volume": [100.0, 110.0, 120.0, 200.0, 220.0, 240.0],
        }
    )


def test_cicc_overrides_reset_returns_at_trade_date_boundary() -> None:
    frame = _split_frame()

    amihud = _cal_liq_amihud_1min(frame).sort("date")
    corr = _cal_corr_prvr(frame).sort("date")

    assert amihud["liq_amihud_1min"].to_list() == [0.0, 0.0]
    assert all(value is None or np.isnan(value) for value in corr["corr_prvr"])


def test_cicc_registry_uses_daily_reset_overrides() -> None:
    registry = dict(build_factor_registry())

    assert registry["liq_amihud_1min"] is _cal_liq_amihud_1min
    assert registry["corr_prvr"] is _cal_corr_prvr
