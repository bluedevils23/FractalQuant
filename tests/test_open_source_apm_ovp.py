from __future__ import annotations

import numpy as np
import pandas as pd

from factor.open_source_crossmarket import (
    add_availability_columns,
    build_daily_apm_raw_features,
    build_open_source_apm_panel,
)


def _minute_frame(symbol: str, dates: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    times = list(pd.date_range("2000-01-01 09:30", periods=121, freq="min").time)
    times += list(pd.date_range("2000-01-01 13:00", periods=121, freq="min").time)
    rows = []
    for day_index, date in enumerate(dates):
        for minute_index, clock in enumerate(times):
            value = base + day_index * 0.1 + minute_index * 0.001
            rows.append(
                {
                    "trade_time": pd.Timestamp.combine(date.date(), clock),
                    "open": value,
                    "high": value + 0.01,
                    "low": value - 0.01,
                    "close": value,
                    "volume": 1000.0,
                    "amount": value * 1000.0,
                    "ts_code": symbol,
                }
            )
    return pd.DataFrame(rows)


def test_apmnew_uses_overnight_pm_and_cross_section_residual() -> None:
    dates = pd.bdate_range("2025-01-02", periods=22)
    rows = []
    for symbol_index in range(20):
        asset = _minute_frame(f"{symbol_index:06d}.SZ", dates, 100.0 + symbol_index)
        benchmark = _minute_frame("000300.SH", dates, 200.0)
        rows.append(
            build_daily_apm_raw_features(
                asset, benchmark, f"{symbol_index:06d}.SZ", "000300.SH"
            )
        )
    raw = pd.concat(rows, ignore_index=True)
    result = add_availability_columns(build_open_source_apm_panel(raw))
    last_date = result["trade_date"].max()
    latest = result.loc[result["trade_date"].eq(last_date)]
    assert latest["apm_cross_section_count"].eq(20).all()
    assert latest["kaiyuan_apmnew_m20"].notna().all()
    assert latest["kaiyuan_ovp_m20"].notna().all()
    assert latest["kaiyuan_avp_m20"].notna().all()
    assert result.loc[result["trade_date"].eq(dates[0]), "available_date"].eq(dates[1]).all()
    assert result.loc[result["trade_date"].eq(dates[-1]), "available_date"].isna().all()


def test_apm_panel_keeps_null_when_cross_section_is_too_small() -> None:
    dates = pd.bdate_range("2025-01-02", periods=22)
    asset = _minute_frame("000001.SZ", dates, 100.0)
    benchmark = _minute_frame("000300.SH", dates, 200.0)
    raw = build_daily_apm_raw_features(asset, benchmark, "000001.SZ", "000300.SH")
    result = build_open_source_apm_panel(raw)
    assert result["kaiyuan_apmnew_m20"].isna().all()
    assert result["apm_cross_section_count"].eq(0).all()
    assert result["kaiyuan_ovp_m20"].notna().any()
