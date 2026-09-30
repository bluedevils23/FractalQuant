from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl

from scripts.generate_etf_cicc_minute_factors import (
    _cal_corr_prvr,
    _cal_liq_amihud_1min,
    build_factor_registry,
    filter_minute_frame,
    normalize_date_arg,
)
import MinuteFrequentFactorCalculateMethodsCICC as cicc_methods
from MinuteFrequentFactorCalculateMethodsCICC import cal_mmt_pm
from MinuteFrequentFactorCalculateMethodsCICC import cal_doc_std


def test_cicc_date_filter_is_inclusive() -> None:
    index = pd.date_range("2021-12-31 09:30", periods=4, freq="D")
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=index)

    filtered = filter_minute_frame(
        frame,
        normalize_date_arg("2022-01-01"),
        normalize_date_arg("2022-01-02"),
    )

    assert filtered.index.normalize().unique().tolist() == list(
        pd.to_datetime(["2022-01-01", "2022-01-02"])
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


def test_doc_vol50_ratio_uses_top_50_volume_bars() -> None:
    frame = pl.DataFrame(
        {
            "code": ["510300.SH"] * 60,
            "date": [dt.date(2026, 1, 2)] * 60,
            "time": [93000000 + i * 100000 for i in range(60)],
            "open": [100.0] * 60,
            "close": [100.0 + i * 0.01 for i in range(60)],
            "high": [100.1] * 60,
            "low": [99.9] * 60,
            "volume": [float(i + 1) for i in range(60)],
        }
    )

    vol5 = cicc_methods.cal_doc_vol5_ratio(frame)["doc_vol5_ratio"][0]
    vol50 = cicc_methods.cal_doc_vol50_ratio(frame)["doc_vol50_ratio"][0]
    expected_vol50 = sum(range(11, 61)) / sum(range(1, 61))

    assert np.isclose(vol50, expected_vol50)
    assert vol50 != vol5


def test_doc_pdf_factors_are_invariant_to_input_order() -> None:
    frame = pl.DataFrame(
        {
            "code": ["510300.SH"] * 6,
            "date": [dt.date(2026, 1, 2)] * 3 + [dt.date(2026, 1, 5)] * 3,
            "time": [93000000, 93100000, 93200000] * 2,
            "open": [100.0] * 3 + [200.0] * 3,
            "close": [100.0, 101.0, 102.0, 200.0, 202.0, 204.0],
            "volume": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )

    for suffix in ("60", "70", "80", "90", "95"):
        function = getattr(cicc_methods, f"cal_doc_pdf{suffix}")
        forward = function(frame).sort(["code", "date"])
        reverse = function(frame.reverse()).sort(["code", "date"])

        assert np.allclose(
            forward[f"doc_pdf{suffix}"].to_numpy(),
            reverse[f"doc_pdf{suffix}"].to_numpy(),
        )


def test_mmt_pm_uses_first_available_afternoon_bar() -> None:
    frame = pl.DataFrame(
        {
            "code": ["510300.SH"] * 4,
            "date": [dt.date(2026, 1, 2)] * 4,
            "time": [130100000, 130200000, 145900000, 150000000],
            "open": [100.0, 101.0, 104.0, 999.0],
            "close": [101.0, 102.0, 105.0, 999.0],
        }
    )

    result = cal_mmt_pm(frame)

    assert np.isclose(result["mmt_pm"][0], 1.05)


def test_mmt_paratio_is_invariant_to_input_order() -> None:
    frame = pl.DataFrame(
        {
            "code": ["510300.SH"] * 4,
            "date": [dt.date(2026, 1, 2)] * 4,
            "time": [93000000, 112900000, 130100000, 145900000],
            "open": [100.0, 100.0, 100.0, 100.0],
            "close": [101.0, 102.0, 103.0, 105.0],
        }
    )

    forward = cicc_methods.cal_mmt_paratio(frame)
    reverse = cicc_methods.cal_mmt_paratio(frame.reverse())

    assert np.isclose(forward["mmt_paratio"][0], 0.03)
    assert np.isclose(forward["mmt_paratio"][0], reverse["mmt_paratio"][0])


def test_doc_std_uses_standard_deviation_of_volume_distribution() -> None:
    frame = pl.DataFrame(
        {
            "code": ["510300.SH"] * 4,
            "date": [dt.date(2026, 1, 2)] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1.0, 2.0, 3.0, 20.0],
        }
    )

    result = cal_doc_std(frame)
    volume_distribution = np.array([1.0, 2.0, 3.0, 20.0]) / 26.0

    assert np.isclose(
        result["doc_std"][0], np.std(volume_distribution, ddof=1)
    )


def test_trade_bottom_ratios_are_independent_of_other_trade_dates() -> None:
    isolated_day = pl.DataFrame(
        {
            "code": ["510300.SH"] * 2,
            "date": [dt.date(2026, 1, 2)] * 2,
            "time": [93000000, 144000000],
            "open": [100.0, 100.0],
            "close": [100.0, 110.0],
            "volume": [100.0, 10.0],
        }
    )
    another_day = isolated_day.with_columns(
        pl.lit(dt.date(2026, 1, 5)).alias("date"),
        pl.col("volume") * 100.0,
    )
    combined = pl.concat([isolated_day, another_day])

    for function, factor_name in (
        (
            cicc_methods.cal_trade_bottom20retRatio,
            "trade_bottom20retRatio",
        ),
        (
            cicc_methods.cal_trade_bottom50retRatio,
            "trade_bottom50retRatio",
        ),
    ):
        isolated_value = function(isolated_day)[factor_name][0]
        combined_value = (
            function(combined)
            .filter(pl.col("date") == dt.date(2026, 1, 2))[factor_name][0]
        )

        assert np.isclose(isolated_value, combined_value)
        assert np.isclose(isolated_value, 10.0 / 110.0 * 0.1)
