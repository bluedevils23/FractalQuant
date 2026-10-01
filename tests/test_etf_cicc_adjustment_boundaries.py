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
    process_file,
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


def _end_labeled_times() -> list[int]:
    """241 end-labeled bars: 09:30 auction, 09:31-11:30, 13:01-15:00."""
    minutes = [9 * 60 + 30]
    minutes += list(range(9 * 60 + 31, 11 * 60 + 31))
    minutes += list(range(13 * 60 + 1, 15 * 60 + 1))
    return [(m // 60) * 10000000 + (m % 60) * 100000 for m in minutes]


def _full_day(**overrides: list[float]) -> pl.DataFrame:
    times = _end_labeled_times()
    n = len(times)
    data = {
        "code": ["510300.SH"] * n,
        "date": [dt.date(2026, 1, 2)] * n,
        "time": times,
        "open": [100.0] * n,
        "high": [100.0] * n,
        "low": [100.0] * n,
        "close": [100.0] * n,
        "volume": [1.0] * n,
    }
    data.update(overrides)
    return pl.DataFrame(data)


def _set_at(times: list[int], values: dict[int, float], default: float) -> list[float]:
    return [values.get(t, default) for t in times]


def test_trade_top_ratios_survive_zero_volume_bars() -> None:
    times = _end_labeled_times()
    # One zero-volume bar inside the first-20 window, one positive and one
    # negative bar return; daily volume = 241 - 1 = 240.
    volume = _set_at(times, {93500000: 0.0}, 1.0)
    close = _set_at(times, {93100000: 101.0, 93200000: 99.0}, 100.0)
    frame = _full_day(volume=volume, close=close)

    top20 = cicc_methods.cal_trade_top20retRatio(frame)["trade_top20retRatio"][0]
    top50 = cicc_methods.cal_trade_top50retRatio(frame)["trade_top50retRatio"][0]
    neg = cicc_methods.cal_trade_topNeg20retRatio(frame)["trade_topNeg20retRatio"][0]
    pos = cicc_methods.cal_trade_topPos20retRatio(frame)["trade_topPos20retRatio"][0]

    share = 1.0 / 240.0
    assert np.isclose(top20, (0.01 - 0.01) * share)
    assert np.isclose(top50, (0.01 - 0.01) * share)
    assert np.isclose(neg, 0.01 * share)
    assert np.isclose(pos, 0.01 * share)


def test_session_momentum_windows_match_end_labeled_bars() -> None:
    times = _end_labeled_times()
    open_ = _set_at(
        times, {93000000: 100.0, 100100000: 101.0, 143100000: 104.0}, 999.0
    )
    close = _set_at(
        times, {113000000: 110.0, 143000000: 111.0, 145900000: 114.4}, 999.0
    )
    frame = _full_day(open=open_, close=close)

    assert np.isclose(cicc_methods.cal_mmt_am(frame)["mmt_am"][0], 1.10)
    assert np.isclose(
        cicc_methods.cal_mmt_between(frame)["mmt_between"][0], 111.0 / 101.0
    )
    assert np.isclose(cicc_methods.cal_mmt_last30(frame)["mmt_last30"][0], 1.10)


def test_closing_volume_windows_use_only_continuous_trading_bars() -> None:
    times = _end_labeled_times()
    # SZSE closes with a call auction (14:58-15:00 bars) while SSE ETFs keep
    # trading continuously, so only bars up to 14:57 are comparable.
    volume = _set_at(
        times,
        {
            145500000: 3.0,
            145600000: 3.0,
            145700000: 4.0,
            145800000: 20.0,
            145900000: 20.0,
            150000000: 30.0,
        },
        0.0,
    )
    volume[1] = 60.0  # 09:31 bar
    frame = _full_day(volume=volume)

    for data in (frame, frame.reverse()):
        closevol = cicc_methods.cal_liq_closevol(data)["liq_closevol"][0]
        prevol = cicc_methods.cal_liq_closeprevol(data)["liq_closeprevol"][0]
        last_call = cicc_methods.cal_liq_lastCallR(data)["liq_lastCallR"][0]

        assert np.isclose(closevol, 10.0)
        assert np.isclose(prevol, 60.0)
        assert np.isclose(last_call, 10.0 / (10.0 + 70.0 + 60.0))


def test_opening_window_factors_exclude_the_0930_bar() -> None:
    times = _end_labeled_times()
    # Only the 09:30 bar carries a big return/volume; it must not leak in.
    volume = _set_at(times, {93000000: 500.0}, 1.0)
    close = _set_at(times, {93000000: 110.0}, 100.0)
    frame = _full_day(volume=volume, close=close)

    top20 = cicc_methods.cal_trade_top20retRatio(frame)["trade_top20retRatio"][0]
    top50 = cicc_methods.cal_trade_top50retRatio(frame)["trade_top50retRatio"][0]
    pos = cicc_methods.cal_trade_topPos20retRatio(frame)["trade_topPos20retRatio"][0]
    head = cicc_methods.cal_trade_headRatio(frame)["trade_headRatio"][0]

    assert np.isclose(top20, 0.0)
    assert np.isclose(top50, 0.0)
    assert np.isclose(pos, 0.0)
    assert np.isclose(head, 30.0 / (500.0 + 240.0))


def test_opening_window_bar_counts() -> None:
    frame = _full_day()
    # Unit volume everywhere, every bar returns +1%: the weighted sum equals
    # the number of bars in the window times 0.01 / 241.
    close = [101.0] * frame.height
    frame = frame.with_columns(pl.Series("close", close))

    top20 = cicc_methods.cal_trade_top20retRatio(frame)["trade_top20retRatio"][0]
    top50 = cicc_methods.cal_trade_top50retRatio(frame)["trade_top50retRatio"][0]
    head = cicc_methods.cal_trade_headRatio(frame)["trade_headRatio"][0]

    assert np.isclose(top20, 20 * 0.01 / 241)
    assert np.isclose(top50, 50 * 0.01 / 241)
    assert np.isclose(head, 30 / 241)


def test_trade_tail_ratio_uses_last_30_bars() -> None:
    times = _end_labeled_times()
    volume = _set_at(times, {143000000: 1000.0}, 1.0)
    frame = _full_day(volume=volume)

    tail = cicc_methods.cal_trade_tailRatio(frame)["trade_tailRatio"][0]

    assert np.isclose(tail, 30.0 / (240.0 + 1000.0))


def _write_minute_source(path, days: list[str]) -> None:
    frames = []
    for day in days:
        times = pd.to_datetime(
            [
                f"{day} {t // 10000000:02d}:{t // 100000 % 100:02d}"
                for t in _end_labeled_times()
            ]
        )
        n = len(times)
        rng = np.random.default_rng(len(frames))
        close = 100.0 + rng.normal(0, 0.05, n).cumsum()
        frames.append(
            pd.DataFrame(
                {
                    "ts_code": "510300.SH",
                    "open": close - 0.01,
                    "high": close + 0.02,
                    "low": close - 0.03,
                    "close": close,
                    "vol": rng.integers(1, 1000, n).astype(float),
                    "trade_date": times.normalize(),
                    "trade_time": times,
                }
            )
        )
    pd.concat(frames).set_index(["trade_date", "trade_time"]).to_parquet(path)


def test_process_file_merges_date_range_into_existing_output(tmp_path) -> None:
    source = tmp_path / "in" / "510300.SH.parquet"
    source.parent.mkdir()
    _write_minute_source(source, ["2026-01-05", "2026-01-06"])
    out_root = tmp_path / "out"
    day1, day2 = normalize_date_arg("2026-01-05"), normalize_date_arg("2026-01-06")

    status1, out_path, *_ = process_file(source, out_root, False, day1, day1)
    status2, *_ = process_file(source, out_root, False, day2, day2)
    status3, *_ = process_file(source, out_root, False, day2, day2)
    merged = pd.read_parquet(out_path)

    assert (status1, status2, status3) == ("written", "merged", "merged")
    assert merged.index.tolist() == [day1, day2]
    assert not list(out_root.glob("*.tmp"))

    full = tmp_path / "full"
    process_file(source, full, False)
    pd.testing.assert_frame_equal(
        merged, pd.read_parquet(full / source.name), check_freq=False
    )


def test_process_file_without_date_range_still_skips_existing(tmp_path) -> None:
    source = tmp_path / "510300.SH.parquet"
    _write_minute_source(source, ["2026-01-05"])
    out_root = tmp_path / "out"

    process_file(source, out_root, False)
    status, *_ = process_file(source, out_root, False)

    assert status == "skipped"


def test_open_volume_factors_use_first_continuous_minute_bar() -> None:
    times = _end_labeled_times()
    # 09:30 bar differs by exchange (SH: auction only, SZ: auction + early
    # continuous trading); 09:31 is the comparable first continuous minute.
    volume = _set_at(times, {93000000: 500.0, 93100000: 7.0}, 1.0)
    frame = _full_day(volume=volume)
    total = 500.0 + 7.0 + 239.0

    for data in (frame, frame.reverse()):
        openvol = cicc_methods.cal_liq_openvol(data)["liq_openvol"][0]
        first_r = cicc_methods.cal_liq_firstCallR(data)["liq_firstCallR"][0]

        assert np.isclose(openvol, 7.0)
        assert np.isclose(first_r, 7.0 / total)
