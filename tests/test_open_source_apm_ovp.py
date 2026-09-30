from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json
import sys

from scripts import generate_open_source_apm_ovp as generator

from factor.open_source_crossmarket import (
    APM_FACTOR_COLUMNS,
    trading_dates,
    _cross_section_residual,
    _rolling_apm_stat,
    add_availability_columns,
    build_daily_apm_raw_features,
    build_open_source_apm_panel,
)


def _minute_frame(symbol: str, dates: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    times = list(pd.date_range("2000-01-01 09:30", periods=121, freq="min").time)
    times += list(pd.date_range("2000-01-01 13:01", periods=120, freq="min").time)
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
                    "adj_factor": 1.0,
                    "ts_code": symbol,
                }
            )
    return pd.DataFrame(rows)


def test_apmnew_uses_overnight_pm_and_cross_section_residual() -> None:
    dates = trading_dates(pd.Timestamp("2025-02-05"), pd.Timestamp("2025-03-31"))[:22]
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
    assert result.loc[result["trade_date"].eq(dates[-1]), "available_date"].eq(pd.Timestamp("2025-03-07")).all()


def test_apm_panel_keeps_null_when_cross_section_is_too_small() -> None:
    dates = trading_dates(pd.Timestamp("2025-02-05"), pd.Timestamp("2025-03-31"))[:22]
    asset = _minute_frame("000001.SZ", dates, 100.0)
    benchmark = _minute_frame("000300.SH", dates, 200.0)
    raw = build_daily_apm_raw_features(asset, benchmark, "000001.SZ", "000300.SH")
    result = build_open_source_apm_panel(raw)
    assert result["kaiyuan_apmnew_m20"].isna().all()
    assert result["apm_cross_section_count"].max() == 1
    assert result["kaiyuan_ovp_m20"].notna().any()


def test_apm_raw_features_are_invariant_to_split_adjustment() -> None:
    dates = trading_dates(pd.Timestamp("2025-02-05"), pd.Timestamp("2025-03-31"))[:22]
    asset = _minute_frame("000001.SZ", dates, 100.0)
    benchmark = _minute_frame("000300.SH", dates, 200.0)
    baseline = build_daily_apm_raw_features(
        asset, benchmark, "000001.SZ", "000300.SH"
    )
    split = asset.copy()
    split_date = split["trade_time"].dt.normalize().max()
    split_mask = split["trade_time"].dt.normalize().eq(split_date)
    split.loc[split_mask, ["open", "high", "low", "close"]] /= 2.0
    split.loc[split_mask, "adj_factor"] = 2.0
    adjusted = build_daily_apm_raw_features(
        split, benchmark, "000001.SZ", "000300.SH"
    )

    columns = ["overnight_return", "am_return", "pm_return", "_asset_close"]
    pd.testing.assert_series_equal(
        adjusted.loc[21, columns], baseline.loc[21, columns], check_names=False
    )


def test_real_bar_grid_and_missing_adjustment_keep_intraday_returns():
    dates = trading_dates(pd.Timestamp("2025-02-05"), pd.Timestamp("2025-02-07"))
    a = _minute_frame("A", dates, 100)
    b = _minute_frame("B", dates, 200)
    baseline = build_daily_apm_raw_features(a, b, "A", "B")
    result = build_daily_apm_raw_features(a.drop(columns="adj_factor"), b, "A", "B")
    assert baseline.pm_return.notna().all()
    pd.testing.assert_frame_equal(result[["am_return", "pm_return"]], baseline[["am_return", "pm_return"]])
    assert result.overnight_return.isna().all()
    assert result._asset_close.isna().all()


def test_missing_benchmark_day_does_not_change_asset_overnight():
    dates = trading_dates(pd.Timestamp("2025-02-05"), pd.Timestamp("2025-02-14"))
    a = _minute_frame("A", dates, 100)
    b = _minute_frame("B", dates, 200)
    baseline = build_daily_apm_raw_features(a, b, "A", "B")
    missing = b.loc[b.trade_time.dt.normalize().ne(dates[2])]
    result = build_daily_apm_raw_features(a, missing, "A", "B")
    pd.testing.assert_series_equal(result.overnight_return, baseline.overnight_return)
    assert np.isnan(result.loc[3, "_benchmark_overnight_return"])
    assert result.loc[2, "reference_valid_session_bars"] == 0
    missing_asset = a.loc[a.trade_time.dt.normalize().ne(dates[2])]
    result = build_daily_apm_raw_features(missing_asset, b, "A", "B")
    assert len(result) == len(dates)
    assert result.loc[2:3, "overnight_return"].isna().all()


@pytest.mark.parametrize("damage", ["missing", "zero", "infinite"])
def test_incomplete_or_invalid_afternoon_is_null(damage):
    dates = trading_dates(pd.Timestamp("2025-02-05"), pd.Timestamp("2025-02-06"))
    a = _minute_frame("A", dates, 100)
    b = _minute_frame("B", dates, 200)
    mask = a.trade_time.eq(dates[1] + pd.Timedelta(hours=14))
    if damage == "missing":
        a = a.loc[~mask]
    else:
        a.loc[mask, "close"] = 0 if damage == "zero" else np.inf
    result = build_daily_apm_raw_features(a, b, "A", "B")
    assert np.isnan(result.loc[1, "pm_return"])
    assert np.isfinite(result.loc[1, "am_return"])
    assert result.loc[1, "valid_session_bars"] == 239


def test_availability_uses_exchange_holidays_and_latest_day():
    frame = pd.DataFrame({"trade_date": pd.to_datetime(["2026-09-24", "2026-09-30"]), "ts_code": ["A", "A"]})
    for col in generator.APM_FACTOR_COLUMNS:
        frame[col] = 1.0
    from factor.open_source_crossmarket import AUDIT_COLUMNS
    result = add_availability_columns(frame.reindex(columns=AUDIT_COLUMNS))
    assert result.available_date.tolist() == [pd.Timestamp("2026-09-28"), pd.Timestamp("2026-10-08")]
    assert (result.available_time - result.available_date).eq(pd.Timedelta(hours=9, minutes=30)).all()


def test_paper_regressions_and_arithmetic_sum():
    rng = np.random.default_rng(7)
    x = rng.normal(0, .01, 40)
    y = 1.2*x + rng.normal(0, .003, 40) + np.r_[np.full(20, .002), np.zeros(20)]
    beta = np.sum((x-x.mean())*(y-y.mean())) / np.sum((x-x.mean())**2)
    delta = y[:20]-y[20:] - beta*(x[:20]-x[20:])
    group = pd.DataFrame({"a": y[:20], "b": x[:20], "c": y[20:], "d": x[20:]})
    assert _rolling_apm_stat(group, "a", "b", "c", "d", 20).iloc[-1] == pytest.approx(delta.mean()/delta.std(ddof=1)*np.sqrt(20))
    frame = pd.DataFrame({"trade_date": [pd.Timestamp("2025-02-05")]*20, "stat": np.arange(20)+5., "ret20": np.arange(20)/100.})
    residual, count, _ = _cross_section_residual(frame, "stat", "ret20")
    slope = np.dot(frame.ret20, frame.stat) / np.dot(frame.ret20, frame.ret20)
    np.testing.assert_allclose(residual, frame.stat-slope*frame.ret20)
    assert count.eq(20).all()
    dates = trading_dates(pd.Timestamp("2025-02-05"), pd.Timestamp("2025-03-31"))[:22]
    raw = build_daily_apm_raw_features(_minute_frame("A", dates, 100), _minute_frame("B", dates, 200), "A", "B")
    result = build_open_source_apm_panel(raw)
    assert result.kaiyuan_ovp_m20.iloc[-1] == pytest.approx((raw.overnight_return-raw.pm_return).iloc[-20:].sum())
    assert result.kaiyuan_avp_m20.iloc[-1] == pytest.approx((raw.am_return-raw.pm_return).iloc[-20:].sum())
    # Removing an interior day must not turn a 21-session span into a 20-day window.
    result = build_open_source_apm_panel(raw.drop(index=10))
    assert result.kaiyuan_ovp_m20.iloc[-1:].isna().all()


def test_index_resolution_never_substitutes_exchange(tmp_path):
    (tmp_path / "000001.SZ.parquet").touch()
    assert generator.find_index_parquet(tmp_path, "000001.SH") is None
    (tmp_path / "000001.SH.parquet").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        generator.find_index_parquet(tmp_path, "000001")


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    dates = trading_dates(pd.Timestamp("2025-02-05"), pd.Timestamp("2025-04-30"))[:45]
    symbols = [f"{i:06d}.SZ" for i in range(20)]
    assets = tmp_path / "assets"
    indices = tmp_path / "indices"
    assets.mkdir(); indices.mkdir()
    for symbol in symbols:
        (assets / f"{symbol}.parquet").touch()
    (indices / "000300.SH.parquet").touch()
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame({"fund_code": symbols, "reference_index_code": ["000300.SH"]*20}).to_csv(mapping, index=False)
    monkeypatch.setattr(generator, "read_minutes", lambda path, start, end: (path, start, end))

    def build(asset, benchmark, symbol, reference):
        rng = np.random.default_rng(int(symbol[:6]))
        raw = pd.DataFrame({"trade_date": dates, "ts_code": symbol, "reference_index_code": reference,
            "_asset_close": 100*np.cumprod(1+rng.normal(.001, .01, len(dates))),
            "valid_session_bars": 240, "reference_valid_session_bars": 240, "mapping_available": True})
        for col in ["overnight_return", "am_return", "pm_return", "_benchmark_overnight_return", "_benchmark_am_return", "_benchmark_pm_return"]:
            raw[col] = rng.normal(.001, .01, len(dates))
        if asset[1] is not None:
            raw = raw.loc[raw.trade_date.ge(asset[1])]
        if asset[2] is not None:
            raw = raw.loc[raw.trade_date.le(asset[2])]
        return raw
    monkeypatch.setattr(generator, "build_daily_apm_raw_features", build)

    def run(output, *extra):
        monkeypatch.setattr(sys, "argv", ["generate", "--input-root", str(assets), "--index-root", str(indices), "--mapping", str(mapping), "--output-root", str(output), *extra])
        return generator.main()
    return run, symbols, dates, assets


def test_cli_warmup_and_partial_resume_match_full_universe(cli_inputs, tmp_path):
    run, symbols, dates, _ = cli_inputs
    full, partial, sliced = [tmp_path / name for name in ("full", "partial", "sliced")]
    assert run(full) == 0
    assert run(partial, "--symbols", symbols[0]) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(full / f"{symbols[0]}.parquet"), pd.read_parquet(partial / f"{symbols[0]}.parquet"))
    assert run(partial) == 0
    for symbol in symbols:
        pd.testing.assert_frame_equal(pd.read_parquet(full / f"{symbol}.parquet"), pd.read_parquet(partial / f"{symbol}.parquet"))
    assert run(sliced, "--date-from", str(dates[25].date()), "--date-to", str(dates[-1].date())) == 0
    for symbol in symbols:
        expected = pd.read_parquet(full / f"{symbol}.parquet").iloc[25:].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, pd.read_parquet(sliced / f"{symbol}.parquet"))
    report = json.loads((sliced / "run_report.json").read_text())
    assert report["status"] == "complete"
    assert all(v["nonnull"]["kaiyuan_apmnew_m20"] == 20 for v in report["coverage"].values())


def test_cli_failure_and_lock_prevent_partial_publication(cli_inputs, tmp_path):
    run, symbols, _, assets = cli_inputs
    (assets / f"{symbols[-1]}.parquet").unlink()
    out = tmp_path / "failed"
    assert run(out) == 1
    assert not list(out.glob("*.parquet"))
    assert json.loads((out / "run_report.json").read_text())["failures"]
    assert not (out / ".apm_generation.lock").exists()
    (out / ".apm_generation.lock").touch()
    assert run(out) == 1
    assert (out / ".apm_generation.lock").exists()


def test_cli_rejects_unversioned_existing_files(cli_inputs, tmp_path):
    run, symbols, _, _ = cli_inputs
    out = tmp_path / "old"
    out.mkdir()
    (out / f"{symbols[0]}.parquet").write_bytes(b"old")
    assert run(out) == 1
    assert (out / f"{symbols[0]}.parquet").read_bytes() == b"old"
