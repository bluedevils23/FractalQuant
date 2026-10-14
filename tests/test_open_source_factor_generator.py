from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor.open_source import (  # noqa: E402
    OPEN_SOURCE_FACTOR_COLUMNS,
    SESSION_BAR_COUNT,
    _regular_session_frame,
    _smart_money_gap_for_sessions,
    _daily_return_series,
    _state_daily_metrics,
    add_availability_columns,
    build_daily_raw_features,
    build_open_source_factor_panel,
    normalize_minute_frame,
    select_output_columns,
)
from scripts import generate_open_source_factors  # noqa: E402


def _minute_frame(
    days: int = 4,
    *,
    symbol: str = "510300.SH",
    with_amount: bool = True,
    offset: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2026-01-05", periods=days)
    for day_number, day in enumerate(dates):
        times = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=30), periods=121, freq="min"
        ).append(
            pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min")
        )
        minute = np.arange(241, dtype=float)
        close = 100.0 + offset + day_number * 0.2 + np.sin((minute + offset) / 9.0) * 0.05
        volume = 1_000.0 + offset * 10.0 + (minute % 17.0) * 25.0
        for position, trade_time in enumerate(times):
            row = {
                "trade_time": trade_time,
                "open": close[position] - 0.01,
                "high": close[position] + 0.02,
                "low": close[position] - 0.02,
                "close": close[position],
                "vol": volume[position],
                "adj_factor": 1.0,
                "ts_code": symbol,
            }
            if with_amount:
                row["amount"] = close[position] * volume[position]
            rows.append(row)
    return pd.DataFrame(rows)


def _raw_panel(symbols: int = 2, days: int = 4) -> pd.DataFrame:
    parts = []
    for number in range(symbols):
        symbol = f"{number + 1:06d}.SZ"
        source = _minute_frame(days, symbol=symbol, offset=float(number))
        parts.append(
            build_daily_raw_features(
                source.set_index("trade_time"),
                symbol,
                factor_window=3,
                smart_money_window=2,
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_241_bar_source_is_normalized_to_240_close_labelled_bars() -> None:
    source = _minute_frame(days=1)
    normalized = normalize_minute_frame(source)
    session = _regular_session_frame(normalized)

    assert len(normalized) == 241
    assert len(session) == SESSION_BAR_COUNT
    assert session.index[0].strftime("%H:%M") == "09:31"
    assert session.index[-1].strftime("%H:%M") == "15:00"


def test_ideal_amplitude_uses_high_and_low_price_states() -> None:
    source = _minute_frame(days=4)
    raw = build_daily_raw_features(
        source.set_index("trade_time"),
        "510300.SH",
        factor_window=4,
        smart_money_window=2,
    )
    expected = raw["daily_amplitude"].iloc[-1] - raw["daily_amplitude"].iloc[0]

    assert np.isclose(raw["_ideal_amplitude"].iloc[-1], expected)


def test_cross_day_price_features_are_invariant_to_split_adjustment() -> None:
    source = _minute_frame(days=4)
    baseline = build_daily_raw_features(
        source.set_index("trade_time"),
        "510300.SH",
        factor_window=4,
        smart_money_window=2,
    )
    split = source.copy()
    split_date = split["trade_time"].dt.normalize().max()
    split_mask = split["trade_time"].dt.normalize().eq(split_date)
    split.loc[split_mask, ["open", "high", "low", "close"]] /= 2.0
    split.loc[split_mask, "adj_factor"] = 2.0
    adjusted = build_daily_raw_features(
        split.set_index("trade_time"),
        "510300.SH",
        factor_window=4,
        smart_money_window=2,
    )

    columns = [
        "daily_close",
        "daily_return",
        "overnight_return",
        "_ideal_amplitude",
        "_ideal_reversal",
        "ideal_turnover_raw",
    ]
    pd.testing.assert_series_equal(
        adjusted.loc[3, columns], baseline.loc[3, columns], check_names=False
    )


def test_smart_money_selection_is_volume_weighted_and_causal() -> None:
    source = _minute_frame(days=1)
    normalized = normalize_minute_frame(source)
    session = _regular_session_frame(normalized)
    baseline, valid_days = _smart_money_gap_for_sessions([session])
    all_vwap = np.average(session["close"], weights=session["volume"])

    assert valid_days == 1
    assert np.isfinite(baseline)
    assert not np.isclose(baseline, 0.0)
    assert np.isfinite(all_vwap)


def test_amount_missing_keeps_reversal_null_but_not_ohlcv_factors() -> None:
    source = _minute_frame(days=4, with_amount=False)
    raw = build_daily_raw_features(
        source.set_index("trade_time"),
        "510300.SH",
        factor_window=3,
        smart_money_window=2,
    )

    assert raw["amount_available"].eq(False).all()
    assert raw["_ideal_reversal"].isna().all()
    assert raw["_ideal_amplitude"].notna().any()
    assert raw["smart_money_vwap_gap_m10"].notna().any()


def test_err_is_rank_added_across_symbols_on_the_same_date() -> None:
    raw = _raw_panel(symbols=2, days=4)
    raw.loc[raw["ts_code"] == "000002.SZ", "extreme_return_m20"] *= 10.0
    raw.loc[raw["ts_code"] == "000002.SZ", "extreme_prior_return_m20"] *= 10.0

    panel = build_open_source_factor_panel(raw, factor_window=3, min_tgd_cross_section=1)
    last_date = panel["trade_date"].max()
    same_day = panel.loc[panel["trade_date"].eq(last_date)]

    assert same_day["kaiyuan_err_m20"].notna().all()
    assert same_day["kaiyuan_err_m20"].nunique() == 2


def test_tgd_is_null_when_cross_section_is_too_small() -> None:
    raw = _raw_panel(symbols=2, days=4)
    panel = build_open_source_factor_panel(raw, factor_window=3, min_tgd_cross_section=20)

    assert panel["tgd_cross_section_count"].max() <= 2
    assert panel["kaiyuan_tgd_m20"].isna().all()


def test_availability_includes_latest_day() -> None:
    raw = _raw_panel(symbols=1, days=4)
    panel = build_open_source_factor_panel(raw, factor_window=3, min_tgd_cross_section=1)
    available = add_availability_columns(panel)

    pd.testing.assert_series_equal(
        available["available_date"].iloc[:-1],
        available["trade_date"].iloc[1:].reset_index(drop=True),
        check_dtype=False, check_names=False,
    )
    assert available["available_time"].iloc[0].hour == 9
    assert available["available_date"].iloc[-1] == pd.Timestamp("2026-01-09")


def test_output_columns_are_unique_and_finite_except_expected_nulls() -> None:
    panel = add_availability_columns(
        build_open_source_factor_panel(
            _raw_panel(symbols=2, days=4), factor_window=3, min_tgd_cross_section=1
        )
    )
    output = select_output_columns(panel)

    assert output.columns.is_unique
    assert set(OPEN_SOURCE_FACTOR_COLUMNS) <= set(output.columns)
    numeric = output.select_dtypes(include=["number"])
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()


def test_minute_state_factor_batch_needs_history_and_preserves_amount_nulls() -> None:
    source = _minute_frame(days=45, with_amount=True)
    day_scale = 1.0 + np.arange(len(source), dtype=float) // 241.0 / 100.0
    source["vol"] = source["vol"] * day_scale
    source["amount"] = source["close"] * source["vol"]
    raw = build_daily_raw_features(
        source.set_index("trade_time"),
        "510300.SH",
        factor_window=3,
        smart_money_window=2,
    )
    panel = select_output_columns(
        add_availability_columns(
            build_open_source_factor_panel(raw, factor_window=3, min_tgd_cross_section=1)
        )
    )
    state_columns = [
        "kaiyuan_volume_peak_count_m20",
        "kaiyuan_volume_ridge_return_m20",
        "kaiyuan_price_peak_count_m20",
        "kaiyuan_intraday_amplitude_cut_mean_m10_l20",
    ]
    assert panel[state_columns].notna().any().all()
    assert panel["kaiyuan_ideal_turnover_m20_q25"].notna().any()

    no_amount_source = _minute_frame(days=45, with_amount=False)
    no_amount_source["vol"] = no_amount_source["vol"] * day_scale
    no_amount = build_daily_raw_features(
        no_amount_source.set_index("trade_time"),
        "510300.SH",
        factor_window=3,
        smart_money_window=2,
    )
    no_amount_panel = select_output_columns(
        add_availability_columns(
            build_open_source_factor_panel(no_amount, factor_window=3, min_tgd_cross_section=1)
        )
    )
    assert no_amount_panel["kaiyuan_volume_peak_count_m20"].notna().any()
    assert no_amount_panel["kaiyuan_ideal_turnover_m20_q25"].isna().all()
    assert no_amount_panel["kaiyuan_volume_peak_ridge_amount_ratio_m20"].isna().all()


def test_cli_smoke_writes_filtered_symbol_output(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "minute"
    output_root = tmp_path / "factors"
    input_root.mkdir()
    for symbol, offset in (("510300.SH", 0.0), ("510500.SH", 1.0)):
        source = _minute_frame(days=3, symbol=symbol, offset=offset)
        source = source.set_index(["trade_time"])
        source.to_parquet(input_root / f"{symbol}.parquet")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_open_source_factors.py",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--symbols",
            "510300.SH",
            "--date-from",
            "20260106",
            "--date-to",
            "20260107",
            "--workers",
            "1",
            "--min-tgd-cross-section",
            "1",
        ],
    )
    assert generate_open_source_factors.main() == 0
    output_path = output_root / "510300.SH.parquet"
    output = pd.read_parquet(output_path)

    assert len(output) == 2
    assert output["trade_date"].astype(str).tolist() == ["2026-01-06", "2026-01-07"]
    assert output.columns.is_unique


def test_smart_money_is_invariant_to_split_and_requires_adjustment() -> None:
    source = _minute_frame(days=10)
    baseline = build_daily_raw_features(source, "510300.SH")
    split = source.copy()
    split_mask = split["trade_time"].dt.normalize().ge(pd.Timestamp("2026-01-12"))
    split.loc[split_mask, ["open", "high", "low", "close"]] /= 2
    split.loc[split_mask, "vol"] *= 2
    split.loc[split_mask, "adj_factor"] = 2
    adjusted = build_daily_raw_features(split, "510300.SH")
    assert np.isfinite(baseline["smart_money_vwap_gap_m10"].iloc[-1])
    np.testing.assert_allclose(
        baseline["smart_money_vwap_gap_m10"], adjusted["smart_money_vwap_gap_m10"],
        equal_nan=True, atol=1e-12,
    )
    missing = build_daily_raw_features(source.drop(columns="adj_factor"), "510300.SH")
    assert missing["smart_money_vwap_gap_m10"].isna().all()


@pytest.mark.parametrize("opening_bar", [True, False])
def test_first_minute_return_reaches_daily_features(opening_bar: bool) -> None:
    source = _minute_frame(days=1)
    source.loc[:, ["open", "high", "low", "close"]] = 110.0
    source.loc[source["trade_time"].dt.strftime("%H:%M").eq("09:30"), "close"] = 100.0
    source.loc[source["trade_time"].dt.strftime("%H:%M").eq("09:31"), "open"] = 100.0
    if not opening_bar:
        source = source.loc[source["trade_time"].dt.strftime("%H:%M").ne("09:30")]
    result = build_daily_raw_features(source, "510300.SH")
    assert result.loc[0, "r1"] == pytest.approx(0.1)
    assert result.loc[0, "_extreme_return_raw"] == pytest.approx(0.1)
    assert result.loc[0, "gu"] == pytest.approx(1.0)


def test_minute_return_does_not_fill_missing_price() -> None:
    session = pd.DataFrame({"open": [100.0] * 4, "close": [101.0, np.nan, 103.0, 104.0]})
    result = _daily_return_series(session)
    assert result.iloc[0] == pytest.approx(0.01)
    assert result.iloc[1:3].isna().all()


def _state_session(volume: float, amplitude: float) -> pd.DataFrame:
    session = _regular_session_frame(normalize_minute_frame(_minute_frame(days=1)))
    session["volume"] = volume
    session["amount"] = 100.0
    session["_minute_return"] = np.r_[np.nan, np.arange(1, 240) / 100000.0]
    session["_minute_amplitude"] = amplitude
    return session


def test_amplitude_cut_excludes_invalid_rows_before_selecting_tails() -> None:
    current = _state_session(150.0, 0.01)
    current["_minute_amplitude"] = np.arange(240) / 10000.0
    # A finite return with missing amplitude must not enter either tail.
    current.iloc[238, current.columns.get_loc("_minute_amplitude")] = np.nan
    result = _state_daily_metrics(current, [])
    valid = current[["_minute_return", "_minute_amplitude"]].dropna().sort_values("_minute_return")
    count = int(np.ceil(len(valid) * 0.2))
    expected = valid["_minute_amplitude"].tail(count).mean() - valid["_minute_amplitude"].head(count).mean()
    assert result["intraday_amplitude_cut_raw"] == pytest.approx(expected)


@pytest.mark.parametrize("constant", ["volume", "amplitude"])
def test_state_families_do_not_block_each_other(constant: str) -> None:
    history = [
        _state_session(100.0 if constant == "volume" else 100.0 + i,
                       0.01 if constant == "amplitude" else 0.01 + i * 0.0001)
        for i in range(20)
    ]
    result = _state_daily_metrics(_state_session(150.0, 0.02), history)
    valid_prefix, invalid_prefix = ("price", "volume") if constant == "volume" else ("volume", "price")
    assert np.isfinite(result[f"{valid_prefix}_peak_count_raw"])
    assert np.isnan(result[f"{invalid_prefix}_peak_count_raw"])
    assert np.isfinite(result["intraday_amplitude_cut_raw"])


def test_eruption_follow_ratio_uses_only_paired_events() -> None:
    history = [_state_session(100.0 + i, 0.01) for i in range(20)]
    current = _state_session(100.0, 0.01)
    current.iloc[[118, 119, 239], current.columns.get_loc("volume")] = 1000.0
    current.iloc[[118, 119, 239], current.columns.get_loc("amount")] = [100.0, 300.0, 900.0]
    # Only position 118 has a same-session successor. 119 and 239 are boundaries.
    assert _state_daily_metrics(current, history)["volume_eruption_follow_ratio_raw"] == pytest.approx(3.0)
    current.iloc[118, current.columns.get_loc("volume")] = 100.0
    assert np.isnan(_state_daily_metrics(current, history)["volume_eruption_follow_ratio_raw"])


def test_availability_uses_exchange_holidays_and_ignores_symbol_gaps() -> None:
    panel = pd.DataFrame({
        "ts_code": ["A", "A", "B", "B"],
        "trade_date": pd.to_datetime(["2026-09-21", "2026-09-24", "2026-09-21", "2026-09-30"]),
    })
    result = add_availability_columns(panel)
    assert result["available_date"].tolist() == list(pd.to_datetime([
        "2026-09-22", "2026-09-28", "2026-09-22", "2026-10-08",
    ]))
    assert result["available_time"].dt.hour.eq(9).all()
    assert result["available_time"].dt.minute.eq(30).all()


def test_availability_rejects_unknown_next_session() -> None:
    from exchange_calendars.exchange_calendar_xshg import XSHGExchangeCalendar
    panel = pd.DataFrame({"ts_code": ["A"], "trade_date": [XSHGExchangeCalendar.bound_max()]})
    with pytest.raises(ValueError):
        add_availability_columns(panel)


def test_cli_subset_preserves_full_cross_section(tmp_path: Path, monkeypatch) -> None:
    import json
    input_root = tmp_path / "input"
    input_root.mkdir()
    for symbol, offset in (("510300.SH", 0.0), ("510500.SH", 1.0)):
        _minute_frame(days=12, symbol=symbol, offset=offset).to_parquet(input_root / f"{symbol}.parquet")
    base_args = ["generate_open_source_factors.py", "--input-root", str(input_root), "--workers", "1"]
    for folder, extra in (("full", []), ("subset", ["--symbols", "510300.SH"]), ("limited", ["--limit", "1"])):
        monkeypatch.setattr(sys, "argv", base_args + ["--output-root", str(tmp_path / folder)] + extra)
        assert generate_open_source_factors.main() == 0
    full = pd.read_parquet(tmp_path / "full" / "510300.SH.parquet")
    assert full["kaiyuan_err_m20"].notna().any()
    for folder in ("subset", "limited"):
        pd.testing.assert_frame_equal(full, pd.read_parquet(tmp_path / folder / "510300.SH.parquet"))
        assert len(list((tmp_path / folder).glob("*.parquet"))) == 1
        manifest = json.loads((tmp_path / folder / "manifest.json").read_text())
        assert manifest["calculation_universe"] == ["510300.SH", "510500.SH"]
    # A changed pool must not be appended to the existing factor set.
    _minute_frame(days=1, symbol="510050.SH").to_parquet(input_root / "510050.SH.parquet")
    monkeypatch.setattr(sys, "argv", base_args + ["--output-root", str(tmp_path / "full"), "--overwrite"])
    with pytest.raises(ValueError, match="contract changed"):
        generate_open_source_factors.main()


@pytest.mark.parametrize("failed_input", ["corrupt", "empty"])
def test_cli_extraction_failure_writes_nothing(tmp_path: Path, monkeypatch, failed_input: str) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    _minute_frame(days=1).to_parquet(input_root / "510300.SH.parquet")
    bad = input_root / "510500.SH.parquet"
    if failed_input == "corrupt":
        bad.write_text("not parquet")
    else:
        _minute_frame(days=1).iloc[:0].to_parquet(bad)
    monkeypatch.setattr(sys, "argv", ["generate_open_source_factors.py", "--input-root", str(input_root),
                                     "--output-root", str(output_root), "--symbols", "510300.SH", "--workers", "1"])
    assert generate_open_source_factors.main() == 1
    assert not output_root.exists()


def test_cli_rejects_unversioned_outputs(tmp_path: Path, monkeypatch) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    _minute_frame(days=1).to_parquet(input_root / "510300.SH.parquet")
    sentinel = output_root / "510300.SH.parquet"
    sentinel.write_bytes(b"existing output")
    monkeypatch.setattr(sys, "argv", ["generate_open_source_factors.py", "--input-root", str(input_root),
                                     "--output-root", str(output_root), "--overwrite"])
    with pytest.raises(ValueError, match="no calculation manifest"):
        generate_open_source_factors.main()
    assert sentinel.read_bytes() == b"existing output"


def test_cli_subset_preserves_nonempty_tgd(tmp_path: Path, monkeypatch) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    input_root.mkdir()
    symbols = [f"{number:06d}.SH" for number in range(1, 9)]
    for symbol in symbols:
        (input_root / f"{symbol}.parquet").touch()
    template = _raw_panel(symbols=1, days=1).iloc[0].to_dict()
    rng = np.random.default_rng(42)
    rows = []
    for date in pd.bdate_range("2026-01-05", periods=25):
        for symbol in symbols:
            row = dict(template, trade_date=date, ts_code=symbol)
            for column in ("gu", "gd", "avg_up_return", "avg_down_return", "r1", "r2", "overnight_return"):
                row[column] = rng.normal()
            rows.append(row)
    raw = pd.DataFrame(rows)

    def extract(files, workers):
        assert [path.stem for path in files] == symbols
        return raw, []

    monkeypatch.setattr(generate_open_source_factors, "build_raw_panel", extract)
    monkeypatch.setattr(sys, "argv", ["generate_open_source_factors.py", "--input-root", str(input_root),
                                     "--output-root", str(output_root), "--symbols", symbols[0],
                                     "--min-tgd-cross-section", "8", "--workers", "1"])
    assert generate_open_source_factors.main() == 0
    output = pd.read_parquet(output_root / f"{symbols[0]}.parquet")
    expected = build_open_source_factor_panel(raw, min_tgd_cross_section=8)
    expected = expected.loc[expected["ts_code"].eq(symbols[0]), "kaiyuan_tgd_m20"].reset_index(drop=True)
    assert output["kaiyuan_tgd_m20"].notna().sum() == 6
    pd.testing.assert_series_equal(output["kaiyuan_tgd_m20"], expected)


@pytest.mark.parametrize("damage", ["truncated", "wrong_grid", "missing_price"])
def test_incomplete_day_cannot_publish_factors(damage: str) -> None:
    source = _minute_frame(days=25)
    last = source["trade_time"].dt.normalize().max()
    last_day = source["trade_time"].dt.normalize().eq(last)
    if damage == "truncated":
        source = source.loc[~(last_day & source["trade_time"].dt.hour.ge(11))]
    elif damage == "wrong_grid":
        row = source.index[last_day][-2]
        source.loc[row, "trade_time"] -= pd.Timedelta(seconds=30)
    else:
        source.loc[source.index[last_day][-2], "close"] = np.nan
    raw = build_daily_raw_features(source, "510300.SH")
    assert raw.iloc[-1][["daily_close", "daily_amplitude", "daily_amount"]].isna().all()
    assert not raw.iloc[-1]["amount_available"]
    assert pd.isna(raw.iloc[-1]["smart_money_vwap_gap_m10"])
    assert pd.isna(raw.iloc[-1]["extreme_return_m20"])
    assert pd.isna(raw.iloc[-1]["extreme_prior_return_m20"])
    panel = build_open_source_factor_panel(raw)
    assert panel.iloc[-1][list(OPEN_SOURCE_FACTOR_COLUMNS)].isna().all()


@pytest.mark.parametrize("bad_amount", [np.nan, np.inf, -1.0])
def test_partial_daily_amount_is_unknown(bad_amount: float) -> None:
    source = _minute_frame(days=20)
    source.loc[source.index[-2], "amount"] = bad_amount
    raw = build_daily_raw_features(source, "510300.SH")
    last = raw.iloc[-1]
    assert not last["amount_available"]
    assert pd.isna(last["daily_amount"])
    assert pd.isna(last["ideal_turnover_raw"])
    assert np.isfinite(last["_ideal_amplitude"])


def test_real_zero_amount_and_optional_opening_bar_remain_valid() -> None:
    source = _minute_frame(days=1)
    source["amount"] = 0.0
    for frame in (source, source.iloc[1:]):
        raw = build_daily_raw_features(frame, "510300.SH")
        assert raw.iloc[0]["amount_available"]
        assert raw.iloc[0]["daily_amount"] == 0.0
        assert np.isfinite(raw.iloc[0]["daily_close"])


@pytest.mark.parametrize("event_position", [10, 20])
def test_missing_event_amount_does_not_become_zero(event_position: int) -> None:
    history = [_state_session(100.0 + i, 0.01) for i in range(20)]
    current = _state_session(100.0, 0.01)
    current.iloc[[10, 20, 21], current.columns.get_loc("volume")] = 1000.0
    assert _state_daily_metrics(current, history)["volume_peak_ridge_amount_ratio_raw"] == pytest.approx(0.5)
    current.iloc[event_position, current.columns.get_loc("amount")] = np.nan
    assert np.isnan(_state_daily_metrics(current, history)["volume_peak_ridge_amount_ratio_raw"])
    current.iloc[event_position, current.columns.get_loc("amount")] = 0.0
    assert np.isfinite(_state_daily_metrics(current, history)["volume_peak_ridge_amount_ratio_raw"])


def test_zero_follow_amount_is_a_real_observation() -> None:
    history = [_state_session(100.0 + i, 0.01) for i in range(20)]
    current = _state_session(100.0, 0.01)
    current.iloc[[10, 20], current.columns.get_loc("volume")] = 1000.0
    current.iloc[11, current.columns.get_loc("amount")] = 0.0
    current.iloc[21, current.columns.get_loc("amount")] = 200.0
    assert _state_daily_metrics(current, history)["volume_eruption_follow_ratio_raw"] == pytest.approx(1.0)
    current.iloc[11, current.columns.get_loc("amount")] = np.nan
    assert _state_daily_metrics(current, history)["volume_eruption_follow_ratio_raw"] == pytest.approx(2.0)


def _cli_inputs(tmp_path: Path, monkeypatch) -> Path:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    input_root.mkdir()
    for symbol in ("510300.SH", "510500.SH"):
        _minute_frame(days=2, symbol=symbol).to_parquet(input_root / f"{symbol}.parquet")
    monkeypatch.setattr(sys, "argv", ["generate_open_source_factors.py", "--input-root", str(input_root),
                                     "--output-root", str(output_root), "--workers", "1"])
    return output_root


def test_output_outside_calculation_pool_fails_before_extraction(tmp_path: Path, monkeypatch) -> None:
    output_root = _cli_inputs(tmp_path, monkeypatch)
    pool = tmp_path / "pool.txt"
    pool.write_text("510300.SH\n", encoding="utf-8")
    sys.argv.extend(["--calculation-symbols-file", str(pool), "--symbols", "510500.SH"])
    monkeypatch.setattr(generate_open_source_factors, "build_raw_panel", lambda *args: pytest.fail("extraction started"))
    with pytest.raises(ValueError, match="outside calculation universe.*510500.SH"):
        generate_open_source_factors.main()
    assert not output_root.exists()


def test_empty_output_date_range_is_not_success(tmp_path: Path, monkeypatch) -> None:
    output_root = _cli_inputs(tmp_path, monkeypatch)
    sys.argv.extend(["--date-from", "20260201"])
    with pytest.raises(ValueError, match="No output rows"):
        generate_open_source_factors.main()
    assert not output_root.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_staging_failure_preserves_batch_and_allows_retry(tmp_path: Path, monkeypatch, existing: bool) -> None:
    output_root = _cli_inputs(tmp_path, monkeypatch)
    if existing:
        assert generate_open_source_factors.main() == 0
        sys.argv.append("--overwrite")
    before = {p.name: p.read_bytes() for p in output_root.glob("*")}
    original = pd.DataFrame.to_parquet

    def fail_second(frame, path, *args, **kwargs):
        if Path(path).name == "510500.SH.parquet":
            raise OSError("simulated write failure")
        return original(frame, path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_parquet", fail_second)
        with pytest.raises(OSError, match="simulated write failure"):
            generate_open_source_factors.main()
    assert {p.name: p.read_bytes() for p in output_root.glob("*")} == before
    assert generate_open_source_factors.main() == 0
    assert len(list(output_root.glob("*.parquet"))) == 2
    assert (output_root / "manifest.json").exists()


def test_publish_failure_restores_existing_directory(tmp_path: Path, monkeypatch) -> None:
    output_root = _cli_inputs(tmp_path, monkeypatch)
    assert generate_open_source_factors.main() == 0
    before = {p.name: p.read_bytes() for p in output_root.glob("*")}
    sys.argv.append("--overwrite")
    original = Path.rename

    def fail_publish(path, target):
        if path.name == "staging":
            raise OSError("simulated publish failure")
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_publish)
        with pytest.raises(OSError, match="simulated publish failure"):
            generate_open_source_factors.main()
    assert {p.name: p.read_bytes() for p in output_root.glob("*")} == before
    assert generate_open_source_factors.main() == 0


def test_next_run_recovers_interrupted_directory_swap(tmp_path: Path, monkeypatch) -> None:
    import json
    output_root = _cli_inputs(tmp_path, monkeypatch)
    assert generate_open_source_factors.main() == 0
    before = {p.name: p.read_bytes() for p in output_root.glob("*")}
    transaction = generate_open_source_factors._transaction_path(output_root)
    transaction.mkdir()
    (transaction / "owner.json").write_text(json.dumps(str(output_root.resolve())), encoding="utf-8")
    output_root.rename(transaction / "backup")
    (transaction / "staging").mkdir()
    (transaction / "staging" / "510300.SH.parquet").write_bytes(b"partial batch")
    assert generate_open_source_factors.main() == 0
    assert {p.name: p.read_bytes() for p in output_root.glob("*")} == before
    assert not transaction.exists()
