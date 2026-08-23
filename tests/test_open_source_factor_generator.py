from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor.open_source import (  # noqa: E402
    OPEN_SOURCE_FACTOR_COLUMNS,
    SESSION_BAR_COUNT,
    _regular_session_frame,
    _smart_money_gap_for_sessions,
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


def test_availability_uses_next_observed_date() -> None:
    raw = _raw_panel(symbols=1, days=4)
    panel = build_open_source_factor_panel(raw, factor_window=3, min_tgd_cross_section=1)
    available = add_availability_columns(panel)

    assert available["available_date"].iloc[:-1].equals(
        available["trade_date"].iloc[1:].reset_index(drop=True)
    )
    assert available["available_time"].iloc[0].hour == 9
    assert pd.isna(available["available_date"].iloc[-1])


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
