from __future__ import annotations

import pandas as pd

from scripts.generate_etf_orderbook_factors import build_tasks as build_etf_tasks
from scripts.generate_etf_orderbook_factors import discover_minute_symbols
from scripts.generate_stock_orderbook_factors import (
    BASE_OUTPUT_COLUMNS,
    FACTOR_COLUMNS,
    OUTPUT_COLUMNS,
    build_output_frame,
    build_minute_file_index,
    discover_symbol_dirs,
    discover_trade_date_dirs,
    factor_columns_for_profile,
    load_minute_frame,
    merge_symbol_output,
    numeric_code,
    output_columns_for_profile,
)


def _minute_frame(trade_date: str) -> pd.DataFrame:
    times = [f"{trade_date} 09:30:00", f"{trade_date} 09:31:00"]
    return pd.DataFrame(
        {
            "trade_date": [trade_date] * 2,
            "trade_time": times,
            "ts_code": ["000001.SZ"] * 2,
            "open": [10.0, 10.1],
            "high": [10.1, 10.2],
            "low": [9.9, 10.0],
            "close": [10.05, 10.15],
            "vol": [1000.0, 1200.0],
            "amount": [10000.0, 12180.0],
            "adj_factor": [1.0, 1.0],
        }
    )[BASE_OUTPUT_COLUMNS]


def test_output_matches_advanced_parquet_layout() -> None:
    minute = _minute_frame("2026-01-05")
    factor_index = pd.DatetimeIndex(
        ["2026-01-05 09:29:59", "2026-01-05 09:30:59"],
        name="trade_time",
    )
    factors = pd.DataFrame(0.0, index=factor_index, columns=FACTOR_COLUMNS)
    factors["mid_price"] = [10.0, 10.1]

    result = build_output_frame(minute, factors)

    assert isinstance(result.index, pd.RangeIndex)
    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result["trade_date"].dtype == minute["trade_date"].dtype
    assert result["trade_time"].dtype == minute["trade_time"].dtype
    assert result["mid_price"].tolist() == [10.0, 10.1]


def test_multiwindow_output_uses_expanded_schema() -> None:
    minute = _minute_frame("2026-01-05")
    factor_columns = factor_columns_for_profile("multi")
    factors = pd.DataFrame(
        0.0,
        index=pd.DatetimeIndex(
            ["2026-01-05 09:29:59", "2026-01-05 09:30:59"],
            name="trade_time",
        ),
        columns=factor_columns,
    )

    result = build_output_frame(minute, factors, factor_columns)

    assert result.columns.tolist() == output_columns_for_profile("multi")
    assert "market_impact_300s" in result.columns


def test_incremental_merge_replaces_only_requested_trade_date(tmp_path) -> None:
    output_path = tmp_path / "000001.SZ.parquet"
    first = build_output_frame(
        _minute_frame("2026-01-05"),
        pd.DataFrame(
            1.0,
            index=pd.DatetimeIndex(
                ["2026-01-05 09:30:00", "2026-01-05 09:31:00"],
                name="trade_time",
            ),
            columns=FACTOR_COLUMNS,
        ),
    )
    first.to_parquet(output_path)
    second = build_output_frame(
        _minute_frame("2026-01-06"),
        pd.DataFrame(
            2.0,
            index=pd.DatetimeIndex(
                ["2026-01-06 09:30:00", "2026-01-06 09:31:00"],
                name="trade_time",
            ),
            columns=FACTOR_COLUMNS,
        ),
    )

    combined = merge_symbol_output(output_path, second, overwrite=True)

    assert combined["trade_date"].tolist() == [
        "2026-01-05",
        "2026-01-05",
        "2026-01-06",
        "2026-01-06",
    ]
    assert isinstance(combined.index, pd.RangeIndex)


def test_nested_dates_and_symbols_are_discovered_by_numeric_code(tmp_path) -> None:
    date_dir = tmp_path / "2025" / "202501" / "20250102"
    wrong_suffix = date_dir / "501001.SZ"
    wrong_suffix.mkdir(parents=True)
    (date_dir / "000001.SZ").mkdir()

    assert discover_trade_date_dirs(tmp_path) == [date_dir]
    assert discover_symbol_dirs(date_dir, ["501001.SH"]) == [wrong_suffix]
    assert numeric_code("501001.SZ") == "501001"


def test_minute_index_prefers_canonical_file_and_ignores_suffix_for_matching(tmp_path) -> None:
    canonical = tmp_path / "603686.SH.parquet"
    duplicate = tmp_path / "603686.SH(1).parquet"
    wrong_exchange_target = tmp_path / "501001.SH.parquet"
    for path in (duplicate, canonical, wrong_exchange_target):
        path.touch()

    index = build_minute_file_index(tmp_path)

    assert index["603686"] == canonical
    assert index["501001"] == wrong_exchange_target


def test_etf_task_builder_filters_non_etf_tick_dirs_by_minute_root(tmp_path) -> None:
    minute_root = tmp_path / "minute"
    minute_root.mkdir()
    (minute_root / "159001.SZ.parquet").touch()
    (minute_root / "513100.SH.parquet").touch()

    date_dir = tmp_path / "ticks" / "2025" / "202511" / "20251103"
    non_etf_dir = date_dir / "000001.SZ"
    etf_dir = date_dir / "159001.SZ"
    non_etf_dir.mkdir(parents=True)
    etf_dir.mkdir()

    symbols = discover_minute_symbols(minute_root)
    tasks, missing_count = build_etf_tasks(
        [date_dir],
        symbols,
        strict_suffix=False,
        log_missing=False,
    )

    assert tasks == [(etf_dir, "159001.SZ")]
    assert missing_count == 1


def test_etf_task_builder_keeps_universe_suffix_when_tick_dir_suffix_is_wrong(tmp_path) -> None:
    date_dir = tmp_path / "ticks" / "2025" / "202511" / "20251103"
    wrong_suffix_dir = date_dir / "511360.SZ"
    wrong_suffix_dir.mkdir(parents=True)

    tasks, missing_count = build_etf_tasks(
        [date_dir],
        ["511360.SH"],
        strict_suffix=False,
        log_missing=False,
    )

    assert tasks == [(wrong_suffix_dir, "511360.SH")]
    assert missing_count == 0


def test_minute_frame_is_cached_per_symbol(tmp_path, monkeypatch) -> None:
    minute_root = tmp_path / "minute"
    minute_root.mkdir()
    minute_path = minute_root / "159001.SZ.parquet"
    _minute_frame("2026-01-05").to_parquet(minute_path)

    calls = {"count": 0}
    real_read_parquet = pd.read_parquet

    def spy(*args, **kwargs):
        calls["count"] += 1
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)

    first = load_minute_frame(minute_root, "159001.SZ", "2026-01-05")
    second = load_minute_frame(minute_root, "159001.SZ", "2026-01-05")

    assert calls["count"] == 1
    assert first.equals(second)
