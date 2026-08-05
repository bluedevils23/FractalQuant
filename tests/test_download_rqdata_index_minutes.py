from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.download_rqdata_index_minutes import (
    INDEX_MAPPINGS,
    OUTPUT_COLUMNS,
    download_start_date,
    is_quota_exceeded,
    merge_minute_frames,
    month_windows,
    normalize_rqdata_frame,
    prioritize_missing_mappings,
    validate_minute_frame,
    write_parquet_atomic,
)


def _raw_frame(day: str = "2026-07-31", rows: int = 240) -> pd.DataFrame:
    times = pd.date_range(f"{day} 09:31:00", periods=rows, freq="min")
    index = pd.MultiIndex.from_arrays(
        [["930914.INDX"] * rows, times], names=["order_book_id", "datetime"]
    )
    return pd.DataFrame(
        {
            "open": np.full(rows, 10.0),
            "high": np.full(rows, 11.0),
            "low": np.full(rows, 9.0),
            "close": np.full(rows, 10.5),
            "volume": np.arange(rows, dtype=float),
            "total_turnover": np.arange(rows, dtype=float) * 100.0,
        },
        index=index,
    )


def test_fixed_index_mappings_match_requested_codes() -> None:
    assert len(INDEX_MAPPINGS) == 16
    assert {(item.output_code, item.rqdata_code) for item in INDEX_MAPPINGS} == {
        ("930914.CSI", "930914.INDX"),
        ("931239.CSI", "931239.INDX"),
        ("932069.CSI", "932069.INDX"),
        ("930931.CSI", "930931.INDX"),
        ("930965.CSI", "930965.INDX"),
        ("931233.CSI", "931233.INDX"),
        ("931454.CSI", "931454.INDX"),
        ("931250.CSI", "931250.INDX"),
        ("931722.CSI", "931722.INDX"),
        ("H11146.CSI", "H11146.XSHG"),
        ("930709.CSI", "930709.INDX"),
        ("930957.CSI", "930957.INDX"),
        ("931028.CSI", "931028.INDX"),
        ("H50069.CSI", "H50069.XSHG"),
        ("931637.CSI", "931637.INDX"),
        ("931573.CSI", "931573.INDX"),
    }


def test_normalize_matches_existing_index_store_schema() -> None:
    result = normalize_rqdata_frame(_raw_frame(), "930914.CSI")

    assert result.index.names == ["trade_date", "trade_time"]
    assert result.columns.tolist() == list(OUTPUT_COLUMNS)
    assert result["ts_code"].eq("930914.CSI").all()
    assert result[["open", "high", "low", "close", "vol", "amount"]].dtypes.eq(
        "float64"
    ).all()
    validate_minute_frame(result, "930914.CSI")


def test_merge_prefers_fresh_duplicate_minutes() -> None:
    existing = normalize_rqdata_frame(_raw_frame(), "930914.CSI")
    fresh = existing.iloc[-1:].copy()
    fresh.loc[:, "close"] = 99.0

    merged = merge_minute_frames(existing, fresh)

    assert len(merged) == 240
    assert merged.iloc[-1]["close"] == 99.0
    validate_minute_frame(merged, "930914.CSI")


def test_download_start_refreshes_recent_trading_dates() -> None:
    trading_dates = [
        date(2026, 3, 19),
        date(2026, 3, 20),
        date(2026, 3, 23),
        date(2026, 3, 24),
        date(2026, 3, 25),
        date(2026, 3, 26),
    ]
    existing = normalize_rqdata_frame(_raw_frame("2026-03-19"), "930914.CSI")
    existing = merge_minute_frames(
        existing,
        normalize_rqdata_frame(_raw_frame("2026-03-20"), "930914.CSI"),
    )
    existing = merge_minute_frames(
        existing,
        normalize_rqdata_frame(_raw_frame("2026-03-24"), "930914.CSI"),
    )

    assert download_start_date(existing, trading_dates, 2) == date(2026, 3, 25)
    assert download_start_date(None, trading_dates, 2) == date(2026, 3, 19)


def test_month_windows_and_validation_failure() -> None:
    assert month_windows(date(2026, 3, 19), date(2026, 5, 2)) == [
        (date(2026, 3, 19), date(2026, 3, 31)),
        (date(2026, 4, 1), date(2026, 4, 30)),
        (date(2026, 5, 1), date(2026, 5, 2)),
    ]
    with pytest.raises(ValueError, match="Expected 240 rows"):
        validate_minute_frame(
            normalize_rqdata_frame(_raw_frame(rows=239), "930914.CSI"),
            "930914.CSI",
        )


def test_quota_error_is_recognized() -> None:
    QuotaExceeded = type("QuotaExceeded", (Exception,), {})

    assert is_quota_exceeded(QuotaExceeded())
    assert not is_quota_exceeded(ValueError("other error"))


def test_missing_outputs_are_prioritized(tmp_path) -> None:
    existing = INDEX_MAPPINGS[0]
    missing = INDEX_MAPPINGS[1]
    (tmp_path / f"{existing.output_code}.parquet").touch()

    ordered = prioritize_missing_mappings((existing, missing), tmp_path)

    assert ordered == (missing, existing)


def test_normalize_drops_all_zero_non_trading_placeholder_day() -> None:
    raw = pd.concat([_raw_frame("2026-04-02"), _raw_frame("2026-04-03")])
    raw.loc[("930914.INDX", slice("2026-04-03", "2026-04-03")), :] = 0.0

    result = normalize_rqdata_frame(raw, "930914.CSI")

    assert result.index.get_level_values("trade_date").unique().tolist() == [
        pd.Timestamp("2026-04-02")
    ]
    validate_minute_frame(result, "930914.CSI")


def test_parquet_output_preserves_index_store_schema(tmp_path) -> None:
    expected = normalize_rqdata_frame(_raw_frame(), "930914.CSI")
    output_path = tmp_path / "930914.CSI.parquet"

    write_parquet_atomic(expected, output_path)
    actual = pd.read_parquet(output_path)

    assert actual.index.names == ["trade_date", "trade_time"]
    assert actual.columns.tolist() == list(OUTPUT_COLUMNS)
    assert actual.dtypes.to_dict() == expected.dtypes.to_dict()
    pd.testing.assert_frame_equal(actual, expected)
