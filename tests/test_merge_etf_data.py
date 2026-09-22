from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.merge_etf_data import (
    MINUTE_OUTPUT_COLUMNS,
    TABLE_SPECS,
    load_minute_source,
    load_table_source,
    merge_minute_dataset,
    merge_table_dataset,
    write_frame_atomic,
)


def _minute_source(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame[
        [
            "trade_date",
            "trade_time",
            "ts_code",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
            "adj_factor",
        ]
    ]


def _minute_target(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = _minute_source(rows)
    return (
        frame.set_index(["trade_date", "trade_time"])
        .loc[:, list(MINUTE_OUTPUT_COLUMNS)]
        .sort_index()
    )


def _minute_row(
    ts_code: str,
    trade_time: str,
    close: float,
) -> dict[str, object]:
    timestamp = pd.Timestamp(trade_time)
    return {
        "trade_date": timestamp.normalize(),
        "trade_time": timestamp,
        "ts_code": ts_code,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "vol": 100.0,
        "amount": close * 100.0,
        "adj_factor": 1.0,
    }


def test_minute_merge_source_wins_creates_new_file_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "etf_1min_data.parquet"
    target_dir = tmp_path / "etf_1min"
    target_dir.mkdir()
    _minute_source(
        [
            _minute_row("510300.SH", "2026-08-07 09:30:00", 4.2),
            _minute_row("510300.SH", "2026-08-07 09:31:00", 4.3),
            _minute_row("159001.SZ", "2026-08-07 09:30:00", 1.2),
        ]
    ).to_parquet(source_path)
    _minute_target(
        [
            _minute_row("510300.SH", "2026-08-06 15:00:00", 4.0),
            _minute_row("510300.SH", "2026-08-07 09:30:00", 4.1),
        ]
    ).to_parquet(target_dir / "510300.SH.parquet")

    first = merge_minute_dataset(source_path, target_dir, dry_run=False)

    assert first["new_files"] == 1
    assert first["updated_files"] == 1
    assert first["overlap_keys"] == 1
    assert first["conflicting_keys"] == 1
    merged = pd.read_parquet(target_dir / "510300.SH.parquet")
    assert merged.index.names == ["trade_date", "trade_time"]
    assert not merged.index.has_duplicates
    assert merged.loc[("2026-08-07", "2026-08-07 09:30:00"), "close"] == 4.2
    assert merged.loc[("2026-08-06", "2026-08-06 15:00:00"), "close"] == 4.0
    assert (target_dir / "159001.SZ.parquet").is_file()

    second = merge_minute_dataset(source_path, target_dir, dry_run=False)

    assert second["new_files"] == 0
    assert second["updated_files"] == 0
    assert second["unchanged_files"] == 2
    assert len(pd.read_parquet(target_dir / "510300.SH.parquet")) == 3


def test_minute_dry_run_does_not_write_target_data(tmp_path: Path) -> None:
    source_path = tmp_path / "etf_5min_data.parquet"
    target_dir = tmp_path / "etf_5min"
    target_dir.mkdir()
    _minute_source(
        [
            _minute_row("510300.SH", "2026-08-07 09:30:00", 4.2),
            _minute_row("159001.SZ", "2026-08-07 09:30:00", 1.2),
        ]
    ).to_parquet(source_path)
    existing_path = target_dir / "510300.SH.parquet"
    _minute_target(
        [_minute_row("510300.SH", "2026-08-07 09:30:00", 4.1)]
    ).to_parquet(existing_path)
    original_bytes = existing_path.read_bytes()

    result = merge_minute_dataset(source_path, target_dir, dry_run=True)

    assert result["new_files"] == 1
    assert result["updated_files"] == 1
    assert result["conflicting_keys"] == 1
    assert existing_path.read_bytes() == original_bytes
    assert not (target_dir / "159001.SZ.parquet").exists()


def _sh_cons_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    columns = list(TABLE_SPECS[1].value_columns)
    return frame.set_index(["trade_date", "ts_code"])[columns]


def _sh_cons_row(
    trade_date: str,
    con_code: str,
    con_name: str,
) -> dict[str, object]:
    return {
        "trade_date": pd.Timestamp(trade_date),
        "ts_code": "510300.SH",
        "con_code": con_code,
        "con_name": con_name,
        "qty": 100,
        "sub_flag": "允许",
        "cpr": "0",
        "rdr": "0",
        "sca": None,
        "exchange": "SH",
    }


def test_constituent_merge_uses_composite_key_and_keeps_target_only_rows(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source_sh.parquet"
    target_path = tmp_path / "target_sh.parquet"
    _sh_cons_frame(
        [
            _sh_cons_row("2026-08-07", "600000.SH", "新名称"),
            _sh_cons_row("2026-08-07", "600001.SH", "新增成分"),
        ]
    ).to_parquet(source_path)
    _sh_cons_frame(
        [
            _sh_cons_row("2026-08-06", "600000.SH", "历史记录"),
            _sh_cons_row("2026-08-07", "600000.SH", "旧名称"),
            _sh_cons_row("2026-08-07", "600002.SH", "目标独有"),
        ]
    ).to_parquet(target_path)

    first = merge_table_dataset(
        source_path,
        target_path,
        TABLE_SPECS[1],
        dry_run=False,
    )

    assert first["overlap_keys"] == 1
    assert first["conflicting_keys"] == 1
    output = pd.read_parquet(target_path).reset_index()
    assert len(output) == 4
    current = output.loc[output["trade_date"].eq(pd.Timestamp("2026-08-07"))]
    names = current.set_index("con_code")["con_name"].to_dict()
    assert names == {
        "600000.SH": "新名称",
        "600001.SH": "新增成分",
        "600002.SH": "目标独有",
    }
    assert not output.duplicated(
        ["trade_date", "ts_code", "con_code"]
    ).any()

    second = merge_table_dataset(
        source_path,
        target_path,
        TABLE_SPECS[1],
        dry_run=False,
    )
    assert second["updated_files"] == 0
    assert second["unchanged_files"] == 1
    assert len(pd.read_parquet(target_path)) == 4


def test_atomic_frame_write_keeps_original_when_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "510300.SH.parquet"
    output_path.write_bytes(b"original")
    frame = _minute_target(
        [_minute_row("510300.SH", "2026-08-07 09:30:00", 4.2)]
    )

    def fail_validation(*_: object, **__: object) -> None:
        raise ValueError("forced validation failure")

    monkeypatch.setattr(
        "scripts.merge_etf_data._validate_parquet_metadata",
        fail_validation,
    )

    with pytest.raises(ValueError, match="forced validation failure"):
        write_frame_atomic(frame, output_path)

    assert output_path.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp"))


def test_minute_source_rejects_duplicate_business_keys(tmp_path: Path) -> None:
    source_path = tmp_path / "duplicate.parquet"
    row = _minute_row("510300.SH", "2026-08-07 09:30:00", 4.2)
    _minute_source([row, row]).to_parquet(source_path)

    with pytest.raises(ValueError, match="duplicate minute keys"):
        load_minute_source(source_path)


def test_daily_table_accepts_fund_code_suffix(tmp_path: Path) -> None:
    source_path = tmp_path / "etf_daily.parquet"
    spec = TABLE_SPECS[0]
    values = {
        column: None
        for column in spec.value_columns
    }
    values.update(
        {
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "pre_close": 1.0,
            "change": 0.05,
            "pct_chg": 5.0,
            "vol": 100.0,
            "amount": 100.0,
            "adj_factor": 1.0,
            "etf_name": "测试基金",
            "total_share": 10.0,
            "total_size": 10.5,
            "nav": 1.05,
            "exchange": "OF",
        }
    )
    frame = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2026-08-28"),
                "ts_code": "158008.OF",
                **values,
            }
        ]
    ).set_index(["trade_date", "ts_code"])[list(spec.value_columns)]
    frame.to_parquet(source_path)

    loaded, _ = load_table_source(source_path, spec)

    assert loaded["ts_code"].tolist() == ["158008.OF"]
