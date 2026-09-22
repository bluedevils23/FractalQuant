"""Merge a downloaded ETF parquet batch into the local ETF data store."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE_ROOT = Path(r"D:\BaiduNetdiskDownload\8.7-8.15\ETF数据")
DEFAULT_TARGET_ROOT = Path(r"D:\workspace\stockdata\etf-data")
MINUTE_DATASETS = (
    ("etf_1min_data.parquet", "etf_1min"),
    ("etf_5min_data.parquet", "etf_5min"),
    ("etf_15min_data.parquet", "etf_15min"),
    ("etf_30min_data.parquet", "etf_30min"),
    ("etf_60min_data.parquet", "etf_60min"),
)
MINUTE_SOURCE_COLUMNS = (
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
)
MINUTE_OUTPUT_COLUMNS = (
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "adj_factor",
)
CODE_PATTERN = re.compile(r"^\d{6}\.[A-Z]{2,5}$")
MINUTE_CODE_PATTERN = re.compile(r"^\d{6}\.(?:SH|SZ)$")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableSpec:
    filename: str
    value_columns: tuple[str, ...]
    key_columns: tuple[str, ...]


TABLE_SPECS = (
    TableSpec(
        filename="etf_daily.parquet",
        value_columns=(
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
            "adj_factor",
            "etf_name",
            "total_share",
            "total_size",
            "nav",
            "exchange",
        ),
        key_columns=("trade_date", "ts_code"),
    ),
    TableSpec(
        filename="etf_sh_cons.parquet",
        value_columns=(
            "con_code",
            "con_name",
            "qty",
            "sub_flag",
            "cpr",
            "rdr",
            "sca",
            "exchange",
        ),
        key_columns=("trade_date", "ts_code", "con_code"),
    ),
    TableSpec(
        filename="etf_sz_cons.parquet",
        value_columns=(
            "con_code",
            "con_name",
            "qty",
            "sub_flag",
            "cpr",
            "rdr",
            "sub_cc",
            "red_cc",
            "exchange",
        ),
        key_columns=("trade_date", "ts_code", "con_code"),
    ),
)


def _temporary_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")


def _json_default(value: object) -> str | int | float | bool | None:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _validate_code_values(
    values: pd.Series,
    label: str,
    pattern: re.Pattern[str] = CODE_PATTERN,
) -> None:
    if values.isna().any():
        raise ValueError(f"{label} contains null ts_code values")
    invalid = sorted(
        {
            str(value)
            for value in values
            if pattern.fullmatch(str(value)) is None
        }
    )
    if invalid:
        raise ValueError(f"{label} contains invalid ts_code values: {invalid[:5]}")


def _validate_arrow_schema(
    actual: pa.Schema,
    expected: pa.Schema,
    label: str,
) -> None:
    if not actual.remove_metadata().equals(expected.remove_metadata()):
        raise ValueError(
            f"{label} has incompatible parquet schema: "
            f"{actual.remove_metadata()} != {expected.remove_metadata()}"
        )


def load_minute_source(source_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(source_path)
    if frame.columns.tolist() != list(MINUTE_SOURCE_COLUMNS):
        raise ValueError(
            f"{source_path.name} columns do not match the minute source contract"
        )

    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    if frame[["trade_date", "trade_time"]].isna().any().any():
        raise ValueError(f"{source_path.name} contains invalid timestamps")
    if not frame["trade_date"].eq(frame["trade_time"].dt.normalize()).all():
        raise ValueError(f"{source_path.name} has inconsistent trade_date values")
    _validate_code_values(frame["ts_code"], source_path.name, MINUTE_CODE_PATTERN)

    key_columns = ["ts_code", "trade_date", "trade_time"]
    if frame.duplicated(key_columns).any():
        raise ValueError(f"{source_path.name} contains duplicate minute keys")

    numeric_columns = [
        column for column in MINUTE_OUTPUT_COLUMNS if column != "ts_code"
    ]
    if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in numeric_columns):
        raise ValueError(f"{source_path.name} contains non-numeric market columns")

    return (
        frame.set_index(["trade_date", "trade_time"])
        .loc[:, list(MINUTE_OUTPUT_COLUMNS)]
        .sort_index()
    )


def validate_minute_output(frame: pd.DataFrame, ts_code: str) -> None:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != [
        "trade_date",
        "trade_time",
    ]:
        raise ValueError(f"Unexpected minute index schema for {ts_code}")
    if frame.columns.tolist() != list(MINUTE_OUTPUT_COLUMNS):
        raise ValueError(f"Unexpected minute columns for {ts_code}")
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate minute timestamps for {ts_code}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"Unsorted minute timestamps for {ts_code}")
    if frame["ts_code"].isna().any() or not frame["ts_code"].eq(ts_code).all():
        raise ValueError(f"Unexpected ts_code values for {ts_code}")


def _count_frame_conflicts(
    existing: pd.DataFrame,
    fresh: pd.DataFrame,
    overlap: pd.Index,
) -> int:
    if len(overlap) == 0:
        return 0
    left = existing.loc[overlap, fresh.columns]
    right = fresh.loc[overlap, fresh.columns]
    equal = left.eq(right) | (left.isna() & right.isna())
    return int((~equal.all(axis=1)).sum())


def merge_minute_frames(
    existing: pd.DataFrame | None,
    fresh: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    if existing is None or existing.empty:
        return fresh.sort_index(), 0, 0
    overlap = existing.index.intersection(fresh.index, sort=False)
    conflicts = _count_frame_conflicts(existing, fresh, overlap)
    merged = pd.concat([existing, fresh])
    merged = merged.loc[~merged.index.duplicated(keep="last")].sort_index()
    return merged, len(overlap), conflicts


def _validate_parquet_metadata(
    path: Path,
    expected_rows: int,
    expected_schema: pa.Schema | None = None,
) -> None:
    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Unexpected row count in {path}: "
            f"{parquet_file.metadata.num_rows} != {expected_rows}"
        )
    if expected_schema is not None:
        _validate_arrow_schema(parquet_file.schema_arrow, expected_schema, str(path))


def write_frame_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(output_path)
    try:
        frame.to_parquet(temporary_path)
        _validate_parquet_metadata(temporary_path, len(frame))
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_minute_target_tail(
    output_path: Path,
    fresh: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    parquet_file = pq.ParquetFile(output_path)
    date_values = fresh.index.get_level_values("trade_date")
    tail = pd.read_parquet(
        output_path,
        filters=[
            ("trade_date", ">=", date_values.min()),
            ("trade_date", "<=", date_values.max()),
        ],
    )
    if not tail.empty:
        validate_minute_output(tail.sort_index(), str(fresh["ts_code"].iloc[0]))
    return tail.sort_index(), parquet_file.metadata.num_rows


def merge_minute_dataset(
    source_path: Path,
    target_dir: Path,
    dry_run: bool,
    progress_every: int = 250,
) -> dict[str, Any]:
    source = load_minute_source(source_path)
    source_dates = source.index.get_level_values("trade_date")
    result: dict[str, Any] = {
        "dataset": source_path.name,
        "target": str(target_dir),
        "source_rows": len(source),
        "source_codes": int(source["ts_code"].nunique()),
        "source_date_from": source_dates.min(),
        "source_date_to": source_dates.max(),
        "new_files": 0,
        "updated_files": 0,
        "unchanged_files": 0,
        "overlap_keys": 0,
        "conflicting_keys": 0,
        "rows_before": 0,
        "rows_after": 0,
        "failures": [],
    }

    grouped = source.groupby("ts_code", sort=True)
    total_codes = grouped.ngroups
    for position, (ts_code, fresh) in enumerate(grouped, start=1):
        output_path = target_dir / f"{ts_code}.parquet"
        try:
            validate_minute_output(fresh, ts_code)
            if not output_path.exists():
                result["new_files"] += 1
                result["rows_after"] += len(fresh)
                if not dry_run:
                    write_frame_atomic(fresh, output_path)
                continue

            if dry_run:
                existing, rows_before = _load_minute_target_tail(
                    output_path, fresh
                )
                overlap = existing.index.intersection(fresh.index, sort=False)
                conflicts = _count_frame_conflicts(existing, fresh, overlap)
                rows_after = rows_before + len(fresh) - len(overlap)
                unchanged = len(overlap) == len(fresh) and conflicts == 0
            else:
                existing = pd.read_parquet(output_path).sort_index()
                validate_minute_output(existing, ts_code)
                rows_before = len(existing)
                merged, overlap_count, conflicts = merge_minute_frames(
                    existing, fresh
                )
                validate_minute_output(merged, ts_code)
                overlap = pd.RangeIndex(overlap_count)
                rows_after = len(merged)
                unchanged = merged.equals(existing)
                if not unchanged:
                    write_frame_atomic(merged, output_path)

            result["rows_before"] += rows_before
            result["rows_after"] += rows_after
            result["overlap_keys"] += len(overlap)
            result["conflicting_keys"] += conflicts
            if unchanged:
                result["unchanged_files"] += 1
            else:
                result["updated_files"] += 1
        except Exception as exc:  # noqa: BLE001
            result["failures"].append(
                {
                    "ts_code": ts_code,
                    "output_path": str(output_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            LOGGER.exception("Failed to merge %s into %s", ts_code, target_dir)
        finally:
            if position % progress_every == 0 or position == total_codes:
                LOGGER.info(
                    "%s: processed %s/%s symbols",
                    source_path.name,
                    position,
                    total_codes,
                )
    return result


def load_table_source(source_path: Path, spec: TableSpec) -> tuple[pd.DataFrame, pa.Schema]:
    frame = pd.read_parquet(source_path)
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != [
        "trade_date",
        "ts_code",
    ]:
        raise ValueError(f"{source_path.name} has an unexpected pandas index")
    if frame.columns.tolist() != list(spec.value_columns):
        raise ValueError(f"{source_path.name} has unexpected columns")

    flat = frame.reset_index()
    flat["trade_date"] = pd.to_datetime(flat["trade_date"], errors="coerce")
    if flat["trade_date"].isna().any():
        raise ValueError(f"{source_path.name} contains invalid trade_date values")
    _validate_code_values(flat["ts_code"], source_path.name)
    if flat.duplicated(list(spec.key_columns)).any():
        raise ValueError(
            f"{source_path.name} contains duplicate logical keys: "
            f"{spec.key_columns}"
        )

    source_schema = pq.ParquetFile(source_path).schema_arrow
    if set(source_schema.names) != set(flat.columns):
        raise ValueError(f"{source_path.name} parquet fields do not match pandas fields")
    flat = flat.loc[:, source_schema.names]
    flat = flat.sort_values(list(spec.key_columns), kind="stable").reset_index(drop=True)
    return flat, source_schema


def _scan_sorted_unique_target(
    target_path: Path,
    key_columns: tuple[str, ...],
) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(target_path)
    previous_key: tuple[object, ...] | None = None
    first_key: tuple[object, ...] | None = None
    rows = 0
    for batch in parquet_file.iter_batches(
        batch_size=262_144,
        columns=list(key_columns),
    ):
        keys = batch.to_pandas(ignore_metadata=True)
        if keys.isna().any().any():
            raise ValueError(f"{target_path} contains null logical keys")
        index = pd.MultiIndex.from_frame(keys.loc[:, list(key_columns)])
        if not index.is_monotonic_increasing:
            raise ValueError(f"{target_path} logical keys are not sorted")
        if index.duplicated().any():
            raise ValueError(f"{target_path} contains duplicate logical keys")
        if len(index):
            current_first = tuple(index[0])
            current_last = tuple(index[-1])
            if previous_key is not None:
                if previous_key == current_first:
                    raise ValueError(f"{target_path} contains duplicate logical keys")
                if previous_key > current_first:
                    raise ValueError(f"{target_path} logical keys are not sorted")
            if first_key is None:
                first_key = current_first
            previous_key = current_last
        rows += len(index)
    return {
        "rows": rows,
        "first_key": first_key,
        "last_key": previous_key,
    }


def _load_table_target_window(
    target_path: Path,
    schema: pa.Schema,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
) -> pd.DataFrame:
    window = pd.read_parquet(
        target_path,
        filters=[
            ("trade_date", ">=", date_from),
            ("trade_date", "<=", date_to),
        ],
    ).reset_index()
    return window.loc[:, schema.names]


def _count_flat_conflicts(
    existing: pd.DataFrame,
    fresh: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> tuple[int, int]:
    if existing.empty or fresh.empty:
        return 0, 0
    keys = list(key_columns)
    existing_indexed = existing.set_index(keys)
    fresh_indexed = fresh.set_index(keys)
    overlap = existing_indexed.index.intersection(fresh_indexed.index, sort=False)
    if len(overlap) == 0:
        return 0, 0
    value_columns = [column for column in fresh.columns if column not in keys]
    left = existing_indexed.loc[overlap, value_columns]
    right = fresh_indexed.loc[overlap, value_columns]
    equal = left.eq(right) | (left.isna() & right.isna())
    return len(overlap), int((~equal.all(axis=1)).sum())


def merge_flat_frames(
    existing: pd.DataFrame,
    fresh: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, int, int]:
    overlap, conflicts = _count_flat_conflicts(existing, fresh, key_columns)
    merged = pd.concat([existing, fresh], ignore_index=True)
    merged = merged.drop_duplicates(list(key_columns), keep="last")
    merged = merged.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    return merged, overlap, conflicts


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.columns.tolist() != right.columns.tolist() or len(left) != len(right):
        return False
    return left.reset_index(drop=True).equals(right.reset_index(drop=True))


def _filter_batch(batch: pa.RecordBatch, mask: np.ndarray) -> pa.RecordBatch:
    return batch.filter(pa.array(mask, type=pa.bool_()))


def write_table_window_atomic(
    target_path: Path,
    target_schema: pa.Schema,
    merged_window: pd.DataFrame,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
    expected_rows: int,
) -> None:
    temporary_path = _temporary_path(target_path)
    merged_table = pa.Table.from_pandas(
        merged_window,
        schema=target_schema,
        preserve_index=False,
        safe=True,
    )
    window_written = False
    try:
        parquet_file = pq.ParquetFile(target_path)
        try:
            with pq.ParquetWriter(
                temporary_path,
                target_schema,
                compression="snappy",
                use_dictionary=True,
            ) as writer:
                for batch in parquet_file.iter_batches(batch_size=262_144):
                    date_index = batch.schema.get_field_index("trade_date")
                    dates = pd.to_datetime(batch.column(date_index).to_pandas())
                    before = np.asarray(dates < date_from)
                    within = np.asarray((dates >= date_from) & (dates <= date_to))
                    after = np.asarray(dates > date_to)

                    if before.any():
                        writer.write_batch(_filter_batch(batch, before))
                    if after.any() and not window_written:
                        if len(merged_table):
                            writer.write_table(merged_table, row_group_size=262_144)
                        window_written = True
                    if after.any():
                        writer.write_batch(_filter_batch(batch, after))
                    if not (before | within | after).all():
                        raise ValueError(
                            f"Unable to classify all rows in {target_path}"
                        )

                if not window_written:
                    if len(merged_table):
                        writer.write_table(merged_table, row_group_size=262_144)
        finally:
            parquet_file.close()

        _validate_parquet_metadata(
            temporary_path,
            expected_rows,
            expected_schema=target_schema,
        )
        written_window = _load_table_target_window(
            temporary_path,
            target_schema,
            date_from,
            date_to,
        )
        if not _frames_equal(written_window, merged_window):
            raise ValueError(f"Written source window does not validate: {target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.replace(target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_new_table_atomic(
    source: pd.DataFrame,
    source_schema: pa.Schema,
    target_path: Path,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(target_path)
    table = pa.Table.from_pandas(
        source,
        schema=source_schema,
        preserve_index=False,
        safe=True,
    )
    try:
        pq.write_table(table, temporary_path, compression="snappy")
        _validate_parquet_metadata(
            temporary_path,
            len(source),
            expected_schema=source_schema,
        )
        temporary_path.replace(target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def merge_table_dataset(
    source_path: Path,
    target_path: Path,
    spec: TableSpec,
    dry_run: bool,
) -> dict[str, Any]:
    source, source_schema = load_table_source(source_path, spec)
    date_from = source["trade_date"].min()
    date_to = source["trade_date"].max()
    result: dict[str, Any] = {
        "dataset": source_path.name,
        "target": str(target_path),
        "source_rows": len(source),
        "source_codes": int(source["ts_code"].nunique()),
        "source_date_from": date_from,
        "source_date_to": date_to,
        "new_files": 0,
        "updated_files": 0,
        "unchanged_files": 0,
        "overlap_keys": 0,
        "conflicting_keys": 0,
        "rows_before": 0,
        "rows_after": 0,
        "failures": [],
    }

    if not target_path.exists():
        result["new_files"] = 1
        result["rows_after"] = len(source)
        if not dry_run:
            write_new_table_atomic(source, source_schema, target_path)
        return result

    target_schema = pq.ParquetFile(target_path).schema_arrow
    _validate_arrow_schema(target_schema, source_schema, str(target_path))
    scan = _scan_sorted_unique_target(target_path, spec.key_columns)
    existing_window = _load_table_target_window(
        target_path,
        target_schema,
        date_from,
        date_to,
    )
    merged_window, overlap, conflicts = merge_flat_frames(
        existing_window,
        source,
        spec.key_columns,
    )
    expected_rows = scan["rows"] - len(existing_window) + len(merged_window)
    unchanged = _frames_equal(existing_window, merged_window)

    result["rows_before"] = scan["rows"]
    result["rows_after"] = expected_rows
    result["overlap_keys"] = overlap
    result["conflicting_keys"] = conflicts
    result["unchanged_files"] = int(unchanged)
    result["updated_files"] = int(not unchanged)

    if not dry_run and not unchanged:
        write_table_window_atomic(
            target_path,
            target_schema,
            merged_window,
            date_from,
            date_to,
            expected_rows,
        )
    return result


def expected_source_paths(source_root: Path) -> list[Path]:
    return [
        *(source_root / filename for filename, _ in MINUTE_DATASETS),
        *(source_root / spec.filename for spec in TABLE_SPECS),
    ]


def preflight_sources(source_root: Path) -> None:
    missing = [path for path in expected_source_paths(source_root) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source parquet files: {missing}")

    for filename, _ in MINUTE_DATASETS:
        source = load_minute_source(source_root / filename)
        LOGGER.info(
            "Validated %s: %s rows, %s symbols",
            filename,
            len(source),
            source["ts_code"].nunique(),
        )
    for spec in TABLE_SPECS:
        source, _ = load_table_source(source_root / spec.filename, spec)
        LOGGER.info(
            "Validated %s: %s rows, %s symbols",
            spec.filename,
            len(source),
            source["ts_code"].nunique(),
        )


def run_merge(
    source_root: Path,
    target_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    preflight_sources(source_root)
    datasets: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for filename, dirname in MINUTE_DATASETS:
        LOGGER.info("Processing %s", filename)
        try:
            dataset = merge_minute_dataset(
                source_root / filename,
                target_root / dirname,
                dry_run,
            )
            datasets.append(dataset)
            failures.extend(
                {
                    "dataset": filename,
                    **failure,
                }
                for failure in dataset["failures"]
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to process %s", filename)
            failures.append(
                {
                    "dataset": filename,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    for spec in TABLE_SPECS:
        LOGGER.info("Processing %s", spec.filename)
        try:
            datasets.append(
                merge_table_dataset(
                    source_root / spec.filename,
                    target_root / spec.filename,
                    spec,
                    dry_run,
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to process %s", spec.filename)
            failures.append(
                {
                    "dataset": spec.filename,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    completed_at = datetime.now().astimezone()
    complete = not failures and len(datasets) == len(MINUTE_DATASETS) + len(TABLE_SPECS)
    status = "dry_run_complete" if dry_run and complete else "complete" if complete else "failed"
    return {
        "status": status,
        "dry_run": dry_run,
        "source_root": str(source_root),
        "target_root": str(target_root),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "datasets": datasets,
        "failures": failures,
        "summary": {
            "expected_datasets": len(MINUTE_DATASETS) + len(TABLE_SPECS),
            "processed_datasets": len(datasets),
            "source_rows": sum(item["source_rows"] for item in datasets),
            "new_files": sum(item["new_files"] for item in datasets),
            "updated_files": sum(item["updated_files"] for item in datasets),
            "unchanged_files": sum(item["unchanged_files"] for item in datasets),
            "overlap_keys": sum(item["overlap_keys"] for item in datasets),
            "conflicting_keys": sum(item["conflicting_keys"] for item in datasets),
            "failure_count": len(failures),
        },
    }


def write_report_atomic(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(report_path)
    try:
        temporary_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(report_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    report_path = args.report_path
    if report_path is None and not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = args.target_root / "logs" / f"merge_etf_data_{timestamp}.json"

    try:
        report = run_merge(args.source_root, args.target_root, args.dry_run)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("ETF data merge failed during preflight")
        now = datetime.now().astimezone()
        report = {
            "status": "failed",
            "dry_run": args.dry_run,
            "source_root": str(args.source_root),
            "target_root": str(args.target_root),
            "completed_at": now,
            "datasets": [],
            "failures": [
                {"dataset": "preflight", "error": f"{type(exc).__name__}: {exc}"}
            ],
            "summary": {
                "expected_datasets": len(MINUTE_DATASETS) + len(TABLE_SPECS),
                "processed_datasets": 0,
                "failure_count": 1,
            },
        }

    if report_path is not None:
        write_report_atomic(report, report_path)
        LOGGER.info("Wrote report to %s", report_path)
    print(json.dumps(report["summary"], ensure_ascii=False, default=_json_default))
    return 0 if report["status"] in {"complete", "dry_run_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
