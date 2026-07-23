"""Audit ETF minute-factor files and their matching raw minute inputs.

The audit is intentionally read-only.  It scans each valid factor parquet file,
reports field-level missing/zero/concentration metrics, and checks the matching
source parquet for price, volume, timestamp, and row-count issues.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_FACTOR_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_factors")
DEFAULT_SOURCE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
RAW_FACTOR_COLUMNS = {
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adj_factor",
    "trade_time",
}
SOURCE_NUMERIC_COLUMNS = ("open", "high", "low", "close", "vol", "amount", "adj_factor")


@dataclass
class NumericStats:
    rows: int = 0
    finite: int = 0
    nonfinite: int = 0
    zero: int = 0
    positive: int = 0
    negative: int = 0
    early_rows: int = 0
    late_rows: int = 0
    early_nonfinite: int = 0
    late_nonfinite: int = 0
    min_value: float = math.inf
    max_value: float = -math.inf
    batches: int = 0
    constant_batches: int = 0
    max_sample_mode_ratio: float = 0.0
    max_sample_mode_value: float = math.nan

    def add(self, values: np.ndarray, early: np.ndarray | None, sample_stride: int) -> None:
        values = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values)
        finite = int(finite_mask.sum())
        missing = len(values) - finite
        self.rows += len(values)
        self.finite += finite
        self.nonfinite += missing
        self.batches += 1
        if early is not None:
            self.early_rows += int(early.sum())
            self.late_rows += int((~early).sum())
            self.early_nonfinite += int((~finite_mask & early).sum())
            self.late_nonfinite += int((~finite_mask & ~early).sum())
        if not finite:
            return

        valid = values[finite_mask]
        self.zero += int((valid == 0.0).sum())
        self.positive += int((valid > 0.0).sum())
        self.negative += int((valid < 0.0).sum())
        self.min_value = min(self.min_value, float(valid.min()))
        self.max_value = max(self.max_value, float(valid.max()))

        sample = valid[::sample_stride]
        if len(sample):
            unique, counts = np.unique(sample, return_counts=True)
            mode_index = int(counts.argmax())
            ratio = float(counts[mode_index] / len(sample))
            if ratio > self.max_sample_mode_ratio:
                self.max_sample_mode_ratio = ratio
                self.max_sample_mode_value = float(unique[mode_index])
            if len(unique) == 1:
                self.constant_batches += 1

    def merge(self, other: "NumericStats") -> None:
        self.rows += other.rows
        self.finite += other.finite
        self.nonfinite += other.nonfinite
        self.zero += other.zero
        self.positive += other.positive
        self.negative += other.negative
        self.early_rows += other.early_rows
        self.late_rows += other.late_rows
        self.early_nonfinite += other.early_nonfinite
        self.late_nonfinite += other.late_nonfinite
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)
        self.batches += other.batches
        self.constant_batches += other.constant_batches
        if other.max_sample_mode_ratio > self.max_sample_mode_ratio:
            self.max_sample_mode_ratio = other.max_sample_mode_ratio
            self.max_sample_mode_value = other.max_sample_mode_value

    def as_row(self, name: str, files_present: int) -> dict[str, object]:
        nonfinite_rate = self.nonfinite / self.rows if self.rows else math.nan
        zero_rate = self.zero / self.finite if self.finite else math.nan
        late_nonfinite_rate = (
            self.late_nonfinite / self.late_rows if self.late_rows else math.nan
        )
        status = "ok"
        if not self.finite:
            status = "critical_all_missing"
        elif late_nonfinite_rate >= 0.05:
            status = "review_late_missing"
        elif zero_rate >= 0.95:
            status = "review_dominant_zero"
        elif self.max_sample_mode_ratio >= 0.95:
            status = "review_dominant_value"
        elif nonfinite_rate >= 0.20:
            status = "likely_session_warmup"
        return {
            "field": name,
            "files_present": files_present,
            "rows": self.rows,
            "finite_rows": self.finite,
            "missing_rows": self.nonfinite,
            "missing_rate": nonfinite_rate,
            "zero_rows": self.zero,
            "zero_rate_among_finite": zero_rate,
            "positive_rows": self.positive,
            "negative_rows": self.negative,
            "late_missing_rows": self.late_nonfinite,
            "late_missing_rate": late_nonfinite_rate,
            "min": None if self.min_value == math.inf else self.min_value,
            "max": None if self.max_value == -math.inf else self.max_value,
            "constant_batch_rate": self.constant_batches / self.batches if self.batches else math.nan,
            "max_sample_mode_rate": self.max_sample_mode_ratio,
            "sample_mode_value": self.max_sample_mode_value,
            "assessment": status,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTOR_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument(
        "--warmup-bars", type=int, default=80,
        help="Per-session opening bars excluded from unexpected-missing-rate checks.",
    )
    parser.add_argument(
        "--sample-stride", type=int, default=32,
        help="Stride used for dominant-value sampling (1 means exact batch modes).",
    )
    return parser.parse_args()


def dates_and_early_mask(
    timestamps: np.ndarray,
    warmup_bars: int,
    prior_day: np.datetime64 | None,
    prior_count: int,
) -> tuple[np.ndarray, np.datetime64 | None, int]:
    values = np.asarray(timestamps).astype("datetime64[ns]")
    days = values.astype("datetime64[D]")
    early = np.zeros(len(days), dtype=bool)
    if not len(days):
        return early, prior_day, prior_count

    starts = np.r_[0, np.flatnonzero(days[1:] != days[:-1]) + 1]
    stops = np.r_[starts[1:], len(days)]
    for start, stop in zip(starts, stops):
        offset = prior_count if start == 0 and days[start] == prior_day else 0
        positions = offset + np.arange(stop - start)
        early[start:stop] = positions < warmup_bars
    last_start = int(starts[-1])
    if days[last_start] == prior_day:
        last_count = prior_count + len(days) - last_start
    else:
        last_count = len(days) - last_start
    return early, days[-1], int(last_count)


def source_quality(source_path: Path, batch_size: int, sample_stride: int) -> tuple[dict[str, NumericStats], dict[str, object]]:
    source_stats: dict[str, NumericStats] = defaultdict(NumericStats)
    if not source_path.exists():
        return source_stats, {"source_exists": False}

    parquet = pq.ParquetFile(source_path)
    available = set(parquet.schema_arrow.names)
    numeric = [column for column in SOURCE_NUMERIC_COLUMNS if column in available]
    timestamp_column = "trade_time" if "trade_time" in available else "trade_date"
    columns = [*numeric, timestamp_column]
    rows = duplicates = nonmonotonic = invalid_ohlc = 0
    day_counts: Counter[str] = Counter()
    previous_time: np.datetime64 | None = None

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        rows += batch.num_rows
        arrays = {name: batch.column(index).to_numpy(zero_copy_only=False) for index, name in enumerate(columns)}
        for column in numeric:
            source_stats[column].add(arrays[column], None, sample_stride)
        if {"open", "high", "low", "close"}.issubset(arrays):
            open_, high, low, close = (np.asarray(arrays[name], dtype=float) for name in ("open", "high", "low", "close"))
            valid = np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
            invalid_ohlc += int(
                (valid & ((high < np.maximum(np.maximum(open_, close), low)) | (low > np.minimum(np.minimum(open_, close), high)))).sum()
            )
        times = np.asarray(arrays[timestamp_column]).astype("datetime64[ns]")
        if len(times):
            time_int = times.astype("int64")
            if previous_time is not None:
                duplicates += int(time_int[0] == previous_time.astype("datetime64[ns]").astype("int64"))
                nonmonotonic += int(time_int[0] < previous_time.astype("datetime64[ns]").astype("int64"))
            diffs = np.diff(time_int)
            duplicates += int((diffs == 0).sum())
            nonmonotonic += int((diffs < 0).sum())
            previous_time = times[-1]
            days, counts = np.unique(times.astype("datetime64[D]").astype(str), return_counts=True)
            day_counts.update(dict(zip(days.tolist(), counts.tolist())))

    return source_stats, {
        "source_exists": True,
        "source_rows": rows,
        "source_duplicate_timestamps": duplicates,
        "source_nonmonotonic_timestamps": nonmonotonic,
        "source_invalid_ohlc_rows": invalid_ohlc,
        "source_trade_days": len(day_counts),
        "source_min_bars_per_day": min(day_counts.values()) if day_counts else 0,
        "source_max_bars_per_day": max(day_counts.values()) if day_counts else 0,
        "source_days_lt_200_bars": sum(count < 200 for count in day_counts.values()),
    }


def audit_factor_file(
    factor_path_text: str,
    source_root_text: str,
    batch_size: int,
    warmup_bars: int,
    sample_stride: int,
) -> dict[str, object]:
    factor_path = Path(factor_path_text)
    source_path = Path(source_root_text) / factor_path.name
    parquet = pq.ParquetFile(factor_path)
    columns = parquet.schema_arrow.names
    factor_columns = [column for column in columns if column not in RAW_FACTOR_COLUMNS]
    read_columns = [*factor_columns, "trade_time"]
    stats: dict[str, NumericStats] = defaultdict(NumericStats)
    prior_day: np.datetime64 | None = None
    prior_count = 0

    for batch in parquet.iter_batches(batch_size=batch_size, columns=read_columns):
        timestamp_index = len(read_columns) - 1
        early, prior_day, prior_count = dates_and_early_mask(
            batch.column(timestamp_index).to_numpy(zero_copy_only=False),
            warmup_bars,
            prior_day,
            prior_count,
        )
        for index, column in enumerate(factor_columns):
            stats[column].add(batch.column(index).to_numpy(zero_copy_only=False), early, sample_stride)

    source_stats, source_summary = source_quality(source_path, batch_size, sample_stride)
    file_summary = {
        "file": factor_path.name,
        "factor_rows": parquet.metadata.num_rows,
        "factor_columns": len(factor_columns),
        **source_summary,
    }
    if source_summary["source_exists"]:
        file_summary["row_count_difference"] = parquet.metadata.num_rows - int(source_summary["source_rows"])
    else:
        file_summary["row_count_difference"] = None
    return {
        "factor_stats": stats,
        "source_stats": source_stats,
        "summary": file_summary,
        "factor_columns": factor_columns,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(
    output_path: Path,
    factor_rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    file_rows: list[dict[str, object]],
    invalid_files: list[dict[str, object]],
) -> None:
    reviewed = [row for row in factor_rows if str(row["assessment"]).startswith("review") or row["assessment"] == "critical_all_missing"]
    source_issues = [
        row for row in file_rows
        if (not row["source_exists"])
        or row.get("row_count_difference") not in (0, None)
        or row.get("source_duplicate_timestamps", 0)
        or row.get("source_nonmonotonic_timestamps", 0)
        or row.get("source_invalid_ohlc_rows", 0)
    ]
    lines = [
        "# ETF minute factor quality audit",
        "",
        f"- Valid factor files audited: {len(file_rows)}",
        f"- Invalid/corrupt factor files: {len(invalid_files)}",
        f"- Distinct factor fields: {len(factor_rows)}",
        f"- Fields requiring review: {len(reviewed)}",
        f"- Source-file linkage issues: {len(source_issues)}",
        "",
        "## Interpretation",
        "",
        "- Missing values in the first 80 bars of each session are treated separately as expected rolling-window warm-up. `late_missing_rate` measures missing values after that opening region.",
        "- `zero_rate_among_finite` is exact. `max_sample_mode_rate` is a batch-level sampled concentration diagnostic; it is intended to flag nearly constant values, not to estimate global cardinality.",
        "- A factor's zero/constant result is not automatically a source-data fault. The detailed source report and the factor implementation must be read together before removing a field.",
        "",
        "## Highest-priority factor fields",
        "",
    ]
    priority = sorted(
        reviewed,
        key=lambda row: (
            row["assessment"] == "critical_all_missing",
            row.get("late_missing_rate") or 0,
            row.get("zero_rate_among_finite") or 0,
            row.get("max_sample_mode_rate") or 0,
        ),
        reverse=True,
    )[:30]
    if priority:
        lines.extend(["| Factor | Assessment | Missing | Late missing | Zero | Sample-mode |", "|---|---:|---:|---:|---:|---:|"])
        for row in priority:
            lines.append(
                "| {field} | {assessment} | {missing_rate:.2%} | {late_missing_rate:.2%} | {zero_rate_among_finite:.2%} | {max_sample_mode_rate:.2%} |".format(
                    **{key: (0.0 if value is None or (isinstance(value, float) and math.isnan(value)) else value) for key, value in row.items()}
                )
            )
    else:
        lines.append("No field triggered the review thresholds.")
    lines.extend(["", "## Invalid factor files", ""])
    if invalid_files:
        lines.extend(["| File | Bytes | Error |", "|---|---:|---|"])
        lines.extend(f"| {row['file']} | {row['bytes']} | {row['error']} |" for row in invalid_files)
    else:
        lines.append("None.")
    lines.extend(["", "Detailed machine-readable results are in the CSV files beside this report."])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.sample_stride < 1:
        raise ValueError("workers, batch-size, and sample-stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.factor_root.glob("*.parquet"))
    valid_paths: list[Path] = []
    invalid_files: list[dict[str, object]] = []
    for path in paths:
        try:
            pq.ParquetFile(path)
            valid_paths.append(path)
        except Exception as exc:  # noqa: BLE001
            invalid_files.append({"file": path.name, "bytes": path.stat().st_size, "error": f"{type(exc).__name__}: {exc}"})

    factor_stats: dict[str, NumericStats] = defaultdict(NumericStats)
    source_stats: dict[str, NumericStats] = defaultdict(NumericStats)
    factor_presence: Counter[str] = Counter()
    source_presence: Counter[str] = Counter()
    file_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                audit_factor_file,
                str(path),
                str(args.source_root),
                args.batch_size,
                args.warmup_bars,
                args.sample_stride,
            ): path
            for path in valid_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                invalid_files.append({"file": path.name, "bytes": path.stat().st_size, "error": f"audit failure: {type(exc).__name__}: {exc}"})
                continue
            for name, stats in result["factor_stats"].items():
                factor_stats[name].merge(stats)
            for name, stats in result["source_stats"].items():
                source_stats[name].merge(stats)
            factor_presence.update(result["factor_columns"])
            source_presence.update(result["source_stats"].keys())
            file_rows.append(result["summary"])
            if completed % 25 == 0 or completed == len(futures):
                print(f"Audited {completed}/{len(futures)} valid factor files", flush=True)

    factor_rows = [factor_stats[name].as_row(name, factor_presence[name]) for name in sorted(factor_stats)]
    source_rows = [source_stats[name].as_row(name, source_presence[name]) for name in sorted(source_stats)]
    file_rows.sort(key=lambda row: str(row["file"]))
    invalid_files.sort(key=lambda row: str(row["file"]))
    write_csv(args.output_dir / "factor_field_quality.csv", factor_rows)
    write_csv(args.output_dir / "source_field_quality.csv", source_rows)
    write_csv(args.output_dir / "file_linkage_quality.csv", file_rows)
    write_csv(args.output_dir / "invalid_factor_files.csv", invalid_files)
    markdown_report(args.output_dir / "README.md", factor_rows, source_rows, file_rows, invalid_files)
    print(f"Wrote audit report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
