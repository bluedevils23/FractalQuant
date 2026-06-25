from __future__ import annotations

"""Generate advanced factor parquet files for A-share minute data."""

import argparse
import inspect
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor import advanced as advanced_module  # noqa: E402
from factor.base import BaseFactor  # noqa: E402


LOGGER = logging.getLogger("generate_stock_advanced_factor")

DEFAULT_MINUTE_ROOT = Path(r"D:\workspace\stockdata\a-share-data")
DEFAULT_TICK_ROOT = Path(r"E:\逐笔数据")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\a-share-data\stock_advanced_factors")

POSITIVE_PRICE_COLUMNS = ("open", "high", "low", "close")
SYMBOL_FILE_PATTERN = re.compile(r"^\d{6}\.[A-Z]{2}$")
_CURRENT_DAY_WORKERS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate advanced factor parquet files from stock minute data."
    )
    parser.add_argument(
        "--minute-root",
        type=Path,
        default=DEFAULT_MINUTE_ROOT,
        help="Directory containing stock minute parquet files, or the parent that holds stock_1min.",
    )
    parser.add_argument(
        "--tick-root",
        type=Path,
        default=DEFAULT_TICK_ROOT,
        help="Directory containing tick folders for validation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where advanced factor parquet files will be written.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbols such as 000001.SZ 600000.SH.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N symbols after filtering.",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Optional inclusive start trade date such as 20260101 or 2026-01-01.",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="Optional inclusive end trade date such as 20260331 or 2026-03-31.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers to use.",
    )
    parser.add_argument(
        "--day-workers",
        type=int,
        default=1,
        help="Parallel workers to use within one symbol across trading days. Use >1 for single-symbol runs.",
    )
    parser.add_argument(
        "--skip-tick-check",
        action="store_true",
        help="Skip tick coverage validation to save time.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def normalize_symbol_id(value: str) -> str:
    symbol = str(value).strip()
    if symbol.lower().endswith(".parquet"):
        symbol = symbol[:-8]
    return symbol


def normalize_trade_date_arg(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 8:
        raise ValueError(f"Invalid trade date: {value}")
    return digits


def resolve_minute_root(minute_root: Path) -> Path:
    candidate = minute_root / "stock_1min"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return minute_root


def discover_minute_files(minute_root: Path, symbols: list[str] | None) -> list[Path]:
    resolved_root = resolve_minute_root(minute_root)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Minute directory does not exist: {resolved_root}")

    files = []
    for path in sorted(resolved_root.glob("*.parquet")):
        symbol_id = normalize_symbol_id(path.name)
        if SYMBOL_FILE_PATTERN.fullmatch(symbol_id):
            files.append(path)

    if not files:
        raise FileNotFoundError(f"No minute parquet files found in: {resolved_root}")

    if symbols:
        wanted = [normalize_symbol_id(symbol) for symbol in symbols]
        file_map = {normalize_symbol_id(path.name): path for path in files}
        missing = [symbol for symbol in wanted if symbol not in file_map]
        if missing:
            raise FileNotFoundError(
                "Missing minute parquet files: " + ", ".join(missing[:10])
            )
        files = [file_map[symbol] for symbol in wanted]

    return files


def normalize_minute_frame(raw_df: pd.DataFrame, ts_code_hint: str) -> pd.DataFrame:
    df = raw_df.copy()

    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename(columns={"volume": "vol"})

    if "ts_code" not in df.columns:
        df["ts_code"] = ts_code_hint
    else:
        df["ts_code"] = df["ts_code"].astype(str)

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name in {"trade_date", "trade_time"}:
        df = df.reset_index()

    for column in ("trade_date", "trade_time"):
        if column in df.columns:
            df[column] = df[column].astype(str)

    numeric_columns = [
        column
        for column in ("open", "high", "low", "close", "vol", "amount", "adj_factor")
        if column in df.columns
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    if isinstance(df.index, pd.MultiIndex):
        df = df.sort_index()
    elif "trade_date" in df.columns and "trade_time" in df.columns:
        df = df.sort_values(["trade_date", "trade_time"], kind="mergesort")
    else:
        df = df.sort_index()

    return df


def filter_raw_frame_by_date(
    df: pd.DataFrame, date_from: str | None, date_to: str | None
) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return df
    if isinstance(df.index, pd.MultiIndex) and "trade_date" in df.index.names:
        trade_dates = pd.Index(df.index.get_level_values("trade_date"))
    elif "trade_date" in df.columns:
        trade_dates = pd.Index(df["trade_date"])
    else:
        return df

    normalized_dates = (
        trade_dates.astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
    )
    mask = pd.Series(True, index=df.index)
    if date_from is not None:
        mask &= normalized_dates >= date_from
    if date_to is not None:
        mask &= normalized_dates <= date_to
    return df.loc[mask].copy()


def filter_minute_frame_by_date(
    df: pd.DataFrame, date_from: str | None, date_to: str | None
) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return df
    if "trade_date" not in df.columns:
        raise ValueError("Missing trade_date column for date filtering.")

    trade_dates = (
        df["trade_date"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 8)
    )
    mask = pd.Series(True, index=df.index)
    if date_from is not None:
        mask &= trade_dates >= date_from
    if date_to is not None:
        mask &= trade_dates <= date_to
    return df.loc[mask].copy()


def prepare_factor_input(df: pd.DataFrame) -> pd.DataFrame:
    factor_input = df.copy()
    for column in POSITIVE_PRICE_COLUMNS:
        if column in factor_input.columns:
            factor_input.loc[factor_input[column] <= 0, column] = np.nan
    return factor_input


@lru_cache(maxsize=1)
def build_advanced_factors() -> tuple[BaseFactor, ...]:
    factors: list[BaseFactor] = []
    for name, obj in advanced_module.__dict__.items():
        if not inspect.isclass(obj):
            continue
        if obj is BaseFactor or not issubclass(obj, BaseFactor):
            continue
        if obj.__module__ != advanced_module.__name__:
            continue
        # 排除前视标签因子（如 future_returns 使用 shift(-window)），
        # 它是预测目标(label)而非特征(feature)，若写入因子文件会造成未来函数泄漏。
        if getattr(obj, "name", None) == "future_returns" or name == "FutureReturnsFactor":
            continue
        try:
            instance = obj()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping factor %s: %s", name, exc)
            continue
        if getattr(instance, "name", None) == "future_returns":
            continue
        factors.append(instance)
    if not factors:
        raise RuntimeError("No advanced factors could be constructed.")
    return tuple(factors)


def _calculate_factors_for_group(factor_input: pd.DataFrame) -> pd.DataFrame:
    factor_series: dict[str, pd.Series] = {}

    for factor in build_advanced_factors():
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                values = factor.calculate(factor_input)
            if isinstance(values, pd.Series):
                series = values.reindex(factor_input.index)
            else:
                series = pd.Series(values, index=factor_input.index)
            series = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            factor_series[factor.name] = series
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Factor %s failed: %s", factor.name, exc)
            factor_series[factor.name] = pd.Series(np.nan, index=factor_input.index)

    return pd.DataFrame(factor_series, index=factor_input.index)


def calculate_factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    factor_input = prepare_factor_input(df)

    # 按交易日分组分别计算滚动窗口因子，避免隔夜跨日污染。
    # 优先用 trade_date 列；缺失时退化为整体计算。
    if "trade_date" in factor_input.columns:
        trade_days = (
            factor_input["trade_date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
        )
        grouped_frames = [group for _, group in factor_input.groupby(trade_days, sort=False)]
        if len(grouped_frames) > 1:
            if _CURRENT_DAY_WORKERS > 1:
                with ProcessPoolExecutor(max_workers=_CURRENT_DAY_WORKERS) as executor:
                    per_day_frames = list(executor.map(_calculate_factors_for_group, grouped_frames))
            else:
                per_day_frames = [_calculate_factors_for_group(group) for group in grouped_frames]
            factor_df = pd.concat(per_day_frames, axis=0)
            return factor_df.reindex(factor_input.index)

    return _calculate_factors_for_group(factor_input)


def extract_trade_dates(df: pd.DataFrame) -> list[str]:
    if isinstance(df.index, pd.MultiIndex) and "trade_date" in df.index.names:
        raw = pd.Index(df.index.get_level_values("trade_date"))
    elif "trade_date" in df.columns:
        raw = pd.Index(df["trade_date"])
    else:
        return []

    trade_dates = (
        raw.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 8)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(trade_dates)


def tick_path_for_date(tick_root: Path, trade_date: str, ts_code: str) -> Path:
    return tick_root / trade_date[:6] / trade_date / ts_code


def validate_tick_coverage(
    tick_root: Path, trade_dates: list[str], ts_code: str
) -> dict[str, object]:
    if not tick_root.exists():
        return {
            "enabled": False,
            "total": len(trade_dates),
            "present": 0,
            "missing": trade_dates[:10],
            "missing_count": len(trade_dates),
        }

    present = 0
    missing: list[str] = []
    for trade_date in trade_dates:
        if tick_path_for_date(tick_root, trade_date, ts_code).exists():
            present += 1
        else:
            missing.append(trade_date)

    return {
        "enabled": True,
        "total": len(trade_dates),
        "present": present,
        "missing": missing[:10],
        "missing_count": len(missing),
    }


def build_output_frame(df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([df, factor_df], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def process_symbol_file(
    input_path: Path,
    output_root: Path,
    tick_root: Path,
    date_from: str | None,
    date_to: str | None,
    day_workers: int,
    skip_tick_check: bool,
    overwrite: bool,
) -> dict[str, object]:
    ts_code = normalize_symbol_id(input_path.name)
    output_path = output_root / f"{ts_code}.parquet"
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "ts_code": ts_code,
            "output_path": output_path,
        }

    raw_df = pd.read_parquet(input_path)
    raw_df = filter_raw_frame_by_date(raw_df, date_from, date_to)
    minute_df = normalize_minute_frame(raw_df, ts_code)
    if minute_df.empty:
        return {
            "status": "empty",
            "ts_code": ts_code,
            "output_path": output_path,
        }
    global _CURRENT_DAY_WORKERS
    _CURRENT_DAY_WORKERS = max(1, int(day_workers))
    factor_df = calculate_factor_frame(minute_df)
    result_df = build_output_frame(minute_df, factor_df)

    trade_dates = extract_trade_dates(minute_df)
    coverage = {"enabled": False, "total": len(trade_dates), "present": 0, "missing": [], "missing_count": 0}
    if not skip_tick_check:
        coverage = validate_tick_coverage(tick_root, trade_dates, ts_code)

    output_root.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path)

    return {
        "status": "written",
        "ts_code": ts_code,
        "output_path": output_path,
        "rows": len(result_df),
        "cols": len(result_df.columns),
        "trade_dates": len(trade_dates),
        "tick_enabled": coverage["enabled"],
        "tick_total": coverage["total"],
        "tick_present": coverage["present"],
        "tick_missing_count": coverage["missing_count"],
        "tick_missing_examples": coverage["missing"],
    }


def log_result(result: dict[str, object]) -> None:
    ts_code = result["ts_code"]
    status = result["status"]
    output_path = result["output_path"]

    if status == "skipped":
        LOGGER.info("Skipping existing output: %s", output_path)
        return
    if status == "empty":
        LOGGER.info("Skipping %s because no rows matched the requested date range", ts_code)
        return

    LOGGER.info(
        "Wrote %s rows and %s columns for %s to %s",
        result["rows"],
        result["cols"],
        ts_code,
        output_path,
    )

    if result.get("tick_enabled"):
        total = int(result.get("tick_total", 0))
        present = int(result.get("tick_present", 0))
        missing = int(result.get("tick_missing_count", 0))
        LOGGER.info(
            "Tick coverage for %s: %s/%s days present, %s missing",
            ts_code,
            present,
            total,
            missing,
        )
        examples = result.get("tick_missing_examples", [])
        if examples:
            LOGGER.info("Missing tick days for %s: %s", ts_code, ", ".join(examples))
    else:
        LOGGER.info("Tick coverage validation skipped for %s", ts_code)


def main() -> int:
    args = parse_args()
    configure_logging()
    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")

    files = discover_minute_files(args.minute_root, args.symbols)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        LOGGER.warning("No minute parquet files matched the requested inputs.")
        return 0

    if not args.tick_root.exists():
        LOGGER.warning("Tick root does not exist: %s", args.tick_root)

    LOGGER.info("Processing %s stock minute parquet files", len(files))
    if date_from or date_to:
        LOGGER.info(
            "Applying trade-date filter: [%s, %s]",
            date_from or "-inf",
            date_to or "+inf",
        )
    worker_count = max(1, args.workers)
    day_workers = max(1, args.day_workers if worker_count == 1 else 1)
    failures: list[tuple[Path, str]] = []

    if worker_count == 1:
        for input_path in files:
            try:
                result = process_symbol_file(
                    input_path,
                    args.output_root,
                    args.tick_root,
                    date_from,
                    date_to,
                    day_workers,
                    args.skip_tick_check,
                    args.overwrite,
                )
                log_result(result)
            except Exception as exc:  # noqa: BLE001
                failures.append((input_path, str(exc)))
                LOGGER.exception("Failed to process %s", input_path)
    else:
        LOGGER.info("Using %s parallel workers", worker_count)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_symbol_file,
                    input_path,
                    args.output_root,
                    args.tick_root,
                    date_from,
                    date_to,
                    day_workers,
                    args.skip_tick_check,
                    args.overwrite,
                ): input_path
                for input_path in files
            }
            for future in as_completed(future_map):
                input_path = future_map[future]
                try:
                    result = future.result()
                    log_result(result)
                except Exception as exc:  # noqa: BLE001
                    failures.append((input_path, str(exc)))
                    LOGGER.exception("Failed to process %s", input_path)

    if failures:
        LOGGER.error("Completed with %s failures", len(failures))
        for failed_path, reason in failures[:10]:
            LOGGER.error("  %s -> %s", failed_path, reason)
        return 1

    LOGGER.info("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
