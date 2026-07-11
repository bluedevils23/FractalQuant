from __future__ import annotations

"""Generate ETF orderbook factor parquet files from raw tick CSV folders."""

import argparse
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.stock_orderbook import build_stock_orderbook_factor_frame  # noqa: E402
from generate_stock_orderbook_factors import (  # noqa: E402
    build_output_frame,
    load_minute_frame,
    merge_symbol_output,
    normalize_order_frame,
    normalize_quote_frame,
    normalize_trade_frame,
)


LOGGER = logging.getLogger("generate_etf_orderbook_factors")

DEFAULT_TICK_ROOT = Path(r"E:\逐笔数据")
DEFAULT_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_orderbook_factors"
)
SYMBOL_PATTERN = re.compile(r"^(\d{6})(?:\.([A-Z]{2}))?$")
DATE_PATTERN = re.compile(r"^\d{8}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ETF orderbook factor parquet files from local tick CSVs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional single trade-date directory containing per-symbol tick folders.",
    )
    parser.add_argument(
        "--tick-root",
        type=Path,
        default=DEFAULT_TICK_ROOT,
        help="Root directory containing tick folders such as <year>/<yyyymm>/<yyyymmdd>.",
    )
    parser.add_argument(
        "--minute-root",
        type=Path,
        default=DEFAULT_MINUTE_ROOT,
        help="Directory containing one 1-minute parquet file per ETF.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where ETF orderbook factor parquet files will be written.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional ETF symbols such as 159941.SZ 513100.SH.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional text file with one or more ETF symbols per line.",
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
        help="Optional inclusive end trade date such as 20260131 or 2026-01-31.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N matched ETF/date tasks after filtering.",
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
        "--strict-suffix",
        action="store_true",
        help="Match symbol directories by full code instead of the default 6-digit code.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def normalize_trade_date_arg(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 8:
        raise ValueError(f"Invalid trade date: {value}")
    return digits


def normalize_symbol(value: str) -> str:
    symbol = str(value).strip().upper()
    if symbol.lower().endswith(".parquet"):
        symbol = symbol[:-8]
    match = SYMBOL_PATTERN.fullmatch(symbol)
    if not match:
        raise ValueError(f"Invalid ETF symbol: {value}")
    code, suffix = match.groups()
    return f"{code}.{suffix}" if suffix else code


def symbol_digits(symbol: str) -> str:
    match = re.search(r"\d{6}", symbol)
    if not match:
        raise ValueError(f"Invalid ETF symbol: {symbol}")
    return match.group(0)


def read_symbol_list_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file does not exist: {path}")

    symbols: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                symbols.append(normalize_symbol(line))

    return dedupe_symbols(symbols)


def load_requested_symbols(
    symbols: list[str] | None,
    symbols_file: Path | None,
) -> list[str] | None:
    requested: list[str] = []
    if symbols_file is not None:
        requested.extend(read_symbol_list_file(symbols_file))
    if symbols:
        requested.extend(normalize_symbol(symbol) for symbol in symbols)
    if not requested:
        return None
    return dedupe_symbols(requested)


def dedupe_symbols(symbols: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def discover_trade_date_dirs(
    tick_root: Path,
    input_root: Path | None,
    date_from: str | None,
    date_to: str | None,
) -> list[Path]:
    if input_root is not None:
        if not input_root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_root}")
        return [input_root]

    if not tick_root.exists():
        raise FileNotFoundError(f"Tick root does not exist: {tick_root}")

    date_dirs: list[Path] = []
    for year_dir in sorted(path for path in tick_root.iterdir() if path.is_dir()):
        for month_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            for date_dir in sorted(path for path in month_dir.iterdir() if path.is_dir()):
                trade_date = date_dir.name
                if not DATE_PATTERN.fullmatch(trade_date):
                    continue
                if date_from is not None and trade_date < date_from:
                    continue
                if date_to is not None and trade_date > date_to:
                    continue
                date_dirs.append(date_dir)

    return date_dirs


def build_symbol_dir_index(date_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    exact: dict[str, Path] = {}
    by_digits: dict[str, list[Path]] = {}
    for symbol_dir in sorted(path for path in date_dir.iterdir() if path.is_dir()):
        name = symbol_dir.name.upper()
        if not re.search(r"\d{6}", name):
            LOGGER.warning("Skipping non-symbol directory: %s", symbol_dir)
            continue
        exact[name] = symbol_dir
        by_digits.setdefault(symbol_digits(name), []).append(symbol_dir)
    return exact, by_digits


def resolve_symbol_dir(
    symbol: str,
    exact_index: dict[str, Path],
    digit_index: dict[str, list[Path]],
    strict_suffix: bool,
) -> Path | None:
    normalized = symbol.upper()
    exact_match = exact_index.get(normalized)
    if exact_match is not None:
        return exact_match
    if strict_suffix:
        return None

    candidates = digit_index.get(symbol_digits(normalized), [])
    if not candidates:
        return None
    if len(candidates) > 1:
        LOGGER.warning(
            "Multiple directories match %s by digits; using %s",
            symbol,
            candidates[0],
        )
    return candidates[0]


def build_tasks(
    date_dirs: list[Path],
    requested_symbols: list[str] | None,
    strict_suffix: bool,
) -> tuple[list[tuple[Path, str]], int]:
    tasks: list[tuple[Path, str]] = []
    missing_count = 0

    for date_dir in date_dirs:
        exact_index, digit_index = build_symbol_dir_index(date_dir)
        if requested_symbols is None:
            tasks.extend((symbol_dir, symbol_dir.name.upper()) for symbol_dir in exact_index.values())
            continue

        missing_for_date: list[str] = []
        for symbol in requested_symbols:
            symbol_dir = resolve_symbol_dir(symbol, exact_index, digit_index, strict_suffix)
            if symbol_dir is None:
                missing_for_date.append(symbol)
                continue
            tasks.append((symbol_dir, symbol_dir.name.upper()))

        if missing_for_date:
            missing_count += len(missing_for_date)
            LOGGER.warning(
                "%s missing %s requested symbols: %s",
                date_dir.name,
                len(missing_for_date),
                ", ".join(missing_for_date[:10]),
            )

    return tasks, missing_count


def process_symbol_tasks(
    symbol_dirs: list[Path],
    output_symbol: str,
    minute_root: Path,
    output_root: Path,
    overwrite: bool,
) -> tuple[str, Path, int | None, int | None]:
    output_path = output_root / f"{output_symbol}.parquet"
    existing_dates: set[str] = set()
    if output_path.exists() and not overwrite:
        existing_dates = set(
            pd.read_parquet(output_path, columns=["trade_date"])["trade_date"].astype(str)
        )

    daily_frames: list[pd.DataFrame] = []
    for symbol_dir in symbol_dirs:
        trade_date = symbol_dir.parent.name
        normalized_date = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        if normalized_date in existing_dates:
            continue
        quotes = normalize_quote_frame(symbol_dir)
        quotes["ts_code"] = output_symbol
        orders = normalize_order_frame(symbol_dir)
        trades = normalize_trade_frame(symbol_dir)
        factors = build_stock_orderbook_factor_frame(quotes, orders, trades)
        minute_df = load_minute_frame(minute_root, output_symbol, trade_date)
        daily_frames.append(build_output_frame(minute_df, factors))

    if not daily_frames:
        return ("skipped", output_path, None, None)

    result = pd.concat(daily_frames, ignore_index=True)
    result = merge_symbol_output(output_path, result, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    return ("written", output_path, len(result), len(result.columns))


def log_task_result(
    status: str,
    output_path: Path,
    row_count: int | None,
    column_count: int | None,
) -> None:
    if status == "skipped":
        LOGGER.info("Skipping existing output: %s", output_path)
    else:
        LOGGER.info(
            "Wrote %s rows and %s columns to %s",
            row_count,
            column_count,
            output_path,
        )


def main() -> int:
    args = parse_args()
    configure_logging()

    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")

    requested_symbols = load_requested_symbols(args.symbols, args.symbols_file)
    if requested_symbols is not None:
        LOGGER.info("Loaded %s requested ETF symbols", len(requested_symbols))

    date_dirs = discover_trade_date_dirs(args.tick_root, args.input_root, date_from, date_to)
    if not date_dirs:
        LOGGER.warning("No trade-date directories matched the requested inputs.")
        return 0

    tasks, missing_count = build_tasks(date_dirs, requested_symbols, args.strict_suffix)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    if not tasks:
        LOGGER.warning("No ETF tick directories matched the requested symbols.")
        return 0

    tasks_by_symbol: dict[str, list[Path]] = {}
    for symbol_dir, output_symbol in tasks:
        tasks_by_symbol.setdefault(output_symbol, []).append(symbol_dir)

    LOGGER.info(
        "Processing %s ETFs and %s daily orderbook inputs across %s trade dates",
        len(tasks_by_symbol),
        len(tasks),
        len(date_dirs),
    )
    if missing_count:
        LOGGER.warning("Skipped %s missing symbol/date combinations", missing_count)

    failures: list[tuple[Path, str]] = []
    written_count = 0
    skipped_count = 0
    worker_count = max(1, args.workers)

    if worker_count == 1:
        for output_symbol, symbol_dirs in tasks_by_symbol.items():
            try:
                status, output_path, row_count, column_count = process_symbol_tasks(
                    symbol_dirs,
                    output_symbol,
                    args.minute_root,
                    args.output_root,
                    args.overwrite,
                )
                written_count += int(status == "written")
                skipped_count += int(status == "skipped")
                log_task_result(status, output_path, row_count, column_count)
            except Exception as exc:  # noqa: BLE001
                failures.append((symbol_dirs[0], str(exc)))
                LOGGER.exception("Failed to process %s", output_symbol)
    else:
        LOGGER.info("Using %s parallel workers", worker_count)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_symbol_tasks,
                    symbol_dirs,
                    output_symbol,
                    args.minute_root,
                    args.output_root,
                    args.overwrite,
                ): (output_symbol, symbol_dirs[0])
                for output_symbol, symbol_dirs in tasks_by_symbol.items()
            }
            for future in as_completed(future_map):
                output_symbol, symbol_dir = future_map[future]
                try:
                    status, output_path, row_count, column_count = future.result()
                    written_count += int(status == "written")
                    skipped_count += int(status == "skipped")
                    log_task_result(status, output_path, row_count, column_count)
                except Exception as exc:  # noqa: BLE001
                    failures.append((symbol_dir, str(exc)))
                    LOGGER.exception("Failed to process %s", symbol_dir)

    LOGGER.info(
        "Completed ETF orderbook run: written=%s skipped=%s failed=%s",
        written_count,
        skipped_count,
        len(failures),
    )

    if failures:
        LOGGER.error("Completed with %s failures", len(failures))
        for failed_path, reason in failures[:10]:
            LOGGER.error("  %s -> %s", failed_path, reason)
        return 1

    LOGGER.info("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
