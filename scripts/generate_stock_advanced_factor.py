from __future__ import annotations

"""Generate advanced factor parquet files for A-share minute data."""

import argparse
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.advanced_runtime import (  # noqa: E402
    log_result,
    normalize_symbol_id,
    normalize_trade_date_arg,
    read_symbol_list_file,
    process_symbol_file,
)


LOGGER = logging.getLogger("generate_stock_advanced_factor")

DEFAULT_MINUTE_ROOT = Path(r"D:\workspace\stockdata\a-share-data")
DEFAULT_TICK_ROOT = Path(r"E:\逐笔数据")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\a-share-data\stock_advanced_factors")
SYMBOL_FILE_PATTERN = re.compile(r"^\d{6}\.[A-Z]{2}$")


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
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional text file with one or more symbols per line.",
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


def load_requested_symbols(
    symbols: list[str] | None, symbols_file: Path | None
) -> list[str] | None:
    requested: list[str] = []
    if symbols_file is not None:
        requested.extend(read_symbol_list_file(symbols_file))
    if symbols:
        requested.extend(normalize_symbol_id(symbol) for symbol in symbols)
    if not requested:
        return None

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in requested:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def main() -> int:
    args = parse_args()
    configure_logging()
    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")

    requested_symbols = load_requested_symbols(args.symbols, args.symbols_file)
    files = discover_minute_files(args.minute_root, requested_symbols)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        LOGGER.warning("No minute parquet files matched the requested inputs.")
        return 0

    if not args.skip_tick_check and not args.tick_root.exists():
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
                    exclude_future_returns=False,
                )
                log_result(LOGGER, result)
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
                    exclude_future_returns=False,
                ): input_path
                for input_path in files
            }
            for future in as_completed(future_map):
                input_path = future_map[future]
                try:
                    result = future.result()
                    log_result(LOGGER, result)
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
