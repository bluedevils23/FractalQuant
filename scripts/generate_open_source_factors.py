"""Generate daily ETF/stock proxies from open-source microstructure research."""

from __future__ import annotations

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

from factor.open_source import (  # noqa: E402
    add_availability_columns,
    build_daily_raw_features,
    build_open_source_factor_panel,
    select_output_columns,
)


LOGGER = logging.getLogger("generate_open_source_factors")
DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-factors\etf_open_source_factors"
)
SYMBOL_PATTERN = re.compile(r"^\d{6}\.[A-Z]{2}$")


def normalize_symbol(value: str) -> str:
    symbol = str(value).strip()
    if symbol.lower().endswith(".parquet"):
        symbol = symbol[:-8]
    return symbol


def normalize_trade_date(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 8:
        raise ValueError(f"Invalid trade date: {value}")
    return digits


def read_symbols_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file does not exist: {path}")
    symbols: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if value:
                symbols.append(normalize_symbol(value))
    return list(dict.fromkeys(symbols))


def resolve_input_root(input_root: Path) -> Path:
    candidate = input_root / "etf_1min"
    if candidate.is_dir():
        return candidate
    return input_root


def discover_input_files(
    input_root: Path,
    symbols: list[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    root = resolve_input_root(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    files = [
        path
        for path in sorted(root.glob("*.parquet"))
        if SYMBOL_PATTERN.fullmatch(normalize_symbol(path.name))
    ]
    if symbols:
        file_map = {normalize_symbol(path.name): path for path in files}
        missing = [symbol for symbol in symbols if symbol not in file_map]
        if missing:
            raise FileNotFoundError(
                "Missing input parquet files: " + ", ".join(missing[:10])
            )
        files = [file_map[symbol] for symbol in symbols]
    if limit is not None:
        files = files[: max(0, limit)]
    if not files:
        raise FileNotFoundError(f"No input parquet files found under {root}")
    return files


def _extract_one(input_path: Path) -> pd.DataFrame:
    symbol = normalize_symbol(input_path.name)
    raw = pd.read_parquet(input_path)
    return build_daily_raw_features(raw, symbol)


def build_raw_panel(files: list[Path], workers: int) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    if workers <= 1:
        for path in files:
            try:
                frames.append(_extract_one(path))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to extract %s", path)
                failures.append((normalize_symbol(path.name), str(exc)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_one, path): path for path in files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    frames.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Failed to extract %s", path)
                    failures.append((normalize_symbol(path.name), str(exc)))
    if not frames:
        raise RuntimeError("No symbol produced daily raw features")
    return pd.concat(frames, ignore_index=True), failures


def _filter_dates(
    frame: pd.DataFrame,
    date_from: str | None,
    date_to: str | None,
) -> pd.DataFrame:
    result = frame.copy()
    if date_from is not None:
        result = result.loc[result["trade_date"].dt.strftime("%Y%m%d").ge(date_from)]
    if date_to is not None:
        result = result.loc[result["trade_date"].dt.strftime("%Y%m%d").le(date_to)]
    return result


def write_outputs(
    panel: pd.DataFrame,
    output_root: Path,
    *,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
) -> list[Path]:
    selected = select_output_columns(_filter_dates(panel, date_from, date_to))
    written: list[Path] = []
    for symbol, frame in selected.groupby("ts_code", sort=True):
        output_path = output_root / f"{symbol}.parquet"
        if output_path.exists() and not overwrite:
            LOGGER.info("Skipping existing output: %s", output_path)
            continue
        output_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        written.append(output_path)
        LOGGER.info("Wrote %s rows to %s", len(frame), output_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel workers for per-symbol daily extraction.",
    )
    parser.add_argument(
        "--min-tgd-cross-section",
        type=int,
        default=20,
        help="Minimum valid symbols per date for the TGD cross-sectional regression.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    date_from = normalize_trade_date(args.date_from)
    date_to = normalize_trade_date(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")
    requested: list[str] = []
    if args.symbols_file is not None:
        requested.extend(read_symbols_file(args.symbols_file))
    if args.symbols:
        requested.extend(normalize_symbol(symbol) for symbol in args.symbols)
    symbols = list(dict.fromkeys(requested)) or None
    files = discover_input_files(args.input_root, symbols, args.limit)
    LOGGER.info("Extracting open-source factors for %s symbols", len(files))
    raw_panel, failures = build_raw_panel(files, max(1, int(args.workers)))
    factor_panel = build_open_source_factor_panel(
        raw_panel, min_tgd_cross_section=max(1, int(args.min_tgd_cross_section))
    )
    factor_panel = add_availability_columns(factor_panel)
    written = write_outputs(
        factor_panel,
        args.output_root,
        date_from=date_from,
        date_to=date_to,
        overwrite=args.overwrite,
    )
    LOGGER.info("Completed: %s files written, %s failures", len(written), len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
