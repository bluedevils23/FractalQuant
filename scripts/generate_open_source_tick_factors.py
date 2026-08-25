"""Generate daily transaction-tick factors from the local A-share tick archive."""

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

from factor.open_source_tick import (  # noqa: E402
    add_availability_columns,
    build_daily_flow_raw_features,
    build_daily_tick_raw_features,
    build_daily_order_raw_features,
    build_open_source_tick_factor_panel,
    load_order_file,
    load_tick_file,
    select_output_columns,
)


LOGGER = logging.getLogger("generate_open_source_tick_factors")
DEFAULT_INPUT_ROOT = Path(r"E:\逐笔数据")
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\stock-factors\open_source_tick_factors"
)
SYMBOL_PATTERN = re.compile(r"^\d{6}\.[A-Z]{2}$")
TRANSACTION_NAMES = ("逐笔成交.csv", "transactions.parquet")
ORDER_NAMES = ("逐笔委托.csv", "orders.parquet")


def normalize_symbol(value: str) -> str:
    symbol = str(value).strip()
    for suffix in (".parquet", ".csv"):
        if symbol.lower().endswith(suffix):
            symbol = symbol[: -len(suffix)]
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
    values: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if value:
                values.append(normalize_symbol(value))
    return list(dict.fromkeys(values))


def discover_symbol_files(
    input_root: Path,
    symbols: list[str] | None = None,
    limit: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, dict[str, list[Path]]]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    grouped: dict[str, dict[str, list[Path]]] = {}
    requested_set = set(symbols or [])
    date_dirs: list[Path] = []
    for year_dir in input_root.iterdir():
        if not year_dir.is_dir() or not re.fullmatch(r"\d{4}", year_dir.name):
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir() or not re.fullmatch(r"\d{6}", month_dir.name):
                continue
            for date_dir in month_dir.iterdir():
                if not date_dir.is_dir() or not re.fullmatch(r"\d{8}", date_dir.name):
                    continue
                if date_from and date_dir.name < date_from:
                    continue
                if date_to and date_dir.name > date_to:
                    continue
                date_dirs.append(date_dir)
    for date_dir in sorted(date_dirs):
        symbol_dirs = (
            [date_dir / symbol for symbol in sorted(requested_set)]
            if requested_set
            else sorted(path for path in date_dir.iterdir() if path.is_dir())
        )
        for symbol_dir in symbol_dirs:
            symbol = normalize_symbol(symbol_dir.name)
            if not SYMBOL_PATTERN.fullmatch(symbol):
                continue
            for name in (*TRANSACTION_NAMES, *ORDER_NAMES):
                path = symbol_dir / name
                if path.is_file():
                    kind = "transactions" if name in TRANSACTION_NAMES else "orders"
                    grouped.setdefault(symbol, {"transactions": [], "orders": []})[kind].append(path)
    selected = list(symbols or sorted(grouped))
    missing = [symbol for symbol in selected if symbol not in grouped]
    if missing:
        raise FileNotFoundError("Missing tick files for: " + ", ".join(missing[:10]))
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    result = {symbol: grouped[symbol] for symbol in selected}
    if not result:
        raise FileNotFoundError(f"No transaction tick files found under {input_root}")
    return result


def _extract_one(item: tuple[str, dict[str, list[Path]]]) -> pd.DataFrame:
    symbol, paths = item
    transaction_frames = [load_tick_file(path, symbol) for path in paths["transactions"]]
    transaction_frames = [frame for frame in transaction_frames if not frame.empty]
    order_frames = [load_order_file(path, symbol) for path in paths["orders"]]
    order_frames = [frame for frame in order_frames if not frame.empty]
    if not transaction_frames and not order_frames:
        return pd.DataFrame()
    if transaction_frames:
        transactions = pd.concat(transaction_frames, ignore_index=True)
        result = build_daily_tick_raw_features(transactions, symbol)
    else:
        transactions = pd.DataFrame()
        result = pd.DataFrame(columns=["trade_date", "ts_code"])
    if order_frames:
        orders = pd.concat(order_frames, ignore_index=True)
        order_result = build_daily_order_raw_features(orders, symbol)
        result = result.merge(order_result, on=["trade_date", "ts_code"], how="outer")
        if not transactions.empty:
            flow_result = build_daily_flow_raw_features(transactions, orders, symbol)
            result = result.merge(flow_result, on=["trade_date", "ts_code"], how="left")
    return result.sort_values("trade_date").reset_index(drop=True)


def build_raw_panel(
    files: dict[str, dict[str, list[Path]]], workers: int
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    items = list(files.items())
    if workers <= 1:
        iterator = ((item, _extract_one(item)) for item in items)
        for (symbol, _), result in iterator:
            if result.empty:
                failures.append((symbol, "no valid transaction rows"))
            else:
                frames.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_one, item): item[0] for item in items}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result.empty:
                        failures.append((symbol, "no valid transaction rows"))
                    else:
                        frames.append(result)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Failed to extract %s", symbol)
                    failures.append((symbol, str(exc)))
    if not frames:
        raise RuntimeError("No symbol produced daily transaction features")
    return pd.concat(frames, ignore_index=True), failures


def write_outputs(
    panel: pd.DataFrame,
    output_root: Path,
    *,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
) -> list[Path]:
    selected = select_output_columns(panel)
    if date_from:
        selected = selected.loc[selected["trade_date"].dt.strftime("%Y%m%d").ge(date_from)]
    if date_to:
        selected = selected.loc[selected["trade_date"].dt.strftime("%Y%m%d").le(date_to)]
    written: list[Path] = []
    for symbol, frame in selected.groupby("ts_code", sort=True):
        output_path = output_root / f"{symbol}.parquet"
        if output_path.exists() and not overwrite:
            LOGGER.info("Skipping existing output: %s", output_path)
            continue
        output_root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        written.append(output_path)
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
        "--workers", type=int, default=min(4, os.cpu_count() or 1)
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
        requested.extend(normalize_symbol(value) for value in args.symbols)
    symbols = list(dict.fromkeys(requested)) or None
    # Keep history before --date-from so the 20-day rolling factors retain a
    # causal warm-up window; date_from is applied only when writing outputs.
    files = discover_symbol_files(args.input_root, symbols, args.limit, None, date_to)
    LOGGER.info("Extracting transaction factors for %s symbols", len(files))
    raw_panel, failures = build_raw_panel(files, max(1, int(args.workers)))
    factor_panel = add_availability_columns(build_open_source_tick_factor_panel(raw_panel))
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
