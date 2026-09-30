"""Generate daily transaction-tick factors from the local A-share tick archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
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
DEFAULT_DAILY_PATH = Path(
    r"D:\workspace\stockdata\stock-data\行情数据\stock_daily.parquet"
)
SYMBOL_PATTERN = re.compile(r"^\d{6}\.[A-Z]{2}$")
TRANSACTION_NAMES = ("逐笔成交.csv", "transactions.parquet")
ORDER_NAMES = ("逐笔委托.csv", "orders.parquet")
PROGRESS_EVERY = 25
WRITE_PROGRESS_EVERY = 50
FACTOR_VERSION = "2"


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _log_extraction_progress(
    completed: int,
    total: int,
    failures: int,
    started: float,
) -> None:
    if completed not in {1, total} and completed % PROGRESS_EVERY:
        return
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else None
    eta = _format_duration(remaining) if remaining is not None else "unknown"
    LOGGER.info(
        "Raw extraction progress: %s/%s (%.1f%%), failures=%s, elapsed=%s, ETA=%s",
        completed,
        total,
        completed / total * 100.0 if total else 100.0,
        failures,
        _format_duration(elapsed),
        eta,
    )


def normalize_symbol(value: str) -> str:
    symbol = str(value).strip().upper()
    for suffix in (".parquet", ".csv"):
        if symbol.lower().endswith(suffix):
            symbol = symbol[: -len(suffix)]
    return symbol


def numeric_code(value: str) -> str | None:
    match = re.match(r"^(\d{6})", str(value).strip())
    return match.group(1) if match else None


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
    requested_symbols = list(dict.fromkeys(normalize_symbol(symbol) for symbol in (symbols or [])))
    requested_by_code: dict[str, str] = {}
    for symbol in requested_symbols:
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")
        code = numeric_code(symbol)
        if code is None:
            raise ValueError(f"Invalid symbol: {symbol}")
        previous = requested_by_code.setdefault(code, symbol)
        if previous != symbol:
            raise ValueError(
                f"Multiple requested names share numeric code {code}: {previous}, {symbol}"
            )
    grouped: dict[str, dict[str, list[Path]]] = {
        symbol: {"transactions": [], "orders": []} for symbol in requested_symbols
    }
    requested_codes = set(requested_by_code)
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
        if requested_codes:
            # The requested universe already carries canonical exchange suffixes
            # (ETF lists use .SH/.SZ).  Probe those paths directly instead of
            # enumerating every symbol directory on the archive volume.
            for code, requested_symbol in requested_by_code.items():
                suffixes = [requested_symbol[6:]] + [
                    suffix for suffix in (".SH", ".SZ", ".BJ")
                    if suffix != requested_symbol[6:]
                ]
                candidates = [
                    date_dir / f"{code}{suffix}"
                    for suffix in suffixes
                    if (date_dir / f"{code}{suffix}").is_dir()
                ]
                if not candidates:
                    continue
                symbol_dir = sorted(candidates)[0]
                symbol = requested_symbol
                for name in (*TRANSACTION_NAMES, *ORDER_NAMES):
                    path = symbol_dir / name
                    if path.is_file():
                        kind = "transactions" if name in TRANSACTION_NAMES else "orders"
                        grouped[symbol][kind].append(path)
            continue
        matches_by_code: dict[str, list[Path]] = {}
        for symbol_dir in date_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            code = numeric_code(symbol_dir.name)
            if code is None or (requested_codes and code not in requested_codes):
                continue
            matches_by_code.setdefault(code, []).append(symbol_dir)
        for code, candidates in matches_by_code.items():
            # Match auction-factor discovery: select one deterministic directory
            # for a numeric code, regardless of the source exchange suffix.
            symbol_dir = sorted(candidates)[0]
            symbol = requested_by_code.get(code, normalize_symbol(symbol_dir.name))
            for name in (*TRANSACTION_NAMES, *ORDER_NAMES):
                path = symbol_dir / name
                if path.is_file():
                    kind = "transactions" if name in TRANSACTION_NAMES else "orders"
                    grouped.setdefault(symbol, {"transactions": [], "orders": []})[kind].append(path)
    selected = requested_symbols or sorted(grouped)
    missing = [
        symbol
        for symbol in selected
        if not grouped[symbol]["transactions"] and not grouped[symbol]["orders"]
    ]
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
    # Raw archives are organized as one symbol directory per trading date.
    # Process one directory at a time so a two-year symbol never occupies the
    # worker's memory as a single concatenated tick table.
    by_day: dict[Path, dict[str, list[Path]]] = {}
    for kind in ("transactions", "orders"):
        for path in paths[kind]:
            by_day.setdefault(path.parent, {"transactions": [], "orders": []})[kind].append(path)

    daily_results: list[pd.DataFrame] = []
    for day_dir in sorted(by_day):
        transaction_frames = []
        for path in by_day[day_dir]["transactions"]:
            frame = load_tick_file(path, symbol)
            if not frame.empty:
                # The universe name is canonical.  Source CSVs may contain a
                # stale or exchange-mismatched code in their first column.
                frame["ts_code"] = symbol
                transaction_frames.append(frame)
        order_frames = []
        for path in by_day[day_dir]["orders"]:
            frame = load_order_file(path, symbol)
            if not frame.empty:
                frame["ts_code"] = symbol
                order_frames.append(frame)
        if not transaction_frames and not order_frames:
            continue

        if transaction_frames:
            transactions = pd.concat(transaction_frames, ignore_index=True)
            result = build_daily_tick_raw_features(transactions, symbol, normalized=True)
        else:
            transactions = pd.DataFrame()
            result = pd.DataFrame(columns=["trade_date", "ts_code"])

        if order_frames:
            orders = pd.concat(order_frames, ignore_index=True)
            order_result = build_daily_order_raw_features(orders, symbol, normalized=True)
            result = result.merge(order_result, on=["trade_date", "ts_code"], how="outer")
            if not transactions.empty:
                flow_result = build_daily_flow_raw_features(
                    transactions, orders, symbol, normalized=True
                )
                result = result.merge(flow_result, on=["trade_date", "ts_code"], how="left")
        if not result.empty:
            daily_results.append(result)

    if not daily_results:
        return pd.DataFrame()
    return pd.concat(daily_results, ignore_index=True).sort_values(
        "trade_date", kind="mergesort"
    ).reset_index(drop=True)


def build_raw_panel(
    files: dict[str, dict[str, list[Path]]], workers: int
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    items = list(files.items())
    started = time.monotonic()
    completed = 0
    total = len(items)
    LOGGER.info("Raw extraction started: %s symbols, workers=%s", total, workers)
    if workers <= 1:
        for item in items:
            symbol = item[0]
            try:
                result = _extract_one(item)
                if result.empty:
                    failures.append((symbol, "no valid transaction or order rows"))
                else:
                    frames.append(result)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to extract %s", symbol)
                failures.append((symbol, str(exc)))
            completed += 1
            _log_extraction_progress(completed, total, len(failures), started)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_one, item): item[0] for item in items}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result.empty:
                        failures.append((symbol, "no valid transaction or order rows"))
                    else:
                        frames.append(result)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Failed to extract %s", symbol)
                    failures.append((symbol, str(exc)))
                completed += 1
                _log_extraction_progress(completed, total, len(failures), started)
    if not frames:
        raise RuntimeError("No symbol produced daily transaction features")
    return pd.concat(frames, ignore_index=True), failures


def load_daily_adjustment_factors(
    path: Path,
    symbols: list[str],
    date_to: str | None,
) -> pd.DataFrame:
    """Load cumulative daily adjustment factors for the extracted tick panel."""

    if not path.is_file():
        raise FileNotFoundError(f"Daily adjustment source does not exist: {path}")
    filters: list[tuple[str, str, object]] = [("ts_code", "in", symbols)]
    if date_to:
        filters.append(("trade_date", "<=", pd.Timestamp(date_to)))
    factors = pd.read_parquet(path, columns=["adj_factor"], filters=filters).reset_index()
    required = {"trade_date", "ts_code", "adj_factor"}
    missing = sorted(required - set(factors.columns))
    if missing:
        raise ValueError(f"Daily adjustment source is missing columns: {missing}")
    factors["trade_date"] = pd.to_datetime(
        factors["trade_date"], errors="coerce"
    ).dt.normalize()
    factors["ts_code"] = factors["ts_code"].astype(str)
    factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
    factors.loc[factors["adj_factor"].le(0), "adj_factor"] = pd.NA
    return factors.dropna(subset=["trade_date", "ts_code"]).drop_duplicates(
        ["trade_date", "ts_code"], keep="last"
    )


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
    total_symbols = int(selected["ts_code"].nunique())
    started = time.monotonic()
    for completed, (symbol, frame) in enumerate(
        selected.groupby("ts_code", sort=True), start=1
    ):
        output_path = output_root / f"{symbol}.parquet"
        if output_path.exists() and not overwrite:
            LOGGER.info("Skipping existing output: %s", output_path)
            written_count = len(written)
        else:
            output_root.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output_path, index=False)
            written.append(output_path)
            written_count = len(written)
        if completed == 1 or completed == total_symbols or completed % WRITE_PROGRESS_EVERY == 0:
            LOGGER.info(
                "Output progress: %s/%s symbols (%.1f%%), files_written=%s, elapsed=%s",
                completed,
                total_symbols,
                completed / total_symbols * 100.0 if total_symbols else 100.0,
                written_count,
                _format_duration(time.monotonic() - started),
            )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--daily-path", type=Path, default=DEFAULT_DAILY_PATH)
    parser.add_argument("--symbols", nargs="*", default=None, help="Output symbols only; all discovered symbols form the calculation universe.")
    parser.add_argument("--symbols-file", type=Path, default=None, help="Output symbols file, not the calculation universe.")
    parser.add_argument("--limit", type=int, default=None, help="Limit output symbols, not the calculation universe.")
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
    files = discover_symbol_files(args.input_root, None, None, None, date_to)
    output_symbols = symbols or sorted(files)
    missing = sorted(set(output_symbols) - set(files))
    if missing:
        raise ValueError("Output symbols absent from calculation universe: " + ", ".join(missing[:10]))
    if args.limit is not None:
        output_symbols = output_symbols[:max(0, args.limit)]
    if not output_symbols:
        raise ValueError("No output symbols selected")
    universe = sorted(files)
    contract = {
        "factor_version": FACTOR_VERSION,
        "input_root": str(args.input_root.resolve()),
        "daily_path": str(args.daily_path.resolve()),
        "calculation_universe": universe,
        "universe_sha256": hashlib.sha256("\n".join(universe).encode("utf-8")).hexdigest(),
        "calendar": "XSHG",
        "exchange_calendars_version": version("exchange-calendars"),
        "price_units": "archive=1/10000 yuan; canonical=yuan; quantity=shares",
    }
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Output calculation contract changed; use a new --output-root")
    elif any(args.output_root.glob("*.parquet")):
        raise ValueError("Existing output has no calculation manifest; use a new --output-root")
    LOGGER.info(
        "Discovered %s symbols, %s transaction files, %s order files",
        len(files),
        sum(len(paths["transactions"]) for paths in files.values()),
        sum(len(paths["orders"]) for paths in files.values()),
    )
    LOGGER.info("Extracting transaction factors for %s symbols", len(files))
    raw_started = time.monotonic()
    raw_panel, failures = build_raw_panel(files, max(1, int(args.workers)))
    if failures:
        LOGGER.error("Aborting without writing: %s symbols failed in the calculation universe", len(failures))
        return 1
    if "daily_close" in raw_panel and raw_panel["daily_close"].notna().any():
        adjustments = load_daily_adjustment_factors(args.daily_path, sorted(files), date_to)
        raw_panel = raw_panel.merge(
            adjustments, on=["trade_date", "ts_code"], how="left", validate="many_to_one"
        )
    else:
        # Order-memory factors do not need transaction prices or adjustments.
        raw_panel["adj_factor"] = float("nan")
    missing_adjustments = int(raw_panel["adj_factor"].isna().sum())
    if missing_adjustments:
        LOGGER.warning(
            "Missing daily adjustment factors for %s/%s raw rows; cross-day price factors will be null",
            missing_adjustments,
            len(raw_panel),
        )
    LOGGER.info(
        "Raw extraction complete: %s rows, %s failures, elapsed=%s",
        len(raw_panel),
        len(failures),
        _format_duration(time.monotonic() - raw_started),
    )
    factor_started = time.monotonic()
    LOGGER.info("Building cross-sectional tick factors")
    factor_panel = add_availability_columns(build_open_source_tick_factor_panel(raw_panel))
    factor_panel = factor_panel.loc[factor_panel["ts_code"].isin(output_symbols)]
    LOGGER.info(
        "Cross-sectional factor build complete: %s rows, elapsed=%s",
        len(factor_panel),
        _format_duration(time.monotonic() - factor_started),
    )
    output_started = time.monotonic()
    written = write_outputs(
        factor_panel,
        args.output_root,
        date_from=date_from,
        date_to=date_to,
        overwrite=args.overwrite,
    )
    if written:
        manifest_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Output write complete: %s files, elapsed=%s",
        len(written),
        _format_duration(time.monotonic() - output_started),
    )
    LOGGER.info("Completed: %s files written, %s failures", len(written), len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
