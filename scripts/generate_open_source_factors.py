"""Generate daily ETF/stock proxies from open-source microstructure research."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from importlib.metadata import version
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
FACTOR_VERSION = "3"


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
                frame = _extract_one(path)
                if frame.empty:
                    raise ValueError("No daily raw features")
                frames.append(frame)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to extract %s", path)
                failures.append((normalize_symbol(path.name), str(exc)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_one, path): path for path in files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    frame = future.result()
                    if frame.empty:
                        raise ValueError("No daily raw features")
                    frames.append(frame)
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


@contextmanager
def _output_lock(output_root: Path):
    """Serialize writers; OS locks are released even if the process is killed."""
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.with_name(f".{output_root.name}.open-source.lock")
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _transaction_path(output_root: Path) -> Path:
    return output_root.with_name(f".{output_root.name}.open-source-transaction")


def _recover_outputs(output_root: Path) -> None:
    """Restore an interrupted directory swap, or clean up a completed one."""
    transaction = _transaction_path(output_root)
    if not transaction.exists():
        return
    # Validate ownership and resolved deletion/move targets before touching them.
    marker = transaction / "owner.json"
    if (
        transaction.resolve() != transaction.absolute()
        or not marker.is_file()
        or json.loads(marker.read_text(encoding="utf-8")) != str(output_root.resolve())
    ):
        raise ValueError(f"Unrecognized output transaction: {transaction}")
    backup = transaction / "backup"
    if backup.resolve().parent != transaction.resolve():
        raise ValueError(f"Invalid backup path: {backup}")
    if backup.exists() and not output_root.exists():
        backup.rename(output_root)
    shutil.rmtree(transaction)


def write_outputs(
    panel: pd.DataFrame,
    output_root: Path,
    *,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
    contract: dict,
    output_symbols: set[str],
) -> list[Path]:
    selected = select_output_columns(_filter_dates(panel, date_from, date_to))
    missing = output_symbols - set(selected["ts_code"])
    if missing:
        raise ValueError("No output rows in the requested date range for: " + ", ".join(sorted(missing)))
    transaction = _transaction_path(output_root)
    transaction.mkdir()
    (transaction / "owner.json").write_text(json.dumps(str(output_root.resolve())), encoding="utf-8")
    staging = transaction / "staging"
    written: list[Path] = []
    try:
        if output_root.exists():
            shutil.copytree(output_root, staging)
        else:
            staging.mkdir()
        for symbol, frame in selected.groupby("ts_code", sort=True):
            output_path = output_root / f"{symbol}.parquet"
            staged_path = staging / output_path.name
            if output_path.exists() and not overwrite:
                existing = pd.read_parquet(staged_path)
                pd.testing.assert_frame_equal(existing, frame.reset_index(drop=True))
                LOGGER.info("Skipping verified existing output: %s", output_path)
                continue
            frame = frame.reset_index(drop=True)
            if frame.duplicated(["ts_code", "trade_date"]).any():
                raise ValueError(f"Duplicate output keys: {symbol}")
            frame.to_parquet(staged_path, index=False)
            pd.testing.assert_frame_equal(pd.read_parquet(staged_path), frame)
            written.append(output_path)
        (staging / "manifest.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_root.exists():
            output_root.rename(transaction / "backup")
        staging.rename(output_root)
    finally:
        _recover_outputs(output_root)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--symbols", nargs="*", default=None, help="Output symbols only; all input symbols form the calculation universe.")
    parser.add_argument("--symbols-file", type=Path, default=None, help="Output symbols file; does not restrict the calculation universe.")
    parser.add_argument("--calculation-symbols-file", type=Path, default=None, help="Full calculation universe. When set, cross-sectional factors use only this file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit output symbols, not the calculation universe.")
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
    args.output_root = args.output_root.resolve()
    with _output_lock(args.output_root):
        _recover_outputs(args.output_root)
        return _generate(args)


def _generate(args: argparse.Namespace) -> int:
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
    output_symbols = {path.stem for path in discover_input_files(args.input_root, symbols, args.limit)}
    calculation_symbols = (
        read_symbols_file(args.calculation_symbols_file)
        if args.calculation_symbols_file is not None
        else None
    )
    files = discover_input_files(args.input_root, calculation_symbols)
    universe = [path.stem for path in files]
    missing = output_symbols - set(universe)
    if missing:
        raise ValueError("Output symbols outside calculation universe: " + ", ".join(sorted(missing)))
    contract = {
        "factor_version": FACTOR_VERSION,
        "input_root": str(resolve_input_root(args.input_root).resolve()),
        "calculation_universe": universe,
        "universe_sha256": hashlib.sha256("\n".join(universe).encode("utf-8")).hexdigest(),
        "min_tgd_cross_section": max(1, int(args.min_tgd_cross_section)),
        "calendar": "XSHG",
        "exchange_calendars_version": version("exchange-calendars"),
    }
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Output calculation contract changed; use a new --output-root to avoid mixing factors")
    elif any(args.output_root.glob("*.parquet")):
        raise ValueError("Existing output has no calculation manifest; use a new --output-root")
    LOGGER.info("Extracting open-source factors for %s symbols", len(files))
    raw_panel, failures = build_raw_panel(files, max(1, int(args.workers)))
    if failures:
        LOGGER.error("Aborting without writing: %s symbols failed in the calculation universe", len(failures))
        return 1
    factor_panel = build_open_source_factor_panel(
        raw_panel, min_tgd_cross_section=max(1, int(args.min_tgd_cross_section))
    )
    factor_panel = add_availability_columns(factor_panel)
    factor_panel = factor_panel.loc[factor_panel["ts_code"].isin(output_symbols)]
    written = write_outputs(
        factor_panel,
        args.output_root,
        date_from=date_from,
        date_to=date_to,
        overwrite=args.overwrite,
        contract=contract,
        output_symbols=output_symbols,
    )
    LOGGER.info("Completed: %s files written, %s failures", len(written), len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
