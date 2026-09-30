"""Generate mapped-index APM/OVP/AVP daily factors from minute parquet files."""

from __future__ import annotations

import argparse
import logging
import hashlib
import json
import tempfile
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"
for root in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from factor.open_source_crossmarket import (  # noqa: E402
    APM_FACTOR_COLUMNS,
    WINDOW,
    trading_dates,
    add_availability_columns,
    build_daily_apm_raw_features,
    build_open_source_apm_panel,
)

from factor.open_source import normalize_minute_frame

LOGGER = logging.getLogger("generate_open_source_apm_ovp")
DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_INDEX_ROOT = Path(r"D:\workspace\stockdata\指数数据\index_1min")
DEFAULT_MAPPING_PATH = Path(r"D:\workspace\stockdata\etf-data\exchange_fund_index_mapping.csv")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_open_source_apm_ovp")


def normalize_symbol(value: object) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"\.(PARQUET|CSV)$", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d{6}", text):
        return text + (".SH" if text.startswith(("5", "6")) else ".SZ")
    return text


def read_symbols_file(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            values.append(normalize_symbol(value))
    return list(dict.fromkeys(values))


def load_mapping(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str)
    fund_col = next((c for c in ("fund_code", "基金代码", "ts_code") if c in frame.columns), None)
    ref_col = next((c for c in ("reference_index_code", "tracking_index_code", "跟踪指数代码") if c in frame.columns), None)
    if fund_col is None or ref_col is None:
        raise ValueError(f"Mapping must contain fund and reference columns: {path}")
    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        fund = normalize_symbol(row[fund_col])
        reference = str(row[ref_col]).strip().upper()
        if reference and reference != "NAN":
            if fund in mapping and mapping[fund] != reference:
                raise ValueError(f"Conflicting reference indices for {fund}")
            mapping[fund] = reference
    return mapping


def find_parquet(root: Path, symbol: str) -> Path | None:
    direct = root / f"{symbol}.parquet"
    return direct if direct.is_file() else None


def find_index_parquet(root: Path, reference: str) -> Path | None:
    reference = str(reference).strip().upper()
    if "." in reference:
        return find_parquet(root, reference)
    candidates = sorted(root.glob(f"{reference}.*.parquet"))
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous unsuffixed index {reference}: {candidates}")
    return candidates[0] if candidates else None


def read_minutes(path: Path, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    schema = pq.read_schema(path)
    time_col = next((c for c in ("trade_time", "datetime") if c in schema.names and pa.types.is_timestamp(schema.field(c).type)), None)
    filters = []
    if time_col and start is not None:
        filters.append((time_col, ">=", start))
    if time_col and end is not None:
        filters.append((time_col, "<", end + pd.Timedelta(days=1)))
    frame = normalize_minute_frame(pd.read_parquet(path, filters=filters or None))
    if start is not None:
        frame = frame.loc[frame.index >= start]
    if end is not None:
        frame = frame.loc[frame.index < end + pd.Timedelta(days=1)]
    return frame


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_date(value: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(value).normalize() if value else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--universe-file", type=Path, help="Fixed calculation universe; defaults to all mapped funds. --symbols/--limit only select outputs.")
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for compatible batch invocation; processing is deterministic and sequential.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mapping = load_mapping(args.mapping)
    requested = [normalize_symbol(v) for v in (args.symbols or [])]
    if args.symbols_file:
        requested.extend(read_symbols_file(args.symbols_file))
    if requested:
        symbols = list(dict.fromkeys(requested))
    else:
        symbols = sorted(mapping)
    if args.limit is not None:
        symbols = symbols[: max(0, args.limit)]
    date_from = parse_date(args.date_from)
    date_to = parse_date(args.date_to)
    if date_from is not None and date_to is not None and date_from > date_to:
        parser.error("--date-from must not exceed --date-to")
    universe = read_symbols_file(args.universe_file) if args.universe_file else sorted(mapping)
    if not requested:
        symbols = universe[:max(0, args.limit)] if args.limit is not None else universe
    if not universe or not symbols or not set(symbols).issubset(universe):
        parser.error("Output symbols must be a nonempty subset of the fixed calculation universe")
    history_start = None
    if date_from is not None:
        history = trading_dates(date_from - pd.Timedelta(days=120), date_from)
        prior = history[history < date_from]
        if len(prior) < WINDOW:
            parser.error("Insufficient calendar history for warmup")
        history_start = prior[-WINDOW]
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / ".apm_generation.lock"
    try:
        lock = lock_path.open("x")
    except FileExistsError:
        LOGGER.error("Output directory is locked: %s", lock_path)
        return 1
    report = {"version": 2, "status": "failed", "failures": {}, "written": [],
              "mapping_pit": False, "mapping_policy": "static snapshot; ETF mapped-index adaptation"}
    try:
        lock.write(str(os.getpid()))
        lock.close()
        sources = {}
        for symbol in universe:
            reference = mapping.get(symbol)
            asset_path = find_parquet(args.input_root, symbol)
            index_path = find_index_parquet(args.index_root, reference) if reference else None
            if not reference or asset_path is None or index_path is None:
                report["failures"][symbol] = f"Missing mapping/input/index: {reference}"
            else:
                sources[symbol] = (reference, asset_path, index_path)
        if report["failures"]:
            raise ValueError("Incomplete fixed universe; see run_report.json")
        paths = sorted({path for _, asset, benchmark in sources.values() for path in (asset, benchmark)})
        config = {"version": 2, "universe": sorted(universe),
                  "date_from": str(date_from), "date_to": str(date_to),
                  "mapping_sha256": hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
                  "sources": {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}}
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        manifest_path = args.output_root / "manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        existing_symbols = {p.stem for p in args.output_root.glob("*.parquet")}
        if existing_symbols and existing.get("signature") != signature:
            if not (args.overwrite and existing_symbols.issubset(symbols)):
                raise ValueError("Existing outputs have incompatible/unverified provenance; use a new directory or overwrite every existing symbol")
        write_symbols = {symbol for symbol in symbols if args.overwrite or not (args.output_root / f"{symbol}.parquet").exists()}
        report.update({"signature": signature, "config": config, "requested": symbols,
                       "skipped": sorted(set(symbols) - write_symbols)})
        if not write_symbols:
            report["status"] = "complete"
            return 0
        # All universe members participate, including those with existing output.
        # A private staging directory plus an exclusive lock prevents writer races.
        with tempfile.TemporaryDirectory(prefix=".apm_", dir=args.output_root) as temporary:
            staging = Path(temporary)
            for symbol, (reference, asset_path, index_path) in sources.items():
                try:
                    raw = build_daily_apm_raw_features(
                        read_minutes(asset_path, history_start, date_to),
                        read_minutes(index_path, history_start, date_to), symbol, reference)
                    if raw.empty:
                        raise ValueError("No daily observations in the requested interval")
                    raw.to_parquet(staging / f"{symbol}.parquet", index=False)
                except Exception as exc:
                    report["failures"][symbol] = str(exc)
                    LOGGER.exception("failed %s", symbol)
            if report["failures"]:
                raise ValueError("Failed universe members; no factor outputs published")
            panel = pd.concat([pd.read_parquet(staging / f"{s}.parquet") for s in universe], ignore_index=True)
            result = add_availability_columns(build_open_source_apm_panel(panel))
            if date_from is not None:
                result = result.loc[result["trade_date"].ge(date_from)]
            if date_to is not None:
                result = result.loc[result["trade_date"].le(date_to)]
            report["coverage"] = {}
            for symbol in sorted(write_symbols):
                group = result.loc[result["ts_code"].eq(symbol)].sort_values("trade_date").reset_index(drop=True)
                valid = group[list(APM_FACTOR_COLUMNS)].notna().sum().astype(int).to_dict()
                report["coverage"][symbol] = {"rows": len(group), "first_date": str(group["trade_date"].min()),
                    "last_date": str(group["trade_date"].max()), "nonnull": valid}
                if group.empty or not any(valid.values()):
                    report["failures"][symbol] = "No valid factor values in requested output interval"
                    continue
                staged = staging / f"output_{symbol}.parquet"
                group.to_parquet(staged, index=False)
                pd.testing.assert_frame_equal(pd.read_parquet(staged), group)
            if report["failures"]:
                raise ValueError("Empty factor outputs; see coverage in run_report.json")
            current_sources = {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
            if current_sources != config["sources"] or hashlib.sha256(args.mapping.read_bytes()).hexdigest() != config["mapping_sha256"]:
                raise ValueError("Inputs changed during generation; no outputs published")
            for symbol in sorted(write_symbols):
                (staging / f"output_{symbol}.parquet").replace(args.output_root / f"{symbol}.parquet")
                report["written"].append(symbol)
        atomic_json(manifest_path, {"signature": signature, "config": config})
        report["status"] = "complete"
        LOGGER.info("wrote %d symbols to %s", len(report["written"]), args.output_root)
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        LOGGER.exception("APM generation failed")
        return 1
    finally:
        try:
            atomic_json(args.output_root / "run_report.json", report)
        finally:
            lock.close()
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
