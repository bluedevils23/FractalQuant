"""Generate mapped-index APM/OVP/AVP daily factors from minute parquet files."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"
for root in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from factor.open_source_crossmarket import (  # noqa: E402
    add_availability_columns,
    build_daily_apm_raw_features,
    build_open_source_apm_panel,
)

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
            mapping[fund] = reference
    return mapping


def find_parquet(root: Path, symbol: str) -> Path | None:
    direct = root / f"{symbol}.parquet"
    if direct.exists():
        return direct
    code = symbol.split(".", 1)[0]
    for path in root.glob(f"{code}.*.parquet"):
        return path
    return None


def find_index_parquet(root: Path, reference: str) -> Path | None:
    reference = str(reference).strip().upper()
    code = reference.split(".", 1)[0]
    candidates = [reference, f"{code}.SH", f"{code}.SZ", f"{code}.CSI", f"{code}.CFE"]
    for candidate in candidates:
        path = root / f"{candidate}.parquet"
        if path.exists():
            return path
    for path in root.glob(f"{code}.*.parquet"):
        return path
    return None


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
    args.output_root.mkdir(parents=True, exist_ok=True)
    for stale in args.output_root.glob("._raw_*.parquet"):
        stale.unlink(missing_ok=True)
    written = 0
    for symbol in symbols:
        output_path = args.output_root / f"{symbol}.parquet"
        if output_path.exists() and not args.overwrite:
            LOGGER.info("skip existing %s", output_path.name)
            continue
        reference = mapping.get(symbol)
        asset_path = find_parquet(args.input_root, symbol)
        index_path = find_index_parquet(args.index_root, reference) if reference else None
        if not reference or asset_path is None or index_path is None:
            LOGGER.warning("skip %s: mapping or parquet missing (reference=%s)", symbol, reference)
            continue
        try:
            asset = pd.read_parquet(asset_path)
            benchmark = pd.read_parquet(index_path)
            if date_from is not None or date_to is not None:
                def _slice_dates(frame: pd.DataFrame) -> pd.DataFrame:
                    if isinstance(frame.index, pd.MultiIndex):
                        names = set(frame.index.names)
                        date_values = frame.index.get_level_values("trade_date") if "trade_date" in names else None
                        if date_values is not None:
                            dates = pd.to_datetime(date_values, errors="coerce").normalize()
                            mask = pd.Series(True, index=frame.index)
                            if date_from is not None:
                                mask &= dates >= date_from - pd.Timedelta(days=30)
                            if date_to is not None:
                                mask &= dates <= date_to
                            return frame.loc[mask.to_numpy()]
                    if "trade_time" in frame.columns:
                        dates = pd.to_datetime(frame["trade_time"], errors="coerce").dt.normalize()
                        mask = pd.Series(True, index=frame.index)
                        if date_from is not None:
                            mask &= dates >= date_from - pd.Timedelta(days=30)
                        if date_to is not None:
                            mask &= dates <= date_to
                        return frame.loc[mask]
                    return frame
                asset = _slice_dates(asset)
                benchmark = _slice_dates(benchmark)
            raw = build_daily_apm_raw_features(asset, benchmark, symbol, reference)
            if date_from is not None:
                raw = raw.loc[pd.to_datetime(raw["trade_date"]).ge(date_from)]
            if date_to is not None:
                raw = raw.loc[pd.to_datetime(raw["trade_date"]).le(date_to)]
            if raw.empty:
                continue
            # Cross-sectional residualization must see all selected symbols.
            # The single-symbol files are therefore staged and finalized below.
            raw.to_parquet(args.output_root / f"._raw_{symbol}.parquet", index=False)
        except Exception as exc:  # pragma: no cover - batch diagnostics
            LOGGER.exception("failed %s: %s", symbol, exc)
    raw_paths = sorted(args.output_root.glob("._raw_*.parquet"))
    if raw_paths:
        panel = pd.concat([pd.read_parquet(path) for path in raw_paths], ignore_index=True)
        result = add_availability_columns(build_open_source_apm_panel(panel))
        for symbol, group in result.groupby("ts_code", sort=True):
            output_path = args.output_root / f"{symbol}.parquet"
            group.sort_values("trade_date", kind="mergesort").to_parquet(output_path, index=False)
            written += 1
        for path in raw_paths:
            path.unlink(missing_ok=True)
    LOGGER.info("wrote %d symbols to %s", written, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
