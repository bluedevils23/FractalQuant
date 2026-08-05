"""Import exported index minute Excel files into the local parquet index store."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_ROOT = Path(r"D:\workspace\stockdata\指数分钟数据")
DEFAULT_MAPPING_PATH = Path(
    r"D:\workspace\stockdata\etf-data\exchange_fund_index_mapping.csv"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\指数数据\index_1min")
REQUIRED_MAPPING_COLUMNS = ("reference_index_code",)
SOURCE_FILE_PATTERN = re.compile(r"K线导出_(.+)_1分钟线数据\.xlsx$", re.IGNORECASE)
REFERENCE_DATA_ALIASES = {
    "931573CNY00.CSI": "931573.CSI",
}
LOGGER = logging.getLogger(__name__)


def normalize_reference_code(value: object) -> str:
    code = str(value).strip().upper()
    if not code or code in {"NAN", "NONE"}:
        raise ValueError("Reference index code is empty")
    return code


def canonical_reference_code(reference_code: str) -> str:
    normalized = normalize_reference_code(reference_code)
    return REFERENCE_DATA_ALIASES.get(normalized, normalized)


def build_output_code_lookup(mapping_path: Path) -> dict[str, str]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file does not exist: {mapping_path}")
    mapping = pd.read_csv(mapping_path, dtype=str)
    missing = [
        column
        for column in REQUIRED_MAPPING_COLUMNS
        if column not in mapping.columns
    ]
    if missing:
        raise ValueError(f"Mapping file is missing columns: {missing}")

    codes_by_stem: dict[str, set[str]] = defaultdict(set)
    for value in mapping["reference_index_code"].dropna():
        code = canonical_reference_code(value)
        codes_by_stem[code.rsplit(".", 1)[0]].add(code)

    ambiguous = {
        stem: sorted(codes)
        for stem, codes in codes_by_stem.items()
        if len(codes) > 1
    }
    if ambiguous:
        raise ValueError(f"Ambiguous mapped reference stems: {ambiguous}")
    return {
        stem: next(iter(codes))
        for stem, codes in codes_by_stem.items()
    }


def source_stem_from_path(path: Path) -> str | None:
    match = SOURCE_FILE_PATTERN.fullmatch(path.name)
    return match.group(1).upper() if match else None


def normalize_source_frame(raw_frame: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    if raw_frame.shape[1] < 11:
        raise ValueError(
            f"Expected at least 11 Excel columns, found {raw_frame.shape[1]}"
        )

    trade_time = pd.to_datetime(raw_frame.iloc[:, 2], errors="coerce")
    frame = pd.DataFrame(
        {
            "ts_code": ts_code,
            "open": pd.to_numeric(raw_frame.iloc[:, 3], errors="coerce").to_numpy(),
            "high": pd.to_numeric(raw_frame.iloc[:, 4], errors="coerce").to_numpy(),
            "low": pd.to_numeric(raw_frame.iloc[:, 5], errors="coerce").to_numpy(),
            "close": pd.to_numeric(raw_frame.iloc[:, 6], errors="coerce").to_numpy(),
            "vol": pd.to_numeric(raw_frame.iloc[:, 9], errors="coerce").to_numpy(),
            "amount": pd.to_numeric(raw_frame.iloc[:, 10], errors="coerce").to_numpy(),
        }
    )
    frame.index = pd.DatetimeIndex(trade_time)
    frame.index.name = "trade_time"
    frame = frame.loc[frame.index.notna()].copy()
    frame = frame.loc[
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    ]
    frame["vol"] = frame["vol"].where(frame["vol"].ge(0))
    frame["amount"] = frame["amount"].where(frame["amount"].ge(0))
    frame = frame.sort_index().loc[~frame.index.duplicated(keep="last")]
    if frame.empty:
        raise ValueError("No valid minute rows after normalization")

    frame["trade_date"] = frame.index.normalize()
    return frame.set_index("trade_date", append=True).reorder_levels(
        ["trade_date", "trade_time"]
    )


def import_source_file(
    source_path: Path,
    output_path: Path,
    ts_code: str,
    overwrite: bool,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "output_path": output_path}

    normalized = normalize_source_frame(pd.read_excel(source_path), ts_code)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path)
    return {
        "status": "written",
        "output_path": output_path,
        "rows": len(normalized),
        "date_from": normalized.index.get_level_values("trade_date").min(),
        "date_to": normalized.index.get_level_values("trade_date").max(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if not args.source_root.exists():
        raise FileNotFoundError(
            f"Excel minute source root does not exist: {args.source_root}"
        )

    output_codes = build_output_code_lookup(args.mapping)
    selected: dict[str, Path] = {}
    ignored = 0
    for source_path in sorted(args.source_root.glob("*.xlsx")):
        source_stem = source_stem_from_path(source_path)
        if source_stem is None or source_stem not in output_codes:
            ignored += 1
            continue
        output_code = output_codes[source_stem]
        existing = selected.get(output_code)
        if existing is not None:
            raise ValueError(
                f"Multiple Excel files resolve to {output_code}: "
                f"{existing.name}, {source_path.name}"
            )
        selected[output_code] = source_path

    failures = 0
    for output_code, source_path in selected.items():
        try:
            result = import_source_file(
                source_path,
                args.output_root / f"{output_code}.parquet",
                output_code,
                args.overwrite,
            )
            LOGGER.info(
                "%s %s from %s%s",
                result["status"].capitalize(),
                output_code,
                source_path.name,
                f" ({result['rows']} rows)"
                if result["status"] == "written"
                else "",
            )
        except Exception:  # noqa: BLE001
            failures += 1
            LOGGER.exception("Failed to import %s", source_path)

    LOGGER.info(
        "Selected %s mapped files; ignored %s unmatched files; failures %s",
        len(selected),
        ignored,
        failures,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
