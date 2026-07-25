from __future__ import annotations

"""Generate ETF minute CrossMarket factors using mapped reference indices."""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

for import_root in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from factor.advanced_runtime import (  # noqa: E402
    normalize_trade_date_arg,
    read_symbol_list_file,
)
from factor.crossmarket import (  # noqa: E402
    ArbitrageOpportunityFactor,
    CointegrationFactor,
    CrossMarketCoherenceFactor,
    CrossMarketCopulaFactor,
    CrossMarketCorrelationFactor,
    CrossMarketDynamicCorrelationFactor,
    CrossMarketEntropyFactor,
    CrossMarketGrangerFactor,
    CrossMarketInformationFlowFactor,
    CrossMarketJointDistributionFactor,
    CrossMarketMultiscaleCorrelationFactor,
    CrossMarketPhaseSynchronizationFactor,
    CrossMarketVolatilityFactor,
    MarketLinkageFactor,
    MarketRegimeSwitchFactor,
    RelativeStrengthFactor,
)
from scripts.generate_etf_minute_factors import (  # noqa: E402
    normalize_minute_frame,
)


LOGGER = logging.getLogger("generate_etf_crossmarket_factors")

DEFAULT_MAPPING_PATH = Path(
    r"D:\workspace\stockdata\etf-data\exchange_fund_index_mapping.csv"
)
DEFAULT_ETF_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_INDEX_ROOT = Path(r"D:\workspace\stockdata\index-data\index_1min")
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_crossmarket_factors"
)
REQUIRED_MAPPING_COLUMNS = ("fund_code", "reference_index_code")


@dataclass(frozen=True)
class MappingRecord:
    fund_code: str
    reference_index_code: str


def build_crossmarket_factors() -> tuple[object, ...]:
    return (
        CrossMarketCorrelationFactor(),
        ArbitrageOpportunityFactor(),
        MarketLinkageFactor(),
        RelativeStrengthFactor(),
        CointegrationFactor(),
        CrossMarketVolatilityFactor(),
        MarketRegimeSwitchFactor(),
        CrossMarketEntropyFactor(),
        CrossMarketCoherenceFactor(),
        CrossMarketGrangerFactor(),
        CrossMarketJointDistributionFactor(),
        CrossMarketCopulaFactor(),
        CrossMarketPhaseSynchronizationFactor(),
        CrossMarketInformationFlowFactor(),
        CrossMarketMultiscaleCorrelationFactor(),
        CrossMarketDynamicCorrelationFactor(),
    )


FACTOR_COLUMNS = tuple(
    factor.name for factor in build_crossmarket_factors()
)


def normalize_fund_code(value: object) -> str:
    code = str(value).strip()
    if code.lower().endswith(".parquet"):
        code = code[:-8]
    code = code.split(".", 1)[0]
    if not code.isdigit():
        raise ValueError(f"Invalid ETF fund code: {value}")
    return code.zfill(6)


def normalize_reference_code(value: object) -> str:
    code = str(value).strip().upper()
    if not code or code in {"NAN", "NONE"}:
        raise ValueError("Reference index code is empty")
    return code


def load_mapping_records(mapping_path: Path) -> dict[str, MappingRecord]:
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

    records: dict[str, MappingRecord] = {}
    for row in mapping.loc[:, REQUIRED_MAPPING_COLUMNS].itertuples(
        index=False
    ):
        try:
            fund_code = normalize_fund_code(row.fund_code)
            reference_code = normalize_reference_code(
                row.reference_index_code
            )
        except ValueError:
            continue

        record = MappingRecord(fund_code, reference_code)
        existing = records.get(fund_code)
        if existing is not None and existing != record:
            raise ValueError(
                f"Conflicting reference mappings for ETF {fund_code}: "
                f"{existing.reference_index_code} and {reference_code}"
            )
        records[fund_code] = record

    if not records:
        raise ValueError(f"No valid mapping records found in: {mapping_path}")
    return records


def discover_etf_files(etf_root: Path) -> dict[str, Path]:
    if not etf_root.exists():
        raise FileNotFoundError(f"ETF minute root does not exist: {etf_root}")

    files: dict[str, Path] = {}
    for path in sorted(etf_root.glob("*.parquet")):
        fund_code = normalize_fund_code(path.name)
        if fund_code in files:
            raise ValueError(
                f"Multiple ETF minute files share fund code {fund_code}"
            )
        files[fund_code] = path
    if not files:
        raise FileNotFoundError(
            f"No ETF minute parquet files found in: {etf_root}"
        )
    return files


def build_index_file_lookup(
    index_root: Path,
) -> tuple[dict[str, Path], dict[str, tuple[Path, ...]]]:
    if not index_root.exists():
        raise FileNotFoundError(
            f"Index minute root does not exist: {index_root}"
        )

    by_full_code: dict[str, Path] = {}
    paths_by_stem: dict[str, list[Path]] = {}
    for path in sorted(index_root.glob("*.parquet")):
        full_code = path.stem.upper()
        by_full_code[full_code] = path
        code_stem = full_code.rsplit(".", 1)[0]
        paths_by_stem.setdefault(code_stem, []).append(path)

    if not by_full_code:
        raise FileNotFoundError(
            f"No index minute parquet files found in: {index_root}"
        )
    return (
        by_full_code,
        {
            code: tuple(paths)
            for code, paths in paths_by_stem.items()
        },
    )


def resolve_reference_path(
    reference_code: str,
    by_full_code: dict[str, Path],
    paths_by_stem: dict[str, tuple[Path, ...]],
) -> tuple[Path | None, str]:
    normalized = normalize_reference_code(reference_code)
    exact = by_full_code.get(normalized)
    if exact is not None:
        return exact, "exact"

    code_stem = normalized.rsplit(".", 1)[0]
    candidates = paths_by_stem.get(code_stem, ())
    if len(candidates) == 1:
        return candidates[0], "stem"
    return None, "missing"


def filter_date_range(
    frame: pd.DataFrame,
    date_from: str | None,
    date_to: str | None,
) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return frame

    trade_dates = frame.index.strftime("%Y%m%d")
    mask = np.ones(len(frame), dtype=bool)
    if date_from is not None:
        mask &= trade_dates >= date_from
    if date_to is not None:
        mask &= trade_dates <= date_to
    return frame.loc[mask].copy()


def normalize_close_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    if isinstance(frame.index, pd.MultiIndex) and (
        "trade_time" in frame.index.names
    ):
        frame.index = pd.to_datetime(
            frame.index.get_level_values("trade_time")
        )
    elif "trade_time" in frame.columns:
        frame.index = pd.to_datetime(frame.pop("trade_time"))
    elif "datetime" in frame.columns:
        frame.index = pd.to_datetime(frame.pop("datetime"))
    else:
        raise ValueError(
            "Cannot locate trade_time/datetime index or column."
        )

    if "close" not in frame.columns:
        raise ValueError("Reference minute frame is missing close")
    frame.index.name = "trade_time"
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_index()
    return frame.loc[~frame.index.duplicated(keep="last"), ["close"]]


def read_reference_close_frame(
    reference_path: Path,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
) -> pd.DataFrame:
    filters = [
        ("trade_date", ">=", date_from.normalize()),
        ("trade_date", "<=", date_to.normalize()),
    ]
    try:
        raw_frame = pd.read_parquet(
            reference_path,
            columns=["close"],
            filters=filters,
        )
    except (TypeError, ValueError, NotImplementedError):
        raw_frame = pd.read_parquet(reference_path, columns=["close"])
    return normalize_close_frame(raw_frame)


def _calculate_day_factors(
    etf_day: pd.DataFrame,
    reference_day: pd.DataFrame,
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan,
        index=etf_day.index,
        columns=FACTOR_COLUMNS,
        dtype=float,
    )
    aligned_reference = reference_day.reindex(etf_day.index)
    current_close = pd.to_numeric(etf_day["close"], errors="coerce")
    reference_close = pd.to_numeric(
        aligned_reference["close"], errors="coerce"
    )
    valid = (
        current_close.notna()
        & reference_close.notna()
        & current_close.gt(0)
        & reference_close.gt(0)
    )
    if valid.sum() < 50:
        return result

    current_input = etf_day.loc[valid]
    reference_input = aligned_reference.loc[valid]
    for factor in build_crossmarket_factors():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = factor.calculate(current_input, reference_input)
        series = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        result.loc[series.index, factor.name] = series
    return result


def calculate_crossmarket_factor_frame(
    etf_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.DataFrame:
    if "close" not in etf_frame.columns:
        raise ValueError("ETF minute frame is missing close")
    if "close" not in reference_frame.columns:
        raise ValueError("Reference minute frame is missing close")

    result = pd.DataFrame(
        np.nan,
        index=etf_frame.index,
        columns=FACTOR_COLUMNS,
        dtype=float,
    )
    for trade_day, etf_day in etf_frame.groupby(
        etf_frame.index.normalize(), sort=False
    ):
        reference_day = reference_frame.loc[
            reference_frame.index.normalize() == trade_day
        ]
        if reference_day.empty:
            continue
        result.loc[etf_day.index] = _calculate_day_factors(
            etf_day, reference_day
        )
    return result


def build_output_frame(
    etf_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    mapping_reference_code: str,
    reference_ts_code: str,
    factor_frame: pd.DataFrame,
) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "reference_index_code": mapping_reference_code,
            "reference_ts_code": reference_ts_code,
            "reference_close": reference_frame["close"].reindex(
                etf_frame.index
            ),
        },
        index=etf_frame.index,
    )
    result = pd.concat([etf_frame, metadata, factor_frame], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def process_mapping_record(
    etf_path: Path,
    reference_path: Path,
    mapping_reference_code: str,
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
) -> dict[str, object]:
    output_path = output_root / etf_path.name
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "etf_code": etf_path.stem,
            "output_path": output_path,
        }

    etf_frame = filter_date_range(
        normalize_minute_frame(pd.read_parquet(etf_path)),
        date_from,
        date_to,
    )
    if etf_frame.empty:
        return {
            "status": "empty",
            "etf_code": etf_path.stem,
            "output_path": output_path,
        }
    reference_frame = read_reference_close_frame(
        reference_path,
        etf_frame.index.min(),
        etf_frame.index.max(),
    )

    factor_frame = calculate_crossmarket_factor_frame(
        etf_frame, reference_frame
    )
    result = build_output_frame(
        etf_frame,
        reference_frame,
        mapping_reference_code,
        reference_path.stem,
        factor_frame,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    factor_values = result.loc[:, FACTOR_COLUMNS]
    return {
        "status": "written",
        "etf_code": etf_path.stem,
        "reference_code": reference_path.stem,
        "output_path": output_path,
        "rows": len(result),
        "cols": len(result.columns),
        "factor_non_null": int(factor_values.notna().sum().sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ETF CrossMarket minute factors from mapped local "
            "reference-index minute data."
        )
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--etf-root", type=Path, default=DEFAULT_ETF_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional ETF symbols such as 159008.SZ or 510300.SH.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional text file with one ETF symbol per line.",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _requested_fund_codes(args: argparse.Namespace) -> list[str] | None:
    requested: list[str] = []
    if args.symbols_file is not None:
        requested.extend(read_symbol_list_file(args.symbols_file))
    if args.symbols:
        requested.extend(args.symbols)
    if not requested:
        return None
    return list(dict.fromkeys(normalize_fund_code(code) for code in requested))


def main() -> int:
    args = parse_args()
    configure_logging()
    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")

    mappings = load_mapping_records(args.mapping)
    etf_files = discover_etf_files(args.etf_root)
    by_full_code, paths_by_stem = build_index_file_lookup(args.index_root)
    requested = _requested_fund_codes(args)
    selected_codes = requested or sorted(mappings)

    if requested:
        missing_mapping = [code for code in requested if code not in mappings]
        missing_etf = [code for code in requested if code not in etf_files]
        if missing_mapping:
            raise FileNotFoundError(
                "ETF codes missing from mapping: "
                + ", ".join(missing_mapping)
            )
        if missing_etf:
            raise FileNotFoundError(
                "ETF minute files missing: " + ", ".join(missing_etf)
            )

    jobs: list[tuple[Path, Path, str]] = []
    missing_references: list[tuple[str, str]] = []
    alias_count = 0
    for fund_code in selected_codes:
        mapping = mappings.get(fund_code)
        etf_path = etf_files.get(fund_code)
        if mapping is None or etf_path is None:
            continue
        reference_path, match_type = resolve_reference_path(
            mapping.reference_index_code,
            by_full_code,
            paths_by_stem,
        )
        if reference_path is None:
            missing_references.append(
                (fund_code, mapping.reference_index_code)
            )
            continue
        alias_count += int(match_type == "stem")
        jobs.append(
            (etf_path, reference_path, mapping.reference_index_code)
        )

    if args.limit is not None:
        jobs = jobs[: args.limit]

    LOGGER.info(
        "Prepared %s ETF/reference jobs; %s use code-stem aliases",
        len(jobs),
        alias_count,
    )
    if missing_references:
        examples = ", ".join(
            f"{fund}->{reference}"
            for fund, reference in missing_references[:10]
        )
        LOGGER.warning(
            "Skipping %s mappings without local index minute files; "
            "examples: %s",
            len(missing_references),
            examples,
        )
    if not jobs:
        LOGGER.error("No ETF/reference jobs can be processed")
        return 1

    failures: list[tuple[str, str]] = []
    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        results = []
        for etf_path, reference_path, mapping_code in jobs:
            try:
                results.append(
                    process_mapping_record(
                        etf_path,
                        reference_path,
                        mapping_code,
                        args.output_root,
                        date_from,
                        date_to,
                        args.overwrite,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((etf_path.stem, str(exc)))
                LOGGER.exception("Failed to process %s", etf_path)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_mapping_record,
                    etf_path,
                    reference_path,
                    mapping_code,
                    args.output_root,
                    date_from,
                    date_to,
                    args.overwrite,
                ): etf_path
                for etf_path, reference_path, mapping_code in jobs
            }
            for future in as_completed(future_map):
                etf_path = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append((etf_path.stem, str(exc)))
                    LOGGER.exception("Failed to process %s", etf_path)

    for result in results:
        if result["status"] == "written":
            LOGGER.info(
                "Wrote %s rows, %s columns and %s non-null factor values "
                "for %s against %s to %s",
                result["rows"],
                result["cols"],
                result["factor_non_null"],
                result["etf_code"],
                result["reference_code"],
                result["output_path"],
            )
        else:
            LOGGER.info(
                "%s %s: %s",
                result["status"].capitalize(),
                result["etf_code"],
                result["output_path"],
            )

    if failures:
        LOGGER.error("Completed with %s failures", len(failures))
        for etf_code, reason in failures[:10]:
            LOGGER.error("  %s -> %s", etf_code, reason)
        return 1

    LOGGER.info("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
