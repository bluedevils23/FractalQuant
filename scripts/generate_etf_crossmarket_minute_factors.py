from __future__ import annotations

"""Append benchmark-relative minute factors to existing ETF outputs."""

import argparse
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scripts.generate_etf_crossmarket_factors import (  # noqa: E402
    DEFAULT_ETF_ROOT,
    DEFAULT_INDEX_ROOT,
    DEFAULT_MAPPING_PATH,
    build_index_file_lookup,
    filter_date_range,
    load_mapping_records,
    normalize_fund_code,
    normalize_reference_frame,
    resolve_reference_path,
)
from scripts.generate_etf_minute_factors import normalize_minute_frame  # noqa: E402


LOGGER = logging.getLogger("generate_etf_crossmarket_minute_factors")

DEFAULT_CROSSMARKET_ROOT = Path(
    r"D:\workspace\stockdata\etf-factors\etf_1min_crossmarket_factors_rqdata"
)

FACTOR_COLUMNS = (
    "rolling_market_beta_60m",
    "beta_residual_momentum_20m",
    "beta_residual_zscore_60m",
    "benchmark_correlation_60m",
)
BETA_WINDOW = 60
BETA_MIN_PERIODS = 30
RESIDUAL_MOMENTUM_WINDOW = 20
RESIDUAL_ZSCORE_WINDOW = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append ETF benchmark-relative minute factors to existing "
            "CrossMarket parquet files."
        )
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--etf-root", type=Path, default=DEFAULT_ETF_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument(
        "--crossmarket-root",
        "--output-root",
        dest="crossmarket_root",
        type=Path,
        default=DEFAULT_CROSSMARKET_ROOT,
    )
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
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def discover_symbol_files(
    root: Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> dict[str, Path]:
    if not root.exists():
        if allow_missing:
            LOGGER.warning("%s root does not exist: %s", label, root)
            return {}
        raise FileNotFoundError(f"{label} root does not exist: {root}")

    files: dict[str, Path] = {}
    for path in sorted(root.glob("*.parquet")):
        fund_code = normalize_fund_code(path.name)
        if fund_code in files:
            raise ValueError(
                f"Multiple {label} files share fund code {fund_code}"
            )
        files[fund_code] = path
    return files


def _timestamps_from_frame(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.MultiIndex) and "trade_time" in frame.index.names:
        timestamps = pd.to_datetime(frame.index.get_level_values("trade_time"))
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.to_datetime(frame.index)
    elif "trade_time" in frame.columns:
        timestamps = pd.to_datetime(frame["trade_time"])
    elif "datetime" in frame.columns:
        timestamps = pd.to_datetime(frame["datetime"])
    else:
        raise ValueError("Cannot locate trade_time/datetime index or column")
    return pd.DatetimeIndex(timestamps, name="trade_time")


def _log_return(close: pd.Series) -> pd.Series:
    valid_close = pd.to_numeric(close, errors="coerce").where(
        lambda values: values.gt(0)
    )
    return np.log(valid_close).diff()


def _calculate_day_factors(
    etf_close: pd.Series,
    reference_close: pd.Series,
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan, index=etf_close.index, columns=FACTOR_COLUMNS, dtype=float
    )
    etf_return = _log_return(etf_close)
    reference_return = _log_return(reference_close)

    historical_etf_return = etf_return.shift(1)
    historical_reference_return = reference_return.shift(1)
    valid_history = (
        historical_etf_return.notna() & historical_reference_return.notna()
    )
    historical_etf_return = historical_etf_return.where(valid_history)
    historical_reference_return = historical_reference_return.where(
        valid_history
    )
    covariance = historical_etf_return.rolling(
        BETA_WINDOW, min_periods=BETA_MIN_PERIODS
    ).cov(historical_reference_return)
    variance = historical_reference_return.rolling(
        BETA_WINDOW, min_periods=BETA_MIN_PERIODS
    ).var()
    beta = covariance.div(variance.where(variance.gt(0)))
    residual = etf_return.sub(beta.mul(reference_return))

    result["rolling_market_beta_60m"] = beta
    result["beta_residual_momentum_20m"] = residual.rolling(
        RESIDUAL_MOMENTUM_WINDOW,
        min_periods=RESIDUAL_MOMENTUM_WINDOW,
    ).sum()
    residual_mean = residual.rolling(
        RESIDUAL_ZSCORE_WINDOW, min_periods=BETA_MIN_PERIODS
    ).mean()
    residual_std = residual.rolling(
        RESIDUAL_ZSCORE_WINDOW, min_periods=BETA_MIN_PERIODS
    ).std()
    result["beta_residual_zscore_60m"] = residual.sub(residual_mean).div(
        residual_std.where(residual_std.gt(0))
    )
    result["benchmark_correlation_60m"] = etf_return.rolling(
        BETA_WINDOW, min_periods=BETA_MIN_PERIODS
    ).corr(reference_return)
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_benchmark_factor_frame(
    etf_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.DataFrame:
    if "close" not in etf_frame.columns:
        raise ValueError("ETF minute frame is missing close")
    if "close" not in reference_frame.columns:
        raise ValueError("Reference minute frame is missing close")

    common_index = etf_frame.index.intersection(reference_frame.index)
    result = pd.DataFrame(
        np.nan, index=common_index, columns=FACTOR_COLUMNS, dtype=float
    )
    for _, day_index in pd.Series(common_index, index=common_index).groupby(
        common_index.normalize(), sort=False
    ):
        timestamps = pd.DatetimeIndex(day_index.to_numpy())
        etf_close = pd.to_numeric(
            etf_frame.loc[timestamps, "close"], errors="coerce"
        )
        reference_close = pd.to_numeric(
            reference_frame.loc[timestamps, "close"], errors="coerce"
        )
        result.loc[timestamps] = _calculate_day_factors(
            etf_close, reference_close
        )
    return result


def merge_factor_columns(
    existing: pd.DataFrame,
    factor_frame: pd.DataFrame,
    overwrite: bool,
    update_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    result = existing.copy()
    timestamps = _timestamps_from_frame(result)
    if timestamps.has_duplicates:
        raise ValueError("CrossMarket output has duplicate minute timestamps")
    aligned = factor_frame.reindex(timestamps)
    incoming = timestamps.isin(factor_frame.index)
    scope = (
        timestamps.isin(update_index)
        if update_index is not None
        else incoming
    )

    for column in FACTOR_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        if overwrite:
            result.loc[scope & ~incoming, column] = np.nan
            result.loc[scope & incoming, column] = aligned.loc[
                scope & incoming, column
            ].to_numpy()
        else:
            missing = result[column].isna() & incoming
            result.loc[missing, column] = aligned.loc[missing, column].to_numpy()
    return result


def _read_reference_frame(reference_path: Path) -> pd.DataFrame:
    return normalize_reference_frame(pd.read_parquet(reference_path))


def process_reference_group(
    reference_code: str,
    target_codes: tuple[str, ...],
    etf_files: dict[str, Path],
    crossmarket_files: dict[str, Path],
    reference_path: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    writable_codes: list[str] = []
    for fund_code in target_codes:
        if fund_code not in crossmarket_files:
            results.append(
                {
                    "status": "missing_crossmarket_output",
                    "etf_code": fund_code,
                    "reference_code": reference_code,
                }
            )
        else:
            writable_codes.append(fund_code)
    if not writable_codes:
        return results

    reference_frame = _read_reference_frame(reference_path)
    for fund_code in writable_codes:
        etf_frame = filter_date_range(
            normalize_minute_frame(pd.read_parquet(etf_files[fund_code])),
            date_from,
            date_to,
        )
        if etf_frame.empty:
            results.append(
                {
                    "status": "empty",
                    "etf_code": fund_code,
                    "reference_code": reference_code,
                }
            )
            continue
        factor_frame = calculate_benchmark_factor_frame(
            etf_frame, reference_frame
        )
        output_path = crossmarket_files[fund_code]
        output = merge_factor_columns(
            pd.read_parquet(output_path),
            factor_frame,
            overwrite,
            update_index=etf_frame.index,
        )
        output.to_parquet(output_path)
        results.append(
            {
                "status": "written" if not factor_frame.empty else "no_overlap",
                "etf_code": fund_code,
                "reference_code": reference_code,
                "output_path": output_path,
                "rows": len(output),
                "factor_non_null": int(
                    output.loc[:, FACTOR_COLUMNS].notna().sum().sum()
                ),
            }
        )
    return results


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
    etf_files = discover_symbol_files(args.etf_root, "ETF minute")
    crossmarket_files = discover_symbol_files(
        args.crossmarket_root, "CrossMarket", allow_missing=True
    )
    by_full_code, paths_by_stem = build_index_file_lookup(args.index_root)
    requested = _requested_fund_codes(args)
    selected_codes = requested or sorted(mappings)
    if requested:
        missing_mapping = [code for code in requested if code not in mappings]
        missing_etf = [code for code in requested if code not in etf_files]
        if missing_mapping:
            raise FileNotFoundError(
                "ETF codes missing from mapping: " + ", ".join(missing_mapping)
            )
        if missing_etf:
            raise FileNotFoundError(
                "ETF minute files missing: " + ", ".join(missing_etf)
            )
    if args.limit is not None:
        selected_codes = selected_codes[: args.limit]

    selected: dict[str, list[str]] = defaultdict(list)
    reference_paths: dict[str, Path] = {}
    missing_references: list[tuple[str, str]] = []
    alias_count = 0
    for fund_code in selected_codes:
        mapping = mappings.get(fund_code)
        if mapping is None or fund_code not in etf_files:
            continue
        reference_path, match_type = resolve_reference_path(
            mapping.reference_index_code, by_full_code, paths_by_stem
        )
        if reference_path is None:
            missing_references.append((fund_code, mapping.reference_index_code))
            continue
        alias_count += int(match_type == "stem" or mapping.uses_data_alias)
        selected[mapping.reference_index_code].append(fund_code)
        reference_paths[mapping.reference_index_code] = reference_path

    jobs = [
        (
            reference_code,
            tuple(sorted(target_codes)),
            etf_files,
            crossmarket_files,
            reference_paths[reference_code],
            date_from,
            date_to,
            args.overwrite,
        )
        for reference_code, target_codes in selected.items()
    ]
    LOGGER.info(
        "Prepared %s ETF/reference groups; %s selected ETFs use reference "
        "code aliases",
        len(jobs),
        alias_count,
    )
    if missing_references:
        examples = ", ".join(
            f"{fund}->{reference}"
            for fund, reference in missing_references[:10]
        )
        LOGGER.warning(
            "Excluding %s ETF(s) across %s reference group(s) without local "
            "index minute files; examples: %s",
            len(missing_references),
            len({reference for _, reference in missing_references}),
            examples,
        )
    if not jobs:
        LOGGER.error("No ETF/reference jobs can be processed")
        return 1

    worker_count = max(1, int(args.workers))
    results: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    if worker_count == 1:
        for job in jobs:
            try:
                results.extend(process_reference_group(*job))
            except Exception as exc:  # noqa: BLE001
                failures.append((job[0], str(exc)))
                LOGGER.exception("Failed to process reference %s", job[0])
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(process_reference_group, *job): job[0]
                for job in jobs
            }
            for future in as_completed(future_map):
                reference_code = future_map[future]
                try:
                    results.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append((reference_code, str(exc)))
                    LOGGER.exception(
                        "Failed to process reference %s", reference_code
                    )

    counts = pd.Series(
        [result["status"] for result in results], dtype="object"
    ).value_counts()
    LOGGER.info("Result counts: %s", counts.to_dict())
    if failures:
        LOGGER.error("Failed %s reference groups", len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
