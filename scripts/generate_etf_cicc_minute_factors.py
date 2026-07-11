from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"
REPLICATION_ROOT = PROJECT_ROOT.parent / "Replication-of-Minute-Frequency-Factor-refer-CICC"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPLICATION_ROOT))

import MinuteFrequentFactorCalculateMethodsCICC as cicc_methods  # noqa: E402


LOGGER = logging.getLogger("generate_etf_cicc_minute_factors")

DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_cicc_factors")
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
FACTOR_FUNCTION_NAMES = (
    "cal_mmt_pm",
    "cal_mmt_last30",
    "cal_mmt_paratio",
    "cal_mmt_am",
    "cal_mmt_between",
    "cal_mmt_ols_qrs",
    "cal_mmt_ols_corr_square_mean",
    "cal_mmt_ols_corr_mean",
    "cal_mmt_ols_beta_mean",
    "cal_mmt_ols_beta_zscore_last",
    "cal_mmt_top50VolumeRet",
    "cal_mmt_bottom50VolumeRet",
    "cal_mmt_top20VolumeRet",
    "cal_mmt_bottom20VolumeRet",
    "cal_vol_volume1min",
    "cal_vol_range1min",
    "cal_vol_return1min",
    "cal_vol_upVol",
    "cal_vol_upRatio",
    "cal_vol_downVol",
    "cal_vol_downRatio",
    "cal_shape_skew",
    "cal_shape_kurt",
    "cal_shape_skratio",
    "cal_shape_skewVol",
    "cal_shape_kurtVol",
    "cal_shape_skratioVol",
    "cal_liq_amihud_1min",
    "cal_liq_closeprevol",
    "cal_liq_closevol",
    "cal_liq_firstCallR",
    "cal_liq_lastCallR",
    "cal_liq_openvol",
    "cal_corr_prv",
    "cal_corr_prvr",
    "cal_corr_pv",
    "cal_corr_pvd",
    "cal_corr_pvl",
    "cal_corr_pvr",
    "cal_doc_kurt",
    "cal_doc_skew",
    "cal_doc_std",
    "cal_doc_pdf60",
    "cal_doc_pdf70",
    "cal_doc_pdf80",
    "cal_doc_pdf90",
    "cal_doc_pdf95",
    "cal_doc_vol10_ratio",
    "cal_doc_vol5_ratio",
    "cal_doc_vol50_ratio",
    "cal_trade_bottom20retRatio",
    "cal_trade_bottom50retRatio",
    "cal_trade_headRatio",
    "cal_trade_tailRatio",
    "cal_trade_top20retRatio",
    "cal_trade_top50retRatio",
    "cal_trade_topNeg20retRatio",
    "cal_trade_topPos20retRatio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ETF CICC minute factors from local parquet files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing ETF minute parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where factor parquet files will be written.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional ETF symbols such as 159001.SZ 510300.SH.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional ETF symbol list file with one symbol per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N parquet files after filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel workers to use.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def normalize_symbol_id(value: str) -> str:
    symbol = str(value).strip()
    if symbol.lower().endswith(".parquet"):
        symbol = symbol[:-8]
    return symbol


def read_symbol_list_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file does not exist: {path}")

    symbols: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                symbols.append(normalize_symbol_id(line))

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


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


def resolve_input_root(input_root: Path) -> Path:
    candidate = input_root / "etf_1min"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return input_root


def discover_input_files(input_root: Path, symbols: list[str] | None) -> list[Path]:
    resolved_root = resolve_input_root(input_root)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {resolved_root}")

    if symbols:
        files = [resolved_root / f"{symbol}.parquet" for symbol in symbols]
    else:
        files = sorted(resolved_root.glob("*.parquet"))

    missing_files = [path for path in files if not path.exists()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(f"Missing input parquet files: {missing_text}")

    return files


def normalize_minute_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    if isinstance(df.index, pd.MultiIndex) and "trade_time" in df.index.names:
        trade_time = pd.to_datetime(df.index.get_level_values("trade_time"))
        df.index = trade_time
        df.index.name = "trade_time"
    elif "trade_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["trade_time"])
        df = df.set_index("trade_time")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df.index.name = "trade_time"
    else:
        raise ValueError("Cannot locate trade_time/datetime index or column.")

    df = df.rename(columns={"vol": "volume"})

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    numeric_columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "amount", "adj_factor")
        if column in df.columns
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str)

    return df


def build_factor_registry() -> list[tuple[str, Callable[[pl.DataFrame], pl.DataFrame]]]:
    registry: list[tuple[str, Callable[[pl.DataFrame], pl.DataFrame]]] = []
    for function_name in FACTOR_FUNCTION_NAMES:
        function = getattr(cicc_methods, function_name, None)
        if function is None:
            raise AttributeError(f"Missing CICC factor function: {function_name}")
        registry.append((function_name.removeprefix("cal_"), function))
    return registry


def convert_time_to_cicc_int(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        index.hour * 10000000 + index.minute * 100000,
        index=index,
        dtype="int64",
    )


def build_cicc_input(df: pd.DataFrame) -> pl.DataFrame:
    trade_time_index = pd.DatetimeIndex(df.index)
    cicc_frame = pd.DataFrame(
        {
            "code": df["ts_code"] if "ts_code" in df.columns else "",
            "date": trade_time_index.normalize(),
            "time": convert_time_to_cicc_int(trade_time_index).to_numpy(),
            "open": df["open"].to_numpy(),
            "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(),
            "close": df["close"].to_numpy(),
            "volume": df["volume"].to_numpy(),
        },
        index=df.index,
    )
    if "ts_code" not in df.columns:
        raise ValueError("Missing ts_code column in source minute parquet.")
    cicc_frame["code"] = cicc_frame["code"].astype(str)
    cicc_frame["date"] = pd.to_datetime(cicc_frame["date"])
    return pl.from_pandas(cicc_frame, include_index=False)


def calculate_daily_factor_exposures(
    cicc_df: pl.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    trade_dates = (
        cicc_df.select(["code", "date"])
        .unique()
        .sort(["code", "date"])
    )
    exposure = trade_dates
    factor_columns: list[str] = []

    for factor_name, function in build_factor_registry():
        factor_df = function(cicc_df)
        value_columns = [
            column for column in factor_df.columns if column not in {"code", "date"}
        ]
        if len(value_columns) != 1:
            raise ValueError(
                f"Factor {factor_name} returned {len(value_columns)} value columns: {value_columns}"
            )
        value_column = value_columns[0]
        if value_column != factor_name:
            factor_df = factor_df.rename({value_column: factor_name})
        exposure = exposure.join(factor_df, on=["code", "date"], how="left")
        factor_columns.append(factor_name)

    exposure_pd = exposure.to_pandas()
    exposure_pd["date"] = pd.to_datetime(exposure_pd["date"])
    return exposure_pd, factor_columns


def merge_daily_factors_back(
    minute_df: pd.DataFrame, daily_exposure: pd.DataFrame
) -> pd.DataFrame:
    result = minute_df.copy()
    result["trade_date"] = pd.to_datetime(result.index).normalize()
    result = result.reset_index()
    result = result.merge(
        daily_exposure.rename(columns={"code": "ts_code", "date": "trade_date"}),
        on=["ts_code", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    result = result.set_index("trade_time")
    result.index.name = "trade_time"
    return result.drop(columns=["trade_date"])


def process_file(
    input_path: Path, output_root: Path, overwrite: bool
) -> tuple[str, Path, int | None, int | None]:
    output_path = output_root / input_path.name
    if output_path.exists() and not overwrite:
        return ("skipped", output_path, None, None)

    raw_df = pd.read_parquet(input_path)
    minute_df = normalize_minute_frame(raw_df)
    cicc_df = build_cicc_input(minute_df)
    daily_exposure, factor_columns = calculate_daily_factor_exposures(cicc_df)
    result_df = merge_daily_factors_back(minute_df, daily_exposure)

    ordered_columns = [
        column
        for column in ("ts_code", "open", "high", "low", "close", "volume", "amount", "adj_factor")
        if column in result_df.columns
    ]
    ordered_columns.extend(
        column for column in factor_columns if column in result_df.columns
    )
    remaining_columns = [
        column for column in result_df.columns if column not in ordered_columns
    ]
    result_df = result_df[ordered_columns + remaining_columns]

    output_root.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path)

    return ("written", output_path, len(result_df), len(result_df.columns))


def main() -> int:
    args = parse_args()
    configure_logging()

    requested_symbols = load_requested_symbols(args.symbols, args.symbols_file)
    files = discover_input_files(args.input_root, requested_symbols)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        LOGGER.warning("No parquet files matched the requested inputs.")
        return 0

    LOGGER.info("Processing %s ETF minute parquet files", len(files))

    failures: list[tuple[Path, str]] = []
    worker_count = max(1, args.workers)

    if worker_count == 1:
        for input_path in files:
            try:
                status, output_path, row_count, column_count = process_file(
                    input_path, args.output_root, args.overwrite
                )
                if status == "skipped":
                    LOGGER.info("Skipping existing output: %s", output_path)
                else:
                    LOGGER.info(
                        "Wrote %s rows and %s columns to %s",
                        row_count,
                        column_count,
                        output_path,
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append((input_path, str(exc)))
                LOGGER.exception("Failed to process %s", input_path)
    else:
        LOGGER.info("Using %s parallel workers", worker_count)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_file, input_path, args.output_root, args.overwrite
                ): input_path
                for input_path in files
            }
            for future in as_completed(future_map):
                input_path = future_map[future]
                try:
                    status, output_path, row_count, column_count = future.result()
                    if status == "skipped":
                        LOGGER.info("Skipping existing output: %s", output_path)
                    else:
                        LOGGER.info(
                            "Wrote %s rows and %s columns to %s",
                            row_count,
                            column_count,
                            output_path,
                        )
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
