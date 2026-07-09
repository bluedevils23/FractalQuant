from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.price import (  # noqa: E402
    LogReturnsFactor,
    OBVFactor,
    PriceMomentumFactor,
    PriceRelativeFactor,
    PriceZScoreFactor,
    ReturnsFactor,
    VolumeMomentumFactor,
    VolumePriceConfirmFactor,
    VolumePriceTrendFactor,
)
from factor.microstructure import (  # noqa: E402
    LiquidityDepthFactor,
    LiquidityMigrationFactor,
    LiquidityRatioFactor,
    LiquidityShockFactor,
    MarketEfficiencyFactor,
    MarketImpactFactor,
    MomentumAccelerationFactor,
    OrderBookAsymmetryFactor,
    OrderBookPressureFactor,
    OrderFlowImbalanceFactor,
    OrderFlowSignificanceFactor,
    PriceVelocityFactor,
    PriceVolumeDecouplingFactor,
    TradeDirectionPersistenceFactor,
    TradeSizeDistributionFactor,
    VolatilityAdjustedVolumeFactor,
    VolumeClusteringFactor,
    VolumeSpikeFactor,
    VolumeWeightedPriceFactor,
)
from factor.trend import (  # noqa: E402
    ADXFactor,
    AOFactor,
    CCIFactor,
    CMOFactor,
    EMAFactor,
    LSMAFactor,
    MACDFactor,
    MovingAverageFactor,
    ROCFactor,
    RSIFactor,
    StochasticFactor,
    TRIXFactor,
    WilliamsRFactor,
)
from factor.volatility import (  # noqa: E402
    ATRFactor,
    AnnualizedVolatilityFactor,
    BollingerBandWidthFactor,
    GarmanKlassVolatilityFactor,
    HistoricalVolatilityFactor,
    ParkinsonVolatilityFactor,
    RealizedVolatilityFactor,
    VolatilityKurtosisFactor,
    VolatilityRegimeFactor,
    VolatilitySkewFactor,
)


LOGGER = logging.getLogger("generate_etf_minute_factors")

DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_factors")
POSITIVE_PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def build_factors() -> list[object]:
    return [
        ReturnsFactor(window=5),
        LogReturnsFactor(window=5),
        PriceMomentumFactor(window=20),
        PriceZScoreFactor(window=20),
        HistoricalVolatilityFactor(window=20),
        ParkinsonVolatilityFactor(window=20),
        MACDFactor(),
        RSIFactor(),
        VolumeMomentumFactor(window=5),
        OBVFactor(),
        PriceRelativeFactor(window=20),
        VolumePriceTrendFactor(window=20),
        VolumePriceConfirmFactor(window=20),
        AnnualizedVolatilityFactor(window=20),
        RealizedVolatilityFactor(window=20),
        GarmanKlassVolatilityFactor(window=20),
        BollingerBandWidthFactor(window=20),
        ATRFactor(window=14),
        VolatilityRegimeFactor(short_window=5, long_window=20),
        VolatilitySkewFactor(window=20),
        VolatilityKurtosisFactor(window=20),
        MovingAverageFactor(window=20),
        EMAFactor(window=20),
        ADXFactor(window=14),
        StochasticFactor(window=14, smooth_k=3, smooth_d=3),
        CMOFactor(window=14),
        WilliamsRFactor(window=14),
        AOFactor(short_window=5, long_window=34),
        CCIFactor(window=20),
        ROCFactor(window=12),
        TRIXFactor(window=15),
        LSMAFactor(window=20),
        LiquidityRatioFactor(window=50),
        VolumeWeightedPriceFactor(window=50),
        VolatilityAdjustedVolumeFactor(window=50),
        PriceVelocityFactor(window=50),
        MomentumAccelerationFactor(window=50),
        VolumeSpikeFactor(window=50, threshold=2.0),
        PriceVolumeDecouplingFactor(window=50),
        MarketEfficiencyFactor(window=50),
        OrderFlowImbalanceFactor(window=50),
        OrderBookPressureFactor(window=50, levels=5),
        TradeSizeDistributionFactor(window=50),
        LiquidityShockFactor(window=50),
        OrderBookAsymmetryFactor(window=50),
        TradeDirectionPersistenceFactor(window=50),
        MarketImpactFactor(window=50, alpha=0.5),
        LiquidityDepthFactor(window=50),
        OrderFlowSignificanceFactor(window=50),
        VolumeClusteringFactor(window=50),
        LiquidityMigrationFactor(window=50),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ETF minute factors from local parquet files."
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


def discover_input_files(input_root: Path, symbols: list[str] | None) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    if symbols:
        files = [input_root / f"{symbol}.parquet" for symbol in symbols]
    else:
        files = sorted(input_root.glob("*.parquet"))

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


def prepare_factor_input(df: pd.DataFrame) -> pd.DataFrame:
    factor_input = df.copy()
    for column in POSITIVE_PRICE_COLUMNS:
        factor_input.loc[factor_input[column] <= 0, column] = np.nan
    return factor_input


def calculate_factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    factor_input = prepare_factor_input(df)

    # 按交易日分组分别计算滚动窗口因子，避免隔夜跨日污染：
    # 每个交易日开头会有窗口预热期（前若干根 bar 为 NaN），这是预期行为。
    trade_days = factor_input.index.normalize()
    unique_days = trade_days.unique()

    if len(unique_days) <= 1:
        factor_df = _calculate_factors_for_group(factor_input)
    else:
        per_day_frames = [
            _calculate_factors_for_group(factor_input.loc[trade_days == day])
            for day in unique_days
        ]
        factor_df = pd.concat(per_day_frames, axis=0)
        factor_df = factor_df.reindex(factor_input.index)

    return pd.concat([df, factor_df], axis=1)


def _calculate_factors_for_group(factor_input: pd.DataFrame) -> pd.DataFrame:
    factor_series: dict[str, pd.Series] = {}

    for factor in build_factors():
        with np.errstate(divide="ignore", invalid="ignore"):
            values = factor.calculate(factor_input)

        series = values
        if not series.index.equals(factor_input.index):
            series = series.reindex(factor_input.index)

        array = series.to_numpy(dtype=float, copy=False)
        if not np.isfinite(array).all():
            cleaned = array.copy()
            cleaned[~np.isfinite(cleaned)] = np.nan
            series = pd.Series(cleaned, index=factor_input.index, copy=False)

        factor_series[factor.name] = series

    return pd.DataFrame(factor_series, index=factor_input.index)


def process_file(
    input_path: Path, output_root: Path, overwrite: bool
) -> tuple[str, Path, int | None, int | None]:
    output_path = output_root / input_path.name
    if output_path.exists() and not overwrite:
        return ("skipped", output_path, None, None)

    raw_df = pd.read_parquet(input_path)
    df = normalize_minute_frame(raw_df)
    result_df = calculate_factor_frame(df)

    output_root.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path)

    return ("written", output_path, len(result_df), len(result_df.columns))


def main() -> int:
    args = parse_args()
    configure_logging()

    files = discover_input_files(args.input_root, args.symbols)
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
