from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
    RollingOBVFactor,
    RollingVolumePriceTrendFactor,
    VolumeMomentumFactor,
    VolumePriceConfirmFactor,
    VolumePriceConfirmRateFactor,
    VolumePriceTrendFactor,
)
from factor.fractional import FractionalDiffLogCloseFactor  # noqa: E402
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
DEFAULT_MULTIWINDOW_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_factors_multiwindow"
)
POSITIVE_PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
WINDOW_PROFILES = ("base", "multi")


@dataclass(frozen=True)
class FactorSpec:
    output_name: str
    factor: object


def _base_factor_specs() -> list[FactorSpec]:
    factors = [
        ReturnsFactor(window=5),
        LogReturnsFactor(window=5),
        PriceMomentumFactor(window=20),
        PriceZScoreFactor(window=20),
        FractionalDiffLogCloseFactor(order=0.4, threshold=1e-3),
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
    return [FactorSpec(factor.name, factor) for factor in factors]


def _window_variants(
    output_base: str,
    windows: tuple[int, ...],
    factory: Callable[[int], object],
) -> list[FactorSpec]:
    return [
        FactorSpec(f"{output_base}_w{window}", factory(window))
        for window in windows
    ]


def _multiwindow_factor_specs() -> list[FactorSpec]:
    specs = _base_factor_specs()

    specs.extend(_window_variants("returns", (3, 10, 20), ReturnsFactor))
    specs.extend(
        _window_variants("volume_momentum", (10, 20), VolumeMomentumFactor)
    )

    for output_name, factor_class in (
        ("price_momentum", PriceMomentumFactor),
        ("price_zscore", PriceZScoreFactor),
        ("price_relative", PriceRelativeFactor),
        ("moving_average", MovingAverageFactor),
        ("ema", EMAFactor),
        ("cci", CCIFactor),
        ("lsma", LSMAFactor),
    ):
        specs.extend(_window_variants(output_name, (10, 40), factor_class))

    for output_name, factor_class in (
        ("rsi", RSIFactor),
        ("atr", ATRFactor),
        ("adx", ADXFactor),
        ("cmo", CMOFactor),
        ("williams_r", WilliamsRFactor),
    ):
        specs.extend(_window_variants(output_name, (7, 28), factor_class))
    specs.extend(
        _window_variants(
            "stochastic",
            (7, 28),
            lambda window: StochasticFactor(window=window, smooth_k=3, smooth_d=3),
        )
    )

    specs.extend(_window_variants("roc", (5, 24), ROCFactor))
    specs.extend(_window_variants("trix", (8, 30), TRIXFactor))
    specs.extend(
        [
            FactorSpec("macd_f6_s13_sig5", MACDFactor(6, 13, 5)),
            FactorSpec("macd_f24_s52_sig18", MACDFactor(24, 52, 18)),
            FactorSpec("awesome_oscillator_s3_l10", AOFactor(3, 10)),
            FactorSpec("awesome_oscillator_s10_l68", AOFactor(10, 68)),
        ]
    )

    for output_name, factor_class in (
        ("historical_volatility", HistoricalVolatilityFactor),
        ("parkinson_volatility", ParkinsonVolatilityFactor),
        ("annualized_volatility", AnnualizedVolatilityFactor),
        ("realized_volatility", RealizedVolatilityFactor),
        ("garman_klass_volatility", GarmanKlassVolatilityFactor),
        ("bollinger_band_width", BollingerBandWidthFactor),
    ):
        specs.extend(_window_variants(output_name, (10, 40), factor_class))
    specs.extend(
        [
            FactorSpec(
                "volatility_regime_s3_l10",
                VolatilityRegimeFactor(short_window=3, long_window=10),
            ),
            FactorSpec(
                "volatility_regime_s10_l40",
                VolatilityRegimeFactor(short_window=10, long_window=40),
            ),
        ]
    )
    specs.extend(
        _window_variants("volatility_skew", (40, 60), VolatilitySkewFactor)
    )
    specs.extend(
        _window_variants(
            "volatility_kurtosis", (40, 60), VolatilityKurtosisFactor
        )
    )

    for window in (5, 20, 50):
        specs.extend(
            [
                FactorSpec(f"obv_delta_w{window}", RollingOBVFactor(window)),
                FactorSpec(
                    f"volume_price_trend_delta_w{window}",
                    RollingVolumePriceTrendFactor(window),
                ),
            ]
        )
    for window in (5, 10, 20):
        specs.append(
            FactorSpec(
                f"volume_price_confirm_rate_w{window}",
                VolumePriceConfirmRateFactor(window),
            )
        )

    direct_microstructure = (
        ("volume_weighted_price", VolumeWeightedPriceFactor),
        ("volatility_adj_volume", VolatilityAdjustedVolumeFactor),
        ("price_velocity", PriceVelocityFactor),
        ("momentum_acceleration", MomentumAccelerationFactor),
        ("order_flow_imbalance", OrderFlowImbalanceFactor),
        ("liquidity_shock", LiquidityShockFactor),
        ("orderbook_asymmetry", OrderBookAsymmetryFactor),
        ("trade_direction_persistence", TradeDirectionPersistenceFactor),
    )
    for output_name, factor_class in direct_microstructure:
        specs.extend(_window_variants(output_name, (10, 20), factor_class))
    specs.extend(
        _window_variants(
            "volume_spike",
            (10, 20),
            lambda window: VolumeSpikeFactor(window=window, threshold=2.0),
        )
    )
    specs.extend(
        _window_variants(
            "orderbook_pressure",
            (10, 20),
            lambda window: OrderBookPressureFactor(window=window, levels=5),
        )
    )

    stable_microstructure = (
        ("liquidity_ratio", LiquidityRatioFactor),
        ("trade_size_distribution", TradeSizeDistributionFactor),
        ("liquidity_depth", LiquidityDepthFactor),
        ("orderflow_significance", OrderFlowSignificanceFactor),
        ("volume_clustering", VolumeClusteringFactor),
        ("price_volume_decoupling", PriceVolumeDecouplingFactor),
        ("market_efficiency", MarketEfficiencyFactor),
        ("liquidity_migration", LiquidityMigrationFactor),
    )
    for output_name, factor_class in stable_microstructure:
        specs.extend(_window_variants(output_name, (30, 80), factor_class))

    specs.extend(
        [
            FactorSpec(
                "market_impact_w10",
                MarketImpactFactor(
                    window=10, flow_window=5, volatility_window=10
                ),
            ),
            FactorSpec(
                "market_impact_w20",
                MarketImpactFactor(
                    window=20, flow_window=10, volatility_window=20
                ),
            ),
        ]
    )
    return specs


def _validate_factor_specs(specs: list[FactorSpec]) -> None:
    names = [spec.output_name for spec in specs]
    duplicates = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate factor output names: {duplicates}")


def build_factor_specs(window_profile: str = "base") -> list[FactorSpec]:
    if window_profile == "base":
        specs = _base_factor_specs()
    elif window_profile == "multi":
        specs = _multiwindow_factor_specs()
    else:
        raise ValueError(f"Unsupported window profile: {window_profile}")
    _validate_factor_specs(specs)
    return specs


def build_factors(window_profile: str = "base") -> list[object]:
    """Return factor objects for backward-compatible registry inspection."""
    return [spec.factor for spec in build_factor_specs(window_profile)]


FACTOR_SPECS_BY_PROFILE = {
    profile: tuple(build_factor_specs(profile)) for profile in WINDOW_PROFILES
}
FACTORS = [spec.factor for spec in FACTOR_SPECS_BY_PROFILE["base"]]


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
        default=None,
        help="Output directory. Defaults depend on --window-profile.",
    )
    parser.add_argument(
        "--window-profile",
        choices=WINDOW_PROFILES,
        default="base",
        help="Use the compatible base factors or the expanded multi-window set.",
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


def resolve_output_root(
    output_root: Path | None, window_profile: str
) -> Path:
    if output_root is not None:
        return output_root
    if window_profile == "multi":
        return DEFAULT_MULTIWINDOW_OUTPUT_ROOT
    return DEFAULT_OUTPUT_ROOT


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


def calculate_factor_frame(
    df: pd.DataFrame, window_profile: str = "base"
) -> pd.DataFrame:
    factor_input = prepare_factor_input(df)
    factor_specs = FACTOR_SPECS_BY_PROFILE[window_profile]

    # 按交易日分组分别计算滚动窗口因子，避免隔夜跨日污染：
    # 每个交易日开头会有窗口预热期（前若干根 bar 为 NaN），这是预期行为。
    trade_days = factor_input.index.normalize()
    trade_day_values = trade_days.to_numpy(dtype="datetime64[ns]", copy=False)

    if len(trade_day_values) <= 1:
        factor_df = _calculate_factors_for_group(factor_input, factor_specs)
    else:
        split_points = np.flatnonzero(trade_day_values[1:] != trade_day_values[:-1]) + 1
        starts = np.concatenate(([0], split_points))
        stops = np.concatenate((split_points, [len(factor_input)]))
        per_day_frames = [
            _calculate_factors_for_group(
                factor_input.iloc[start:stop], factor_specs
            )
            for start, stop in zip(starts, stops)
        ]
        factor_df = pd.concat(per_day_frames, axis=0)

    return pd.concat([df, factor_df], axis=1)


def _calculate_factors_for_group(
    factor_input: pd.DataFrame, factor_specs: tuple[FactorSpec, ...]
) -> pd.DataFrame:
    factor_series: dict[str, pd.Series] = {}

    for spec in factor_specs:
        factor = spec.factor
        with np.errstate(divide="ignore", invalid="ignore"):
            series = factor.calculate(factor_input)

        if not series.index.equals(factor_input.index):
            series = series.reindex(factor_input.index)

        array = series.to_numpy(dtype=float, copy=False)
        if not np.isfinite(array).all():
            cleaned = array.copy()
            cleaned[~np.isfinite(cleaned)] = np.nan
            series = pd.Series(cleaned, index=factor_input.index, copy=False)

        factor_series[spec.output_name] = series

    return pd.DataFrame(factor_series, index=factor_input.index)


def process_file(
    input_path: Path,
    output_root: Path,
    overwrite: bool,
    window_profile: str = "base",
) -> tuple[str, Path, int | None, int | None]:
    output_path = output_root / input_path.name
    if output_path.exists() and not overwrite:
        return ("skipped", output_path, None, None)

    raw_df = pd.read_parquet(input_path)
    df = normalize_minute_frame(raw_df)
    result_df = calculate_factor_frame(df, window_profile)

    output_root.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path)

    return ("written", output_path, len(result_df), len(result_df.columns))


def main() -> int:
    args = parse_args()
    configure_logging()
    output_root = resolve_output_root(args.output_root, args.window_profile)

    files = discover_input_files(args.input_root, args.symbols)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        LOGGER.warning("No parquet files matched the requested inputs.")
        return 0

    LOGGER.info("Processing %s ETF minute parquet files", len(files))
    LOGGER.info(
        "Using %s window profile with %s factor columns",
        args.window_profile,
        len(FACTOR_SPECS_BY_PROFILE[args.window_profile]),
    )

    failures: list[tuple[Path, str]] = []
    worker_count = max(1, args.workers)

    if worker_count == 1:
        for input_path in files:
            try:
                status, output_path, row_count, column_count = process_file(
                    input_path,
                    output_root,
                    args.overwrite,
                    args.window_profile,
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
                    process_file,
                    input_path,
                    output_root,
                    args.overwrite,
                    args.window_profile,
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
