from __future__ import annotations

"""Generate stock and ETF orderbook factors from raw per-symbol tick CSV folders.

It reads one trade-date directory of stock quote/order/trade CSV files, computes
snapshot-level orderbook factors via ``factor.stock_orderbook``, aligns them to
the stock 1-minute bars, and writes one advanced-compatible parquet per symbol.
"""

import argparse
import logging
import os
import re
import sys
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.stock_orderbook import build_stock_orderbook_factor_frame  # noqa: E402


LOGGER = logging.getLogger("generate_stock_orderbook_factors")

# Raw tick root organized as YYYY/YYYYMM/YYYYMMDD/symbol.
DEFAULT_INPUT_ROOT = Path(r"E:\逐笔数据")
DEFAULT_STOCK_MINUTE_ROOT = Path(r"D:\workspace\stockdata\a-share-data\行情数据\stock_1min")
DEFAULT_ETF_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_STOCK_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\stock_1min_orderbook_factors"
)
DEFAULT_ETF_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_orderbook_factors"
)
DEFAULT_STOCK_MULTIWINDOW_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\stock_1min_orderbook_factors_multiwindow"
)
DEFAULT_ETF_MULTIWINDOW_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_orderbook_factors_multiwindow"
)
WINDOW_PROFILES = ("base", "multi")

QUOTE_USECOLS = [0, 2, 3, *range(17, 27), *range(27, 37), *range(37, 47), *range(47, 57)]
QUOTE_COLUMN_NAMES = [
    "ts_code",
    "trade_date",
    "raw_time",
    *[f"ask_price{i}" for i in range(1, 11)],
    *[f"ask_qty{i}" for i in range(1, 11)],
    *[f"bid_price{i}" for i in range(1, 11)],
    *[f"bid_qty{i}" for i in range(1, 11)],
]
ORDER_USECOLS = [0, 2, 3, 7, 8, 9]
ORDER_COLUMN_NAMES = ["ts_code", "trade_date", "raw_time", "side", "price", "qty"]
TRADE_USECOLS = [0, 2, 3, 7, 8, 9]
TRADE_COLUMN_NAMES = ["ts_code", "trade_date", "raw_time", "side", "price", "qty"]
BASE_OUTPUT_COLUMNS = [
    "trade_date",
    "trade_time",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "adj_factor",
]
BASE_FACTOR_COLUMNS = [
    # Snapshot quote factors.
    "mid_price",
    "spread_bps",
    "depth_imbalance_l1",
    "depth_imbalance_l5",
    "normalized_ofi_l1",
    "normalized_ofi_l1_60s",
    "ofi_spread_scaled_impact",
    "ofi_level_entropy_l5",
    "normalized_mlofi_l5",
    "normalized_mlofi_l5_60s",
    "mlofi_event_50_l5",
    "mlofi_deep_divergence_l5",
    "mlofi_impact_beta",
    "weighted_depth_imbalance_l5",
    "weighted_depth_pressure_l5",
    "soir_l5_decay",
    "weighted_imbalance_velocity_l5",
    "contextual_lob_surprise_l5",
    "contextual_imbalance_surprise_l5",
    "bid_refill_intensity_l5",
    "ask_refill_intensity_l5",
    "bid_ask_qty_ratio_l1",
    "depth_l5_total",
    "bid_resilience_30s",
    "ask_resilience_30s",
    "resilience_imbalance_30s",
    "orderbook_decay_l5",
    "orderbook_asymmetry_l5",
    "depth_concentration_l5",
    "orderbook_liquidity_l5",
    "book_pressure_wap5",
    "book_slope_diff_l5",
    "mci_bid_l5",
    "mci_ask_l5",
    "mpc_1m_mean_5m",
    "mpc_1m_max_5m",
    "mpc_1m_skew_5m",
    "mpc_5m_mean_5m",
    "mpc_5m_max_5m",
    "mpc_5m_skew_5m",
    "orderbook_velocity_l5",
    # 60s order-flow factors aligned backward to snapshots.
    "order_count_imbalance_60s",
    "order_qty_imbalance_60s",
    "order_notional_imbalance_60s",
    # 60s trade-flow and trade-impact factors aligned backward to snapshots.
    "trade_count_imbalance_60s",
    "trade_qty_imbalance_60s",
    "trade_vwap_gap_60s",
    "trade_size_distribution_60s",
    "trade_direction_persistence_60s",
    "liquidity_shock_60s",
    "market_impact_60s",
    "orderflow_significance_60s",
    "volatility_adj_volume_60s",
    "price_velocity_60s",
    "momentum_acceleration_60s",
    "volume_spike_60s",
    "volume_clustering_60s",
    "liquidity_depth_60s",
    "price_volume_decoupling_60s",
    "market_efficiency_60s",
    "liquidity_migration_60s",
    "order_flow_imbalance_60s",
    "liquidity_ratio_60s",
    "volume_weighted_price_60s",
    "orderbook_pressure_60s",
    "vpin_50bucket",
    "adverse_selection_markout_30s",
    "cautious_to_aggressive_buy_ratio_60s",
    "trade_notional_quantile_position_60s",
    "price_band_high_trade_count_share_60s",
    "price_band_low_trade_count_share_60s",
    "price_band_high_trade_size_rel_60s",
    "price_band_low_trade_size_rel_60s",
    # Causal contextual anomaly-segment factors.
    "contextual_flow_surprise_60s",
    "contextual_segment_anomaly_60s",
    "contextual_segment_selected_60s",
    "contextual_selected_flow_imbalance_60s",
    "contextual_selected_lob_surprise_60s",
]
FACTOR_COLUMNS = BASE_FACTOR_COLUMNS
OUTPUT_COLUMNS = BASE_OUTPUT_COLUMNS + FACTOR_COLUMNS


def factor_columns_for_profile(window_profile: str) -> list[str]:
    if window_profile not in WINDOW_PROFILES:
        raise ValueError(f"Unsupported window profile: {window_profile}")
    if window_profile == "base":
        return list(BASE_FACTOR_COLUMNS)

    additional: list[str] = []
    for window in ("10s", "30s", "300s"):
        additional.extend(
            [
                f"normalized_ofi_l1_{window}",
                f"normalized_mlofi_l5_{window}",
                f"ofi_spread_scaled_impact_{window}",
            ]
        )
    for window in ("10s", "30s", "300s"):
        additional.extend(
            f"order_{metric}_imbalance_{window}"
            for metric in ("count", "qty", "notional")
        )
    for window in ("10s", "30s", "300s"):
        additional.extend(
            f"trade_{metric}_{window}"
            for metric in ("count_imbalance", "qty_imbalance", "vwap_gap")
        )
    impact_bases = (
        "trade_size_distribution",
        "trade_direction_persistence",
        "liquidity_shock",
        "market_impact",
        "orderflow_significance",
        "volatility_adj_volume",
        "price_velocity",
        "momentum_acceleration",
        "volume_spike",
        "volume_clustering",
        "liquidity_depth",
        "price_volume_decoupling",
        "market_efficiency",
        "liquidity_migration",
        "order_flow_imbalance",
        "liquidity_ratio",
        "volume_weighted_price",
        "orderbook_pressure",
    )
    for window in ("30s", "300s"):
        additional.extend(f"{base}_{window}" for base in impact_bases)
    for window in ("10s", "60s", "300s"):
        additional.extend(
            [
                f"bid_resilience_{window}",
                f"ask_resilience_{window}",
                f"resilience_imbalance_{window}",
            ]
        )
    columns = list(BASE_FACTOR_COLUMNS) + additional
    if len(columns) != len(set(columns)):
        raise ValueError("Multi-window factor columns must be unique")
    return columns


def output_columns_for_profile(window_profile: str) -> list[str]:
    return BASE_OUTPUT_COLUMNS + factor_columns_for_profile(window_profile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stock and ETF orderbook factor parquet files from local tick CSVs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Tick root organized as YYYY/YYYYMM/YYYYMMDD/symbol.",
    )
    parser.add_argument(
        "--stock-minute-root",
        type=Path,
        default=DEFAULT_STOCK_MINUTE_ROOT,
        help="Directory containing one 1-minute parquet file per stock.",
    )
    parser.add_argument(
        "--etf-minute-root",
        type=Path,
        default=DEFAULT_ETF_MINUTE_ROOT,
        help="Directory containing one 1-minute parquet file per ETF.",
    )
    parser.add_argument(
        "--stock-output-root",
        type=Path,
        default=None,
        help="Directory where stock orderbook factor parquet files will be written.",
    )
    parser.add_argument(
        "--etf-output-root",
        type=Path,
        default=None,
        help="Directory where ETF orderbook factor parquet files will be written.",
    )
    parser.add_argument(
        "--window-profile",
        choices=WINDOW_PROFILES,
        default="base",
        help="Factor window profile; base preserves legacy output and multi adds short/long windows.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbols or six-digit codes; exchange suffixes are ignored.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N matched date/symbol tasks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers to use.",
    )
    return parser.parse_args()


def resolve_output_root(
    requested_root: Path | None,
    window_profile: str,
    base_root: Path,
    multiwindow_root: Path,
) -> Path:
    if requested_root is not None:
        return requested_root
    return multiwindow_root if window_profile == "multi" else base_root


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def numeric_code(name: str) -> str | None:
    match = re.match(r"^(\d{6})", name)
    return match.group(1) if match else None


def discover_trade_date_dirs(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    date_dirs = [
        path
        for path in input_root.glob("*/*/*")
        if path.is_dir() and re.fullmatch(r"\d{8}", path.name)
    ]
    return sorted(date_dirs)


def build_minute_file_index(minute_root: Path) -> dict[str, Path]:
    if not minute_root.exists():
        raise FileNotFoundError(f"Minute directory does not exist: {minute_root}")
    candidates: dict[str, list[Path]] = {}
    for path in minute_root.glob("*.parquet"):
        code = numeric_code(path.stem)
        if code is not None:
            candidates.setdefault(code, []).append(path)

    result: dict[str, Path] = {}
    canonical = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$", re.IGNORECASE)
    for code, paths in candidates.items():
        result[code] = sorted(
            paths,
            key=lambda path: (not bool(canonical.fullmatch(path.stem)), path.name),
        )[0]
    return result


def discover_symbol_dirs(input_root: Path, symbols: list[str] | None) -> list[Path]:
    """Discover symbol directories inside one YYYYMMDD directory."""
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    if symbols:
        requested_codes = {code for symbol in symbols if (code := numeric_code(symbol))}
        symbol_dirs = sorted(
            path
            for path in input_root.iterdir()
            if path.is_dir() and numeric_code(path.name) in requested_codes
        )
    else:
        symbol_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    return symbol_dirs


def parse_trade_time(trade_date: pd.Series, raw_time: pd.Series) -> pd.Series:
    date_text = trade_date.astype(str).str.zfill(8)
    time_text = raw_time.astype(str).str.zfill(9)
    combined = date_text + time_text
    return pd.to_datetime(combined, format="%Y%m%d%H%M%S%f", errors="raise")


def read_csv_subset(path: Path, usecols: list[int], names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="gbk",
        usecols=usecols,
        dtype=str,
        low_memory=False,
    )
    df.columns = names
    return df


@lru_cache(maxsize=4)
def _load_minute_table(minute_path: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(minute_path))
    if isinstance(df.index, pd.MultiIndex) or df.index.name in {
        "trade_date",
        "trade_time",
    }:
        df = df.reset_index()
    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename(columns={"volume": "vol"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df.sort_values(["trade_date", "trade_time"], kind="mergesort")
    return df.set_index("trade_date", drop=False)


def normalize_quote_frame(symbol_dir: Path) -> pd.DataFrame:
    quote_path = symbol_dir / "行情.csv"
    if not quote_path.exists():
        raise FileNotFoundError(f"Missing quote file: {quote_path}")

    df = read_csv_subset(quote_path, QUOTE_USECOLS, QUOTE_COLUMN_NAMES)
    df["trade_time"] = parse_trade_time(df["trade_date"], df["raw_time"])
    numeric_columns = [column for column in df.columns if column.startswith(("ask_", "bid_"))]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    price_columns = [column for column in numeric_columns if "price" in column]
    qty_columns = [column for column in numeric_columns if "qty" in column]
    df[price_columns] = df[price_columns] / 10000.0
    df[qty_columns] = df[qty_columns].fillna(0.0)

    for level in range(1, 11):
        ask_price_column = f"ask_price{level}"
        ask_qty_column = f"ask_qty{level}"
        bid_price_column = f"bid_price{level}"
        bid_qty_column = f"bid_qty{level}"

        invalid_ask = ~(df[ask_price_column] > 0)
        invalid_bid = ~(df[bid_price_column] > 0)

        df.loc[invalid_ask, ask_price_column] = np.nan
        df.loc[invalid_ask, ask_qty_column] = 0.0
        df.loc[invalid_bid, bid_price_column] = np.nan
        df.loc[invalid_bid, bid_qty_column] = 0.0

    df["ts_code"] = df["ts_code"].astype(str)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_time")
    df = df.drop_duplicates("trade_time", keep="last")
    df = df.set_index("trade_time")
    df.index.name = "trade_time"
    return df


def normalize_order_frame(symbol_dir: Path) -> pd.DataFrame:
    order_path = symbol_dir / "逐笔委托.csv"
    if not order_path.exists():
        raise FileNotFoundError(f"Missing order file: {order_path}")

    df = read_csv_subset(order_path, ORDER_USECOLS, ORDER_COLUMN_NAMES)
    df["event_time"] = parse_trade_time(df["trade_date"], df["raw_time"])
    df["side"] = df["side"].astype(str).str.strip().str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce") / 10000.0
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df[df["side"].isin(["B", "S"])]
    df = df[df["price"] > 0]
    df = df[df["qty"] > 0]
    df["notional"] = df["price"] * df["qty"]
    df = df.sort_values("event_time")
    return df[["event_time", "side", "price", "qty", "notional"]]


def normalize_trade_frame(symbol_dir: Path) -> pd.DataFrame:
    trade_path = symbol_dir / "逐笔成交.csv"
    if not trade_path.exists():
        raise FileNotFoundError(f"Missing trade file: {trade_path}")

    df = read_csv_subset(trade_path, TRADE_USECOLS, TRADE_COLUMN_NAMES)
    df["event_time"] = parse_trade_time(df["trade_date"], df["raw_time"])
    df["side"] = df["side"].astype(str).str.strip().str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce") / 10000.0
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df[df["side"].isin(["B", "S"])]
    df = df[df["price"] > 0]
    df = df[df["qty"] > 0]
    df["notional"] = df["price"] * df["qty"]
    df = df.sort_values("event_time")
    return df[["event_time", "side", "price", "qty", "notional"]]


def load_minute_frame(
    minute_root: Path, symbol: str, trade_date: str
) -> pd.DataFrame:
    minute_path = minute_root / f"{symbol}.parquet"
    if not minute_path.exists():
        raise FileNotFoundError(f"Missing minute parquet: {minute_path}")

    trade_day = pd.Timestamp(trade_date)
    minute_table = _load_minute_table(str(minute_path))
    try:
        df = minute_table.loc[trade_day].copy()
    except KeyError as exc:
        raise ValueError(f"No minute rows for {symbol} on {trade_date}") from exc
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    if df.empty:
        raise ValueError(f"No minute rows for {symbol} on {trade_date}")
    df = df.sort_values("trade_time", kind="mergesort").drop_duplicates(
        "trade_time", keep="last"
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df["trade_time"] = df["trade_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["ts_code"] = symbol
    missing = [column for column in BASE_OUTPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing minute columns for {symbol}: {missing}")
    return df[BASE_OUTPUT_COLUMNS]


def build_output_frame(
    minute_df: pd.DataFrame,
    factors: pd.DataFrame,
    factor_columns: list[str] = FACTOR_COLUMNS,
) -> pd.DataFrame:
    minute_times = pd.to_datetime(minute_df["trade_time"])
    factor_frame = factors.reset_index().sort_values("trade_time")
    aligned = pd.merge_asof(
        pd.DataFrame({"trade_time": minute_times}).sort_values("trade_time"),
        factor_frame,
        on="trade_time",
        direction="backward",
        tolerance=pd.Timedelta("60s"),
    )
    aligned.index = minute_df.index
    result = pd.concat(
        [minute_df.reset_index(drop=True), aligned[factor_columns].reset_index(drop=True)],
        axis=1,
    )
    result = result[BASE_OUTPUT_COLUMNS + factor_columns]
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def merge_symbol_output(
    output_path: Path,
    result: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    if not output_path.exists():
        return result.reset_index(drop=True)

    existing = pd.read_parquet(output_path)
    incoming_dates = set(result["trade_date"].astype(str))
    existing_dates = existing["trade_date"].astype(str)
    if overwrite:
        existing = existing.loc[~existing_dates.isin(incoming_dates)]
    else:
        result = result.loc[~result["trade_date"].astype(str).isin(set(existing_dates))]
    combined = pd.concat([existing, result], ignore_index=True)
    return combined.sort_values(["trade_date", "trade_time"], kind="mergesort").reset_index(
        drop=True
    )


def process_symbol_dir(
    symbol_dir: Path,
    minute_path: Path,
    output_root: Path,
    date_dir_name: str,
    overwrite: bool,
    window_profile: str,
    factor_columns: list[str],
) -> tuple[str, Path, int | None, int | None]:
    canonical_symbol = minute_path.stem
    output_path = output_root / f"{canonical_symbol}.parquet"
    if output_path.exists() and not overwrite:
        existing_dates = pd.read_parquet(output_path, columns=["trade_date"])[
            "trade_date"
        ].astype(str)
        normalized_date = pd.Timestamp(date_dir_name).strftime("%Y-%m-%d")
        if normalized_date in set(existing_dates):
            return ("skipped", output_path, None, None)

    quotes = normalize_quote_frame(symbol_dir)
    orders = normalize_order_frame(symbol_dir)
    trades = normalize_trade_frame(symbol_dir)
    factors = build_stock_orderbook_factor_frame(
        quotes, orders, trades, window_profile=window_profile
    )
    minute_df = load_minute_frame(minute_path.parent, canonical_symbol, date_dir_name)
    result = build_output_frame(minute_df, factors, factor_columns)
    result = merge_symbol_output(output_path, result, overwrite)

    output_root.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    return ("written", output_path, len(result), len(result.columns))


def main() -> int:
    args = parse_args()
    configure_logging()
    date_dirs = discover_trade_date_dirs(args.input_root)
    stock_minutes = build_minute_file_index(args.stock_minute_root)
    etf_minutes = build_minute_file_index(args.etf_minute_root)
    stock_output_root = resolve_output_root(
        args.stock_output_root,
        args.window_profile,
        DEFAULT_STOCK_OUTPUT_ROOT,
        DEFAULT_STOCK_MULTIWINDOW_OUTPUT_ROOT,
    )
    etf_output_root = resolve_output_root(
        args.etf_output_root,
        args.window_profile,
        DEFAULT_ETF_OUTPUT_ROOT,
        DEFAULT_ETF_MULTIWINDOW_OUTPUT_ROOT,
    )
    factor_columns = factor_columns_for_profile(args.window_profile)
    failures: list[tuple[Path, str]] = []
    worker_count = max(1, args.workers)
    processed_tasks = 0
    skipped_without_minute = 0

    for date_dir in date_dirs:
        tasks: list[tuple[Path, Path, Path]] = []
        for symbol_dir in discover_symbol_dirs(date_dir, args.symbols):
            code = numeric_code(symbol_dir.name)
            if code in stock_minutes:
                tasks.append(
                    (symbol_dir, stock_minutes[code], stock_output_root)
                )
            elif code in etf_minutes:
                tasks.append((symbol_dir, etf_minutes[code], etf_output_root))
            else:
                skipped_without_minute += 1

        if args.limit is not None:
            remaining = args.limit - processed_tasks
            if remaining <= 0:
                break
            tasks = tasks[:remaining]
        if not tasks:
            continue

        LOGGER.info("Processing %s matched symbols for %s", len(tasks), date_dir.name)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_symbol_dir,
                    symbol_dir,
                    minute_path,
                    output_root,
                    date_dir.name,
                    args.overwrite,
                    args.window_profile,
                    factor_columns,
                ): symbol_dir
                for symbol_dir, minute_path, output_root in tasks
            }
            for future in as_completed(future_map):
                symbol_dir = future_map[future]
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
                    failures.append((symbol_dir, str(exc)))
                    LOGGER.exception("Failed to process %s", symbol_dir)
        processed_tasks += len(tasks)

    LOGGER.info(
        "Matched %s date/symbol tasks; skipped %s without minute data",
        processed_tasks,
        skipped_without_minute,
    )
    if failures:
        LOGGER.error("Completed with %s failures", len(failures))
        for failed_path, reason in failures[:10]:
            LOGGER.error("  %s -> %s", failed_path, reason)
        return 1

    LOGGER.info("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
