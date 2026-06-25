from __future__ import annotations

"""Generate A-share stock orderbook factors from raw per-symbol tick CSV folders.

This entrypoint is intentionally separate from ``generate_etf_minute_factors.py``.
It reads one trade-date directory of stock quote/order/trade CSV files, computes
snapshot-level orderbook factors via ``factor.stock_orderbook``, and writes the
results to ``a-share-data/orderbook_factors/<trade_date>/*.parquet``.

It does not read ETF minute parquet inputs and does not share the ETF minute
factor set.
"""

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

from factor.stock_orderbook import build_stock_orderbook_factor_frame  # noqa: E402


LOGGER = logging.getLogger("generate_stock_orderbook_factors")

# Raw A-share tick CSV root for one trade date. This is not the ETF minute parquet input tree.
DEFAULT_INPUT_ROOT = Path(r"D:\BaiduNetdiskDownload\2025\202501\20250102")
# Stock orderbook outputs stay under a-share-data and are kept separate from ETF factor outputs.
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\a-share-data\orderbook_factors")

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
OUTPUT_COLUMNS = [
    "ts_code",
    "trade_date",
    # Snapshot quote factors.
    "mid_price",
    "spread_bps",
    "depth_imbalance_l1",
    "depth_imbalance_l5",
    "bid_ask_qty_ratio_l1",
    "depth_l5_total",
    "orderbook_decay_l5",
    "orderbook_asymmetry_l5",
    "depth_concentration_l5",
    "orderbook_liquidity_l5",
    "book_pressure_wap5",
    "book_slope_diff_l5",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stock orderbook factor parquet files from local tick CSVs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing per-symbol folders with quote/order/trade CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where orderbook factor parquet files will be written.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional stock symbols such as 000001.SZ 600000.SH.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N symbol directories after filtering.",
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


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def discover_symbol_dirs(input_root: Path, symbols: list[str] | None) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    if symbols:
        symbol_dirs = [input_root / symbol for symbol in symbols]
    else:
        symbol_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())

    missing_dirs = [path for path in symbol_dirs if not path.exists()]
    if missing_dirs:
        missing_text = ", ".join(str(path) for path in missing_dirs[:5])
        raise FileNotFoundError(f"Missing symbol directories: {missing_text}")

    return symbol_dirs


def parse_trade_time(trade_date: pd.Series, raw_time: pd.Series) -> pd.Series:
    date_text = trade_date.astype(str).str.extract(r"(\d{8})", expand=False)
    time_text = raw_time.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(9)
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


def build_output_frame(quotes: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([quotes[["ts_code", "trade_date"]], factors], axis=1)
    result = result[OUTPUT_COLUMNS]
    result = result.replace([np.inf, -np.inf], np.nan)
    result.index.name = "trade_time"
    return result


def process_symbol_dir(
    symbol_dir: Path, output_root: Path, date_dir_name: str, overwrite: bool
) -> tuple[str, Path, int | None, int | None]:
    output_dir = output_root / date_dir_name
    output_path = output_dir / f"{symbol_dir.name}.parquet"
    if output_path.exists() and not overwrite:
        return ("skipped", output_path, None, None)

    quotes = normalize_quote_frame(symbol_dir)
    orders = normalize_order_frame(symbol_dir)
    trades = normalize_trade_frame(symbol_dir)
    factors = build_stock_orderbook_factor_frame(quotes, orders, trades)
    result = build_output_frame(quotes, factors)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    return ("written", output_path, len(result), len(result.columns))


def main() -> int:
    args = parse_args()
    configure_logging()

    symbol_dirs = discover_symbol_dirs(args.input_root, args.symbols)
    if args.limit is not None:
        symbol_dirs = symbol_dirs[: args.limit]

    if not symbol_dirs:
        LOGGER.warning("No symbol directories matched the requested inputs.")
        return 0

    date_dir_name = args.input_root.name
    LOGGER.info(
        "Processing %s stock orderbook directories for %s",
        len(symbol_dirs),
        date_dir_name,
    )

    failures: list[tuple[Path, str]] = []
    worker_count = max(1, args.workers)

    if worker_count == 1:
        for symbol_dir in symbol_dirs:
            try:
                status, output_path, row_count, column_count = process_symbol_dir(
                    symbol_dir, args.output_root, date_dir_name, args.overwrite
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
                failures.append((symbol_dir, str(exc)))
                LOGGER.exception("Failed to process %s", symbol_dir)
    else:
        LOGGER.info("Using %s parallel workers", worker_count)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_symbol_dir,
                    symbol_dir,
                    args.output_root,
                    date_dir_name,
                    args.overwrite,
                ): symbol_dir
                for symbol_dir in symbol_dirs
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

    if failures:
        LOGGER.error("Completed with %s failures", len(failures))
        for failed_path, reason in failures[:10]:
            LOGGER.error("  %s -> %s", failed_path, reason)
        return 1

    LOGGER.info("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
