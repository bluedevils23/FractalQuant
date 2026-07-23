"""Generate daily auction factors and optional causal minute path companions."""

from __future__ import annotations

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("generate_auction_factors")

DEFAULT_TICK_ROOT = Path(r"E:\逐笔数据")
DEFAULT_STOCK_MINUTE_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\行情数据\stock_1min"
)
DEFAULT_ETF_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_STOCK_DAILY_PATH = Path(
    r"D:\workspace\stockdata\a-share-data\stock_daily.parquet"
)
DEFAULT_ETF_DAILY_PATH = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_STOCK_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\stock_auction_factors"
)
DEFAULT_ETF_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_auction_factors")
DEFAULT_STOCK_SESSION_PATH_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\stock_intraday_session_path_factors"
)
DEFAULT_ETF_SESSION_PATH_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_intraday_session_path_factors"
)
DEFAULT_BENCHMARK_TS_CODE = "510300.SH"

ASSET_TYPES = ("stock", "etf", "both")
DATE_PATTERN = re.compile(r"^\d{8}$")
SYMBOL_PATTERN = re.compile(r"^(\d{6})(?:\.(?:SH|SZ|BJ))?$", re.IGNORECASE)

KEY_COLUMNS = ["trade_date", "available_time", "ts_code"]
DIAGNOSTIC_COLUMNS = [
    "auction_has_match",
    "snapshot_count_stage1",
    "snapshot_count_stage2",
    "auction_event_reconstruction_ok",
    "auction_add_count_stage1",
    "auction_cancel_count_stage1",
    "auction_add_count_stage2",
    "auction_large_order_history_days",
    "auction_stage2_twap_coverage_ratio",
    "benchmark_auction_has_match",
]
REFERENCE_COLUMNS = [
    "previous_close",
    "auction_open_price",
    "auction_final_indicative_price",
    "auction_amount",
    "auction_matched_volume",
    "auction_stage1_add_notional",
    "auction_stage1_cancel_notional",
    "auction_stage2_add_notional",
    "auction_large_order_threshold",
    "previous_20d_average_daily_amount",
    "previous_5d_average_daily_amount",
    "auction_stage2_twap_price",
    "auction_submitted_volume",
    "benchmark_ts_code",
    "benchmark_available_time",
]
CORE_FACTOR_COLUMNS = [
    "auction_overnight_return",
    "auction_return_stage1",
    "auction_return_stage2",
    "auction_amount_ratio_5d",
    "auction_imbalance_change_stage1",
    "auction_imbalance_change_stage2",
    "auction_commitment_shift",
    "auction_stage2_slope_bps_per_min",
    "auction_stage2_range_bps",
    "auction_stage2_efficiency_ratio",
    "auction_matched_volume_ratio_5d",
    "auction_unmatched_imbalance",
]
EVENT_FACTOR_COLUMNS = [
    "auction_bid_cancel_qty_ratio_stage1",
    "auction_ask_cancel_qty_ratio_stage1",
    "auction_cancel_notional_ratio_stage1",
    "auction_cancel_imbalance_stage1",
    "auction_late_cancel_notional_share",
    "auction_large_order_cancel_ratio_stage1",
    "auction_large_cancel_imbalance_stage1",
    "auction_stage2_add_imbalance",
    "auction_stage2_commitment_ratio",
    "auction_stage2_last60s_add_share",
    "auction_fake_pressure_proxy",
    "auction_stage_reversal_strength_bps",
]
PATH_FACTOR_COLUMNS = [
    "auction_stage2_mid_mean_return",
    "auction_stage2_mid_max_return",
    "auction_stage2_mid_min_return",
    "auction_stage2_total_variation_bps",
    "auction_stage2_up_step_ratio",
    "auction_stage2_reversal_count",
]
ROBUST_IMBALANCE_FACTOR_COLUMNS = [
    "auction_imbalance_relative_change_stage1",
    "auction_imbalance_relative_change_stage2",
    "auction_imbalance_fisher_change_stage1",
    "auction_imbalance_fisher_change_stage2",
]
PARTICIPATION_FACTOR_COLUMNS = [
    "auction_amount_to_prev5d_adv_240",
    "auction_amount_to_prev20d_adv",
    "auction_amount_zscore_20d",
    "auction_matched_volume_to_submitted_ratio",
]
PRIORITY_REPORT_FACTOR_COLUMNS = [
    "auction_final_vs_stage2_twap",
    "auction_l3_imbalance_twap_stage2",
    "auction_relative_spread_twap_stage2",
    "prevday_intraday_drawdown_from_session_high",
    "prevday_intraday_rebound_from_session_low",
    "prevday_intraday_return_from_prev_close",
    "prev_2d_return_rank_cs",
    "prev_20d_return_rank_cs",
    "market_return_from_prev_close",
    "market_above_ma20_prevclose",
    "market_momentum_2d_prevclose",
    "auction_gap_excess_benchmark",
    "auction_stage2_excess_return_benchmark",
]
FACTOR_COLUMNS = (
    CORE_FACTOR_COLUMNS
    + EVENT_FACTOR_COLUMNS
    + PATH_FACTOR_COLUMNS
    + ROBUST_IMBALANCE_FACTOR_COLUMNS
    + PARTICIPATION_FACTOR_COLUMNS
    + PRIORITY_REPORT_FACTOR_COLUMNS
)
OUTPUT_COLUMNS = KEY_COLUMNS + DIAGNOSTIC_COLUMNS + REFERENCE_COLUMNS + FACTOR_COLUMNS
SESSION_PATH_OUTPUT_COLUMNS = [
    "trade_date",
    "bar_time",
    "available_time",
    "ts_code",
    "intraday_drawdown_from_session_high",
    "intraday_rebound_from_session_low",
    "intraday_return_from_prev_close",
]

EVENT_COLUMNS = [
    "trade_time",
    "event_type",
    "side",
    "order_id",
    "price",
    "quantity",
    "notional",
    "original_notional",
]
LARGE_ORDER_LOOKBACK_DAYS = 20
LARGE_ORDER_QUANTILE = 0.90
HISTORICAL_AMOUNT_LOOKBACK_DAYS = 20
SHORT_DAILY_AMOUNT_LOOKBACK_DAYS = 5
MINUTES_PER_TRADING_DAY = 240
STAGE2_TWAP_MIN_COVERAGE = 0.80
IMBALANCE_RELATIVE_FLOOR = 0.05
IMBALANCE_FISHER_CLIP = 1.0 - 1e-6

RAW_COLUMN_MAP = {
    "自然日": "raw_trade_date",
    "时间": "raw_time",
    "成交价": "trade_price",
    "成交量": "trade_volume",
    "成交额": "trade_amount",
    "开盘价": "open_price",
    "前收盘": "previous_close",
    **{f"申卖价{level}": f"ask_price{level}" for level in range(1, 4)},
    **{f"申卖量{level}": f"ask_qty{level}" for level in range(1, 4)},
    **{f"申买价{level}": f"bid_price{level}" for level in range(1, 4)},
    **{f"申买量{level}": f"bid_qty{level}" for level in range(1, 4)},
}
PRICE_COLUMNS = [
    "trade_price",
    "open_price",
    "previous_close",
    *[f"ask_price{level}" for level in range(1, 4)],
    *[f"bid_price{level}" for level in range(1, 4)],
]
QUANTITY_COLUMNS = [
    "trade_volume",
    *[f"ask_qty{level}" for level in range(1, 4)],
    *[f"bid_qty{level}" for level in range(1, 4)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate daily opening-auction factors for stocks and ETFs."
    )
    parser.add_argument("--tick-root", type=Path, default=DEFAULT_TICK_ROOT)
    parser.add_argument(
        "--asset-type",
        choices=ASSET_TYPES,
        default="both",
        help="Asset universe to process.",
    )
    parser.add_argument(
        "--stock-minute-root", type=Path, default=DEFAULT_STOCK_MINUTE_ROOT
    )
    parser.add_argument("--etf-minute-root", type=Path, default=DEFAULT_ETF_MINUTE_ROOT)
    parser.add_argument("--stock-daily-path", type=Path, default=DEFAULT_STOCK_DAILY_PATH)
    parser.add_argument("--etf-daily-path", type=Path, default=DEFAULT_ETF_DAILY_PATH)
    parser.add_argument(
        "--stock-output-root", type=Path, default=DEFAULT_STOCK_OUTPUT_ROOT
    )
    parser.add_argument("--etf-output-root", type=Path, default=DEFAULT_ETF_OUTPUT_ROOT)
    parser.add_argument(
        "--stock-session-path-output-root",
        type=Path,
        default=DEFAULT_STOCK_SESSION_PATH_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--etf-session-path-output-root",
        type=Path,
        default=DEFAULT_ETF_SESSION_PATH_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbols or six-digit codes.",
    )
    parser.add_argument(
        "--symbols-file", type=Path, default=None, help="Optional UTF-8 symbol list."
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--benchmark-ts-code",
        type=str,
        default=DEFAULT_BENCHMARK_TS_CODE,
        help="Tradable auction benchmark proxy; default: 510300.SH.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N matched symbols.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace requested dates while preserving dates outside the requested range.",
    )
    parser.add_argument(
        "--write-session-path-factors",
        action="store_true",
        help="Also write causal minute-level session path companion parquets.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def normalize_trade_date_arg(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 8:
        raise ValueError(f"Invalid trade date: {value}")
    return digits


def numeric_code(value: str) -> str | None:
    match = re.match(r"^(\d{6})", str(value).strip())
    return match.group(1) if match else None


def normalize_requested_symbol(value: str) -> str:
    symbol = str(value).strip().upper()
    if symbol.lower().endswith(".parquet"):
        symbol = symbol[:-8]
    match = SYMBOL_PATTERN.fullmatch(symbol)
    if not match:
        raise ValueError(f"Invalid symbol: {value}")
    return match.group(1)


def read_symbol_list_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file does not exist: {path}")
    symbols: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                symbols.append(normalize_requested_symbol(line))
    return symbols


def load_requested_codes(
    symbols: list[str] | None, symbols_file: Path | None
) -> set[str] | None:
    requested: list[str] = []
    if symbols_file is not None:
        requested.extend(read_symbol_list_file(symbols_file))
    if symbols:
        requested.extend(normalize_requested_symbol(symbol) for symbol in symbols)
    return set(requested) if requested else None


def build_universe_index(minute_root: Path) -> dict[str, str]:
    if not minute_root.exists():
        raise FileNotFoundError(f"Minute directory does not exist: {minute_root}")

    candidates: dict[str, list[str]] = {}
    for path in minute_root.glob("*.parquet"):
        code = numeric_code(path.stem)
        if code is not None:
            candidates.setdefault(code, []).append(path.stem.upper())

    canonical = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$", re.IGNORECASE)
    result: dict[str, str] = {}
    for code, symbols in candidates.items():
        result[code] = sorted(
            symbols,
            key=lambda symbol: (not bool(canonical.fullmatch(symbol)), symbol),
        )[0]
    return result


def build_asset_universe(
    asset_type: str,
    stock_minute_root: Path,
    etf_minute_root: Path,
    requested_codes: set[str] | None,
) -> list[tuple[str, str, str]]:
    assets: list[tuple[str, str, str]] = []
    indexes: list[tuple[str, dict[str, str]]] = []
    if asset_type in {"stock", "both"}:
        indexes.append(("stock", build_universe_index(stock_minute_root)))
    if asset_type in {"etf", "both"}:
        indexes.append(("etf", build_universe_index(etf_minute_root)))

    owners: dict[str, str] = {}
    for kind, index in indexes:
        for code, symbol in index.items():
            if requested_codes is not None and code not in requested_codes:
                continue
            previous_owner = owners.get(code)
            if previous_owner is not None and previous_owner != kind:
                raise ValueError(
                    f"Numeric code {code} exists in both stock and ETF universes. "
                    "Use --asset-type to disambiguate."
                )
            owners[code] = kind
            assets.append((kind, code, symbol))

    if requested_codes is not None:
        found = {code for _, code, _ in assets}
        missing = sorted(requested_codes - found)
        if missing:
            raise FileNotFoundError(
                "Requested symbols not found in selected minute universes: "
                + ", ".join(missing[:20])
            )
    return sorted(assets, key=lambda item: (item[0], item[2]))


def discover_trade_date_dirs(tick_root: Path, date_to: str | None) -> list[Path]:
    if not tick_root.exists():
        raise FileNotFoundError(f"Tick root does not exist: {tick_root}")
    date_dirs = [
        path
        for path in tick_root.glob("*/*/*")
        if path.is_dir()
        and DATE_PATTERN.fullmatch(path.name)
        and (date_to is None or path.name <= date_to)
    ]
    return sorted(date_dirs)


def group_symbol_paths(
    date_dirs: list[Path], selected_codes: set[str]
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {code: [] for code in selected_codes}
    if len(selected_codes) <= 200:
        suffixes = (".SH", ".SZ", ".BJ", "")
        for date_dir in date_dirs:
            for code in selected_codes:
                matches = sorted(
                    candidate
                    for suffix in suffixes
                    if (candidate := date_dir / f"{code}{suffix}").is_dir()
                )
                if matches:
                    grouped[code].append(matches[0])
        return grouped

    for date_dir in date_dirs:
        matches_by_code: dict[str, list[Path]] = {}
        for symbol_dir in date_dir.iterdir():
            code = numeric_code(symbol_dir.name)
            if symbol_dir.is_dir() and code in grouped:
                matches_by_code.setdefault(code, []).append(symbol_dir)
        for code, matches in matches_by_code.items():
            grouped[code].append(sorted(matches)[0])
    return grouped


def parse_trade_time(trade_date: pd.Series, raw_time: pd.Series) -> pd.Series:
    date_text = trade_date.astype(str).str.zfill(8)
    time_text = raw_time.astype(str).str.zfill(9)
    return pd.to_datetime(
        date_text + time_text,
        format="%Y%m%d%H%M%S%f",
        errors="coerce",
    )


def load_quote_frame(symbol_dir: Path) -> pd.DataFrame:
    quote_path = symbol_dir / "行情.csv"
    if not quote_path.exists():
        raise FileNotFoundError(f"Missing quote file: {quote_path}")

    frame = pd.read_csv(
        quote_path,
        encoding="gbk",
        usecols=list(RAW_COLUMN_MAP),
        dtype=str,
        low_memory=False,
    ).rename(columns=RAW_COLUMN_MAP)
    frame["trade_time"] = parse_trade_time(frame["raw_trade_date"], frame["raw_time"])

    numeric_columns = PRICE_COLUMNS + QUANTITY_COLUMNS + ["trade_amount"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[PRICE_COLUMNS] = frame[PRICE_COLUMNS] / 10000.0
    frame[PRICE_COLUMNS] = frame[PRICE_COLUMNS].where(frame[PRICE_COLUMNS] > 0)
    frame[QUANTITY_COLUMNS + ["trade_amount"]] = frame[
        QUANTITY_COLUMNS + ["trade_amount"]
    ].fillna(0.0)

    frame = frame.dropna(subset=["trade_time"])
    frame = frame.sort_values("trade_time", kind="mergesort")
    return frame.drop_duplicates("trade_time", keep="last").reset_index(drop=True)


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _load_raw_orders(symbol_dir: Path) -> pd.DataFrame:
    path = symbol_dir / "逐笔委托.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing order file: {path}")
    frame = pd.read_csv(
        path,
        encoding="gbk",
        usecols=[
            "自然日",
            "时间",
            "交易所委托号",
            "委托类型",
            "委托代码",
            "委托价格",
            "委托数量",
        ],
        dtype=str,
        low_memory=False,
    ).rename(
        columns={
            "自然日": "raw_trade_date",
            "时间": "raw_time",
            "交易所委托号": "order_id",
            "委托类型": "order_type",
            "委托代码": "side",
            "委托价格": "price",
            "委托数量": "quantity",
        }
    )
    frame["trade_time"] = parse_trade_time(frame["raw_trade_date"], frame["raw_time"])
    frame["order_id"] = pd.to_numeric(frame["order_id"], errors="coerce").astype(
        "Int64"
    )
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce") / 10000.0
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["order_type"] = frame["order_type"].astype(str).str.strip().str.upper()
    frame["side"] = frame["side"].astype(str).str.strip().str.upper()
    return frame


def _load_sz_cancellations(symbol_dir: Path) -> pd.DataFrame:
    path = symbol_dir / "逐笔成交.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing transaction file: {path}")
    frame = pd.read_csv(
        path,
        encoding="gbk",
        usecols=[
            "自然日",
            "时间",
            "成交代码",
            "成交数量",
            "叫卖序号",
            "叫买序号",
        ],
        dtype=str,
        low_memory=False,
    ).rename(
        columns={
            "自然日": "raw_trade_date",
            "时间": "raw_time",
            "成交代码": "trade_code",
            "成交数量": "quantity",
            "叫卖序号": "ask_order_id",
            "叫买序号": "bid_order_id",
        }
    )
    frame["trade_time"] = parse_trade_time(frame["raw_trade_date"], frame["raw_time"])
    frame["trade_code"] = frame["trade_code"].astype(str).str.strip().str.upper()
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["ask_order_id"] = (
        pd.to_numeric(frame["ask_order_id"], errors="coerce").fillna(0).astype("int64")
    )
    frame["bid_order_id"] = (
        pd.to_numeric(frame["bid_order_id"], errors="coerce").fillna(0).astype("int64")
    )
    return frame


def _auction_event_bounds(frame: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    valid_times = frame["trade_time"].dropna()
    if valid_times.empty:
        raise ValueError("Auction event file contains no valid timestamp")
    trade_day = pd.Timestamp(valid_times.iloc[0]).normalize()
    return (
        trade_day + pd.Timedelta(hours=9, minutes=15),
        trade_day + pd.Timedelta(hours=9, minutes=20),
        trade_day + pd.Timedelta(hours=9, minutes=25),
    )


def _finalize_event_frame(
    adds: pd.DataFrame, cancellations: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    valid = True
    required_add = adds[
        adds["trade_time"].notna()
        & adds["order_id"].notna()
        & adds["side"].isin(["B", "S"])
        & adds["price"].gt(0)
        & adds["quantity"].gt(0)
    ].copy()
    if len(required_add) != len(adds) or required_add["order_id"].duplicated().any():
        valid = False

    required_add["event_type"] = "A"
    required_add["notional"] = required_add["price"] * required_add["quantity"]
    required_add["original_notional"] = required_add["notional"]

    if cancellations.empty:
        return required_add.reindex(columns=EVENT_COLUMNS), valid

    required_cancel = cancellations[
        cancellations["trade_time"].notna()
        & cancellations["order_id"].notna()
        & cancellations["side"].isin(["B", "S"])
        & cancellations["quantity"].gt(0)
    ].copy()
    if len(required_cancel) != len(cancellations):
        valid = False

    lookup = required_add[
        ["order_id", "trade_time", "side", "price", "quantity", "original_notional"]
    ].rename(
        columns={
            "trade_time": "add_time",
            "side": "add_side",
            "price": "add_price",
            "quantity": "add_quantity",
        }
    )
    joined = required_cancel.merge(
        lookup, on="order_id", how="left", validate="many_to_one"
    )
    matched = joined["add_time"].notna()
    valid &= bool(matched.all())
    if matched.any():
        valid &= bool(
            (joined.loc[matched, "side"] == joined.loc[matched, "add_side"]).all()
        )
        valid &= bool(
            (joined.loc[matched, "trade_time"] >= joined.loc[matched, "add_time"]).all()
        )
        cancelled_by_order = joined.loc[matched].groupby("order_id")["quantity"].sum()
        original_by_order = lookup.set_index("order_id")["add_quantity"]
        valid &= bool(
            (
                cancelled_by_order <= original_by_order.loc[cancelled_by_order.index]
            ).all()
        )

    joined["price"] = joined["add_price"]
    joined["notional"] = joined["price"] * joined["quantity"]
    joined["event_type"] = "C"
    joined["original_notional"] = joined["original_notional"]
    cancel_events = joined.reindex(columns=EVENT_COLUMNS)
    events = pd.concat(
        [required_add.reindex(columns=EVENT_COLUMNS), cancel_events],
        ignore_index=True,
    ).sort_values("trade_time", kind="mergesort")
    return events.reset_index(drop=True), valid


def _reconstruct_sh_events(orders: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    start_time, split_time, end_time = _auction_event_bounds(orders)
    auction = orders.loc[
        orders["trade_time"].ge(start_time) & orders["trade_time"].lt(end_time)
    ].copy()
    known_types = auction["order_type"].isin(["A", "D"])
    valid = bool(known_types.all())
    adds = auction.loc[auction["order_type"].eq("A")].copy()
    cancellations = auction.loc[auction["order_type"].eq("D")].copy()
    events, linked = _finalize_event_frame(adds, cancellations)
    valid &= linked
    valid &= not bool(cancellations["trade_time"].ge(split_time).any())
    return events, valid


def _reconstruct_sz_events(
    orders: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    start_time, split_time, end_time = _auction_event_bounds(orders)
    adds = orders.loc[
        orders["trade_time"].ge(start_time) & orders["trade_time"].lt(end_time)
    ].copy()
    cancellations = transactions.loc[
        transactions["trade_time"].ge(start_time)
        & transactions["trade_time"].lt(end_time)
        & transactions["trade_code"].eq("C")
    ].copy()
    ask_present = cancellations["ask_order_id"].gt(0)
    bid_present = cancellations["bid_order_id"].gt(0)
    sequence_ok = ask_present ^ bid_present
    cancellations["order_id"] = (
        cancellations["ask_order_id"]
        .where(ask_present, cancellations["bid_order_id"])
        .astype("Int64")
    )
    cancellations["side"] = np.where(ask_present, "S", "B")
    events, linked = _finalize_event_frame(adds, cancellations)
    valid = bool(sequence_ok.all()) and linked
    valid &= not bool(cancellations["trade_time"].ge(split_time).any())
    return events, valid


def load_auction_event_frame(
    symbol_dir: Path, ts_code: str
) -> tuple[pd.DataFrame, bool]:
    exchange = ts_code.rsplit(".", 1)[-1].upper()
    try:
        orders = _load_raw_orders(symbol_dir)
        if exchange == "SH":
            return _reconstruct_sh_events(orders)
        if exchange == "SZ":
            return _reconstruct_sz_events(orders, _load_sz_cancellations(symbol_dir))
        LOGGER.warning("Unsupported exchange for auction events: %s", ts_code)
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.warning("Could not reconstruct auction events for %s: %s", ts_code, exc)
    return _empty_event_frame(), False


def _safe_return(end_value: float, start_value: float) -> float:
    if not np.isfinite(end_value) or not np.isfinite(start_value) or start_value <= 0:
        return np.nan
    return float(end_value / start_value - 1.0)


def _first_finite(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values) & (values > 0)]
    return float(values.iloc[0]) if not values.empty else np.nan


def _calculate_indicative_price(frame: pd.DataFrame) -> pd.Series:
    ask = frame["ask_price1"]
    bid = frame["bid_price1"]
    valid = ask.notna() & bid.notna() & (ask > 0) & (bid > 0)
    return ((ask + bid) / 2.0).where(valid)


def _calculate_l3_imbalance(frame: pd.DataFrame) -> pd.Series:
    bid_qty = frame[[f"bid_qty{level}" for level in range(1, 4)]].sum(axis=1)
    ask_qty = frame[[f"ask_qty{level}" for level in range(1, 4)]].sum(axis=1)
    total = bid_qty + ask_qty
    return ((bid_qty - ask_qty) / total).where(total > 0)


def _calculate_relative_spread(frame: pd.DataFrame) -> pd.Series:
    ask = pd.to_numeric(frame["ask_price1"], errors="coerce")
    bid = pd.to_numeric(frame["bid_price1"], errors="coerce")
    midpoint = (ask + bid) / 2.0
    valid = ask.notna() & bid.notna() & (bid > 0) & (ask >= bid)
    return ((ask - bid) / midpoint).where(valid)


def _time_weighted_mean(
    stage: pd.DataFrame,
    column: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> tuple[float, float]:
    duration_seconds = float((end_time - start_time).total_seconds())
    if stage.empty or duration_seconds <= 0:
        return np.nan, 0.0

    ordered = (
        stage.loc[
            stage["trade_time"].ge(start_time) & stage["trade_time"].lt(end_time),
            ["trade_time", column],
        ]
        .sort_values("trade_time", kind="mergesort")
        .drop_duplicates("trade_time", keep="last")
    )
    if ordered.empty:
        return np.nan, 0.0

    next_times = ordered["trade_time"].shift(-1).fillna(end_time)
    weights = (next_times - ordered["trade_time"]).dt.total_seconds().clip(lower=0)
    values = pd.to_numeric(ordered[column], errors="coerce")
    valid = np.isfinite(values) & weights.gt(0)
    covered_seconds = float(weights.loc[valid].sum())
    coverage_ratio = covered_seconds / duration_seconds
    if covered_seconds <= 0 or coverage_ratio < STAGE2_TWAP_MIN_COVERAGE:
        return np.nan, coverage_ratio
    weighted_mean = float(np.average(values.loc[valid], weights=weights.loc[valid]))
    return weighted_mean, coverage_ratio


def _apply_stage2_twap_factors(
    row: dict[str, object],
    stage2: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> None:
    price_twap, price_coverage = _time_weighted_mean(
        stage2, "indicative_price", start_time, end_time
    )
    imbalance_twap, imbalance_coverage = _time_weighted_mean(
        stage2, "l3_imbalance", start_time, end_time
    )
    spread_twap, spread_coverage = _time_weighted_mean(
        stage2, "relative_spread", start_time, end_time
    )
    row["auction_stage2_twap_coverage_ratio"] = float(
        min(price_coverage, imbalance_coverage, spread_coverage)
    )
    row["auction_stage2_twap_price"] = price_twap
    row["auction_l3_imbalance_twap_stage2"] = imbalance_twap
    row["auction_relative_spread_twap_stage2"] = spread_twap

    open_price = row["auction_open_price"]
    if np.isfinite(open_price) and np.isfinite(price_twap) and price_twap > 0:
        row["auction_final_vs_stage2_twap"] = float(open_price / price_twap - 1.0)


def _stage2_slope(stage2: pd.DataFrame, previous_close: float) -> float:
    valid = stage2.dropna(subset=["indicative_price"])
    if len(valid) < 3 or not np.isfinite(previous_close) or previous_close <= 0:
        return np.nan
    elapsed_minutes = (
        valid["trade_time"] - valid["trade_time"].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float) / 60.0
    if np.unique(elapsed_minutes).size < 3 or np.ptp(elapsed_minutes) <= 0:
        return np.nan
    relative_log_price = np.log(
        valid["indicative_price"].to_numpy(dtype=float) / previous_close
    )
    return float(np.polyfit(elapsed_minutes, relative_log_price, 1)[0] * 10000.0)


def _stage2_efficiency(stage2: pd.DataFrame) -> float:
    prices = stage2["indicative_price"].dropna().to_numpy(dtype=float)
    if len(prices) < 2:
        return np.nan
    total_variation = float(np.abs(np.diff(prices)).sum())
    if total_variation == 0:
        return 0.0
    return float(abs(prices[-1] - prices[0]) / total_variation)


def _apply_stage2_path_factors(
    row: dict[str, object], stage2: pd.DataFrame, previous_close: float
) -> None:
    prices = stage2["indicative_price"].dropna().to_numpy(dtype=float)
    if prices.size == 0:
        return
    if np.isfinite(previous_close) and previous_close > 0:
        row["auction_stage2_mid_mean_return"] = float(
            prices.mean() / previous_close - 1.0
        )
        row["auction_stage2_mid_max_return"] = float(
            prices.max() / previous_close - 1.0
        )
        row["auction_stage2_mid_min_return"] = float(
            prices.min() / previous_close - 1.0
        )
    if prices.size < 2:
        return

    changes = np.diff(prices)
    if np.isfinite(previous_close) and previous_close > 0:
        row["auction_stage2_total_variation_bps"] = float(
            np.abs(changes).sum() / previous_close * 10000.0
        )
    row["auction_stage2_up_step_ratio"] = float(
        np.count_nonzero(changes > 0) / len(changes)
    )
    nonzero_directions = np.sign(changes[changes != 0])
    row["auction_stage2_reversal_count"] = int(
        np.count_nonzero(nonzero_directions[1:] != nonzero_directions[:-1])
    )


def _relative_imbalance_change(end_value: float, start_value: float) -> float:
    if not np.isfinite(end_value) or not np.isfinite(start_value):
        return np.nan
    denominator = max(abs(float(start_value)), IMBALANCE_RELATIVE_FLOOR)
    return float((end_value - start_value) / denominator)


def _fisher_imbalance_change(end_value: float, start_value: float) -> float:
    if not np.isfinite(end_value) or not np.isfinite(start_value):
        return np.nan
    end_clipped = np.clip(end_value, -IMBALANCE_FISHER_CLIP, IMBALANCE_FISHER_CLIP)
    start_clipped = np.clip(start_value, -IMBALANCE_FISHER_CLIP, IMBALANCE_FISHER_CLIP)
    return float(np.arctanh(end_clipped) - np.arctanh(start_clipped))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _signed_imbalance(bid_value: float, ask_value: float) -> float:
    total = bid_value + ask_value
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float((bid_value - ask_value) / total)


def _event_slice(
    events: pd.DataFrame,
    event_type: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    return events.loc[
        events["event_type"].eq(event_type)
        & events["trade_time"].ge(start_time)
        & events["trade_time"].lt(end_time)
    ]


def _side_sum(frame: pd.DataFrame, side: str, column: str) -> float:
    return float(frame.loc[frame["side"].eq(side), column].sum())


def _apply_event_factors(
    row: dict[str, object],
    trade_day: pd.Timestamp,
    events: pd.DataFrame | None,
    reconstruction_ok: bool,
) -> None:
    row["auction_event_reconstruction_ok"] = bool(reconstruction_ok)
    if events is None or events.empty:
        if reconstruction_ok:
            row["auction_cancel_imbalance_stage1"] = 0.0
            row["auction_late_cancel_notional_share"] = 0.0
        return

    start_time = trade_day + pd.Timedelta(hours=9, minutes=15)
    late_cancel_time = trade_day + pd.Timedelta(hours=9, minutes=19)
    split_time = trade_day + pd.Timedelta(hours=9, minutes=20)
    late_add_time = trade_day + pd.Timedelta(hours=9, minutes=24)
    end_time = trade_day + pd.Timedelta(hours=9, minutes=25)

    stage1_adds = _event_slice(events, "A", start_time, split_time)
    stage1_cancels = _event_slice(events, "C", start_time, split_time)
    stage2_adds = _event_slice(events, "A", split_time, end_time)
    row["auction_add_count_stage1"] = int(len(stage1_adds))
    row["auction_cancel_count_stage1"] = int(len(stage1_cancels))
    row["auction_add_count_stage2"] = int(len(stage2_adds))

    stage1_add_notional = float(stage1_adds["notional"].sum())
    stage1_cancel_notional = float(stage1_cancels["notional"].sum())
    stage2_add_notional = float(stage2_adds["notional"].sum())
    row["auction_stage1_add_notional"] = stage1_add_notional
    row["auction_stage1_cancel_notional"] = stage1_cancel_notional
    row["auction_stage2_add_notional"] = stage2_add_notional
    row["auction_submitted_volume"] = float(
        stage1_adds["quantity"].sum() + stage2_adds["quantity"].sum()
    )
    if not reconstruction_ok:
        return

    add_qty = {side: _side_sum(stage1_adds, side, "quantity") for side in ("B", "S")}
    cancel_qty = {
        side: _side_sum(stage1_cancels, side, "quantity") for side in ("B", "S")
    }
    add_notional = {
        side: _side_sum(stage1_adds, side, "notional") for side in ("B", "S")
    }
    cancel_notional = {
        side: _side_sum(stage1_cancels, side, "notional") for side in ("B", "S")
    }
    stage2_notional = {
        side: _side_sum(stage2_adds, side, "notional") for side in ("B", "S")
    }

    row["auction_bid_cancel_qty_ratio_stage1"] = _safe_ratio(
        cancel_qty["B"], add_qty["B"]
    )
    row["auction_ask_cancel_qty_ratio_stage1"] = _safe_ratio(
        cancel_qty["S"], add_qty["S"]
    )
    row["auction_cancel_notional_ratio_stage1"] = _safe_ratio(
        stage1_cancel_notional, stage1_add_notional
    )
    if stage1_cancel_notional > 0:
        row["auction_cancel_imbalance_stage1"] = float(
            (cancel_notional["S"] - cancel_notional["B"]) / stage1_cancel_notional
        )
        late_cancels = _event_slice(events, "C", late_cancel_time, split_time)
        row["auction_late_cancel_notional_share"] = float(
            late_cancels["notional"].sum() / stage1_cancel_notional
        )
    else:
        row["auction_cancel_imbalance_stage1"] = 0.0
        row["auction_late_cancel_notional_share"] = 0.0

    row["auction_stage2_add_imbalance"] = _signed_imbalance(
        stage2_notional["B"], stage2_notional["S"]
    )
    remaining = {
        side: max(add_notional[side] - cancel_notional[side], 0.0)
        for side in ("B", "S")
    }
    row["auction_stage2_commitment_ratio"] = _safe_ratio(
        stage2_add_notional,
        remaining["B"] + remaining["S"] + stage2_add_notional,
    )
    if stage2_add_notional > 0:
        late_adds = _event_slice(events, "A", late_add_time, end_time)
        row["auction_stage2_last60s_add_share"] = float(
            late_adds["notional"].sum() / stage2_add_notional
        )

    initial_imbalance = _signed_imbalance(add_notional["B"], add_notional["S"])
    surviving_imbalance = _signed_imbalance(remaining["B"], remaining["S"])
    if np.isfinite(initial_imbalance) and np.isfinite(surviving_imbalance):
        row["auction_fake_pressure_proxy"] = float(
            initial_imbalance - surviving_imbalance
        )


def _apply_matched_volume_participation(row: dict[str, object]) -> None:
    if not bool(row["auction_event_reconstruction_ok"]):
        return
    matched_volume = row["auction_matched_volume"]
    submitted_volume = row["auction_submitted_volume"]
    row["auction_matched_volume_to_submitted_ratio"] = _safe_ratio(
        matched_volume, submitted_volume
    )


def _apply_stage_reversal(row: dict[str, object]) -> None:
    stage1_return = row["auction_return_stage1"]
    stage2_return = row["auction_return_stage2"]
    if not np.isfinite(stage1_return) or not np.isfinite(stage2_return):
        return
    if stage1_return * stage2_return < 0:
        row["auction_stage_reversal_strength_bps"] = float(
            np.sign(stage2_return)
            * min(abs(stage1_return), abs(stage2_return))
            * 10000.0
        )
    else:
        row["auction_stage_reversal_strength_bps"] = 0.0


def _empty_output_row(trade_date: str, ts_code: str) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in OUTPUT_COLUMNS}
    row.update(
        {
            "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
            "available_time": pd.NaT,
            "ts_code": ts_code,
            "auction_has_match": False,
            "snapshot_count_stage1": 0,
            "snapshot_count_stage2": 0,
            "auction_event_reconstruction_ok": False,
            "auction_add_count_stage1": 0,
            "auction_cancel_count_stage1": 0,
            "auction_add_count_stage2": 0,
            "auction_large_order_history_days": 0,
        }
    )
    return row


def calculate_daily_auction_factors(
    quotes: pd.DataFrame,
    ts_code: str,
    events: pd.DataFrame | None = None,
    event_reconstruction_ok: bool = False,
) -> dict[str, object]:
    if quotes.empty:
        raise ValueError(f"Empty quote frame for {ts_code}")

    trade_day = pd.Timestamp(quotes["trade_time"].iloc[0]).normalize()
    trade_date = trade_day.strftime("%Y-%m-%d")
    row = _empty_output_row(trade_date, ts_code)
    _apply_event_factors(row, trade_day, events, event_reconstruction_ok)

    start_time = trade_day + pd.Timedelta(hours=9, minutes=15)
    split_time = trade_day + pd.Timedelta(hours=9, minutes=20)
    nominal_end_time = trade_day + pd.Timedelta(hours=9, minutes=25)
    match_deadline = trade_day + pd.Timedelta(hours=9, minutes=30)

    match_mask = (
        quotes["trade_time"].ge(nominal_end_time)
        & quotes["trade_time"].lt(match_deadline)
        & quotes["open_price"].notna()
        & quotes["trade_volume"].gt(0)
        & quotes["trade_amount"].gt(0)
    )
    match_rows = quotes.loc[match_mask]
    has_match = not match_rows.empty
    match_row = match_rows.iloc[0] if has_match else None
    auction = quotes.loc[
        quotes["trade_time"].ge(start_time)
        & quotes["trade_time"].lt(nominal_end_time)
    ].copy()
    auction["indicative_price"] = _calculate_indicative_price(auction)
    auction["l3_imbalance"] = _calculate_l3_imbalance(auction)
    auction["relative_spread"] = _calculate_relative_spread(auction)

    valid_price = auction.dropna(subset=["indicative_price"])
    stage1 = auction.loc[auction["trade_time"].lt(split_time)]
    stage2 = auction.loc[auction["trade_time"].ge(split_time)]
    stage1_valid = stage1.dropna(subset=["indicative_price"])
    stage2_valid = stage2.dropna(subset=["indicative_price"])

    previous_close = _first_finite(quotes["previous_close"])
    row["previous_close"] = previous_close
    row["auction_has_match"] = has_match
    row["snapshot_count_stage1"] = int(len(stage1_valid))
    row["snapshot_count_stage2"] = int(len(stage2_valid))

    if match_row is not None:
        row["available_time"] = pd.Timestamp(match_row["trade_time"])
        row["auction_open_price"] = float(match_row["open_price"])
        row["auction_amount"] = float(match_row["trade_amount"])
        row["auction_matched_volume"] = float(match_row["trade_volume"])
        row["auction_overnight_return"] = _safe_return(
            float(match_row["open_price"]), previous_close
        )
    elif not auction.empty or (events is not None and not events.empty):
        row["available_time"] = nominal_end_time

    _apply_matched_volume_participation(row)
    _apply_stage2_twap_factors(row, stage2, split_time, nominal_end_time)

    if not valid_price.empty:
        final = valid_price.iloc[-1]
        row["auction_final_indicative_price"] = float(final["indicative_price"])

    if len(stage1_valid) >= 2:
        stage1_first = stage1_valid.iloc[0]
        stage1_final = stage1_valid.iloc[-1]
        row["auction_return_stage1"] = _safe_return(
            float(stage1_final["indicative_price"]),
            float(stage1_first["indicative_price"]),
        )
        first_imbalance = stage1_first["l3_imbalance"]
        final_imbalance = stage1_final["l3_imbalance"]
        if np.isfinite(first_imbalance) and np.isfinite(final_imbalance):
            row["auction_imbalance_change_stage1"] = float(
                final_imbalance - first_imbalance
            )
            row["auction_imbalance_relative_change_stage1"] = _relative_imbalance_change(
                final_imbalance, first_imbalance
            )
            row["auction_imbalance_fisher_change_stage1"] = _fisher_imbalance_change(
                final_imbalance, first_imbalance
            )

    if len(stage2_valid) >= 2:
        stage2_first = stage2_valid.iloc[0]
        stage2_final = stage2_valid.iloc[-1]
        row["auction_return_stage2"] = _safe_return(
            float(stage2_final["indicative_price"]),
            float(stage2_first["indicative_price"]),
        )
        first_imbalance = stage2_first["l3_imbalance"]
        final_imbalance = stage2_final["l3_imbalance"]
        if np.isfinite(first_imbalance) and np.isfinite(final_imbalance):
            row["auction_imbalance_change_stage2"] = float(
                final_imbalance - first_imbalance
            )
            row["auction_imbalance_relative_change_stage2"] = _relative_imbalance_change(
                final_imbalance, first_imbalance
            )
            row["auction_imbalance_fisher_change_stage2"] = _fisher_imbalance_change(
                final_imbalance, first_imbalance
            )

    stage1_imbalance = stage1["l3_imbalance"].dropna()
    stage2_imbalance = stage2["l3_imbalance"].dropna()
    if not stage1_imbalance.empty and not stage2_imbalance.empty:
        row["auction_commitment_shift"] = float(
            stage2_imbalance.median() - stage1_imbalance.median()
        )

    row["auction_stage2_slope_bps_per_min"] = _stage2_slope(
        stage2_valid, previous_close
    )
    if not stage2_valid.empty and np.isfinite(previous_close) and previous_close > 0:
        row["auction_stage2_range_bps"] = float(
            (
                stage2_valid["indicative_price"].max()
                - stage2_valid["indicative_price"].min()
            )
            / previous_close
            * 10000.0
        )
    row["auction_stage2_efficiency_ratio"] = _stage2_efficiency(stage2_valid)
    _apply_stage2_path_factors(row, stage2_valid, previous_close)

    if not valid_price.empty:
        final = valid_price.iloc[-1]
        unmatched_bid = float(final["bid_qty2"])
        unmatched_ask = float(final["ask_qty2"])
        unmatched_total = unmatched_bid + unmatched_ask
        row["auction_unmatched_imbalance"] = (
            float((unmatched_bid - unmatched_ask) / unmatched_total)
            if unmatched_total > 0
            else 0.0
        )
    _apply_stage_reversal(row)
    return row


def _apply_large_order_factors(
    result: pd.DataFrame,
    index: int,
    events: pd.DataFrame,
    threshold: float,
) -> None:
    trade_day = pd.Timestamp(result.at[index, "trade_date"])
    start_time = trade_day + pd.Timedelta(hours=9, minutes=15)
    split_time = trade_day + pd.Timedelta(hours=9, minutes=20)
    stage1_adds = _event_slice(events, "A", start_time, split_time)
    stage1_cancels = _event_slice(events, "C", start_time, split_time)
    large_adds = stage1_adds.loc[stage1_adds["original_notional"].ge(threshold)]
    if large_adds.empty:
        return

    large_cancels = stage1_cancels.loc[
        stage1_cancels["original_notional"].ge(threshold)
    ]
    large_add_notional = float(large_adds["notional"].sum())
    large_cancel_notional = float(large_cancels["notional"].sum())
    result.at[index, "auction_large_order_cancel_ratio_stage1"] = _safe_ratio(
        large_cancel_notional, large_add_notional
    )
    if large_cancel_notional == 0:
        result.at[index, "auction_large_cancel_imbalance_stage1"] = 0.0
    else:
        bid_cancel = _side_sum(large_cancels, "B", "notional")
        ask_cancel = _side_sum(large_cancels, "S", "notional")
        result.at[index, "auction_large_cancel_imbalance_stage1"] = float(
            (ask_cancel - bid_cancel) / large_cancel_notional
        )


def load_daily_amount_history(minute_path: Path) -> pd.Series:
    if not minute_path.exists():
        raise FileNotFoundError(f"Minute file does not exist: {minute_path}")
    frame = pd.read_parquet(minute_path, columns=["amount"])
    if isinstance(frame.index, pd.MultiIndex):
        level = "trade_date" if "trade_date" in frame.index.names else 0
        trade_dates = pd.to_datetime(frame.index.get_level_values(level))
    elif frame.index.name == "trade_date":
        trade_dates = pd.to_datetime(frame.index)
    elif "trade_date" in frame.columns:
        trade_dates = pd.to_datetime(frame["trade_date"])
    else:
        raise ValueError(
            f"Minute file has no trade_date index or column: {minute_path}"
        )

    amounts = pd.to_numeric(frame["amount"], errors="coerce")
    daily = amounts.groupby(trade_dates.normalize()).sum(min_count=1).sort_index()
    return daily.astype(float)


def build_historical_context(
    daily_path: Path,
    target_dates: list[str],
    requested_codes: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if not target_dates or not daily_path.exists():
        return {}

    normalized_targets = sorted({pd.Timestamp(value).normalize() for value in target_dates})
    read_start = normalized_targets[0] - pd.Timedelta(days=90)
    read_end = normalized_targets[-1]
    columns = ["close", "high", "low", "pre_close", "adj_factor"]
    try:
        daily = pd.read_parquet(
            daily_path,
            columns=columns,
            filters=[("trade_date", ">=", read_start), ("trade_date", "<=", read_end)],
        )
    except (KeyError, TypeError, ValueError):
        daily = pd.read_parquet(daily_path, columns=columns)

    work = daily.reset_index()
    if "trade_date" not in work.columns or "ts_code" not in work.columns:
        raise ValueError(f"Daily file must expose trade_date and ts_code: {daily_path}")
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work["ts_code"] = work["ts_code"].astype(str).str.upper()
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[
        work["trade_date"].between(read_start, read_end)
    ].sort_values(["ts_code", "trade_date"], kind="mergesort")
    work = work.drop_duplicates(["ts_code", "trade_date"], keep="last")

    valid_close = work["close"].where(work["close"].gt(0))
    valid_high = work["high"].where(work["high"].gt(0))
    valid_low = work["low"].where(work["low"].gt(0))
    valid_pre_close = work["pre_close"].where(work["pre_close"].gt(0))
    valid_adj_factor = work["adj_factor"].where(work["adj_factor"].gt(0))
    work["_adj_close"] = valid_close * valid_adj_factor
    work["prevday_intraday_drawdown_from_session_high"] = valid_close / valid_high - 1.0
    work["prevday_intraday_rebound_from_session_low"] = valid_close / valid_low - 1.0
    work["prevday_intraday_return_from_prev_close"] = (
        valid_close / valid_pre_close - 1.0
    )

    available_daily_dates = np.sort(work["trade_date"].dropna().unique())
    session_numbers = {
        pd.Timestamp(trade_date): number
        for number, trade_date in enumerate(available_daily_dates)
    }
    work["_session_number"] = work["trade_date"].map(session_numbers)
    lag_lookup = work[["ts_code", "_session_number", "_adj_close"]]
    for periods, target in [(2, "_prev_2d_return"), (20, "_prev_20d_return")]:
        lagged = lag_lookup.rename(columns={"_adj_close": "_lagged_adj_close"}).copy()
        lagged["_session_number"] += periods
        work = work.merge(
            lagged,
            on=["ts_code", "_session_number"],
            how="left",
            validate="one_to_one",
        )
        work[target] = work["_adj_close"] / work.pop("_lagged_adj_close") - 1.0

    grouped = work.groupby("ts_code", sort=False)
    rolling_ma20 = grouped["_adj_close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    rolling_first_session = grouped["_session_number"].transform(
        lambda values: values.rolling(20, min_periods=20).min()
    )
    has_consecutive_20d = work["_session_number"].sub(rolling_first_session).eq(19)
    work["_market_above_ma20"] = (
        work["_adj_close"]
        .gt(rolling_ma20)
        .astype(float)
        .where(rolling_ma20.notna() & has_consecutive_20d)
    )
    work["prev_2d_return_rank_cs"] = work.groupby("trade_date", sort=False)[
        "_prev_2d_return"
    ].rank(method="average", pct=True)
    work["prev_20d_return_rank_cs"] = work.groupby("trade_date", sort=False)[
        "_prev_20d_return"
    ].rank(method="average", pct=True)

    target_to_source: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for target in normalized_targets:
        prior_dates = available_daily_dates[available_daily_dates < target.to_datetime64()]
        if prior_dates.size:
            target_to_source.append((target, pd.Timestamp(prior_dates[-1])))
    if not target_to_source:
        return {}

    mappings = pd.DataFrame(target_to_source, columns=["target_date", "trade_date"])
    context_columns = [
        "trade_date",
        "ts_code",
        "prevday_intraday_drawdown_from_session_high",
        "prevday_intraday_rebound_from_session_low",
        "prevday_intraday_return_from_prev_close",
        "prev_2d_return_rank_cs",
        "prev_20d_return_rank_cs",
        "_prev_2d_return",
        "_market_above_ma20",
    ]
    context = work[context_columns].merge(mappings, on="trade_date", how="inner")
    context["trade_date"] = context.pop("target_date").dt.strftime("%Y-%m-%d")
    if requested_codes is not None:
        context = context.loc[context["ts_code"].isin(requested_codes)]
    return {
        ts_code: group.drop(columns="ts_code").reset_index(drop=True)
        for ts_code, group in context.groupby("ts_code", sort=False)
    }


def build_benchmark_context(
    benchmark_ts_code: str,
    symbol_paths: list[Path],
    target_dates: list[str],
    historical_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    paths_by_date = {path.parent.name: path for path in symbol_paths}
    historical_by_date = (
        historical_context.set_index("trade_date", drop=False)
        if historical_context is not None and not historical_context.empty
        else pd.DataFrame()
    )
    records: list[dict[str, object]] = []
    for trade_date in sorted(set(target_dates)):
        record: dict[str, object] = {
            "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
            "benchmark_ts_code": benchmark_ts_code,
            "benchmark_available_time": pd.NaT,
            "benchmark_auction_has_match": False,
            "market_return_from_prev_close": np.nan,
            "_benchmark_auction_return_stage2": np.nan,
            "market_above_ma20_prevclose": np.nan,
            "market_momentum_2d_prevclose": np.nan,
        }
        path = paths_by_date.get(pd.Timestamp(trade_date).strftime("%Y%m%d"))
        if path is not None:
            benchmark_row = calculate_daily_auction_factors(
                load_quote_frame(path), benchmark_ts_code
            )
            record["benchmark_available_time"] = benchmark_row["available_time"]
            record["benchmark_auction_has_match"] = benchmark_row["auction_has_match"]
            if bool(benchmark_row["auction_has_match"]):
                record["market_return_from_prev_close"] = benchmark_row[
                    "auction_overnight_return"
                ]
                record["_benchmark_auction_return_stage2"] = benchmark_row[
                    "auction_return_stage2"
                ]
        date_key = record["trade_date"]
        if not historical_by_date.empty and date_key in historical_by_date.index:
            historical = historical_by_date.loc[date_key]
            if isinstance(historical, pd.DataFrame):
                historical = historical.iloc[-1]
            record["market_above_ma20_prevclose"] = historical[
                "_market_above_ma20"
            ]
            record["market_momentum_2d_prevclose"] = historical["_prev_2d_return"]
        records.append(record)
    return pd.DataFrame(records)


def apply_external_context(
    frame: pd.DataFrame,
    symbol_context: pd.DataFrame | None = None,
    benchmark_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    result["benchmark_ts_code"] = result["benchmark_ts_code"].astype(object)
    result["benchmark_auction_has_match"] = result[
        "benchmark_auction_has_match"
    ].astype(object)
    result["benchmark_available_time"] = pd.to_datetime(
        result["benchmark_available_time"], errors="coerce"
    )
    symbol_by_date = (
        symbol_context.set_index("trade_date", drop=False)
        if symbol_context is not None and not symbol_context.empty
        else pd.DataFrame()
    )
    benchmark_by_date = (
        benchmark_context.set_index("trade_date", drop=False)
        if benchmark_context is not None and not benchmark_context.empty
        else pd.DataFrame()
    )
    symbol_columns = [
        "prevday_intraday_drawdown_from_session_high",
        "prevday_intraday_rebound_from_session_low",
        "prevday_intraday_return_from_prev_close",
        "prev_2d_return_rank_cs",
        "prev_20d_return_rank_cs",
    ]
    for index, row in result.iterrows():
        trade_date = pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d")
        if not symbol_by_date.empty and trade_date in symbol_by_date.index:
            context = symbol_by_date.loc[trade_date]
            if isinstance(context, pd.DataFrame):
                context = context.iloc[-1]
            for column in symbol_columns:
                result.at[index, column] = context[column]

        if benchmark_by_date.empty or trade_date not in benchmark_by_date.index:
            continue
        benchmark = benchmark_by_date.loc[trade_date]
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[-1]
        for column in [
            "benchmark_ts_code",
            "benchmark_available_time",
            "benchmark_auction_has_match",
            "market_return_from_prev_close",
            "market_above_ma20_prevclose",
            "market_momentum_2d_prevclose",
        ]:
            result.at[index, column] = benchmark[column]

        if bool(benchmark["benchmark_auction_has_match"]):
            asset_available = pd.to_datetime(row["available_time"], errors="coerce")
            benchmark_available = pd.to_datetime(
                benchmark["benchmark_available_time"], errors="coerce"
            )
            if pd.notna(benchmark_available) and (
                pd.isna(asset_available) or benchmark_available > asset_available
            ):
                result.at[index, "available_time"] = benchmark_available
            market_return = benchmark["market_return_from_prev_close"]
            if np.isfinite(row["auction_overnight_return"]) and np.isfinite(market_return):
                result.at[index, "auction_gap_excess_benchmark"] = float(
                    row["auction_overnight_return"] - market_return
                )
            benchmark_stage2 = benchmark["_benchmark_auction_return_stage2"]
            if np.isfinite(row["auction_return_stage2"]) and np.isfinite(benchmark_stage2):
                result.at[index, "auction_stage2_excess_return_benchmark"] = float(
                    row["auction_return_stage2"] - benchmark_stage2
                )
    return result[OUTPUT_COLUMNS]


def build_session_path_factor_frame(
    minute_path: Path,
    ts_code: str,
    requested_dates: set[str] | None = None,
) -> pd.DataFrame:
    if not minute_path.exists():
        raise FileNotFoundError(f"Minute file does not exist: {minute_path}")
    minute = pd.read_parquet(minute_path, columns=["high", "low", "close"])
    work = minute.reset_index()
    if "trade_date" not in work.columns or "trade_time" not in work.columns:
        raise ValueError(f"Minute file must expose trade_date and trade_time: {minute_path}")
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work["bar_time"] = pd.to_datetime(work["trade_time"], errors="coerce")
    for column in ["high", "low", "close"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["trade_date", "bar_time"]).sort_values(
        ["trade_date", "bar_time"], kind="mergesort"
    )
    work = work.drop_duplicates(["trade_date", "bar_time"], keep="last")

    daily_close = work.groupby("trade_date", sort=True)["close"].last()
    previous_close = daily_close.shift(1)
    work["_previous_close"] = work["trade_date"].map(previous_close)
    work["_session"] = np.where(work["bar_time"].dt.hour < 13, "am", "pm")
    session_groups = work.groupby(["trade_date", "_session"], sort=False)
    session_high = session_groups["high"].cummax()
    session_low = session_groups["low"].cummin()
    valid_close = work["close"].where(work["close"].gt(0))
    work["intraday_drawdown_from_session_high"] = (
        valid_close / session_high.where(session_high.gt(0)) - 1.0
    )
    work["intraday_rebound_from_session_low"] = (
        valid_close / session_low.where(session_low.gt(0)) - 1.0
    )
    work["intraday_return_from_prev_close"] = (
        valid_close / work["_previous_close"].where(work["_previous_close"].gt(0))
        - 1.0
    )
    work["available_time"] = work["bar_time"] + pd.Timedelta(minutes=1)
    work["ts_code"] = ts_code
    work["trade_date"] = work["trade_date"].dt.strftime("%Y-%m-%d")
    if requested_dates is not None:
        normalized_dates = {
            pd.Timestamp(value).strftime("%Y-%m-%d") for value in requested_dates
        }
        work = work.loc[work["trade_date"].isin(normalized_dates)]
    result = work[SESSION_PATH_OUTPUT_COLUMNS].reset_index(drop=True)
    numeric = result[
        [
            "intraday_drawdown_from_session_high",
            "intraday_rebound_from_session_low",
            "intraday_return_from_prev_close",
        ]
    ].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError(f"Infinite session path factor produced for {ts_code}")
    return result


def merge_session_path_output(
    output_path: Path,
    requested: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    if output_path.exists():
        existing = pd.read_parquet(output_path).reindex(columns=SESSION_PATH_OUTPUT_COLUMNS)
    else:
        existing = pd.DataFrame(columns=SESSION_PATH_OUTPUT_COLUMNS)
    requested_keys = pd.MultiIndex.from_frame(requested[["trade_date", "bar_time"]])
    existing_keys = pd.MultiIndex.from_frame(existing[["trade_date", "bar_time"]])
    if overwrite:
        existing = existing.loc[~existing_keys.isin(requested_keys)]
        additions = requested
    else:
        additions = requested.loc[~requested_keys.isin(existing_keys)]
    combined = pd.concat([existing, additions], ignore_index=True)
    if combined.empty:
        return combined.reindex(columns=SESSION_PATH_OUTPUT_COLUMNS)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    combined["bar_time"] = pd.to_datetime(combined["bar_time"], errors="coerce")
    combined["available_time"] = pd.to_datetime(
        combined["available_time"], errors="coerce"
    )
    return combined.sort_values(["trade_date", "bar_time"], kind="mergesort").reset_index(
        drop=True
    )[SESSION_PATH_OUTPUT_COLUMNS]


def apply_historical_ratios(
    frame: pd.DataFrame,
    event_frames: dict[str, pd.DataFrame] | None = None,
    daily_amount_history: pd.Series | None = None,
) -> pd.DataFrame:
    result = (
        frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True).copy()
    )
    histories: dict[str, list[float]] = {
        "auction_amount": [],
        "auction_matched_volume": [],
    }
    targets = {
        "auction_amount": "auction_amount_ratio_5d",
        "auction_matched_volume": "auction_matched_volume_ratio_5d",
    }
    order_notional_history: list[np.ndarray] = []
    event_frames = event_frames or {}

    for index, row in result.iterrows():
        for source, target in targets.items():
            history = histories[source]
            value = row[source]
            result.at[index, target] = np.nan
            if len(history) >= 5:
                mean = float(np.mean(history[-5:]))
                if np.isfinite(value) and mean > 0:
                    result.at[index, target] = float(value / mean)
            if source == "auction_amount":
                result.at[index, "auction_amount_zscore_20d"] = np.nan
                if len(history) >= HISTORICAL_AMOUNT_LOOKBACK_DAYS:
                    recent = np.asarray(
                        history[-HISTORICAL_AMOUNT_LOOKBACK_DAYS:], dtype=float
                    )
                    standard_deviation = float(recent.std(ddof=0))
                    if np.isfinite(value) and standard_deviation > 0:
                        result.at[index, "auction_amount_zscore_20d"] = float(
                            (value - recent.mean()) / standard_deviation
                        )
            if np.isfinite(value) and value > 0:
                history.append(float(value))

        trade_date = pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d")
        result.at[index, "previous_5d_average_daily_amount"] = np.nan
        result.at[index, "auction_amount_to_prev5d_adv_240"] = np.nan
        result.at[index, "previous_20d_average_daily_amount"] = np.nan
        result.at[index, "auction_amount_to_prev20d_adv"] = np.nan
        if daily_amount_history is not None:
            prior_daily_amounts = daily_amount_history.loc[
                daily_amount_history.index < pd.Timestamp(trade_date)
            ]
            previous_5d = prior_daily_amounts.tail(SHORT_DAILY_AMOUNT_LOOKBACK_DAYS)
            previous_5d_valid = (
                len(previous_5d) == SHORT_DAILY_AMOUNT_LOOKBACK_DAYS
                and np.isfinite(previous_5d).all()
                and previous_5d.gt(0).all()
            )
            if previous_5d_valid:
                average_5d_amount = float(previous_5d.mean())
                result.at[index, "previous_5d_average_daily_amount"] = average_5d_amount
                if np.isfinite(row["auction_amount"]):
                    result.at[index, "auction_amount_to_prev5d_adv_240"] = float(
                        row["auction_amount"] / (average_5d_amount / MINUTES_PER_TRADING_DAY)
                    )

            previous_20d = prior_daily_amounts.loc[
                np.isfinite(prior_daily_amounts) & prior_daily_amounts.gt(0)
            ].tail(HISTORICAL_AMOUNT_LOOKBACK_DAYS)
            if len(previous_20d) >= HISTORICAL_AMOUNT_LOOKBACK_DAYS:
                average_daily_amount = float(previous_20d.mean())
                result.at[index, "previous_20d_average_daily_amount"] = (
                    average_daily_amount
                )
                if np.isfinite(row["auction_amount"]) and average_daily_amount > 0:
                    result.at[index, "auction_amount_to_prev20d_adv"] = float(
                        row["auction_amount"] / average_daily_amount
                    )

        events = event_frames.get(trade_date)
        result.at[index, "auction_large_order_history_days"] = min(
            len(order_notional_history), LARGE_ORDER_LOOKBACK_DAYS
        )
        result.at[index, "auction_large_order_threshold"] = np.nan
        result.at[index, "auction_large_order_cancel_ratio_stage1"] = np.nan
        result.at[index, "auction_large_cancel_imbalance_stage1"] = np.nan
        if len(order_notional_history) >= LARGE_ORDER_LOOKBACK_DAYS:
            historical_notionals = np.concatenate(
                order_notional_history[-LARGE_ORDER_LOOKBACK_DAYS:]
            )
            threshold = float(np.quantile(historical_notionals, LARGE_ORDER_QUANTILE))
            result.at[index, "auction_large_order_threshold"] = threshold
            if (
                bool(row["auction_event_reconstruction_ok"])
                and events is not None
                and not events.empty
            ):
                _apply_large_order_factors(result, index, events, threshold)

        if (
            bool(row["auction_event_reconstruction_ok"])
            and events is not None
            and not events.empty
        ):
            valid_adds = events.loc[
                events["event_type"].eq("A")
                & events["notional"].gt(0)
                & np.isfinite(events["notional"])
            ]
            if not valid_adds.empty:
                order_notional_history.append(
                    valid_adds["notional"].to_numpy(dtype=float)
                )
    return result[OUTPUT_COLUMNS]


def merge_symbol_output(
    output_path: Path,
    requested: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        missing_columns = [
            column for column in OUTPUT_COLUMNS if column not in existing.columns
        ]
        if missing_columns and not overwrite:
            LOGGER.warning(
                "%s uses an older schema; %s new columns will remain missing on "
                "existing dates unless those dates are rerun with --overwrite.",
                output_path,
                len(missing_columns),
            )
        existing = existing.reindex(columns=OUTPUT_COLUMNS)
    else:
        existing = pd.DataFrame(columns=OUTPUT_COLUMNS)

    requested_dates = set(requested["trade_date"].astype(str))
    if overwrite:
        existing = existing.loc[
            ~existing["trade_date"].astype(str).isin(requested_dates)
        ]
        additions = requested
    else:
        existing_dates = set(existing["trade_date"].astype(str))
        additions = requested.loc[
            ~requested["trade_date"].astype(str).isin(existing_dates)
        ]

    combined = pd.concat([existing, additions], ignore_index=True)
    if combined.empty:
        return combined.reindex(columns=OUTPUT_COLUMNS)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    combined["available_time"] = pd.to_datetime(
        combined["available_time"], errors="coerce"
    )
    combined = combined.drop_duplicates("trade_date", keep="last")
    return combined.sort_values("trade_date", kind="mergesort").reset_index(drop=True)[
        OUTPUT_COLUMNS
    ]


def _date_in_requested_range(
    date_text: str, date_from: str | None, date_to: str | None
) -> bool:
    return (date_from is None or date_text >= date_from) and (
        date_to is None or date_text <= date_to
    )


def process_symbol_series(
    asset_type: str,
    ts_code: str,
    symbol_paths: list[Path],
    minute_path: Path,
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
    symbol_context: pd.DataFrame | None = None,
    benchmark_context: pd.DataFrame | None = None,
    session_path_output_root: Path | None = None,
) -> tuple[str, Path, int]:
    ordered_paths = sorted(symbol_paths, key=lambda path: path.parent.name)
    requested_paths = [
        path
        for path in ordered_paths
        if _date_in_requested_range(path.parent.name, date_from, date_to)
    ]
    output_path = output_root / f"{ts_code}.parquet"
    if not requested_paths:
        return ("skipped", output_path, 0)

    first_requested_date = requested_paths[0].parent.name
    prior_paths = [
        path for path in ordered_paths if path.parent.name < first_requested_date
    ]
    warmup_records: list[tuple[dict[str, object], pd.DataFrame]] = []
    valid_amount_history_count = 0
    valid_event_history_count = 0
    for path in reversed(prior_paths):
        events, event_ok = load_auction_event_frame(path, ts_code)
        daily = calculate_daily_auction_factors(
            load_quote_frame(path), ts_code, events, event_ok
        )
        warmup_records.append((daily, events))
        if (
            np.isfinite(daily["auction_amount"])
            and daily["auction_amount"] > 0
            and np.isfinite(daily["auction_matched_volume"])
            and daily["auction_matched_volume"] > 0
        ):
            valid_amount_history_count += 1
        if event_ok and not events.loc[events["event_type"].eq("A")].empty:
            valid_event_history_count += 1
        if (
            valid_amount_history_count >= HISTORICAL_AMOUNT_LOOKBACK_DAYS
            and valid_event_history_count >= LARGE_ORDER_LOOKBACK_DAYS
        ):
            break

    requested_records: list[tuple[dict[str, object], pd.DataFrame]] = []
    for path in requested_paths:
        events, event_ok = load_auction_event_frame(path, ts_code)
        daily = calculate_daily_auction_factors(
            load_quote_frame(path), ts_code, events, event_ok
        )
        requested_records.append((daily, events))

    all_records = list(reversed(warmup_records)) + requested_records
    all_rows = [row for row, _ in all_records]
    event_frames = {row["trade_date"]: events for row, events in all_records}
    daily_amount_history = load_daily_amount_history(minute_path)
    factor_frame = apply_historical_ratios(
        pd.DataFrame(all_rows),
        event_frames=event_frames,
        daily_amount_history=daily_amount_history,
    )
    factor_frame = apply_external_context(
        factor_frame,
        symbol_context=symbol_context,
        benchmark_context=benchmark_context,
    )
    requested_dates = {
        pd.Timestamp(path.parent.name).strftime("%Y-%m-%d") for path in requested_paths
    }
    requested_frame = factor_frame.loc[
        factor_frame["trade_date"].isin(requested_dates)
    ].copy()

    combined = merge_symbol_output(output_path, requested_frame, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    if session_path_output_root is not None:
        session_path_output = session_path_output_root / f"{ts_code}.parquet"
        session_requested = build_session_path_factor_frame(
            minute_path, ts_code, requested_dates
        )
        session_combined = merge_session_path_output(
            session_path_output, session_requested, overwrite
        )
        session_path_output_root.mkdir(parents=True, exist_ok=True)
        session_combined.to_parquet(session_path_output, index=False)
    return (asset_type, output_path, len(requested_frame))


def main() -> int:
    args = parse_args()
    configure_logging()
    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    requested_codes = load_requested_codes(args.symbols, args.symbols_file)
    assets = build_asset_universe(
        args.asset_type,
        args.stock_minute_root,
        args.etf_minute_root,
        requested_codes,
    )
    if args.limit is not None:
        assets = assets[: args.limit]
    if not assets:
        LOGGER.warning("No symbols matched the requested asset universe.")
        return 0

    date_dirs = discover_trade_date_dirs(args.tick_root, date_to)
    benchmark_ts_code = args.benchmark_ts_code.strip().upper()
    benchmark_numeric_code = numeric_code(benchmark_ts_code)
    if benchmark_numeric_code is None:
        raise ValueError(f"Invalid --benchmark-ts-code: {args.benchmark_ts_code}")
    grouped_paths = group_symbol_paths(
        date_dirs,
        {code for _, code, _ in assets} | {benchmark_numeric_code},
    )
    output_roots = {
        "stock": args.stock_output_root,
        "etf": args.etf_output_root,
    }
    minute_roots = {
        "stock": args.stock_minute_root,
        "etf": args.etf_minute_root,
    }
    session_path_output_roots = {
        "stock": args.stock_session_path_output_root,
        "etf": args.etf_session_path_output_root,
    }
    tasks = [
        (kind, code, symbol, grouped_paths.get(code, []))
        for kind, code, symbol in assets
        if grouped_paths.get(code)
    ]
    target_dates = sorted(
        {
            path.parent.name
            for _, _, _, paths in tasks
            for path in paths
            if _date_in_requested_range(path.parent.name, date_from, date_to)
        }
    )
    requested_by_kind = {
        kind: {symbol for asset_kind, _, symbol, _ in tasks if asset_kind == kind}
        for kind in ("stock", "etf")
    }
    historical_context_by_kind: dict[str, dict[str, pd.DataFrame]] = {
        "stock": (
            build_historical_context(
                args.stock_daily_path, target_dates, requested_by_kind["stock"]
            )
            if requested_by_kind["stock"]
            else {}
        ),
        "etf": build_historical_context(
            args.etf_daily_path,
            target_dates,
            requested_by_kind["etf"] | {benchmark_ts_code},
        ),
    }
    benchmark_historical = historical_context_by_kind["etf"].get(
        benchmark_ts_code
    )
    if benchmark_historical is None:
        benchmark_historical = historical_context_by_kind["stock"].get(
            benchmark_ts_code
        )
    benchmark_context = build_benchmark_context(
        benchmark_ts_code,
        grouped_paths.get(benchmark_numeric_code, []),
        target_dates,
        benchmark_historical,
    )
    LOGGER.info(
        "Processing %s symbols from %s matched stock/ETF universe entries",
        len(tasks),
        len(assets),
    )

    failures: list[tuple[str, str]] = []
    written = 0
    worker_count = max(1, args.workers)
    if worker_count == 1:
        for kind, _, symbol, paths in tasks:
            try:
                _, output_path, row_count = process_symbol_series(
                    kind,
                    symbol,
                    paths,
                    minute_roots[kind] / f"{symbol}.parquet",
                    output_roots[kind],
                    date_from,
                    date_to,
                    args.overwrite,
                    historical_context_by_kind[kind].get(symbol),
                    benchmark_context,
                    session_path_output_roots[kind]
                    if args.write_session_path_factors
                    else None,
                )
                written += int(row_count > 0)
                LOGGER.info("Wrote %s requested rows to %s", row_count, output_path)
            except Exception as exc:  # noqa: BLE001
                failures.append((symbol, str(exc)))
                LOGGER.exception("Failed to process %s", symbol)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_symbol_series,
                    kind,
                    symbol,
                    paths,
                    minute_roots[kind] / f"{symbol}.parquet",
                    output_roots[kind],
                    date_from,
                    date_to,
                    args.overwrite,
                    historical_context_by_kind[kind].get(symbol),
                    benchmark_context,
                    session_path_output_roots[kind]
                    if args.write_session_path_factors
                    else None,
                ): symbol
                for kind, _, symbol, paths in tasks
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, output_path, row_count = future.result()
                    written += int(row_count > 0)
                    LOGGER.info("Wrote %s requested rows to %s", row_count, output_path)
                except Exception as exc:  # noqa: BLE001
                    failures.append((symbol, str(exc)))
                    LOGGER.exception("Failed to process %s", symbol)

    LOGGER.info(
        "Completed: %s symbol files written, %s failures", written, len(failures)
    )
    if failures:
        for symbol, error in failures[:20]:
            LOGGER.error("%s: %s", symbol, error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
