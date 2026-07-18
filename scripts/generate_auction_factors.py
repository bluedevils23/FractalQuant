"""Generate one-row-per-day opening-auction factors from local tick quotes."""

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
DEFAULT_STOCK_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\stock_auction_factors"
)
DEFAULT_ETF_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_auction_factors")

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
    "auction_submitted_volume",
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
    "auction_amount_to_prev20d_adv",
    "auction_amount_zscore_20d",
    "auction_matched_volume_to_submitted_ratio",
]
FACTOR_COLUMNS = (
    CORE_FACTOR_COLUMNS
    + EVENT_FACTOR_COLUMNS
    + PATH_FACTOR_COLUMNS
    + ROBUST_IMBALANCE_FACTOR_COLUMNS
    + PARTICIPATION_FACTOR_COLUMNS
)
OUTPUT_COLUMNS = KEY_COLUMNS + DIAGNOSTIC_COLUMNS + REFERENCE_COLUMNS + FACTOR_COLUMNS

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
    parser.add_argument(
        "--stock-output-root", type=Path, default=DEFAULT_STOCK_OUTPUT_ROOT
    )
    parser.add_argument("--etf-output-root", type=Path, default=DEFAULT_ETF_OUTPUT_ROOT)
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
    auction_end_time = (
        pd.Timestamp(match_row["trade_time"])
        if match_row is not None
        else nominal_end_time
    )

    auction = quotes.loc[
        quotes["trade_time"].ge(start_time) & quotes["trade_time"].lt(auction_end_time)
    ].copy()
    auction["indicative_price"] = _calculate_indicative_price(auction)
    auction["l3_imbalance"] = _calculate_l3_imbalance(auction)

    valid_price = auction.dropna(subset=["indicative_price"])
    split_candidates = valid_price.loc[valid_price["trade_time"].le(split_time)]
    final_candidates = valid_price.loc[valid_price["trade_time"].lt(auction_end_time)]

    previous_close = _first_finite(quotes["previous_close"])
    row["previous_close"] = previous_close
    row["auction_has_match"] = has_match

    if match_row is not None:
        row["available_time"] = pd.Timestamp(match_row["trade_time"])
        row["auction_open_price"] = float(match_row["open_price"])
        row["auction_amount"] = float(match_row["trade_amount"])
        row["auction_matched_volume"] = float(match_row["trade_volume"])
        row["auction_overnight_return"] = _safe_return(
            float(match_row["open_price"]), previous_close
        )
    elif not auction.empty:
        row["available_time"] = pd.Timestamp(auction["trade_time"].iloc[-1])

    _apply_matched_volume_participation(row)

    if split_candidates.empty or final_candidates.empty:
        _apply_stage_reversal(row)
        return row

    first = valid_price.iloc[0]
    split = split_candidates.iloc[-1]
    final = final_candidates.iloc[-1]
    split_timestamp = pd.Timestamp(split["trade_time"])

    stage1 = auction.loc[
        auction["trade_time"].ge(pd.Timestamp(first["trade_time"]))
        & auction["trade_time"].le(split_timestamp)
    ]
    stage2 = auction.loc[
        auction["trade_time"].ge(split_timestamp)
        & auction["trade_time"].le(pd.Timestamp(final["trade_time"]))
    ]
    stage1_valid = stage1.dropna(subset=["indicative_price"])
    stage2_valid = stage2.dropna(subset=["indicative_price"])

    row["snapshot_count_stage1"] = int(len(stage1_valid))
    row["snapshot_count_stage2"] = int(len(stage2_valid))
    row["auction_final_indicative_price"] = float(final["indicative_price"])
    row["auction_return_stage1"] = _safe_return(
        float(split["indicative_price"]), float(first["indicative_price"])
    )
    row["auction_return_stage2"] = _safe_return(
        float(final["indicative_price"]), float(split["indicative_price"])
    )

    first_imbalance = first["l3_imbalance"]
    split_imbalance = split["l3_imbalance"]
    final_imbalance = final["l3_imbalance"]
    if np.isfinite(first_imbalance) and np.isfinite(split_imbalance):
        row["auction_imbalance_change_stage1"] = float(
            split_imbalance - first_imbalance
        )
        row["auction_imbalance_relative_change_stage1"] = _relative_imbalance_change(
            split_imbalance, first_imbalance
        )
        row["auction_imbalance_fisher_change_stage1"] = _fisher_imbalance_change(
            split_imbalance, first_imbalance
        )
    if np.isfinite(split_imbalance) and np.isfinite(final_imbalance):
        row["auction_imbalance_change_stage2"] = float(
            final_imbalance - split_imbalance
        )
        row["auction_imbalance_relative_change_stage2"] = _relative_imbalance_change(
            final_imbalance, split_imbalance
        )
        row["auction_imbalance_fisher_change_stage2"] = _fisher_imbalance_change(
            final_imbalance, split_imbalance
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
    return daily.where(np.isfinite(daily) & daily.gt(0)).dropna().astype(float)


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
        result.at[index, "previous_20d_average_daily_amount"] = np.nan
        result.at[index, "auction_amount_to_prev20d_adv"] = np.nan
        if daily_amount_history is not None:
            prior_daily_amounts = daily_amount_history.loc[
                daily_amount_history.index < pd.Timestamp(trade_date)
            ].tail(HISTORICAL_AMOUNT_LOOKBACK_DAYS)
            if len(prior_daily_amounts) >= HISTORICAL_AMOUNT_LOOKBACK_DAYS:
                average_daily_amount = float(prior_daily_amounts.mean())
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
    requested_dates = {
        pd.Timestamp(path.parent.name).strftime("%Y-%m-%d") for path in requested_paths
    }
    requested_frame = factor_frame.loc[
        factor_frame["trade_date"].isin(requested_dates)
    ].copy()

    combined = merge_symbol_output(output_path, requested_frame, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
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
    grouped_paths = group_symbol_paths(date_dirs, {code for _, code, _ in assets})
    output_roots = {
        "stock": args.stock_output_root,
        "etf": args.etf_output_root,
    }
    minute_roots = {
        "stock": args.stock_minute_root,
        "etf": args.etf_minute_root,
    }
    tasks = [
        (kind, code, symbol, grouped_paths.get(code, []))
        for kind, code, symbol in assets
        if grouped_paths.get(code)
    ]
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
