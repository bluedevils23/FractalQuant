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
import pyarrow.parquet as pq

if __package__:
    from scripts.auction_tick_cache import AuctionTickCache
    from scripts import session_path_factors as _session_path_factors
else:
    from auction_tick_cache import AuctionTickCache
    import session_path_factors as _session_path_factors

SESSION_PATH_OUTPUT_COLUMNS = _session_path_factors.SESSION_PATH_OUTPUT_COLUMNS
build_session_path_factor_frame = _session_path_factors.build_session_path_factor_frame
merge_session_path_output = _session_path_factors.merge_session_path_output
process_session_path_only = _session_path_factors.process_session_path_only


LOGGER = logging.getLogger("generate_auction_factors")

DEFAULT_TICK_ROOT = Path(r"E:\逐笔数据")
DEFAULT_STOCK_MINUTE_ROOT = Path(
    r"D:\workspace\stockdata\stock-data\行情数据\stock_1min"
)
DEFAULT_ETF_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_QMT_TICK_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_tick_qmt")
DEFAULT_QMT_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_qmt")
DEFAULT_STOCK_DAILY_PATH = Path(
    r"D:\workspace\stockdata\stock-data\行情数据\stock_daily.parquet"
)
DEFAULT_ETF_DAILY_PATH = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_STOCK_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\stock-factors\stock_auction_factors"
)
DEFAULT_ETF_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-factors\etf_auction_factors")
DEFAULT_QMT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-factors\etf_auction_factors_qmt_0930_match"
)
DEFAULT_STOCK_SESSION_PATH_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\stock-factors\stock_intraday_session_path_factors"
)
DEFAULT_ETF_SESSION_PATH_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-factors\etf_intraday_session_path_factors"
)
DEFAULT_BENCHMARK_TS_CODE = "510300.SH"
DEFAULT_AUCTION_CACHE_ROOT = Path(r"D:\workspace\stockdata\auction_tick_cache")

ASSET_TYPES = ("stock", "etf", "both")
DATE_PATTERN = re.compile(r"^\d{8}$")
SYMBOL_PATTERN = re.compile(r"^(\d{6})(?:\.(?:SH|SZ|BJ))?$", re.IGNORECASE)

KEY_COLUMNS = ["trade_date", "available_time", "ts_code"]
DIAGNOSTIC_COLUMNS = [
    "auction_has_match",
    "auction_match_source",
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
    "auction_stage1_end_time",
    "auction_stage2_end_time",
    "previous_day_volume_shares",
    "previous_day_high",
    "previous_7d_close_max",
    "previous_day_float_market_cap_cny",
    "auction_limit_up_price",
    "auction_limit_down_price",
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
TRANSACTION_ORDER_RATIO_COLUMNS = [
    "auction_transaction_submitted_ratio",
    "auction_transaction_net_order_ratio",
    "auction_stage1_net_order_qty",
    "auction_stage2_order_qty",
    "auction_buy_order_imbalance",
    "auction_stage1_order_participation",
    "auction_stage2_order_participation",
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
    "market_return_from_prev_close",
    "auction_gap_excess_benchmark",
    "auction_stage2_excess_return_benchmark",
]
REPORT_SUPPLEMENT_FACTOR_COLUMNS = [
    "auction_range_ratio",
    "auction_stage1_range_ratio",
    "auction_stage2_range_ratio",
    "auction_stage1_end_return_from_prev_close",
    "auction_stage2_end_return_from_stage1_end",
    "auction_up_step_ratio",
    "auction_down_step_ratio",
    "auction_snapshot_count_total",
    "auction_l3_buy_share_final",
    "auction_l3_buy_share_stage1_end",
    "auction_l3_buy_share_change_stage2",
    "auction_stage1_max_return_from_prev_close",
    "auction_stage1_min_return_from_prev_close",
    "auction_open_pullback_from_stage1_max",
    "auction_open_rebound_from_stage1_min",
    "auction_last60s_price_return",
    "auction_final_to_full_max",
]
REPORT_SMOOTHED_SOURCE_COLUMNS = [
    "auction_range_ratio",
    "auction_stage1_range_ratio",
    "auction_stage2_range_ratio",
    "auction_overnight_return",
    "auction_stage1_end_return_from_prev_close",
    "auction_stage2_end_return_from_stage1_end",
    "auction_up_step_ratio",
    "auction_down_step_ratio",
    "auction_snapshot_count_total",
    "auction_matched_volume",
    "auction_l3_buy_share_final",
    "auction_l3_buy_share_stage1_end",
    "auction_l3_buy_share_change_stage2",
]
REPORT_SMOOTHED_FACTOR_COLUMNS = [
    f"{column}_mean_20d" for column in REPORT_SMOOTHED_SOURCE_COLUMNS
]
CONTEXT_SUPPLEMENT_FACTOR_COLUMNS = [
    "auction_volume_to_prevday_volume",
    "auction_amount_to_float_mcap_prevclose",
    "auction_open_to_prev_high",
    "auction_open_to_prev7d_close_max",
    "auction_stage1_touched_limit_up",
    "auction_stage1_touched_limit_down",
    "auction_stage1_limit_up_distance_bps",
    "auction_stage1_limit_down_distance_bps",
]
VOLUME_RATIO_FACTOR_COLUMNS = [
    "auction_volume_ratio_5d",
    "auction_volume_ratio_20d",
    "auction_volume_ratio_5d_zscore",
]
TURNOVER_FACTOR_COLUMNS = [
    "auction_turnover_rate",
    "auction_turnover_rate_free",
]
SUPPLEMENT_REFERENCE_COLUMNS = [
    "auction_stage1_end_time",
    "auction_stage2_end_time",
    "previous_day_volume_shares",
    "previous_day_high",
    "previous_7d_close_max",
    "previous_day_float_market_cap_cny",
    "previous_day_float_share",
    "previous_day_free_share",
    "auction_limit_up_price",
    "auction_limit_down_price",
]
SUPPLEMENT_OUTPUT_COLUMNS = (
    SUPPLEMENT_REFERENCE_COLUMNS
    + REPORT_SUPPLEMENT_FACTOR_COLUMNS
    + CONTEXT_SUPPLEMENT_FACTOR_COLUMNS
)
FACTOR_COLUMNS = (
    CORE_FACTOR_COLUMNS
    + EVENT_FACTOR_COLUMNS
    + TRANSACTION_ORDER_RATIO_COLUMNS
    + PATH_FACTOR_COLUMNS
    + ROBUST_IMBALANCE_FACTOR_COLUMNS
    + PARTICIPATION_FACTOR_COLUMNS
    + PRIORITY_REPORT_FACTOR_COLUMNS
    + REPORT_SUPPLEMENT_FACTOR_COLUMNS
    + CONTEXT_SUPPLEMENT_FACTOR_COLUMNS
    + REPORT_SMOOTHED_FACTOR_COLUMNS
    + VOLUME_RATIO_FACTOR_COLUMNS
    + TURNOVER_FACTOR_COLUMNS
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
    parser.add_argument("--qmt-tick-root", type=Path, default=DEFAULT_QMT_TICK_ROOT)
    parser.add_argument("--qmt-minute-root", type=Path, default=DEFAULT_QMT_MINUTE_ROOT)
    parser.add_argument("--stock-daily-path", type=Path, default=DEFAULT_STOCK_DAILY_PATH)
    parser.add_argument("--etf-daily-path", type=Path, default=DEFAULT_ETF_DAILY_PATH)
    parser.add_argument(
        "--stock-output-root", type=Path, default=DEFAULT_STOCK_OUTPUT_ROOT
    )
    parser.add_argument("--etf-output-root", type=Path, default=DEFAULT_ETF_OUTPUT_ROOT)
    parser.add_argument("--qmt-output-root", type=Path, default=DEFAULT_QMT_OUTPUT_ROOT)
    parser.add_argument(
        "--use-qmt-auction-source",
        action="store_true",
        help="Generate ETF auction factors from QMT ticks and the 09:30 minute match bar.",
    )
    parser.add_argument(
        "--use-qmt-match-fallback",
        action="store_true",
        help="Use QMT ETF match data after native quote and transaction fallbacks.",
    )
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
    parser.add_argument(
        "--existing-output-only",
        action="store_true",
        help="Restrict processing to symbols with an existing auction output parquet.",
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
        "--auction-cache-root",
        type=Path,
        default=DEFAULT_AUCTION_CACHE_ROOT,
        help="Read-through cache root for normalized opening-auction tick slices.",
    )
    parser.add_argument(
        "--refresh-auction-cache",
        action="store_true",
        help="Rebuild cached tick slices from the source CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace requested dates while preserving dates outside the requested range.",
    )
    parser.add_argument(
        "--refresh-existing-factors",
        action="store_true",
        help="Recompute and replace requested dates after factor formula changes.",
    )
    parser.add_argument(
        "--write-session-path-factors",
        action="store_true",
        help="Deprecated; use generate_etf_minute_factors.py instead.",
    )
    parser.add_argument(
        "--session-path-only",
        action="store_true",
        help="Deprecated compatibility entry; use generate_etf_minute_factors.py instead.",
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


def existing_output_codes(output_root: Path) -> set[str]:
    if not output_root.exists():
        return set()
    return {
        code
        for path in output_root.glob("*.parquet")
        if (code := numeric_code(path.stem)) is not None
    }


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


def load_quote_frame(
    symbol_dir: Path, cache: AuctionTickCache | None = None
) -> pd.DataFrame:
    if cache is not None:
        return cache.load_quote(symbol_dir)
    return AuctionTickCache(None).load_quote(symbol_dir)


def load_open_transaction_match(
    symbol_dir: Path, cache: AuctionTickCache | None = None
) -> dict[str, object] | None:
    """Extract a conservative 09:25 opening match from transaction prints."""
    transaction_frame = (
        cache.load_open_transactions(symbol_dir)
        if cache is not None
        else AuctionTickCache(None).load_open_transactions(symbol_dir)
    )
    if transaction_frame.empty:
        return None
    candidates = transaction_frame.loc[
        transaction_frame["trade_code"].ne("C")
        & transaction_frame["bs_flag"].isin(["B", "S"])
        & transaction_frame["price"].gt(0)
        & transaction_frame["quantity"].gt(0)
    ].copy()
    if candidates.empty:
        return None
    candidates["notional"] = candidates["price"] * candidates["quantity"]
    price_volume = candidates.groupby("price", sort=False)["quantity"].sum()
    selected_price = float(price_volume.idxmax())
    selected = candidates.loc[candidates["price"].eq(selected_price)]
    matched_volume = float(selected["quantity"].sum())
    if not np.isfinite(matched_volume) or matched_volume <= 0:
        return None
    trade_time = pd.Timestamp(selected["trade_time"].min()).normalize() + pd.Timedelta(
        hours=9, minutes=25
    )
    return {
        "trade_time": trade_time,
        "open_price": selected_price,
        "trade_volume": matched_volume,
        "trade_amount": float(selected_price * matched_volume),
    }


def load_qmt_quote_frame(tick_path: Path) -> pd.DataFrame:
    """Normalize QMT tick snapshots to the opening-auction quote contract."""
    columns = [
        "last_price",
        "previous_close",
        *[f"ask_price{level}" for level in range(1, 4)],
        *[f"ask_vol{level}" for level in range(1, 4)],
        *[f"bid_price{level}" for level in range(1, 4)],
        *[f"bid_vol{level}" for level in range(1, 4)],
    ]
    frame = pd.read_parquet(tick_path, columns=columns).reset_index()
    required = {"trade_date", "trade_time"}
    if not required.issubset(frame.columns):
        raise ValueError(f"QMT tick file must expose trade_date and trade_time: {tick_path}")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.dropna(subset=["trade_time"])
    clock = frame["trade_time"].dt.time
    frame = frame.loc[
        (clock >= pd.Timestamp("09:15").time())
        & (clock < pd.Timestamp("09:25").time())
    ].copy()
    result = pd.DataFrame({"trade_time": frame["trade_time"]})
    result["trade_price"] = pd.to_numeric(frame["last_price"], errors="coerce")
    result["trade_volume"] = 0.0
    result["trade_amount"] = 0.0
    result["open_price"] = np.nan
    result["previous_close"] = pd.to_numeric(
        frame["previous_close"], errors="coerce"
    )
    for level in range(1, 4):
        result[f"ask_price{level}"] = pd.to_numeric(
            frame[f"ask_price{level}"], errors="coerce"
        )
        result[f"ask_qty{level}"] = pd.to_numeric(
            frame[f"ask_vol{level}"], errors="coerce"
        )
        result[f"bid_price{level}"] = pd.to_numeric(
            frame[f"bid_price{level}"], errors="coerce"
        )
        result[f"bid_qty{level}"] = pd.to_numeric(
            frame[f"bid_vol{level}"], errors="coerce"
        )
    return result.sort_values("trade_time", kind="mergesort").drop_duplicates(
        "trade_time", keep="last"
    ).reset_index(drop=True)


def load_qmt_0930_matches(minute_path: Path) -> dict[str, dict[str, object]]:
    """Read the QMT 09:30 bar as an explicitly labelled match fallback."""
    frame = pd.read_parquet(minute_path, columns=["open", "vol", "amount"]).reset_index()
    required = {"trade_date", "trade_time"}
    if not required.issubset(frame.columns):
        raise ValueError(f"QMT minute file must expose trade_date and trade_time: {minute_path}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["vol"] = pd.to_numeric(frame["vol"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    matches = frame.loc[
        frame["trade_date"].notna()
        & frame["trade_time"].notna()
        & frame["trade_time"].dt.strftime("%H:%M:%S").eq("09:30:00")
        & frame["open"].gt(0)
        & frame["vol"].gt(0)
        & frame["amount"].gt(0)
    ].copy()
    matches = matches.drop_duplicates("trade_date", keep="last")
    return {
        trade_date.strftime("%Y-%m-%d"): {
            # Requested 09:25 backfill convention; this is not the bar publish time.
            "trade_time": trade_date + pd.Timedelta(hours=9, minutes=25),
            "open_price": float(row["open"]),
            "trade_volume": float(row["vol"]),
            "trade_amount": float(row["amount"]),
        }
        for trade_date, row in matches.set_index("trade_date").iterrows()
    }


def load_qmt_0925_tick_matches(tick_path: Path) -> dict[str, dict[str, object]]:
    """Read the last valid QMT tick snapshot in the 09:25 opening-match minute."""
    frame = pd.read_parquet(
        tick_path, columns=["last_price", "volume", "amount"]
    ).reset_index()
    required = {"trade_date", "trade_time"}
    if not required.issubset(frame.columns):
        raise ValueError(f"QMT tick file must expose trade_date and trade_time: {tick_path}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame["last_price"] = pd.to_numeric(frame["last_price"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    matches = frame.loc[
        frame["trade_date"].notna()
        & frame["trade_time"].notna()
        & frame["trade_time"].dt.strftime("%H:%M:%S").between(
            "09:25:00", "09:25:59"
        )
        & frame["last_price"].gt(0)
        & frame["volume"].gt(0)
        & frame["amount"].gt(0)
    ].copy()
    matches = matches.sort_values("trade_time", kind="mergesort").drop_duplicates(
        "trade_date", keep="last"
    )
    return {
        trade_date.strftime("%Y-%m-%d"): {
            "trade_time": trade_date + pd.Timedelta(hours=9, minutes=25),
            "open_price": float(row["last_price"]),
            # The local QMT tick archive already stores cumulative volume in shares.
            "trade_volume": float(row["volume"]),
            "trade_amount": float(row["amount"]),
        }
        for trade_date, row in matches.set_index("trade_date").iterrows()
    }


def _load_qmt_match_fallbacks(
    ts_code: str,
    tick_path: Path | None,
    minute_path: Path | None,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    minute_matches: dict[str, dict[str, object]] = {}
    tick_matches: dict[str, dict[str, object]] = {}
    if minute_path is None or not minute_path.exists():
        LOGGER.warning("%s QMT minute fallback file is unavailable", ts_code)
    else:
        try:
            minute_matches = load_qmt_0930_matches(minute_path)
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning("%s QMT minute fallback failed: %s", ts_code, exc)
    if tick_path is None or not tick_path.exists():
        LOGGER.warning("%s QMT tick fallback file is unavailable", ts_code)
    else:
        try:
            tick_matches = load_qmt_0925_tick_matches(tick_path)
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning("%s QMT tick fallback failed: %s", ts_code, exc)
    LOGGER.info(
        "%s QMT fallback matches: minute=%s tick=%s",
        ts_code,
        len(minute_matches),
        len(tick_matches),
    )
    return minute_matches, tick_matches


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _load_raw_orders(
    symbol_dir: Path, cache: AuctionTickCache | None = None
) -> pd.DataFrame:
    if cache is not None:
        return cache.load_orders(symbol_dir)
    return AuctionTickCache(None).load_orders(symbol_dir)


def _load_sz_cancellations(
    symbol_dir: Path, cache: AuctionTickCache | None = None
) -> pd.DataFrame:
    if cache is not None:
        return cache.load_transactions(symbol_dir)
    return AuctionTickCache(None).load_transactions(symbol_dir)


def _auction_event_bounds(
    frame: pd.DataFrame,
    expected_trade_date: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, ...]:
    valid_times = frame["trade_time"].dropna()
    if valid_times.empty:
        raise ValueError("Auction event file contains no valid timestamp")
    if expected_trade_date is None:
        normalized_dates = valid_times.dt.normalize()
        trade_day = pd.Timestamp(normalized_dates.mode().iloc[0])
    else:
        trade_day = pd.Timestamp(expected_trade_date).normalize()
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
    duplicate_order_ids = bool(required_add["order_id"].duplicated().any())
    if len(required_add) != len(adds) or duplicate_order_ids:
        valid = False
    if duplicate_order_ids:
        return _empty_event_frame(), False

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


def _reconstruct_sh_events(
    orders: pd.DataFrame,
    expected_trade_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, bool]:
    start_time, split_time, end_time = _auction_event_bounds(
        orders, expected_trade_date
    )
    auction = orders.loc[
        orders["trade_time"].ge(start_time) & orders["trade_time"].lt(end_time)
    ].copy()
    if auction.empty:
        return _empty_event_frame(), False
    known_types = auction["order_type"].isin(["A", "D"])
    valid = bool(known_types.all())
    adds = auction.loc[auction["order_type"].eq("A")].copy()
    cancellations = auction.loc[auction["order_type"].eq("D")].copy()
    events, linked = _finalize_event_frame(adds, cancellations)
    valid &= linked
    valid &= not bool(cancellations["trade_time"].ge(split_time).any())
    return events, valid


def _reconstruct_sz_events(
    orders: pd.DataFrame,
    transactions: pd.DataFrame,
    expected_trade_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, bool]:
    start_time, split_time, end_time = _auction_event_bounds(
        orders, expected_trade_date
    )
    adds = orders.loc[
        orders["trade_time"].ge(start_time) & orders["trade_time"].lt(end_time)
    ].copy()
    cancellations = transactions.loc[
        transactions["trade_time"].ge(start_time)
        & transactions["trade_time"].lt(end_time)
        & transactions["trade_code"].eq("C")
    ].copy()
    if adds.empty and cancellations.empty:
        return _empty_event_frame(), False
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
    symbol_dir: Path,
    ts_code: str,
    *,
    expected_trade_date: str | pd.Timestamp | None = None,
    cache: AuctionTickCache | None = None,
) -> tuple[pd.DataFrame, bool]:
    exchange = ts_code.rsplit(".", 1)[-1].upper()
    try:
        orders = _load_raw_orders(symbol_dir, cache=cache)
        if exchange == "SH":
            return _reconstruct_sh_events(orders, expected_trade_date)
        if exchange == "SZ":
            return _reconstruct_sz_events(
                orders,
                _load_sz_cancellations(symbol_dir, cache=cache),
                expected_trade_date,
            )
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
    bid_qty = frame[[f"bid_qty{level}" for level in range(2, 4)]].sum(axis=1)
    ask_qty = frame[[f"ask_qty{level}" for level in range(2, 4)]].sum(axis=1)
    total = bid_qty + ask_qty
    return ((bid_qty - ask_qty) / total).where(total > 0)


def _calculate_relative_spread(frame: pd.DataFrame) -> pd.Series:
    """
    Calculate relative spread during call auction.

    During the opening auction (09:15-09:25), bid_price1 == ask_price1 (virtual
    matched price), and bid_price2/ask_price2 have no data (0% valid rate in
    production). Therefore, the spread is always zero and this factor carries no
    signal. Return NaN to explicitly mark it as unavailable for auction data.
    """
    return pd.Series(np.nan, index=frame.index)


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
    # During call auction, relative_spread is unavailable (always NaN), so we
    # exclude spread_coverage from the overall coverage_ratio calculation.
    row["auction_stage2_twap_coverage_ratio"] = float(
        min(price_coverage, imbalance_coverage)
    )
    row["auction_stage2_twap_price"] = price_twap
    row["auction_l3_imbalance_twap_stage2"] = imbalance_twap
    row["auction_relative_spread_twap_stage2"] = spread_twap

    open_price = row["auction_open_price"]
    if np.isfinite(open_price) and np.isfinite(price_twap) and price_twap > 0:
        row["auction_final_vs_stage2_twap"] = float(open_price / price_twap - 1.0)


def _stage2_slope(stage2: pd.DataFrame, previous_close: float) -> float:
    valid = stage2.dropna(subset=["trade_time", "indicative_price"])
    if len(valid) < 3 or not np.isfinite(previous_close) or previous_close <= 0:
        return np.nan
    elapsed_minutes = (
        valid["trade_time"] - valid["trade_time"].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float) / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_log_price = np.log(
            valid["indicative_price"].to_numpy(dtype=float) / previous_close
        )
    finite = np.isfinite(elapsed_minutes) & np.isfinite(relative_log_price)
    finite_elapsed = elapsed_minutes[finite]
    finite_log_price = relative_log_price[finite]
    if (
        len(finite_elapsed) < 3
        or np.unique(finite_elapsed).size < 3
        or np.ptp(finite_elapsed) <= 0
    ):
        return np.nan
    slope = float(np.polyfit(finite_elapsed, finite_log_price, 1)[0] * 10000.0)
    return slope if np.isfinite(slope) else np.nan


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


def _indicative_price_range_ratio(frame: pd.DataFrame) -> float:
    prices = pd.to_numeric(frame["indicative_price"], errors="coerce").to_numpy(
        dtype=float
    )
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size == 0:
        return np.nan
    minimum = float(prices.min())
    return float((prices.max() - minimum) / minimum)


def _finite_indicative_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prices = pd.to_numeric(frame["indicative_price"], errors="coerce")
    return frame.loc[np.isfinite(prices) & prices.gt(0)]


def _l3_buy_share(endpoint: pd.Series) -> float:
    imbalance = pd.to_numeric(
        pd.Series([endpoint["l3_imbalance"]]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(imbalance):
        return np.nan
    return float((imbalance + 1.0) / 2.0)


def _stage2_tail_buy_share(
    stage2: pd.DataFrame,
    nominal_end_time: pd.Timestamp,
) -> float:
    """Return the mean L3 buy share from the final Stage2 quote window."""
    if stage2.empty:
        return np.nan
    ordered = (
        stage2.copy()
        .sort_values("trade_time", kind="mergesort")
        .drop_duplicates("trade_time", keep="last")
    )
    prices = pd.to_numeric(ordered["indicative_price"], errors="coerce")
    imbalance = pd.to_numeric(ordered["l3_imbalance"], errors="coerce")
    valid = ordered.loc[
        np.isfinite(prices)
        & prices.gt(0)
        & np.isfinite(imbalance)
        & ordered["trade_time"].lt(nominal_end_time)
    ]
    if valid.empty:
        return np.nan

    tail_start = nominal_end_time - pd.Timedelta(seconds=60)
    tail = valid.loc[valid["trade_time"].ge(tail_start)]
    selected = tail if len(tail) >= 3 else valid.tail(min(3, len(valid)))
    shares = (pd.to_numeric(selected["l3_imbalance"], errors="coerce") + 1.0) / 2.0
    shares = shares[np.isfinite(shares)]
    return float(shares.mean()) if not shares.empty else np.nan


def _stage2_imbalance_endpoints(
    stage2: pd.DataFrame,
) -> tuple[float, float]:
    """Return smoothed Stage2 start/end L3 imbalance values."""
    if stage2.empty:
        return np.nan, np.nan
    ordered = (
        stage2.copy()
        .sort_values("trade_time", kind="mergesort")
        .drop_duplicates("trade_time", keep="last")
    )
    prices = pd.to_numeric(ordered["indicative_price"], errors="coerce")
    imbalance = pd.to_numeric(ordered["l3_imbalance"], errors="coerce")
    valid = ordered.loc[np.isfinite(prices) & prices.gt(0) & np.isfinite(imbalance)]
    values = imbalance.loc[valid.index].to_numpy(dtype=float)
    if values.size < 2:
        return np.nan, np.nan
    window = max(1, int(np.ceil(values.size * 0.1)))
    return float(values[:window].mean()), float(values[-window:].mean())


def _apply_report_supplement_factors(
    row: dict[str, object],
    valid_price: pd.DataFrame,
    stage1_valid: pd.DataFrame,
    stage2_valid: pd.DataFrame,
    previous_close: float,
    nominal_end_time: pd.Timestamp,
) -> None:
    valid_price = _finite_indicative_price_frame(valid_price)
    stage1_valid = _finite_indicative_price_frame(stage1_valid)
    stage2_valid = _finite_indicative_price_frame(stage2_valid)
    row["auction_snapshot_count_total"] = int(len(valid_price))
    row["auction_range_ratio"] = _indicative_price_range_ratio(valid_price)
    row["auction_stage1_range_ratio"] = _indicative_price_range_ratio(stage1_valid)
    row["auction_stage2_range_ratio"] = _indicative_price_range_ratio(stage2_valid)

    if not valid_price.empty:
        prices = valid_price["indicative_price"].to_numpy(dtype=float)
        final = valid_price.iloc[-1]
        changes = np.diff(prices)
        if len(changes) > 0:
            row["auction_up_step_ratio"] = float(
                np.count_nonzero(changes > 0) / len(changes)
            )
            row["auction_down_step_ratio"] = float(
                np.count_nonzero(changes < 0) / len(changes)
            )
        else:
            row["auction_up_step_ratio"] = 0.0
            row["auction_down_step_ratio"] = 0.0
        row["auction_final_to_full_max"] = _safe_return(
            float(final["indicative_price"]), float(prices.max())
        )
        last_minute = valid_price.loc[
            valid_price["trade_time"].ge(nominal_end_time - pd.Timedelta(minutes=1))
        ]
        if len(last_minute) >= 2:
            row["auction_last60s_price_return"] = _safe_return(
                float(last_minute.iloc[-1]["indicative_price"]),
                float(last_minute.iloc[0]["indicative_price"]),
            )

    stage1_final = None
    stage1_buy_share = np.nan
    if not stage1_valid.empty:
        stage1_final = stage1_valid.iloc[-1]
        stage1_prices = stage1_valid["indicative_price"].to_numpy(dtype=float)
        row["auction_stage1_end_time"] = pd.Timestamp(stage1_final["trade_time"])
        row["auction_stage1_end_return_from_prev_close"] = _safe_return(
            float(stage1_final["indicative_price"]), previous_close
        )
        row["auction_stage1_max_return_from_prev_close"] = _safe_return(
            float(stage1_prices.max()), previous_close
        )
        row["auction_stage1_min_return_from_prev_close"] = _safe_return(
            float(stage1_prices.min()), previous_close
        )
        stage1_buy_share = _l3_buy_share(stage1_final)
        row["auction_l3_buy_share_stage1_end"] = stage1_buy_share

        open_price = row["auction_open_price"]
        if np.isfinite(open_price):
            stage1_max = float(stage1_prices.max())
            stage1_min = float(stage1_prices.min())
            row["auction_open_pullback_from_stage1_max"] = float(
                1.0 - open_price / stage1_max
            )
            row["auction_open_rebound_from_stage1_min"] = _safe_return(
                open_price, stage1_min
            )

    if not stage2_valid.empty:
        stage2_final = stage2_valid.iloc[-1]
        row["auction_stage2_end_time"] = pd.Timestamp(stage2_final["trade_time"])
        if stage1_final is not None:
            row["auction_stage2_end_return_from_stage1_end"] = _safe_return(
                float(stage2_final["indicative_price"]),
                float(stage1_final["indicative_price"]),
            )
        stage2_buy_share = _l3_buy_share(stage2_final)
        if np.isfinite(stage1_buy_share) and np.isfinite(stage2_buy_share):
            row["auction_l3_buy_share_change_stage2"] = float(
                stage2_buy_share - stage1_buy_share
            )

    row["auction_l3_buy_share_final"] = _stage2_tail_buy_share(
        stage2_valid, nominal_end_time
    )
    if not valid_price.empty and not np.isfinite(final["l3_imbalance"]):
        row["auction_l3_buy_share_final"] = np.nan


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

    # Apply transaction/order ratio factors
    _apply_transaction_order_ratio_factors(
        row, trade_day, events, stage1_adds, stage1_cancels, stage2_adds
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


def _apply_transaction_order_ratio_factors(
    row: dict[str, object],
    trade_day: pd.Timestamp,
    events: pd.DataFrame,
    stage1_adds: pd.DataFrame,
    stage1_cancels: pd.DataFrame,
    stage2_adds: pd.DataFrame,
) -> None:
    """Calculate transaction/order ratio factors based on events data."""

    # Get matched volume (final transaction volume)
    matched_volume = row.get("auction_matched_volume", np.nan)

    # Initialize all factors to NaN
    row["auction_transaction_submitted_ratio"] = np.nan
    row["auction_transaction_net_order_ratio"] = np.nan
    row["auction_stage1_net_order_qty"] = np.nan
    row["auction_stage2_order_qty"] = np.nan
    row["auction_buy_order_imbalance"] = np.nan
    row["auction_stage1_order_participation"] = np.nan
    row["auction_stage2_order_participation"] = np.nan

    # Calculate order quantities by side
    stage1_buy_qty = _side_sum(stage1_adds, "B", "quantity")
    stage1_sell_qty = _side_sum(stage1_adds, "S", "quantity")
    stage1_buy_cancel_qty = _side_sum(stage1_cancels, "B", "quantity")
    stage1_sell_cancel_qty = _side_sum(stage1_cancels, "S", "quantity")
    stage2_buy_qty = _side_sum(stage2_adds, "B", "quantity")
    stage2_sell_qty = _side_sum(stage2_adds, "S", "quantity")

    # Calculate total order volumes
    total_add_qty = float(stage1_adds["quantity"].sum() + stage2_adds["quantity"].sum())
    total_cancel_qty = float(stage1_cancels["quantity"].sum())
    stage1_net_buy = stage1_buy_qty - stage1_buy_cancel_qty
    stage1_net_sell = stage1_sell_qty - stage1_sell_cancel_qty
    stage1_net_qty = stage1_net_buy + stage1_net_sell
    stage2_total_qty = stage2_buy_qty + stage2_sell_qty
    net_order_qty = stage1_net_qty + stage2_total_qty

    # Store intermediate values
    row["auction_stage1_net_order_qty"] = float(stage1_net_qty) if stage1_net_qty > 0 else np.nan
    row["auction_stage2_order_qty"] = float(stage2_total_qty) if stage2_total_qty > 0 else np.nan

    # Calculate transaction/submitted ratio
    if total_add_qty > 0 and np.isfinite(matched_volume):
        row["auction_transaction_submitted_ratio"] = float(matched_volume / total_add_qty)

    # Calculate transaction/net order ratio (after cancellations)
    if net_order_qty > 0 and np.isfinite(matched_volume):
        row["auction_transaction_net_order_ratio"] = float(matched_volume / net_order_qty)

    # Calculate buy order imbalance
    total_buy = stage1_buy_qty - stage1_buy_cancel_qty + stage2_buy_qty
    total_sell = stage1_sell_qty - stage1_sell_cancel_qty + stage2_sell_qty
    total_orders = total_buy + total_sell
    if total_orders > 0:
        row["auction_buy_order_imbalance"] = float((total_buy - total_sell) / total_orders)

    # Calculate order participation by stage
    if total_add_qty > 0:
        stage1_add_qty = float(stage1_adds["quantity"].sum())
        row["auction_stage1_order_participation"] = float(stage1_add_qty / total_add_qty)
        row["auction_stage2_order_participation"] = float(stage2_total_qty / total_add_qty)


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
            "auction_stage1_end_time": pd.NaT,
            "auction_stage2_end_time": pd.NaT,
            "auction_has_match": False,
            "auction_match_source": "none",
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
    match_override: dict[str, object] | None = None,
    match_source: str | None = None,
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
    if match_override is not None:
        override_price = pd.to_numeric(match_override.get("open_price"), errors="coerce")
        override_volume = pd.to_numeric(match_override.get("trade_volume"), errors="coerce")
        override_amount = pd.to_numeric(match_override.get("trade_amount"), errors="coerce")
        if (
            np.isfinite(override_price)
            and override_price > 0
            and np.isfinite(override_volume)
            and override_volume > 0
            and np.isfinite(override_amount)
            and override_amount > 0
        ):
            match_row = pd.Series(
                {
                    "trade_time": match_override.get("trade_time", nominal_end_time),
                    "open_price": float(override_price),
                    "trade_volume": float(override_volume),
                    "trade_amount": float(override_amount),
                }
            )
            has_match = True
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
    if has_match:
        row["auction_match_source"] = match_source or "quote"
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

    _apply_report_supplement_factors(
        row,
        valid_price,
        stage1_valid,
        stage2_valid,
        previous_close,
        nominal_end_time,
    )
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
        first_imbalance, final_imbalance = _stage2_imbalance_endpoints(stage2_valid)
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
        unmatched_bid = float(final["bid_qty2"] + final["bid_qty3"])
        unmatched_ask = float(final["ask_qty2"] + final["ask_qty3"])
        unmatched_total = unmatched_bid + unmatched_ask
        if not np.isfinite(unmatched_bid) or not np.isfinite(unmatched_ask):
            row["auction_unmatched_imbalance"] = np.nan
        elif unmatched_total > 0:
            row["auction_unmatched_imbalance"] = float(
                (unmatched_bid - unmatched_ask) / unmatched_total
            )
        else:
            row["auction_unmatched_imbalance"] = np.nan
    _apply_stage_reversal(row)
    return row


def _calculate_daily_with_match_fallback(
    quotes: pd.DataFrame,
    ts_code: str,
    events: pd.DataFrame | None = None,
    event_reconstruction_ok: bool = False,
    symbol_dir: Path | None = None,
    cache: AuctionTickCache | None = None,
    qmt_minute_matches: dict[str, dict[str, object]] | None = None,
    qmt_tick_matches: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    daily = calculate_daily_auction_factors(
        quotes, ts_code, events, event_reconstruction_ok
    )
    if bool(daily["auction_has_match"]):
        return daily

    if symbol_dir is not None:
        try:
            transaction_match = load_open_transaction_match(symbol_dir, cache=cache)
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning("%s native 09:25 transaction fallback unavailable: %s", ts_code, exc)
        else:
            if transaction_match is not None:
                daily = calculate_daily_auction_factors(
                    quotes,
                    ts_code,
                    events,
                    event_reconstruction_ok,
                    match_override=transaction_match,
                    match_source="transaction_0925",
                )
                if bool(daily["auction_has_match"]):
                    return daily

    trade_date = str(daily["trade_date"])
    for source, matches in (
        ("qmt_0930_minute", qmt_minute_matches),
        ("qmt_tick_0925", qmt_tick_matches),
    ):
        if not matches or trade_date not in matches:
            continue
        candidate = calculate_daily_auction_factors(
            quotes,
            ts_code,
            events,
            event_reconstruction_ok,
            match_override=matches[trade_date],
            match_source=source,
        )
        if bool(candidate["auction_has_match"]):
            return candidate
    return daily


def calculate_supplemental_auction_fields(
    quotes: pd.DataFrame, ts_code: str
) -> dict[str, object]:
    """Calculate only columns introduced by the report-supplement extension."""
    if quotes.empty:
        raise ValueError(f"Empty auction quote frame for {ts_code}")
    trade_day = pd.Timestamp(quotes["trade_time"].iloc[0]).normalize()
    nominal_end_time = trade_day + pd.Timedelta(hours=9, minutes=25)
    match_deadline = trade_day + pd.Timedelta(hours=9, minutes=30)
    row: dict[str, object] = {column: np.nan for column in SUPPLEMENT_OUTPUT_COLUMNS}
    row["trade_date"] = trade_day.strftime("%Y-%m-%d")
    row["auction_stage1_end_time"] = pd.NaT
    row["auction_stage2_end_time"] = pd.NaT
    row["previous_close"] = _first_finite(quotes["previous_close"])
    row["auction_open_price"] = np.nan
    row["auction_amount"] = np.nan
    row["auction_matched_volume"] = np.nan

    match_rows = quotes.loc[
        quotes["trade_time"].ge(nominal_end_time)
        & quotes["trade_time"].lt(match_deadline)
        & quotes["open_price"].notna()
        & quotes["trade_volume"].gt(0)
        & quotes["trade_amount"].gt(0)
    ]
    if not match_rows.empty:
        match_row = match_rows.iloc[0]
        row["auction_open_price"] = float(match_row["open_price"])
        row["auction_amount"] = float(match_row["trade_amount"])
        row["auction_matched_volume"] = float(match_row["trade_volume"])

    auction = quotes.loc[
        quotes["trade_time"].ge(trade_day + pd.Timedelta(hours=9, minutes=15))
        & quotes["trade_time"].lt(nominal_end_time)
    ].copy()
    auction["indicative_price"] = _calculate_indicative_price(auction)
    auction["l3_imbalance"] = _calculate_l3_imbalance(auction)
    valid_price = auction.dropna(subset=["indicative_price"])
    stage1 = valid_price.loc[
        valid_price["trade_time"].lt(trade_day + pd.Timedelta(hours=9, minutes=20))
    ]
    stage2 = valid_price.loc[
        valid_price["trade_time"].ge(trade_day + pd.Timedelta(hours=9, minutes=20))
    ]
    _apply_report_supplement_factors(
        row,
        valid_price,
        stage1,
        stage2,
        row["previous_close"],
        nominal_end_time,
    )
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


def load_daily_volume_history(minute_path: Path) -> pd.Series:
    """Load historical daily volume series (in shares)."""
    if not minute_path.exists():
        raise FileNotFoundError(f"Minute file does not exist: {minute_path}")

    frame = pd.read_parquet(minute_path, columns=["vol"])

    # Extract trade dates
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

    # Convert volume to shares (multiply by 100 if in lots)
    volumes = pd.to_numeric(frame["vol"], errors="coerce")

    # Check if unit is in lots (手) by examining median value
    # If median < 1,000,000, likely in lots, need to multiply by 100
    median_vol = volumes.median()
    if median_vol > 0 and median_vol < 1e6:
        volumes = volumes * 100

    # Aggregate by date
    daily = volumes.groupby(trade_dates.normalize()).sum(min_count=1).sort_index()
    return daily.astype(float)


def build_historical_context(
    daily_path: Path,
    target_dates: list[str],
    requested_codes: set[str] | None = None,
    include_daily_factor_fields: bool = True,
) -> dict[str, pd.DataFrame]:
    if not target_dates or not daily_path.exists():
        return {}

    normalized_targets = sorted({pd.Timestamp(value).normalize() for value in target_dates})
    read_start = normalized_targets[0] - pd.Timedelta(days=90)
    read_end = normalized_targets[-1]
    required_columns = ["close", "high", "low", "pre_close", "adj_factor"]
    optional_columns = ["vol", "circ_mv", "up_limit", "down_limit", "float_share", "free_share"]
    available_columns = set(pq.read_schema(daily_path).names)
    missing_required = [
        column for column in required_columns if column not in available_columns
    ]
    if missing_required:
        raise ValueError(
            f"Daily file is missing required columns {missing_required}: {daily_path}"
        )
    columns = required_columns + [
        column for column in optional_columns if column in available_columns
    ]
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
    for column in optional_columns:
        if column not in work.columns:
            work[column] = np.nan
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
    work["previous_day_volume_shares"] = work["vol"].where(work["vol"].gt(0)) * 100.0
    work["previous_day_high"] = valid_high
    work["previous_day_float_market_cap_cny"] = (
        work["circ_mv"].where(work["circ_mv"].gt(0)) * 10000.0
    )
    work["previous_day_float_share"] = work["float_share"].where(work["float_share"].gt(0))
    work["previous_day_free_share"] = work["free_share"].where(work["free_share"].gt(0))
    if include_daily_factor_fields:
        work["prevday_intraday_drawdown_from_session_high"] = (
            valid_close / valid_high - 1.0
        )
        work["prevday_intraday_rebound_from_session_low"] = (
            valid_close / valid_low - 1.0
        )
        work["prevday_intraday_return_from_prev_close"] = (
            valid_close / valid_pre_close - 1.0
        )

    available_daily_dates = np.sort(work["trade_date"].dropna().unique())
    session_numbers = {
        pd.Timestamp(trade_date): number
        for number, trade_date in enumerate(available_daily_dates)
    }
    work["_session_number"] = work["trade_date"].map(session_numbers)
    if include_daily_factor_fields:
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
    rolling_7d_close_max = grouped["close"].transform(
        lambda values: values.rolling(7, min_periods=7).max()
    )
    rolling_first_session_7d = grouped["_session_number"].transform(
        lambda values: values.rolling(7, min_periods=7).min()
    )
    has_consecutive_7d = work["_session_number"].sub(rolling_first_session_7d).eq(6)
    work["previous_7d_close_max"] = rolling_7d_close_max.where(has_consecutive_7d)
    if include_daily_factor_fields:
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
        "previous_day_volume_shares",
        "previous_day_high",
        "previous_7d_close_max",
        "previous_day_float_market_cap_cny",
        "previous_day_float_share",
        "previous_day_free_share",
    ]
    if include_daily_factor_fields:
        context_columns.extend(
            [
                "prevday_intraday_drawdown_from_session_high",
                "prevday_intraday_rebound_from_session_low",
                "prevday_intraday_return_from_prev_close",
                "prev_2d_return_rank_cs",
                "prev_20d_return_rank_cs",
                "_prev_2d_return",
                "_market_above_ma20",
            ]
        )
    context = work[context_columns].merge(mappings, on="trade_date", how="inner")
    target_limits = work.loc[
        work["trade_date"].isin(normalized_targets),
        ["trade_date", "ts_code", "up_limit", "down_limit"],
    ].rename(
        columns={
            "trade_date": "target_date",
            "up_limit": "auction_limit_up_price",
            "down_limit": "auction_limit_down_price",
        }
    )
    context = context.merge(
        target_limits,
        on=["target_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    context["source_trade_date"] = context["trade_date"]
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
    cache: AuctionTickCache | None = None,
    qmt_minute_matches: dict[str, dict[str, object]] | None = None,
    qmt_tick_matches: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    paths_by_date = {path.parent.name: path for path in symbol_paths}
    records: list[dict[str, object]] = []
    for trade_date in sorted(set(target_dates)):
        record: dict[str, object] = {
            "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
            "benchmark_ts_code": benchmark_ts_code,
            "benchmark_available_time": pd.NaT,
            "benchmark_auction_has_match": False,
            "market_return_from_prev_close": np.nan,
            "_benchmark_auction_return_stage2": np.nan,
        }
        path = paths_by_date.get(pd.Timestamp(trade_date).strftime("%Y%m%d"))
        if path is not None:
            quotes = load_quote_frame(path, cache=cache)
            if quotes.empty:
                LOGGER.warning(
                    "Empty benchmark auction quote frame for %s on %s",
                    benchmark_ts_code,
                    record["trade_date"],
                )
            else:
                benchmark_row = _calculate_daily_with_match_fallback(
                    quotes,
                    benchmark_ts_code,
                    symbol_dir=path,
                    cache=cache,
                    qmt_minute_matches=qmt_minute_matches,
                    qmt_tick_matches=qmt_tick_matches,
                )
                record["benchmark_available_time"] = benchmark_row["available_time"]
                record["benchmark_auction_has_match"] = benchmark_row[
                    "auction_has_match"
                ]
                if bool(benchmark_row["auction_has_match"]):
                    record["market_return_from_prev_close"] = benchmark_row[
                        "auction_overnight_return"
                    ]
                    record["_benchmark_auction_return_stage2"] = benchmark_row[
                        "auction_return_stage2"
                    ]
        records.append(record)
    return pd.DataFrame(records)


def _apply_context_supplement_factors(result: pd.DataFrame, index: int) -> None:
    matched_volume = result.at[index, "auction_matched_volume"]
    auction_amount = result.at[index, "auction_amount"]
    open_price = result.at[index, "auction_open_price"]
    result.at[index, "auction_volume_to_prevday_volume"] = _safe_ratio(
        matched_volume, result.at[index, "previous_day_volume_shares"]
    )
    result.at[index, "auction_amount_to_float_mcap_prevclose"] = _safe_ratio(
        auction_amount, result.at[index, "previous_day_float_market_cap_cny"]
    )
    result.at[index, "auction_open_to_prev_high"] = _safe_return(
        open_price, result.at[index, "previous_day_high"]
    )
    result.at[index, "auction_open_to_prev7d_close_max"] = _safe_return(
        open_price, result.at[index, "previous_7d_close_max"]
    )

    previous_close = result.at[index, "previous_close"]
    stage1_max_return = result.at[
        index, "auction_stage1_max_return_from_prev_close"
    ]
    stage1_min_return = result.at[
        index, "auction_stage1_min_return_from_prev_close"
    ]
    if not np.isfinite(previous_close) or previous_close <= 0:
        return

    if np.isfinite(stage1_max_return):
        stage1_max = float(previous_close * (1.0 + stage1_max_return))
        limit_up = result.at[index, "auction_limit_up_price"]
        if np.isfinite(limit_up) and limit_up > 0:
            result.at[index, "auction_stage1_touched_limit_up"] = float(
                stage1_max >= limit_up
            )
            result.at[index, "auction_stage1_limit_up_distance_bps"] = float(
                (limit_up - stage1_max) / limit_up * 10000.0
            )

    if np.isfinite(stage1_min_return):
        stage1_min = float(previous_close * (1.0 + stage1_min_return))
        limit_down = result.at[index, "auction_limit_down_price"]
        if np.isfinite(limit_down) and limit_down > 0:
            result.at[index, "auction_stage1_touched_limit_down"] = float(
                stage1_min <= limit_down
            )
            result.at[index, "auction_stage1_limit_down_distance_bps"] = float(
                (stage1_min - limit_down) / limit_down * 10000.0
            )


def _apply_volume_ratio_factors(
    result: pd.DataFrame,
    index: int,
    daily_volume_history: pd.Series,
) -> None:
    """Calculate volume ratio factors."""
    trade_date = pd.Timestamp(result.at[index, "trade_date"])
    auction_volume = result.at[index, "auction_matched_volume"]

    # Initialize columns
    result.at[index, "auction_volume_ratio_5d"] = np.nan
    result.at[index, "auction_volume_ratio_20d"] = np.nan

    if not np.isfinite(auction_volume) or auction_volume <= 0:
        return

    # Get historical volumes before trade date
    prior_volumes = daily_volume_history.loc[
        daily_volume_history.index < trade_date
    ]

    if prior_volumes.empty:
        return

    # 5-day average volume ratio
    if len(prior_volumes) >= 5:
        recent_5d = prior_volumes.iloc[-5:]
        avg_5d = float(recent_5d.mean())
        if avg_5d > 0:
            result.at[index, "auction_volume_ratio_5d"] = float(
                auction_volume / avg_5d
            )

    # 20-day average volume ratio
    if len(prior_volumes) >= 20:
        recent_20d = prior_volumes.iloc[-20:]
        avg_20d = float(recent_20d.mean())
        if avg_20d > 0:
            result.at[index, "auction_volume_ratio_20d"] = float(
                auction_volume / avg_20d
            )


def _apply_volume_ratio_zscore(result: pd.DataFrame) -> None:
    """Calculate rolling Z-score for volume ratio."""
    volume_ratios = []

    for index, row in result.iterrows():
        ratio = row["auction_volume_ratio_5d"]
        result.at[index, "auction_volume_ratio_5d_zscore"] = np.nan

        # Need at least 20 historical ratios to compute Z-score
        if len(volume_ratios) >= 20 and np.isfinite(ratio):
            recent = np.asarray(volume_ratios[-20:], dtype=float)
            mean = float(recent.mean())
            std = float(recent.std(ddof=0))
            if std > 0:
                result.at[index, "auction_volume_ratio_5d_zscore"] = float(
                    (ratio - mean) / std
                )

        # Append current ratio to history
        if np.isfinite(ratio):
            volume_ratios.append(float(ratio))


def _apply_turnover_rate_factors(result: pd.DataFrame, index: int) -> None:
    """Calculate auction turnover rate factors.

    auction_turnover_rate = auction_matched_volume / float_share * 100
    auction_turnover_rate_free = auction_matched_volume / free_share * 100
    """
    auction_volume = result.at[index, "auction_matched_volume"]
    float_share = result.at[index, "previous_day_float_share"]
    free_share = result.at[index, "previous_day_free_share"]

    # Initialize columns
    result.at[index, "auction_turnover_rate"] = np.nan
    result.at[index, "auction_turnover_rate_free"] = np.nan

    if not np.isfinite(auction_volume) or auction_volume <= 0:
        return

    # Calculate turnover rate using float_share
    if np.isfinite(float_share) and float_share > 0:
        result.at[index, "auction_turnover_rate"] = float(
            auction_volume / float_share * 100.0
        )

    # Calculate turnover rate using free_share
    if np.isfinite(free_share) and free_share > 0:
        result.at[index, "auction_turnover_rate_free"] = float(
            auction_volume / free_share * 100.0
        )


def apply_supplemental_context(
    frame: pd.DataFrame, symbol_context: pd.DataFrame | None
) -> pd.DataFrame:
    """Apply only daily context needed by the report-supplement columns."""
    result = frame.copy()
    context_by_date = (
        symbol_context.set_index("trade_date", drop=False)
        if symbol_context is not None and not symbol_context.empty
        else pd.DataFrame()
    )
    context_columns = [
        "previous_day_volume_shares",
        "previous_day_high",
        "previous_7d_close_max",
        "previous_day_float_market_cap_cny",
        "previous_day_float_share",
        "previous_day_free_share",
        "auction_limit_up_price",
        "auction_limit_down_price",
    ]
    for index, row in result.iterrows():
        trade_date = pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d")
        if not context_by_date.empty and trade_date in context_by_date.index:
            context = context_by_date.loc[trade_date]
            if isinstance(context, pd.DataFrame):
                context = context.iloc[-1]
            for column in context_columns:
                if column in context.index:
                    result.at[index, column] = context[column]
        _apply_context_supplement_factors(result, index)
    return result


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
        "previous_day_volume_shares",
        "previous_day_high",
        "previous_7d_close_max",
        "previous_day_float_market_cap_cny",
        "auction_limit_up_price",
        "auction_limit_down_price",
    ]
    for index, row in result.iterrows():
        trade_date = pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d")
        if not symbol_by_date.empty and trade_date in symbol_by_date.index:
            context = symbol_by_date.loc[trade_date]
            if isinstance(context, pd.DataFrame):
                context = context.iloc[-1]
            for column in symbol_columns:
                if column in context.index:
                    result.at[index, column] = context[column]

        _apply_context_supplement_factors(result, index)

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


def apply_historical_ratios(
    frame: pd.DataFrame,
    event_frames: dict[str, pd.DataFrame] | None = None,
    daily_amount_history: pd.Series | None = None,
    daily_volume_history: pd.Series | None = None,
    daily_path: Path | None = None,
    symbol_context: pd.DataFrame | None = None,
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
            previous_5d = prior_daily_amounts.loc[
                np.isfinite(prior_daily_amounts) & prior_daily_amounts.gt(0)
            ].tail(SHORT_DAILY_AMOUNT_LOOKBACK_DAYS)
            if len(previous_5d) >= SHORT_DAILY_AMOUNT_LOOKBACK_DAYS:
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

    # Calculate volume ratio factors
    if daily_volume_history is not None:
        for index, row in result.iterrows():
            _apply_volume_ratio_factors(result, index, daily_volume_history)

        # Calculate Z-score (needs to be done after all basic ratios are computed)
        _apply_volume_ratio_zscore(result)

    # Calculate turnover rate factors
    for index, row in result.iterrows():
        _apply_turnover_rate_factors(result, index)

    return apply_report_smoothed_factors(result)[OUTPUT_COLUMNS]


def apply_report_smoothed_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the report's inclusive 20-trading-day means without future data."""
    result = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True).copy()
    for source, target in zip(
        REPORT_SMOOTHED_SOURCE_COLUMNS,
        REPORT_SMOOTHED_FACTOR_COLUMNS,
        strict=True,
    ):
        if source not in result:
            result[target] = np.nan
            continue
        values = pd.to_numeric(result[source], errors="coerce")
        result[target] = values.rolling(20, min_periods=20).mean()
    return result


def merge_symbol_output(
    output_path: Path,
    requested: pd.DataFrame,
    overwrite: bool,
    replace_existing_dates: set[str] | None = None,
) -> pd.DataFrame:
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        existing = existing.reindex(columns=OUTPUT_COLUMNS)
    else:
        existing = pd.DataFrame(columns=OUTPUT_COLUMNS)

    requested_dates = set(requested["trade_date"].astype(str))
    replace_dates = requested_dates if overwrite else (replace_existing_dates or set())
    if replace_dates:
        existing = existing.loc[
            ~existing["trade_date"].astype(str).isin(replace_dates)
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
    for time_column in (
        "available_time",
        "auction_stage1_end_time",
        "auction_stage2_end_time",
    ):
        combined[time_column] = pd.to_datetime(combined[time_column], errors="coerce")
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


def _existing_trade_dates(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_parquet(output_path, columns=["trade_date"])
    except (OSError, ValueError, KeyError):
        return set()
    return set(pd.to_datetime(existing["trade_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))


def _output_uses_current_schema(output_path: Path) -> bool:
    if not output_path.exists():
        return True
    try:
        columns = set(pq.ParquetFile(output_path).schema_arrow.names)
    except (OSError, ValueError):
        return False
    return set(OUTPUT_COLUMNS).issubset(columns)


def _missing_output_columns(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        columns = set(pq.ParquetFile(output_path).schema_arrow.names)
    except (OSError, ValueError):
        return set(OUTPUT_COLUMNS)
    return set(OUTPUT_COLUMNS) - columns


def merge_supplement_output(
    output_path: Path,
    supplement: pd.DataFrame,
    columns_to_update: list[str],
) -> pd.DataFrame:
    existing = pd.read_parquet(output_path).reindex(columns=OUTPUT_COLUMNS)
    existing["trade_date"] = pd.to_datetime(
        existing["trade_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    for column in ("auction_stage1_end_time", "auction_stage2_end_time"):
        existing[column] = pd.to_datetime(existing[column], errors="coerce")
    updates = supplement.set_index("trade_date")
    for trade_date, values in updates.iterrows():
        matching = existing["trade_date"].eq(trade_date)
        if not matching.any():
            continue
        for column in columns_to_update:
            existing.loc[matching, column] = values[column]
    for column in ("available_time", "auction_stage1_end_time", "auction_stage2_end_time"):
        existing[column] = pd.to_datetime(existing[column], errors="coerce")
    return existing.sort_values("trade_date", kind="mergesort").drop_duplicates(
        "trade_date", keep="last"
    ).reset_index(drop=True)[OUTPUT_COLUMNS]


def _missing_paths_with_warmup(
    ordered_paths: list[Path], missing_paths: list[Path]
) -> tuple[list[Path], list[Path]]:
    if not missing_paths:
        return [], []
    # Each missing-date block gets only the preceding history it needs.
    missing_keys = {path.parent.name for path in missing_paths}
    missing_indexes = [
        index for index, path in enumerate(ordered_paths) if path.parent.name in missing_keys
    ]
    warmup_keys: set[str] = set()
    block_start = missing_indexes[0]
    previous_index = block_start
    blocks: list[tuple[int, int]] = []
    for index in missing_indexes[1:]:
        if index != previous_index + 1:
            blocks.append((block_start, previous_index))
            block_start = index
        previous_index = index
    blocks.append((block_start, previous_index))
    for start, _ in blocks:
        prior = ordered_paths[:start]
        warmup_keys.update(path.parent.name for path in prior[-HISTORICAL_AMOUNT_LOOKBACK_DAYS:])
    warmup_paths = [path for path in ordered_paths if path.parent.name in warmup_keys]
    return missing_paths, warmup_paths


def _missing_dates_with_warmup(
    ordered_dates: list[str], missing_dates: list[str]
) -> tuple[list[str], list[str]]:
    if not missing_dates:
        return [], []
    missing_keys = set(missing_dates)
    indexes = [index for index, date in enumerate(ordered_dates) if date in missing_keys]
    warmup_keys: set[str] = set()
    block_start = indexes[0]
    previous_index = block_start
    blocks: list[tuple[int, int]] = []
    for index in indexes[1:]:
        if index != previous_index + 1:
            blocks.append((block_start, previous_index))
            block_start = index
        previous_index = index
    blocks.append((block_start, previous_index))
    for start, _ in blocks:
        warmup_keys.update(ordered_dates[max(0, start - HISTORICAL_AMOUNT_LOOKBACK_DAYS) : start])
    return missing_dates, [date for date in ordered_dates if date in warmup_keys]


def process_qmt_symbol_series(
    ts_code: str,
    tick_path: Path,
    minute_path: Path,
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
    symbol_context: pd.DataFrame | None = None,
    benchmark_context: pd.DataFrame | None = None,
    daily_path: Path | None = None,
) -> tuple[str, Path, int]:
    output_path = output_root / f"{ts_code}.parquet"
    missing_output_columns = _missing_output_columns(output_path)
    if (
        missing_output_columns
        and missing_output_columns.issubset(REPORT_SMOOTHED_FACTOR_COLUMNS)
        and not overwrite
    ):
        existing = pd.read_parquet(output_path)
        combined = apply_report_smoothed_factors(existing).reindex(columns=OUTPUT_COLUMNS)
        output_root.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)
        LOGGER.info(
            "%s QMT backfilled %s report-smoothed columns from existing output",
            ts_code,
            len(missing_output_columns),
        )
        return "etf", output_path, len(combined)
    quotes = load_qmt_quote_frame(tick_path)
    if quotes.empty:
        LOGGER.warning("QMT tick has no 09:15-09:25 quotes for %s", ts_code)
        return "skipped", output_path, 0
    matches = load_qmt_0930_matches(minute_path)
    quotes["trade_date"] = quotes["trade_time"].dt.strftime("%Y-%m-%d")
    grouped = {
        date: group.drop(columns="trade_date").reset_index(drop=True)
        for date, group in quotes.groupby("trade_date", sort=True)
    }
    # Keep the full source calendar so dates before date_from can warm up
    # rolling auction histories without being written to the requested output.
    ordered_dates = sorted(grouped)
    requested_calendar_dates = [
        date for date in ordered_dates if _date_in_requested_range(date, date_from, date_to)
    ]
    existing_dates = _existing_trade_dates(output_path)
    missing_dates = [
        date
        for date in requested_calendar_dates
        if overwrite or date not in existing_dates
    ]
    if not missing_dates:
        LOGGER.info(
            "%s QMT skipped: requested=%s existing=%s missing=0",
            ts_code,
            len(ordered_dates),
            len(existing_dates),
        )
        return "skipped", output_path, 0
    requested_dates, warmup_dates = _missing_dates_with_warmup(
        ordered_dates, missing_dates
    )
    calculation_dates = warmup_dates + requested_dates
    records: list[dict[str, object]] = []
    event_frames: dict[str, pd.DataFrame] = {}
    for date in calculation_dates:
        daily = calculate_daily_auction_factors(
            grouped[date],
            ts_code,
            _empty_event_frame(),
            False,
            match_override=matches.get(date),
            match_source="qmt_0930_minute" if date in matches else None,
        )
        records.append(daily)
        event_frames[date] = _empty_event_frame()
    daily_amount_history = load_daily_amount_history(minute_path)
    daily_volume_history = load_daily_volume_history(minute_path)
    factor_frame = apply_historical_ratios(
        pd.DataFrame(records),
        event_frames=event_frames,
        daily_amount_history=daily_amount_history,
        daily_volume_history=daily_volume_history,
        daily_path=daily_path,
        symbol_context=symbol_context,
    )
    factor_frame = apply_external_context(
        factor_frame,
        symbol_context=symbol_context,
        benchmark_context=benchmark_context,
    )
    requested_frame = factor_frame.loc[
        factor_frame["trade_date"].isin(set(requested_dates))
    ].copy()
    combined = merge_symbol_output(
        output_path,
        requested_frame,
        overwrite,
        replace_existing_dates=set(requested_dates) if overwrite else None,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    LOGGER.info(
        "%s QMT dates: requested=%s existing=%s missing=%s warmup=%s matches=%s",
        ts_code,
        len(ordered_dates),
        len(existing_dates),
        len(requested_dates),
        len(warmup_dates),
        sum(date in matches for date in requested_dates),
    )
    return "etf", output_path, len(requested_frame)


def _qmt_date_bounds(paths: list[Path]) -> tuple[pd.Timestamp, pd.Timestamp]:
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                if column.path_in_schema != "trade_date" or column.statistics is None:
                    continue
                start = pd.Timestamp(column.statistics.min).normalize()
                end = pd.Timestamp(column.statistics.max).normalize()
                minimum = start if minimum is None or start < minimum else minimum
                maximum = end if maximum is None or end > maximum else maximum
    if minimum is None or maximum is None:
        raise ValueError("QMT tick files have no trade_date statistics")
    return minimum, maximum


def build_qmt_benchmark_context(
    benchmark_ts_code: str,
    tick_path: Path | None,
    minute_path: Path | None,
    target_dates: list[str],
    historical_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if tick_path is None or minute_path is None:
        return build_benchmark_context(benchmark_ts_code, [], target_dates, historical_context)
    quotes = load_qmt_quote_frame(tick_path)
    matches = load_qmt_0930_matches(minute_path)
    quotes["trade_date"] = quotes["trade_time"].dt.strftime("%Y-%m-%d")
    grouped = {
        date: group.drop(columns="trade_date").reset_index(drop=True)
        for date, group in quotes.groupby("trade_date", sort=True)
    }
    records: list[dict[str, object]] = []
    for date in sorted(set(target_dates)):
        record = {
            "trade_date": date,
            "benchmark_ts_code": benchmark_ts_code,
            "benchmark_available_time": pd.NaT,
            "benchmark_auction_has_match": False,
            "market_return_from_prev_close": np.nan,
            "_benchmark_auction_return_stage2": np.nan,
        }
        if date in grouped:
            row = calculate_daily_auction_factors(
                grouped[date],
                benchmark_ts_code,
                _empty_event_frame(),
                False,
                match_override=matches.get(date),
                match_source="qmt_0930_minute" if date in matches else None,
            )
            record["benchmark_available_time"] = row["available_time"]
            record["benchmark_auction_has_match"] = row["auction_has_match"]
            if bool(row["auction_has_match"]):
                record["market_return_from_prev_close"] = row["auction_overnight_return"]
                record["_benchmark_auction_return_stage2"] = row["auction_return_stage2"]
        records.append(record)
    return pd.DataFrame(records)


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
    auction_cache_root: Path | None = DEFAULT_AUCTION_CACHE_ROOT,
    refresh_auction_cache: bool = False,
    refresh_existing_factors: bool = False,
    use_qmt_match_fallback: bool = False,
    qmt_tick_path: Path | None = None,
    qmt_minute_path: Path | None = None,
    daily_path: Path | None = None,
) -> tuple[str, Path, int]:
    ordered_paths = sorted(symbol_paths, key=lambda path: path.parent.name)
    all_requested_paths = [
        path
        for path in ordered_paths
        if _date_in_requested_range(path.parent.name, date_from, date_to)
    ]
    output_path = output_root / f"{ts_code}.parquet"
    existing_dates = _existing_trade_dates(output_path)
    output_uses_current_schema = _output_uses_current_schema(output_path)
    missing_output_columns = _missing_output_columns(output_path)
    smoothed_columns = [
        column
        for column in REPORT_SMOOTHED_FACTOR_COLUMNS
        if column in missing_output_columns
    ]
    derived_only = bool(smoothed_columns) and missing_output_columns.issubset(
        REPORT_SMOOTHED_FACTOR_COLUMNS
    )
    if derived_only and not (overwrite or refresh_existing_factors):
        existing = pd.read_parquet(output_path)
        combined = apply_report_smoothed_factors(existing).reindex(columns=OUTPUT_COLUMNS)
        output_root.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)
        LOGGER.info(
            "%s backfilled %s report-smoothed columns from %s existing dates",
            ts_code,
            len(smoothed_columns),
            len(combined),
        )
        return (asset_type, output_path, len(combined))
    supplement_columns = [
        column for column in SUPPLEMENT_OUTPUT_COLUMNS if column in missing_output_columns
    ]
    supplement_only = bool(supplement_columns) and missing_output_columns.issubset(
        set(SUPPLEMENT_OUTPUT_COLUMNS) | set(REPORT_SMOOTHED_FACTOR_COLUMNS)
    )
    if supplement_only and not (overwrite or refresh_existing_factors):
        if not all_requested_paths:
            return ("skipped", output_path, 0)
        LOGGER.info(
            "%s backfilling %s supplement columns for %s dates",
            ts_code,
            len(supplement_columns),
            len(all_requested_paths),
        )
        cache = AuctionTickCache(auction_cache_root, refresh=refresh_auction_cache)
        records: list[dict[str, object]] = []
        for path in all_requested_paths:
            quotes = load_quote_frame(path, cache=cache)
            if quotes.empty:
                LOGGER.warning(
                    "Empty auction quote frame for %s on %s; skipping date",
                    ts_code,
                    path.parent.name,
                )
                continue
            records.append(calculate_supplemental_auction_fields(quotes, ts_code))
        if records:
            supplement = apply_supplemental_context(
                pd.DataFrame(records), symbol_context
            )
            combined = merge_supplement_output(
                output_path,
                supplement,
                supplement_columns,
            )
            if smoothed_columns:
                combined = apply_report_smoothed_factors(combined).reindex(
                    columns=OUTPUT_COLUMNS
                )
            output_root.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(output_path, index=False)
        LOGGER.info(
            "%s cache: hits=%s rebuilds=%s",
            ts_code,
            cache.stats.hits,
            cache.stats.rebuilds,
        )
        return (asset_type, output_path, len(records))
    replace_existing_dates: set[str] = set()
    if overwrite or refresh_existing_factors:
        missing_paths = all_requested_paths
        replace_existing_dates = {
            pd.Timestamp(path.parent.name).strftime("%Y-%m-%d")
            for path in all_requested_paths
        }
    elif not output_uses_current_schema:
        missing_paths = all_requested_paths
        replace_existing_dates = {
            pd.Timestamp(path.parent.name).strftime("%Y-%m-%d")
            for path in all_requested_paths
        }
        LOGGER.info("%s backfilling current output schema for %s dates", ts_code, len(missing_paths))
    else:
        missing_paths = [
            path
            for path in all_requested_paths
            if pd.Timestamp(path.parent.name).strftime("%Y-%m-%d") not in existing_dates
        ]
    if not missing_paths:
        LOGGER.info(
            "%s skipped: requested=%s existing=%s missing=0",
            ts_code,
            len(all_requested_paths),
            len(existing_dates),
        )
        return ("skipped", output_path, 0)

    requested_paths, warmup_paths = _missing_paths_with_warmup(
        ordered_paths, missing_paths
    )
    LOGGER.info(
        "%s dates: requested=%s existing=%s missing=%s warmup=%s",
        ts_code,
        len(all_requested_paths),
        len(existing_dates),
        len(requested_paths),
        len(warmup_paths),
    )
    cache = AuctionTickCache(auction_cache_root, refresh=refresh_auction_cache)
    qmt_minute_matches: dict[str, dict[str, object]] = {}
    qmt_tick_matches: dict[str, dict[str, object]] = {}
    if use_qmt_match_fallback and asset_type == "etf":
        qmt_minute_matches, qmt_tick_matches = _load_qmt_match_fallbacks(
            ts_code, qmt_tick_path, qmt_minute_path
        )

    def calculate_path_record(
        path: Path,
    ) -> tuple[dict[str, object], pd.DataFrame] | None:
        quotes = load_quote_frame(path, cache=cache)
        if quotes.empty:
            LOGGER.warning(
                "Empty auction quote frame for %s on %s; skipping date",
                ts_code,
                path.parent.name,
            )
            return None
        events, event_ok = load_auction_event_frame(
            path, ts_code, expected_trade_date=path.parent.name, cache=cache
        )
        daily = _calculate_daily_with_match_fallback(
            quotes,
            ts_code,
            events,
            event_ok,
            symbol_dir=path,
            cache=cache,
            qmt_minute_matches=qmt_minute_matches,
            qmt_tick_matches=qmt_tick_matches,
        )
        return daily, events

    warmup_records: list[tuple[dict[str, object], pd.DataFrame]] = []
    for path in warmup_paths:
        record = calculate_path_record(path)
        if record is not None:
            warmup_records.append(record)

    requested_records: list[tuple[dict[str, object], pd.DataFrame]] = []
    for path in requested_paths:
        record = calculate_path_record(path)
        if record is not None:
            requested_records.append(record)

    all_records = list(reversed(warmup_records)) + requested_records
    if not all_records:
        LOGGER.warning("%s skipped: no valid auction quote dates", ts_code)
    requested_dates = {
        pd.Timestamp(path.parent.name).strftime("%Y-%m-%d") for path in requested_paths
    }
    if all_records:
        all_rows = [row for row, _ in all_records]
        event_frames = {row["trade_date"]: events for row, events in all_records}
        daily_amount_history = load_daily_amount_history(minute_path)
        daily_volume_history = load_daily_volume_history(minute_path)
        factor_frame = apply_historical_ratios(
            pd.DataFrame(all_rows),
            event_frames=event_frames,
            daily_amount_history=daily_amount_history,
            daily_volume_history=daily_volume_history,
            daily_path=daily_path,
            symbol_context=symbol_context,
        )
        factor_frame = apply_external_context(
            factor_frame,
            symbol_context=symbol_context,
            benchmark_context=benchmark_context,
        )
        requested_frame = factor_frame.loc[
            factor_frame["trade_date"].isin(requested_dates)
        ].copy()
    else:
        requested_frame = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if not requested_frame.empty or output_path.exists():
        combined = merge_symbol_output(
            output_path,
            requested_frame,
            overwrite,
            replace_existing_dates=replace_existing_dates,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)
    LOGGER.info(
        "%s cache: hits=%s rebuilds=%s",
        ts_code,
        cache.stats.hits,
        cache.stats.rebuilds,
    )
    return (asset_type, output_path, len(requested_frame))


def run_qmt_auction_generation(
    args: argparse.Namespace,
    requested_codes: set[str] | None,
    date_from: str | None,
    date_to: str | None,
) -> int:
    if args.asset_type not in {"etf", "both"}:
        raise ValueError("--use-qmt-auction-source supports ETF symbols only")
    tick_index = build_universe_index(args.qmt_tick_root)
    minute_index = build_universe_index(args.qmt_minute_root)
    existing_codes = existing_output_codes(args.etf_output_root)
    selected_codes = sorted(set(tick_index) & set(minute_index) & existing_codes)
    if requested_codes is not None:
        selected_codes = [code for code in selected_codes if code in requested_codes]
        missing = sorted(requested_codes - set(selected_codes))
        if missing:
            raise FileNotFoundError(
                "Requested QMT ETF symbols need existing auction, tick, and minute files: "
                + ", ".join(missing[:20])
            )
    if args.limit is not None:
        selected_codes = selected_codes[: args.limit]
    if not selected_codes:
        LOGGER.warning("No QMT ETF symbols matched the existing auction universe.")
        return 0
    benchmark_ts_code = args.benchmark_ts_code.strip().upper()
    benchmark_code = numeric_code(benchmark_ts_code)
    if benchmark_code is None:
        raise ValueError(f"Invalid --benchmark-ts-code: {args.benchmark_ts_code}")
    benchmark_tick_path = (
        args.qmt_tick_root / f"{tick_index[benchmark_code]}.parquet"
        if benchmark_code in tick_index
        else None
    )
    benchmark_minute_path = (
        args.qmt_minute_root / f"{minute_index[benchmark_code]}.parquet"
        if benchmark_code in minute_index
        else None
    )
    task_tick_paths = [
        args.qmt_tick_root / f"{tick_index[code]}.parquet" for code in selected_codes
    ]
    if benchmark_tick_path is not None:
        task_tick_paths.append(benchmark_tick_path)
    min_date, max_date = _qmt_date_bounds(task_tick_paths)
    if date_from is not None:
        min_date = max(min_date, pd.Timestamp(date_from))
    if date_to is not None:
        max_date = min(max_date, pd.Timestamp(date_to))
    if min_date > max_date:
        LOGGER.warning("No QMT dates remain after the requested date range.")
        return 0
    target_dates = [
        date.strftime("%Y-%m-%d") for date in pd.bdate_range(min_date, max_date)
    ]
    requested_symbols = {tick_index[code] for code in selected_codes}
    historical_context = build_historical_context(
        args.etf_daily_path,
        target_dates,
        requested_symbols | {benchmark_ts_code},
        include_daily_factor_fields=False,
    )
    benchmark_context = build_qmt_benchmark_context(
        benchmark_ts_code,
        benchmark_tick_path,
        benchmark_minute_path,
        target_dates,
        historical_context.get(benchmark_ts_code),
    )
    tasks = [
        (
            tick_index[code],
            args.qmt_tick_root / f"{tick_index[code]}.parquet",
            args.qmt_minute_root / f"{minute_index[code]}.parquet",
        )
        for code in selected_codes
    ]
    LOGGER.info(
        "Generating QMT auction factors for %s ETF symbols, %s to %s, output=%s",
        len(tasks),
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
        args.qmt_output_root,
    )
    failures: list[tuple[str, str]] = []
    written = 0
    worker_count = max(1, args.workers)
    if worker_count == 1:
        for symbol, tick_path, minute_path in tasks:
            try:
                _, output_path, row_count = process_qmt_symbol_series(
                    symbol,
                    tick_path,
                    minute_path,
                    args.qmt_output_root,
                    date_from,
                    date_to,
                    args.overwrite,
                    historical_context.get(symbol),
                    benchmark_context,
                    args.etf_daily_path,
                )
                written += int(row_count > 0)
                LOGGER.info("Wrote %s QMT rows to %s", row_count, output_path)
            except Exception as exc:  # noqa: BLE001
                failures.append((symbol, str(exc)))
                LOGGER.exception("Failed to process QMT auction factors for %s", symbol)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_qmt_symbol_series,
                    symbol,
                    tick_path,
                    minute_path,
                    args.qmt_output_root,
                    date_from,
                    date_to,
                    args.overwrite,
                    historical_context.get(symbol),
                    benchmark_context,
                    args.etf_daily_path,
                ): symbol
                for symbol, tick_path, minute_path in tasks
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, output_path, row_count = future.result()
                    written += int(row_count > 0)
                    LOGGER.info("Wrote %s QMT rows to %s", row_count, output_path)
                except Exception as exc:  # noqa: BLE001
                    failures.append((symbol, str(exc)))
                    LOGGER.exception("Failed to process QMT auction factors for %s", symbol)
    LOGGER.info(
        "Completed QMT auction factors: %s symbol files written, %s failures",
        written,
        len(failures),
    )
    if failures:
        for symbol, error in failures[:20]:
            LOGGER.error("%s: %s", symbol, error)
        return 1
    return 0


def run_legacy_session_path_generation(
    assets: list[tuple[str, str, str]],
    args: argparse.Namespace,
    date_from: str | None,
    date_to: str | None,
) -> int:
    session_path_output_roots = {
        "stock": args.stock_session_path_output_root,
        "etf": args.etf_session_path_output_root,
    }
    minute_roots = {
        "stock": args.stock_minute_root,
        "etf": args.etf_minute_root,
    }
    LOGGER.info("Generating session path factors for %s symbols", len(assets))
    failures: list[tuple[str, str]] = []
    written = 0
    worker_count = max(1, args.workers)
    if worker_count == 1:
        for kind, _, symbol in assets:
            try:
                output_path, row_count = process_session_path_only(
                    symbol,
                    minute_roots[kind] / f"{symbol}.parquet",
                    session_path_output_roots[kind],
                    date_from,
                    date_to,
                    args.overwrite,
                )
                written += int(row_count > 0)
                LOGGER.info("Wrote %s session path rows to %s", row_count, output_path)
            except Exception as exc:  # noqa: BLE001
                failures.append((symbol, str(exc)))
                LOGGER.exception("Failed to process session path factors for %s", symbol)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_session_path_only,
                    symbol,
                    minute_roots[kind] / f"{symbol}.parquet",
                    session_path_output_roots[kind],
                    date_from,
                    date_to,
                    args.overwrite,
                ): symbol
                for kind, _, symbol in assets
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    output_path, row_count = future.result()
                    written += int(row_count > 0)
                    LOGGER.info(
                        "Wrote %s session path rows to %s", row_count, output_path
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append((symbol, str(exc)))
                    LOGGER.exception(
                        "Failed to process session path factors for %s", symbol
                    )

    LOGGER.info(
        "Completed session path factors: %s symbol files written, %s failures",
        written,
        len(failures),
    )
    if failures:
        for symbol, error in failures[:20]:
            LOGGER.error("%s: %s", symbol, error)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    configure_logging()
    if args.write_session_path_factors or args.session_path_only:
        LOGGER.warning(
            "Session-path generation is deprecated in generate_auction_factors.py; "
            "use generate_etf_minute_factors.py."
        )
    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    requested_codes = load_requested_codes(args.symbols, args.symbols_file)
    if args.use_qmt_auction_source:
        return run_qmt_auction_generation(args, requested_codes, date_from, date_to)
    assets = build_asset_universe(
        args.asset_type,
        args.stock_minute_root,
        args.etf_minute_root,
        requested_codes,
    )
    if args.existing_output_only:
        existing_by_kind = {
            "stock": existing_output_codes(args.stock_output_root),
            "etf": existing_output_codes(args.etf_output_root),
        }
        assets = [
            asset for asset in assets if asset[1] in existing_by_kind[asset[0]]
        ]
    if args.limit is not None:
        assets = assets[: args.limit]
    if not assets:
        LOGGER.warning("No symbols matched the requested asset universe.")
        return 0

    if args.session_path_only:
        return run_legacy_session_path_generation(assets, args, date_from, date_to)
    if args.write_session_path_factors:
        status = run_legacy_session_path_generation(assets, args, date_from, date_to)
        if status:
            return status

    benchmark_ts_code = args.benchmark_ts_code.strip().upper()
    benchmark_numeric_code = numeric_code(benchmark_ts_code)
    if benchmark_numeric_code is None:
        raise ValueError(f"Invalid --benchmark-ts-code: {args.benchmark_ts_code}")
    date_dirs = discover_trade_date_dirs(args.tick_root, date_to)
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
    tasks = [
        (kind, code, symbol, grouped_paths.get(code, []))
        for kind, code, symbol in assets
        if grouped_paths.get(code)
    ]
    target_dates: set[str] = set()
    pending_symbols: dict[str, set[str]] = {"stock": set(), "etf": set()}
    requires_full_context_by_kind = {"stock": False, "etf": False}
    for kind, _, symbol, paths in tasks:
        output_path = output_roots[kind] / f"{symbol}.parquet"
        existing_dates = _existing_trade_dates(output_path)
        output_uses_current_schema = _output_uses_current_schema(output_path)
        missing_output_columns = _missing_output_columns(output_path)
        supplement_only = bool(missing_output_columns) and missing_output_columns.issubset(
            set(SUPPLEMENT_OUTPUT_COLUMNS) | set(REPORT_SMOOTHED_FACTOR_COLUMNS)
        )
        pending = [
            path
            for path in paths
            if _date_in_requested_range(path.parent.name, date_from, date_to)
            and (
                args.overwrite
                or args.refresh_existing_factors
                or not output_uses_current_schema
                or pd.Timestamp(path.parent.name).strftime("%Y-%m-%d")
                not in existing_dates
            )
        ]
        if pending:
            pending_symbols[kind].add(symbol)
            target_dates.update(path.parent.name for path in pending)
            requires_full_context_by_kind[kind] |= (
                args.overwrite
                or args.refresh_existing_factors
                or not supplement_only
            )
    target_dates = sorted(target_dates)
    requested_by_kind = {
        kind: pending_symbols[kind]
        for kind in ("stock", "etf")
    }
    historical_context_by_kind: dict[str, dict[str, pd.DataFrame]] = {
        "stock": (
            build_historical_context(
                args.stock_daily_path,
                target_dates,
                requested_by_kind["stock"],
                include_daily_factor_fields=False,
            )
            if requested_by_kind["stock"]
            else {}
        ),
        "etf": build_historical_context(
            args.etf_daily_path,
            target_dates,
            requested_by_kind["etf"] | {benchmark_ts_code},
            include_daily_factor_fields=False,
        ),
    }
    need_etf_qmt_fallback = (
        args.use_qmt_match_fallback and requires_full_context_by_kind["etf"]
    )
    qmt_tick_index: dict[str, str] = {}
    qmt_minute_index: dict[str, str] = {}
    if need_etf_qmt_fallback:
        if args.qmt_tick_root.exists():
            qmt_tick_index = build_universe_index(args.qmt_tick_root)
        else:
            LOGGER.warning("QMT tick root is unavailable: %s", args.qmt_tick_root)
        if args.qmt_minute_root.exists():
            qmt_minute_index = build_universe_index(args.qmt_minute_root)
        else:
            LOGGER.warning("QMT minute root is unavailable: %s", args.qmt_minute_root)
    benchmark_historical = historical_context_by_kind["etf"].get(
        benchmark_ts_code
    )
    if benchmark_historical is None:
        benchmark_historical = historical_context_by_kind["stock"].get(
            benchmark_ts_code
        )
    benchmark_context_by_kind = {"stock": pd.DataFrame(), "etf": pd.DataFrame()}
    if requires_full_context_by_kind["stock"]:
        benchmark_context_by_kind["stock"] = build_benchmark_context(
            benchmark_ts_code,
            grouped_paths.get(benchmark_numeric_code, []),
            target_dates,
            benchmark_historical,
            cache=AuctionTickCache(
                args.auction_cache_root, refresh=args.refresh_auction_cache
            ),
        )
    if requires_full_context_by_kind["etf"]:
        benchmark_qmt_minute_matches: dict[str, dict[str, object]] = {}
        benchmark_qmt_tick_matches: dict[str, dict[str, object]] = {}
        if need_etf_qmt_fallback:
            benchmark_tick_path = (
                args.qmt_tick_root / f"{qmt_tick_index[benchmark_numeric_code]}.parquet"
                if benchmark_numeric_code in qmt_tick_index
                else None
            )
            benchmark_minute_path = (
                args.qmt_minute_root
                / f"{qmt_minute_index[benchmark_numeric_code]}.parquet"
                if benchmark_numeric_code in qmt_minute_index
                else None
            )
            benchmark_qmt_minute_matches, benchmark_qmt_tick_matches = (
                _load_qmt_match_fallbacks(
                    benchmark_ts_code, benchmark_tick_path, benchmark_minute_path
                )
            )
        benchmark_context_by_kind["etf"] = build_benchmark_context(
            benchmark_ts_code,
            grouped_paths.get(benchmark_numeric_code, []),
            target_dates,
            benchmark_historical,
            cache=AuctionTickCache(
                args.auction_cache_root, refresh=args.refresh_auction_cache
            ),
            qmt_minute_matches=benchmark_qmt_minute_matches,
            qmt_tick_matches=benchmark_qmt_tick_matches,
        )
    LOGGER.info(
        "Processing %s symbols from %s matched stock/ETF universe entries",
        len(tasks),
        len(assets),
    )

    failures: list[tuple[str, str]] = []
    written = 0
    worker_count = max(1, args.workers)
    daily_paths = {
        "stock": args.stock_daily_path,
        "etf": args.etf_daily_path,
    }
    if worker_count == 1:
        for kind, code, symbol, paths in tasks:
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
                    benchmark_context_by_kind[kind],
                    args.auction_cache_root,
                    args.refresh_auction_cache,
                    args.refresh_existing_factors,
                    args.use_qmt_match_fallback,
                    (
                        args.qmt_tick_root / f"{qmt_tick_index[code]}.parquet"
                        if kind == "etf" and code in qmt_tick_index
                        else None
                    ),
                    (
                        args.qmt_minute_root / f"{qmt_minute_index[code]}.parquet"
                        if kind == "etf" and code in qmt_minute_index
                        else None
                    ),
                    daily_paths[kind],
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
                    benchmark_context_by_kind[kind],
                    args.auction_cache_root,
                    args.refresh_auction_cache,
                    args.refresh_existing_factors,
                    args.use_qmt_match_fallback,
                    (
                        args.qmt_tick_root / f"{qmt_tick_index[code]}.parquet"
                        if kind == "etf" and code in qmt_tick_index
                        else None
                    ),
                    (
                        args.qmt_minute_root / f"{qmt_minute_index[code]}.parquet"
                        if kind == "etf" and code in qmt_minute_index
                        else None
                    ),
                    daily_paths[kind],
                ): symbol
                for kind, code, symbol, paths in tasks
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
