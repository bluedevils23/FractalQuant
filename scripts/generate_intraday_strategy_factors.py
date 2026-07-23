"""Generate opt-in P0 or combined P0+P1 intraday-strategy factors."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.intraday_strategy import (  # noqa: E402
    OUTPUT_COLUMNS,
    PRIORITY_PROFILE_P0,
    PRIORITY_PROFILE_P0_P1,
    PRIORITY_PROFILES,
    build_intraday_strategy_factor_frame,
    factor_columns_for_profile,
    normalize_minute_frame,
    output_columns_for_profile,
)
from factor.intraday_strategy_p1 import (  # noqa: E402
    MARKET_FLOW_FACTOR_COLUMNS,
    active_notional_imbalance,
    aggregate_directional_notional,
)


LOGGER = logging.getLogger("generate_intraday_strategy_factors")
DEFAULT_STOCK_MINUTE_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\行情数据\stock_1min"
)
DEFAULT_ETF_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_STOCK_DAILY_PATH = Path(
    r"D:\workspace\stockdata\a-share-data\stock_daily.parquet"
)
DEFAULT_ETF_DAILY_PATH = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_STOCK_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\stock_1min_intraday_strategy_p0_factors"
)
DEFAULT_ETF_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_intraday_strategy_p0_factors"
)
DEFAULT_STOCK_P0_P1_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\a-share-data\stock_1min_intraday_strategy_p0_p1_factors"
)
DEFAULT_ETF_P0_P1_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_intraday_strategy_p0_p1_factors"
)
DEFAULT_TICK_ROOT = Path(r"E:\逐笔数据")
DEFAULT_MARKET_FLOW_CACHE_ROOT = Path(
    r"D:\workspace\stockdata\intraday_strategy_market_flow_context"
)
ASSET_TYPES = ("stock", "etf", "both")
SYMBOL_PATTERN = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$", re.IGNORECASE)
MEMBERSHIP_COLUMNS = ["pool_id", "member_ts_code", "effective_from", "effective_to"]
TARGET_POOL_COLUMNS = ["target_ts_code", "pool_id", "effective_from", "effective_to"]
TRADE_USECOLS = [2, 3, 7, 8, 9]
TRADE_COLUMNS = ["trade_date", "raw_time", "side", "price", "qty"]
MARKET_FLOW_CACHE_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate causal P0 or combined P0+P1 intraday-strategy factors."
    )
    parser.add_argument("--asset-type", choices=ASSET_TYPES, default="both")
    parser.add_argument(
        "--priority-profile", choices=PRIORITY_PROFILES, default=PRIORITY_PROFILE_P0
    )
    parser.add_argument(
        "--stock-minute-root", type=Path, default=DEFAULT_STOCK_MINUTE_ROOT
    )
    parser.add_argument("--etf-minute-root", type=Path, default=DEFAULT_ETF_MINUTE_ROOT)
    parser.add_argument(
        "--stock-daily-path", type=Path, default=DEFAULT_STOCK_DAILY_PATH
    )
    parser.add_argument("--etf-daily-path", type=Path, default=DEFAULT_ETF_DAILY_PATH)
    parser.add_argument("--stock-output-root", type=Path, default=None)
    parser.add_argument("--etf-output-root", type=Path, default=None)
    parser.add_argument("--tick-root", type=Path, default=DEFAULT_TICK_ROOT)
    parser.add_argument("--pool-membership-path", type=Path, default=None)
    parser.add_argument("--target-pool-path", type=Path, default=None)
    parser.add_argument(
        "--market-flow-cache-root",
        type=Path,
        default=DEFAULT_MARKET_FLOW_CACHE_ROOT,
    )
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace requested dates while preserving dates outside the request.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def normalize_symbol(value: str) -> str:
    symbol = str(value).strip().upper()
    if symbol.endswith(".PARQUET"):
        symbol = symbol[:-8]
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"Invalid symbol: {value}")
    return symbol


def normalize_date_arg(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 8:
        raise ValueError(f"Invalid trade date: {value}")
    pd.Timestamp(digits)
    return digits


def read_symbol_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file does not exist: {path}")
    symbols: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if value:
                symbols.append(normalize_symbol(value))
    return symbols


def requested_symbols(
    symbols: list[str] | None, symbols_file: Path | None
) -> list[str] | None:
    values: list[str] = []
    if symbols_file is not None:
        values.extend(read_symbol_list(symbols_file))
    if symbols:
        values.extend(normalize_symbol(symbol) for symbol in symbols)
    if not values:
        return None
    return list(dict.fromkeys(values))


def resolve_output_root(
    asset_type: str,
    priority_profile: str,
    explicit_root: Path | None,
) -> Path:
    if explicit_root is not None:
        return explicit_root
    if asset_type == "stock":
        return (
            DEFAULT_STOCK_P0_P1_OUTPUT_ROOT
            if priority_profile == PRIORITY_PROFILE_P0_P1
            else DEFAULT_STOCK_OUTPUT_ROOT
        )
    if asset_type == "etf":
        return (
            DEFAULT_ETF_P0_P1_OUTPUT_ROOT
            if priority_profile == PRIORITY_PROFILE_P0_P1
            else DEFAULT_ETF_OUTPUT_ROOT
        )
    raise ValueError(f"Unsupported asset type: {asset_type}")


def _read_mapping_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _validate_non_overlapping_intervals(
    frame: pd.DataFrame,
    group_columns: list[str],
    label: str,
) -> None:
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        ordered = group.sort_values("effective_from", kind="mergesort")
        previous_end: pd.Timestamp | None = None
        for row in ordered.itertuples(index=False):
            if previous_end is not None and row.effective_from <= previous_end:
                raise ValueError(f"Overlapping {label} intervals for {keys}")
            previous_end = row.effective_to


def load_pool_membership(path: Path) -> pd.DataFrame:
    frame = _read_mapping_table(path)
    missing = [column for column in MEMBERSHIP_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Pool membership is missing required columns: {missing}")
    work = frame[MEMBERSHIP_COLUMNS].copy()
    work["pool_id"] = work["pool_id"].astype(str).str.strip()
    work["member_ts_code"] = work["member_ts_code"].map(normalize_symbol)
    return _normalize_effective_intervals(
        work,
        ["pool_id", "member_ts_code"],
        "pool membership",
    )


def load_target_pool_mapping(path: Path) -> pd.DataFrame:
    frame = _read_mapping_table(path)
    missing = [column for column in TARGET_POOL_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Target-pool mapping is missing required columns: {missing}")
    work = frame[TARGET_POOL_COLUMNS].copy()
    work["target_ts_code"] = work["target_ts_code"].map(normalize_symbol)
    work["pool_id"] = work["pool_id"].astype(str).str.strip()
    return _normalize_effective_intervals(
        work,
        ["target_ts_code"],
        "target-pool mapping",
    )


def _normalize_effective_intervals(
    frame: pd.DataFrame,
    group_columns: list[str],
    label: str,
) -> pd.DataFrame:
    work = frame.copy()
    work["effective_from"] = pd.to_datetime(
        work["effective_from"], errors="coerce"
    ).dt.normalize()
    raw_end = pd.to_datetime(work["effective_to"], errors="coerce").dt.normalize()
    work["effective_to"] = raw_end.fillna(pd.Timestamp.max.normalize())
    if work["pool_id"].eq("").any() or work["effective_from"].isna().any():
        raise ValueError(f"{label} contains blank pool IDs or invalid start dates")
    if work["effective_to"].lt(work["effective_from"]).any():
        raise ValueError(f"{label} contains an end date before its start date")
    work = work.drop_duplicates().reset_index(drop=True)
    _validate_non_overlapping_intervals(work, group_columns, label)
    return work


def resolve_target_pool(
    target_mapping: pd.DataFrame,
    target_ts_code: str,
    trade_date: pd.Timestamp,
) -> str:
    active = target_mapping.loc[
        target_mapping["target_ts_code"].eq(target_ts_code)
        & target_mapping["effective_from"].le(trade_date)
        & target_mapping["effective_to"].ge(trade_date)
    ]
    if len(active) != 1:
        raise ValueError(
            f"Expected exactly one active pool for {target_ts_code} on {trade_date:%Y%m%d}; "
            f"found {len(active)}"
        )
    return str(active.iloc[0]["pool_id"])


def resolve_pool_members(
    membership: pd.DataFrame,
    pool_id: str,
    trade_date: pd.Timestamp,
) -> list[str]:
    active = membership.loc[
        membership["pool_id"].eq(pool_id)
        & membership["effective_from"].le(trade_date)
        & membership["effective_to"].ge(trade_date),
        "member_ts_code",
    ]
    members = sorted(active.unique())
    if not members:
        raise ValueError(f"Pool {pool_id} has no active members on {trade_date:%Y%m%d}")
    return members


def _parse_trade_times(trade_date: pd.Series, raw_time: pd.Series) -> pd.Series:
    date_text = trade_date.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    time_text = raw_time.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(9)
    return pd.to_datetime(
        date_text + time_text,
        format="%Y%m%d%H%M%S%f",
        errors="raise",
    )


def read_trade_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing trade file: {path}")
    frame = pd.read_csv(
        path,
        encoding="gbk",
        usecols=TRADE_USECOLS,
        dtype=str,
        low_memory=False,
    )
    frame.columns = TRADE_COLUMNS
    frame["event_time"] = _parse_trade_times(frame["trade_date"], frame["raw_time"])
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce") / 10000.0
    frame["qty"] = pd.to_numeric(frame["qty"], errors="coerce")
    return frame[["event_time", "side", "price", "qty"]]


def tick_symbol_dir(tick_root: Path, trade_date: pd.Timestamp, symbol: str) -> Path:
    date_text = trade_date.strftime("%Y%m%d")
    return tick_root / date_text[:4] / date_text[:6] / date_text / symbol


def load_minute_trade_times(path: Path, trade_date: pd.Timestamp) -> pd.DatetimeIndex:
    if not path.exists():
        raise FileNotFoundError(f"Missing member minute parquet: {path}")
    try:
        frame = pd.read_parquet(
            path,
            columns=["close"],
            filters=[("trade_date", "=", trade_date)],
        )
    except (KeyError, TypeError, ValueError):
        frame = pd.read_parquet(path, columns=["close"])
    work = frame.reset_index()
    if "trade_date" not in work or "trade_time" not in work:
        raise ValueError(f"Member minute parquet has no date/time index: {path}")
    dates = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    times = pd.to_datetime(
        work.loc[dates.eq(trade_date), "trade_time"], errors="coerce"
    )
    return pd.DatetimeIndex(times.dropna().sort_values().drop_duplicates())


def discover_minute_files(root: Path, symbols: list[str] | None) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Minute directory does not exist: {root}")
    files = {
        path.stem.upper(): path
        for path in sorted(root.glob("*.parquet"))
        if SYMBOL_PATTERN.fullmatch(path.stem)
    }
    if symbols is None:
        return files
    return {symbol: files[symbol] for symbol in symbols if symbol in files}


def load_daily_histories(
    path: Path,
    symbols: list[str],
    date_from: str | None,
    date_to: str | None,
) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Daily parquet does not exist: {path}")
    if not symbols:
        return {}

    filters: list[tuple[str, str, object]] = []
    if date_from is not None:
        read_start = pd.Timestamp(date_from) - pd.Timedelta(days=180)
        filters.append(("trade_date", ">=", read_start))
    if date_to is not None:
        filters.append(("trade_date", "<=", pd.Timestamp(date_to)))
    filters.append(("ts_code", "in", symbols))
    columns = ["open", "high", "low", "close", "adj_factor"]
    try:
        daily = pd.read_parquet(path, columns=columns, filters=filters)
    except (KeyError, TypeError, ValueError):
        daily = pd.read_parquet(path, columns=columns)

    work = daily.reset_index()
    if "trade_date" not in work.columns or "ts_code" not in work.columns:
        raise ValueError(f"Daily parquet must expose trade_date and ts_code: {path}")
    work["trade_date"] = pd.to_datetime(
        work["trade_date"], errors="coerce"
    ).dt.normalize()
    work["ts_code"] = work["ts_code"].astype(str).str.upper()
    work = work.loc[work["ts_code"].isin(symbols)]
    if date_from is not None:
        work = work.loc[
            work["trade_date"].ge(pd.Timestamp(date_from) - pd.Timedelta(days=180))
        ]
    if date_to is not None:
        work = work.loc[work["trade_date"].le(pd.Timestamp(date_to))]
    return {
        symbol: group.reset_index(drop=True)
        for symbol, group in work.groupby("ts_code", sort=False)
    }


def filter_dates(
    minute: pd.DataFrame,
    date_from: str | None,
    date_to: str | None,
) -> pd.DataFrame:
    result = minute
    if date_from is not None:
        result = result.loc[result["trade_date"].ge(pd.Timestamp(date_from))]
    if date_to is not None:
        result = result.loc[result["trade_date"].le(pd.Timestamp(date_to))]
    return result.reset_index(drop=True)


def merge_output(
    output_path: Path,
    requested: pd.DataFrame,
    *,
    replace_all: bool,
    output_columns: list[str] = OUTPUT_COLUMNS,
) -> pd.DataFrame:
    existing: pd.DataFrame | None = None
    if output_path.exists():
        existing_columns = pq.read_schema(output_path).names
        if existing_columns != list(output_columns):
            raise ValueError(
                f"Existing output schema does not match requested profile: {output_path}"
            )
    if replace_all or not output_path.exists():
        combined = requested.copy()
    else:
        existing = pd.read_parquet(output_path)
        existing["trade_date"] = pd.to_datetime(
            existing["trade_date"], errors="coerce"
        ).dt.normalize()
        existing["trade_time"] = pd.to_datetime(existing["trade_time"], errors="coerce")
        requested_keys = pd.MultiIndex.from_frame(
            requested[["trade_date", "trade_time"]]
        )
        existing_keys = pd.MultiIndex.from_frame(existing[["trade_date", "trade_time"]])
        existing = existing.loc[~existing_keys.isin(requested_keys)]
        combined = pd.concat([existing, requested], ignore_index=True)
    return combined.sort_values(
        ["trade_date", "trade_time"], kind="mergesort"
    ).reset_index(drop=True)


def write_parquet_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".parquet.tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _member_hash(members: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()


def _pool_cache_path(
    cache_root: Path,
    pool_id: str,
    trade_date: pd.Timestamp,
) -> Path:
    safe_pool = re.sub(r"[^A-Za-z0-9_.-]+", "_", pool_id)
    return cache_root / safe_pool / f"{trade_date:%Y%m%d}.parquet"


def _read_valid_pool_cache(
    cache_path: Path,
    bar_times: pd.DatetimeIndex,
    expected_hash: str,
    expected_count: int,
) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    required = {
        "trade_date",
        "trade_time",
        "buy_notional_1m",
        "sell_notional_1m",
        "market_active_notional_imbalance_1m",
        "market_active_notional_imbalance_cum_session",
        "member_count",
        "member_hash",
        "cache_version",
    }
    cached = pd.read_parquet(cache_path)
    if not required.issubset(cached.columns) or cached.empty:
        return None
    cached_times = pd.DatetimeIndex(
        pd.to_datetime(cached["trade_time"], errors="coerce")
    )
    if not cached_times.equals(bar_times):
        return None
    if not cached["member_hash"].eq(expected_hash).all():
        return None
    if not cached["member_count"].eq(expected_count).all():
        return None
    if not cached["cache_version"].eq(MARKET_FLOW_CACHE_VERSION).all():
        return None
    return cached


def build_pool_market_flow(
    pool_id: str,
    trade_date: pd.Timestamp,
    members: list[str],
    bar_times: pd.DatetimeIndex,
    *,
    tick_root: Path,
    stock_minute_root: Path,
    cache_root: Path,
) -> pd.DataFrame:
    """Build or reuse a strict point-in-time pool flow for one date."""
    expected_hash = _member_hash(members)
    cache_path = _pool_cache_path(cache_root, pool_id, trade_date)
    cached = _read_valid_pool_cache(
        cache_path,
        bar_times,
        expected_hash,
        len(members),
    )
    if cached is not None:
        return cached

    total_buy = np.zeros(len(bar_times), dtype=float)
    total_sell = np.zeros(len(bar_times), dtype=float)
    for member in members:
        minute_path = stock_minute_root / f"{member}.parquet"
        member_times = load_minute_trade_times(minute_path, trade_date)
        if member_times.empty:
            continue
        if not member_times.equals(bar_times):
            raise ValueError(
                f"Incomplete member minute coverage for {member} on {trade_date:%Y%m%d}"
            )
        trade_path = tick_symbol_dir(tick_root, trade_date, member) / "逐笔成交.csv"
        events = read_trade_events(trade_path)
        aggregated = aggregate_directional_notional(events, bar_times)
        total_buy += aggregated["buy_notional_1m"].to_numpy(dtype=float)
        total_sell += aggregated["sell_notional_1m"].to_numpy(dtype=float)

    cumulative_buy = np.cumsum(total_buy)
    cumulative_sell = np.cumsum(total_sell)
    result = pd.DataFrame(
        {
            "trade_date": np.repeat(trade_date, len(bar_times)),
            "trade_time": bar_times,
            "buy_notional_1m": total_buy,
            "sell_notional_1m": total_sell,
            "market_active_notional_imbalance_1m": active_notional_imbalance(
                total_buy, total_sell
            ),
            "market_active_notional_imbalance_cum_session": active_notional_imbalance(
                cumulative_buy, cumulative_sell
            ),
            "member_count": np.repeat(len(members), len(bar_times)),
            "member_hash": np.repeat(expected_hash, len(bar_times)),
            "cache_version": np.repeat(MARKET_FLOW_CACHE_VERSION, len(bar_times)),
        }
    )
    write_parquet_atomically(result, cache_path)
    return result


def build_active_flow_context(
    target_ts_code: str,
    requested_minute: pd.DataFrame,
    membership: pd.DataFrame,
    target_mapping: pd.DataFrame,
    *,
    tick_root: Path,
    stock_minute_root: Path,
    cache_root: Path,
) -> pd.DataFrame:
    """Build all three strict market-flow fields for a target symbol."""
    per_day: list[pd.DataFrame] = []
    for trade_date, raw_day in requested_minute.groupby("trade_date", sort=True):
        day = raw_day.sort_values("trade_time", kind="mergesort")
        bar_times = pd.DatetimeIndex(day["trade_time"])
        pool_id = resolve_target_pool(target_mapping, target_ts_code, trade_date)
        members = resolve_pool_members(membership, pool_id, trade_date)
        market = build_pool_market_flow(
            pool_id,
            trade_date,
            members,
            bar_times,
            tick_root=tick_root,
            stock_minute_root=stock_minute_root,
            cache_root=cache_root,
        )

        target_trade_path = (
            tick_symbol_dir(tick_root, trade_date, target_ts_code) / "逐笔成交.csv"
        )
        target_events = read_trade_events(target_trade_path)
        target = aggregate_directional_notional(target_events, bar_times)
        asset_imbalance = active_notional_imbalance(
            target["buy_notional_1m"], target["sell_notional_1m"]
        )
        market_imbalance = market["market_active_notional_imbalance_1m"].to_numpy(
            dtype=float
        )
        per_day.append(
            pd.DataFrame(
                {
                    "trade_date": np.repeat(trade_date, len(day)),
                    "trade_time": bar_times,
                    "market_active_notional_imbalance_1m": market_imbalance,
                    "market_active_notional_imbalance_cum_session": market[
                        "market_active_notional_imbalance_cum_session"
                    ].to_numpy(dtype=float),
                    "asset_minus_market_active_flow_1m": asset_imbalance
                    - market_imbalance,
                }
            )
        )
    if not per_day:
        return pd.DataFrame(
            columns=["trade_date", "trade_time", *MARKET_FLOW_FACTOR_COLUMNS]
        )
    return pd.concat(per_day, ignore_index=True)


def process_symbol_file(
    input_path: Path,
    output_root: Path,
    daily: pd.DataFrame,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
    priority_profile: str = PRIORITY_PROFILE_P0,
    active_flow_context: pd.DataFrame | None = None,
) -> dict[str, object]:
    ts_code = normalize_symbol(input_path.stem)
    output_path = output_root / f"{ts_code}.parquet"
    if output_path.exists() and not overwrite:
        expected_columns = output_columns_for_profile(priority_profile)
        if pq.read_schema(output_path).names != expected_columns:
            raise ValueError(
                f"Existing output schema does not match requested profile: {output_path}"
            )
        return {"status": "skipped", "ts_code": ts_code, "path": output_path}

    minute_history = normalize_minute_frame(pd.read_parquet(input_path), ts_code)
    requested_minute = filter_dates(minute_history, date_from, date_to)
    if requested_minute.empty:
        return {"status": "empty", "ts_code": ts_code, "path": output_path}

    requested = build_intraday_strategy_factor_frame(
        requested_minute,
        daily,
        ts_code,
        priority_profile=priority_profile,
        minute_history=minute_history,
        active_flow_context=active_flow_context,
    )
    output_columns = output_columns_for_profile(priority_profile)
    factor_columns = factor_columns_for_profile(priority_profile)
    combined = merge_output(
        output_path,
        requested,
        replace_all=date_from is None and date_to is None,
        output_columns=output_columns,
    )
    write_parquet_atomically(combined, output_path)
    finite_counts = requested[factor_columns].notna().sum()
    return {
        "status": "written",
        "ts_code": ts_code,
        "path": output_path,
        "rows": len(requested),
        "dates": requested["trade_date"].nunique(),
        "finite_factors": int(finite_counts.gt(0).sum()),
        "factor_count": len(factor_columns),
    }


def log_result(result: dict[str, object]) -> None:
    if result["status"] == "written":
        LOGGER.info(
            "Wrote %s rows across %s dates with %s/%s populated factors: %s",
            result["rows"],
            result["dates"],
            result["finite_factors"],
            result["factor_count"],
            result["path"],
        )
    else:
        LOGGER.info("%s %s: %s", result["status"], result["ts_code"], result["path"])


def main() -> int:
    args = parse_args()
    configure_logging()
    date_from = normalize_date_arg(args.date_from)
    date_to = normalize_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.priority_profile == PRIORITY_PROFILE_P0_P1:
        if args.pool_membership_path is None or args.target_pool_path is None:
            raise ValueError(
                "--pool-membership-path and --target-pool-path are required for p0_p1"
            )
        if not args.tick_root.exists():
            raise FileNotFoundError(f"Tick root does not exist: {args.tick_root}")
    symbols = requested_symbols(args.symbols, args.symbols_file)

    configurations: list[tuple[str, Path, Path, Path]] = []
    if args.asset_type in {"stock", "both"}:
        configurations.append(
            (
                "stock",
                args.stock_minute_root,
                args.stock_daily_path,
                resolve_output_root(
                    "stock", args.priority_profile, args.stock_output_root
                ),
            )
        )
    if args.asset_type in {"etf", "both"}:
        configurations.append(
            (
                "etf",
                args.etf_minute_root,
                args.etf_daily_path,
                resolve_output_root("etf", args.priority_profile, args.etf_output_root),
            )
        )

    assets: list[tuple[str, Path, Path, pd.DataFrame]] = []
    found_symbols: set[str] = set()
    remaining = args.limit
    for kind, minute_root, daily_path, output_root in configurations:
        discovered = discover_minute_files(minute_root, symbols)
        found_symbols.update(discovered)
        files = discovered
        if remaining is not None:
            files = dict(list(discovered.items())[:remaining])
            remaining -= len(files)
        histories = load_daily_histories(
            daily_path,
            list(files),
            date_from,
            date_to,
        )
        for symbol, input_path in files.items():
            daily = histories.get(symbol, pd.DataFrame())
            assets.append((kind, input_path, output_root, daily))

    if symbols is not None:
        missing = [symbol for symbol in symbols if symbol not in found_symbols]
        if missing:
            raise FileNotFoundError(
                "Missing requested minute parquet files: " + ", ".join(missing[:10])
            )
    if not assets:
        LOGGER.warning("No minute parquet files matched the request")
        return 0

    LOGGER.info(
        "Processing %s assets with the opt-in %s-factor %s schema",
        len(assets),
        len(factor_columns_for_profile(args.priority_profile)),
        args.priority_profile,
    )
    failures: list[tuple[Path, str]] = []
    active_flow_contexts: dict[Path, pd.DataFrame] = {}
    if args.priority_profile == PRIORITY_PROFILE_P0_P1:
        membership = load_pool_membership(args.pool_membership_path)
        target_mapping = load_target_pool_mapping(args.target_pool_path)
        for _, input_path, output_root, _ in assets:
            output_path = output_root / f"{normalize_symbol(input_path.stem)}.parquet"
            if output_path.exists() and not args.overwrite:
                continue
            try:
                history = normalize_minute_frame(
                    pd.read_parquet(input_path), normalize_symbol(input_path.stem)
                )
                requested_minute = filter_dates(history, date_from, date_to)
                if requested_minute.empty:
                    continue
                active_flow_contexts[input_path] = build_active_flow_context(
                    normalize_symbol(input_path.stem),
                    requested_minute,
                    membership,
                    target_mapping,
                    tick_root=args.tick_root,
                    stock_minute_root=args.stock_minute_root,
                    cache_root=args.market_flow_cache_root,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((input_path, str(exc)))
                LOGGER.exception(
                    "Failed to build active-flow context for %s", input_path
                )

    workers = max(1, int(args.workers))
    if workers == 1:
        for _, input_path, output_root, daily in assets:
            if any(path == input_path for path, _ in failures):
                continue
            try:
                log_result(
                    process_symbol_file(
                        input_path,
                        output_root,
                        daily,
                        date_from,
                        date_to,
                        args.overwrite,
                        args.priority_profile,
                        active_flow_contexts.get(input_path),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((input_path, str(exc)))
                LOGGER.exception("Failed to process %s", input_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    process_symbol_file,
                    input_path,
                    output_root,
                    daily,
                    date_from,
                    date_to,
                    args.overwrite,
                    args.priority_profile,
                    active_flow_contexts.get(input_path),
                ): input_path
                for _, input_path, output_root, daily in assets
                if not any(path == input_path for path, _ in failures)
            }
            for future in as_completed(future_map):
                input_path = future_map[future]
                try:
                    log_result(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append((input_path, str(exc)))
                    LOGGER.exception("Failed to process %s", input_path)

    if failures:
        LOGGER.error("Completed with %s failures", len(failures))
        for path, error in failures[:10]:
            LOGGER.error("%s: %s", path, error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
