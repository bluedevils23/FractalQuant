"""Read-through cache for opening-auction tick slices."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


CACHE_VERSION = 2
DEFAULT_CHUNK_SIZE = 100_000
DATE_PATTERN = re.compile(r"^\d{8}$")

QUOTE_COLUMNS = {
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
QUOTE_PRICE_COLUMNS = [
    "trade_price",
    "open_price",
    "previous_close",
    *[f"ask_price{level}" for level in range(1, 4)],
    *[f"bid_price{level}" for level in range(1, 4)],
]
QUOTE_QUANTITY_COLUMNS = [
    "trade_volume",
    *[f"ask_qty{level}" for level in range(1, 4)],
    *[f"bid_qty{level}" for level in range(1, 4)],
]
ORDER_COLUMNS = {
    "自然日": "raw_trade_date",
    "时间": "raw_time",
    "交易所委托号": "order_id",
    "委托类型": "order_type",
    "委托代码": "side",
    "委托价格": "price",
    "委托数量": "quantity",
}
TRANSACTION_COLUMNS = {
    "自然日": "raw_trade_date",
    "时间": "raw_time",
    "成交代码": "trade_code",
    "BS标志": "bs_flag",
    "成交价格": "price",
    "成交数量": "quantity",
    "叫卖序号": "ask_order_id",
    "叫买序号": "bid_order_id",
}


def parse_trade_time(trade_date: pd.Series, raw_time: pd.Series) -> pd.Series:
    date_text = trade_date.astype(str).str.zfill(8)
    time_text = raw_time.astype(str).str.zfill(9)
    return pd.to_datetime(
        date_text + time_text,
        format="%Y%m%d%H%M%S%f",
        errors="coerce",
    )


def _date_bounds(
    symbol_dir: Path, start_time: str, end_time: str
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    date_text = symbol_dir.parent.name
    if not DATE_PATTERN.fullmatch(date_text):
        return None, None
    day = pd.Timestamp(date_text)
    start = day + pd.Timedelta(hours=int(start_time[:2]), minutes=int(start_time[3:]))
    return start, day + pd.Timedelta(hours=int(end_time[:2]), minutes=int(end_time[3:]))


def _source_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_write_frame(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp.parquet")
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def _read_csv_chunks(
    path: Path,
    usecols: list[str],
    transform: Callable[[pd.DataFrame], pd.DataFrame],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    start_time: str,
    end_time: str,
    inclusive_end: bool,
    chunksize: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing tick file: {path}")
    header = pd.read_csv(path, encoding="gbk", nrows=0).columns
    available_usecols = [column for column in usecols if column in header]
    if "自然日" not in available_usecols or "时间" not in available_usecols:
        raise ValueError(f"Tick file is missing date/time columns: {path}")
    pieces: list[pd.DataFrame] = []
    # Raw tick times use HHMMSSmmm, so HHMM needs five trailing zeroes.
    start_raw_time = start_time.replace(":", "") + "00000"
    end_raw_time = end_time.replace(":", "") + "00000"
    for chunk in pd.read_csv(
        path,
        encoding="gbk",
        usecols=available_usecols,
        dtype=str,
        low_memory=False,
        chunksize=chunksize,
    ):
        for column in usecols:
            if column not in chunk:
                chunk[column] = pd.NA
        # Most tick rows are outside the requested auction window.  Compare
        # the fixed-width raw HHMMSSmmm value first, then parse timestamps only
        # for the handful of candidate rows.
        raw_time = chunk["时间"].astype(str).str.zfill(9)
        before_end = raw_time.le(end_raw_time) if inclusive_end else raw_time.lt(end_raw_time)
        raw_mask = raw_time.ge(start_raw_time) & before_end
        if not raw_mask.any():
            continue
        selected = chunk.loc[raw_mask].copy()
        trade_time = parse_trade_time(selected["自然日"], selected["时间"])
        if start is not None and end is not None:
            before_end = trade_time.le(end) if inclusive_end else trade_time.lt(end)
            time_mask = trade_time.ge(start) & before_end
            selected = selected.loc[time_mask].copy()
            trade_time = trade_time.loc[time_mask]
        if selected.empty:
            continue
        selected["trade_time"] = trade_time
        pieces.append(transform(selected))
    if not pieces:
        return transform(pd.DataFrame(columns=usecols + ["trade_time"]))
    return pd.concat(pieces, ignore_index=True)


def _normalize_quote(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns=QUOTE_COLUMNS)
    numeric_columns = QUOTE_PRICE_COLUMNS + QUOTE_QUANTITY_COLUMNS + ["trade_amount"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[QUOTE_PRICE_COLUMNS] = frame[QUOTE_PRICE_COLUMNS] / 10000.0
    frame[QUOTE_PRICE_COLUMNS] = frame[QUOTE_PRICE_COLUMNS].where(
        frame[QUOTE_PRICE_COLUMNS] > 0
    )
    frame[QUOTE_QUANTITY_COLUMNS + ["trade_amount"]] = frame[
        QUOTE_QUANTITY_COLUMNS + ["trade_amount"]
    ].fillna(0.0)
    frame = frame.dropna(subset=["trade_time"])
    return frame.sort_values("trade_time", kind="mergesort").drop_duplicates(
        "trade_time", keep="last"
    ).reset_index(drop=True)


def _normalize_orders(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns=ORDER_COLUMNS)
    frame["order_id"] = pd.to_numeric(frame["order_id"], errors="coerce").astype("Int64")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce") / 10000.0
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["order_type"] = frame["order_type"].astype(str).str.strip().str.upper()
    frame["side"] = frame["side"].astype(str).str.strip().str.upper()
    return frame


def _normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns=TRANSACTION_COLUMNS)
    frame["trade_code"] = frame["trade_code"].astype(str).str.strip().str.upper()
    frame["bs_flag"] = frame["bs_flag"].astype(str).str.strip().str.upper()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce") / 10000.0
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["ask_order_id"] = (
        pd.to_numeric(frame["ask_order_id"], errors="coerce").fillna(0).astype("int64")
    )
    frame["bid_order_id"] = (
        pd.to_numeric(frame["bid_order_id"], errors="coerce").fillna(0).astype("int64")
    )
    return frame


def resolve_data_source(
    symbol_path: Path | str | None = None,
    explicit_source: str = "auto",
) -> str:
    """Resolve data source type among 'clickhouse', 'wind', 'qmt'."""
    explicit = (explicit_source or "auto").strip().lower()
    if explicit in ("clickhouse", "wind", "qmt"):
        return explicit
    if explicit in ("e_drive", "l2_filtered", "edrive"):
        return "wind"
    if explicit in ("l2_raw", "ch", "miniqmt"):
        return "clickhouse"

    if symbol_path is not None:
        p_str = str(symbol_path).replace("\\", "/").lower()
        if "etf_tick_qmt" in p_str or p_str.endswith(".parquet"):
            return "qmt"
        if p_str.startswith("e:/") or "逐笔数据" in p_str or "wind" in p_str:
            return "wind"
        if "clickhouse" in p_str or "miniqmt" in p_str or "outputs" in p_str:
            return "clickhouse"

    return "clickhouse"


def filter_quote_heartbeats(
    frame: pd.DataFrame,
    *,
    data_source: str = "auto",
    symbol_path: Path | None = None,
) -> pd.DataFrame:
    """Filter heartbeat snapshots based on specific data source format.

    Supported data sources:
    - 'clickhouse': Raw Level-2 snapshots with dense 3s (09:15-09:25) and
      60s (09:25-09:30) heartbeat snapshots. Filter consecutive unchanged
      states where volume/turnover deltas are 0 across all auction phases.
    - 'wind': Vendor Level-2 snapshots where 09:15-09:25 heartbeats were
      pre-filtered upstream, but 09:25-09:30 silent-period heartbeats remain.
      Filter consecutive unchanged snapshots during 09:25-09:30 and any
      auction duplicates, preparing the stream for 09:20 anchor restoration.
    - 'qmt': QMT 5-level tick snapshots. Skipped if already processed by
      load_qmt_quote_frame (checked via `_qmt_state_anchor`), or delegates
      to QMT cumulative volume/amount heartbeat filtering.
    """
    if frame.empty or "_qmt_state_anchor" in frame.columns:
        return frame

    source = resolve_data_source(symbol_path, data_source)

    ordered = (
        frame.sort_values("trade_time", kind="mergesort")
        .drop_duplicates("trade_time", keep="last")
        .reset_index(drop=True)
    )
    if ordered.empty:
        return ordered

    day_key = (
        ordered["trade_date"]
        if "trade_date" in ordered.columns
        else ordered["trade_time"].dt.normalize()
    )

    state_columns = [
        "open_price",
        "previous_close",
        *[f"ask_price{level}" for level in range(1, 4)],
        *[f"ask_qty{level}" for level in range(1, 4)],
        *[f"bid_price{level}" for level in range(1, 4)],
        *[f"bid_qty{level}" for level in range(1, 4)],
    ]
    avail_cols = [c for c in state_columns if c in ordered.columns]
    previous = ordered[avail_cols].groupby(day_key, sort=False).shift(1)
    same_state = (
        ordered[avail_cols].eq(previous)
        | (ordered[avail_cols].isna() & previous.isna())
    ).all(axis=1)

    vol = pd.to_numeric(ordered.get("trade_volume", 0), errors="coerce").fillna(0)
    amt = pd.to_numeric(ordered.get("trade_amount", 0), errors="coerce").fillna(0)
    prev_vol = vol.groupby(day_key, sort=False).shift(1)
    prev_amt = amt.groupby(day_key, sort=False).shift(1)
    no_trade = (vol.le(0) & amt.le(0)) | (vol.eq(prev_vol) & amt.eq(prev_amt))

    first_row = ordered.groupby(day_key, sort=False).cumcount().eq(0)
    heartbeat = same_state & no_trade & ~first_row
    keep = ~heartbeat
    return ordered.loc[keep].reset_index(drop=True)


def ensure_stage2_quote_anchor(quotes: pd.DataFrame) -> pd.DataFrame:
    """Ensure stage 2 (09:20:00) has an initial state anchor if stage 1 has valid quotes.

    Carries forward the latest valid stage 1 quote (ask_price1 > 0 and bid_price1 > 0)
    as a zero-order hold state anchor at 09:20:00 tagged with ``_qmt_state_anchor = True``.
    This preserves the 09:20:00 baseline and stage 2 duration for TWAP and return
    calculations without inflating event / snapshot counts.
    """
    if quotes.empty or "trade_time" not in quotes.columns:
        return quotes

    first_time = quotes["trade_time"].iloc[0]
    if pd.isna(first_time):
        return quotes
    trade_day = pd.Timestamp(first_time).normalize()
    split_time = trade_day + pd.Timedelta(hours=9, minutes=20)

    # An exact 09:20:00 snapshot already exists
    if quotes["trade_time"].eq(split_time).any():
        return quotes

    stage1 = quotes.loc[quotes["trade_time"].lt(split_time)]
    if stage1.empty:
        return quotes

    valid_mask = stage1["ask_price1"].gt(0) & stage1["bid_price1"].gt(0)
    valid_stage1 = stage1.loc[valid_mask]
    if valid_stage1.empty:
        return quotes
    stage2 = quotes.loc[quotes["trade_time"].ge(split_time)]
    if stage2.empty or stage2["trade_time"].iloc[0] <= split_time:
        return quotes

    last_valid = valid_stage1.iloc[-1]
    anchor = last_valid.to_dict()
    anchor["trade_time"] = split_time
    anchor["_qmt_state_anchor"] = True
    anchor["trade_volume"] = 0.0
    anchor["trade_amount"] = 0.0
    anchor["open_price"] = np.nan
    if "trade_price" in anchor:
        anchor["trade_price"] = np.nan
    if "raw_time" in anchor:
        anchor["raw_time"] = "092000000"

    augmented = pd.concat([quotes, pd.DataFrame([anchor])], ignore_index=True)
    augmented["_qmt_state_anchor"] = augmented["_qmt_state_anchor"].eq(True)
    return augmented.sort_values("trade_time", kind="mergesort").reset_index(drop=True)


@dataclass
class CacheStats:
    hits: int = 0
    rebuilds: int = 0


class AuctionTickCache:
    def __init__(
        self,
        root: Path | None,
        refresh: bool = False,
        chunksize: int = DEFAULT_CHUNK_SIZE,
        validate_source: bool = False,
    ):
        self.root = Path(root) if root is not None else None
        self.refresh = refresh
        self.chunksize = chunksize
        self.validate_source = validate_source
        self.stats = CacheStats()

    def _load(
        self,
        symbol_dir: Path,
        kind: str,
        source_name: str,
        usecols: list[str],
        start_time: str,
        end_time: str,
        inclusive_end: bool,
        transform: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        source = symbol_dir / source_name
        start, end = _date_bounds(symbol_dir, start_time, end_time)
        if self.root is None:
            return _read_csv_chunks(
                source,
                usecols,
                transform,
                start,
                end,
                start_time,
                end_time,
                inclusive_end,
                self.chunksize,
            )

        cache_dir = self.root / symbol_dir.parent.parent.parent.name / symbol_dir.parent.parent.name / symbol_dir.parent.name / symbol_dir.name
        target = cache_dir / f"{kind}.parquet"
        metadata = target.with_suffix(".json")
        # Fast path: If cache exists and not refreshing, directly return cached parquet!
        # Completely bypasses slow external disk stat() and eliminates millions of JSON reads/parses.
        if not self.refresh and target.exists():
            if not self.validate_source:
                try:
                    frame = pd.read_parquet(target)
                    self.stats.hits += 1
                    return frame
                except (OSError, ValueError, TypeError):
                    # Corrupted cache fallback: will rebuild below
                    pass
            elif metadata.exists():
                signature = _source_signature(source)
                try:
                    cached = json.loads(metadata.read_text(encoding="utf-8"))
                    if cached.get("version") == CACHE_VERSION and cached.get("source") == signature:
                        self.stats.hits += 1
                        return pd.read_parquet(target)
                except (OSError, ValueError, TypeError):
                    pass

        signature = _source_signature(source)

        frame = _read_csv_chunks(
            source,
            usecols,
            transform,
            start,
            end,
            start_time,
            end_time,
            inclusive_end,
            self.chunksize,
        )
        _atomic_write_frame(frame, target)
        _atomic_write_json({"version": CACHE_VERSION, "source": signature}, metadata)
        self.stats.rebuilds += 1
        return frame

    def load_quote(
        self, symbol_dir: Path, *, data_source: str = "auto"
    ) -> pd.DataFrame:
        frame = self._load(
            symbol_dir,
            "quote",
            "行情.csv",
            list(QUOTE_COLUMNS),
            "09:15",
            "09:30",
            False,
            _normalize_quote,
        )
        return ensure_stage2_quote_anchor(
            filter_quote_heartbeats(
                frame, data_source=data_source, symbol_path=symbol_dir
            )
        )

    def load_orders(self, symbol_dir: Path) -> pd.DataFrame:
        return self._load(
            symbol_dir,
            "orders",
            "逐笔委托.csv",
            list(ORDER_COLUMNS),
            "09:15",
            "09:25",
            False,
            _normalize_orders,
        )

    def load_transactions(self, symbol_dir: Path) -> pd.DataFrame:
        return self._load(
            symbol_dir,
            "transactions",
            "逐笔成交.csv",
            list(TRANSACTION_COLUMNS),
            "09:15",
            "09:25",
            False,
            _normalize_transactions,
        )

    def load_open_transactions(self, symbol_dir: Path) -> pd.DataFrame:
        """Load the 09:25 opening-match transaction minute separately."""
        return self._load(
            symbol_dir,
            "open_transactions",
            "逐笔成交.csv",
            list(TRANSACTION_COLUMNS),
            "09:25",
            "09:26",
            False,
            _normalize_transactions,
        )

    def load_close_orders(self, symbol_dir: Path) -> pd.DataFrame:
        return self._load(
            symbol_dir,
            "close_orders",
            "逐笔委托.csv",
            list(ORDER_COLUMNS),
            "14:57",
            "15:00",
            False,
            _normalize_orders,
        )

    def load_close_order_history(self, symbol_dir: Path) -> pd.DataFrame:
        """Include earlier orders carried into the closing auction."""
        return self._load(
            symbol_dir, "close_order_history", "逐笔委托.csv", list(ORDER_COLUMNS),
            "09:15", "15:00", False, _normalize_orders,
        )

    def load_close_transactions(self, symbol_dir: Path) -> pd.DataFrame:
        return self._load(
            symbol_dir,
            "close_transactions_before_postclose_v4",
            "逐笔成交.csv",
            list(TRANSACTION_COLUMNS),
            "14:57",
            "15:05",
            False,
            _normalize_transactions,
        )
