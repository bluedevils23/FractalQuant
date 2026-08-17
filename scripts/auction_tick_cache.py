"""Read-through cache for opening-auction tick slices."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
        trade_time = parse_trade_time(chunk["自然日"], chunk["时间"])
        if start is None or end is None:
            minutes = trade_time.dt.hour * 60 + trade_time.dt.minute
            seconds = trade_time.dt.second + trade_time.dt.microsecond / 1_000_000
            elapsed = minutes + seconds / 60.0
            start_minutes = int(start_time[:2]) * 60 + int(start_time[3:])
            end_minutes = int(end_time[:2]) * 60 + int(end_time[3:])
            before_end = elapsed.le(end_minutes) if inclusive_end else elapsed.lt(end_minutes)
            mask = elapsed.ge(start_minutes) & before_end
        else:
            before_end = trade_time.le(end) if inclusive_end else trade_time.lt(end)
            mask = trade_time.ge(start) & before_end
        if not mask.any():
            continue
        selected = chunk.loc[mask].copy()
        selected["trade_time"] = trade_time.loc[mask]
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


@dataclass
class CacheStats:
    hits: int = 0
    rebuilds: int = 0


class AuctionTickCache:
    def __init__(self, root: Path | None, refresh: bool = False, chunksize: int = DEFAULT_CHUNK_SIZE):
        self.root = Path(root) if root is not None else None
        self.refresh = refresh
        self.chunksize = chunksize
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
        signature = _source_signature(source)
        if not self.refresh and target.exists() and metadata.exists():
            try:
                cached = json.loads(metadata.read_text(encoding="utf-8"))
                if cached.get("version") == CACHE_VERSION and cached.get("source") == signature:
                    self.stats.hits += 1
                    return pd.read_parquet(target)
            except (OSError, ValueError, TypeError):
                pass

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

    def load_quote(self, symbol_dir: Path) -> pd.DataFrame:
        return self._load(
            symbol_dir,
            "quote",
            "行情.csv",
            list(QUOTE_COLUMNS),
            "09:15",
            "09:30",
            False,
            _normalize_quote,
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

    def load_close_transactions(self, symbol_dir: Path) -> pd.DataFrame:
        return self._load(
            symbol_dir,
            "close_transactions",
            "逐笔成交.csv",
            list(TRANSACTION_COLUMNS),
            "14:57",
            "15:00",
            True,
            _normalize_transactions,
        )
