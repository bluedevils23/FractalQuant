"""Download mapped index 1-minute bars from RQData into the local parquet schema."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\指数数据\index_1min_rqdata")
DEFAULT_DATE_FROM = date(2026, 3, 19)
OUTPUT_COLUMNS = ("ts_code", "open", "high", "low", "close", "vol", "amount")
PRICE_FIELDS = ("open", "high", "low", "close", "volume", "total_turnover")
LOGGER = logging.getLogger("download_rqdata_index_minutes")


@dataclass(frozen=True)
class IndexMapping:
    output_code: str
    rqdata_code: str
    name: str


INDEX_MAPPINGS = (
    IndexMapping("930914.CSI", "930914.INDX", "中证港股通高股息"),
    IndexMapping("931239.CSI", "931239.INDX", "港股通汽车"),
    IndexMapping("932069.CSI", "932069.INDX", "港股通医疗主题"),
    IndexMapping("930931.CSI", "930931.INDX", "中证港股通50"),
    IndexMapping("930965.CSI", "930965.INDX", "港股通医药C"),
    IndexMapping("931233.CSI", "931233.INDX", "港股通央企红利"),
    IndexMapping("931454.CSI", "931454.INDX", "港股通消费"),
    IndexMapping("931250.CSI", "931250.INDX", "港股通创新药"),
    IndexMapping("931722.CSI", "931722.INDX", "国新港股通央企红利"),
    IndexMapping("H11146.CSI", "H11146.XSHG", "港股通内地金融"),
    IndexMapping("930709.CSI", "930709.INDX", "中证香港证券"),
    IndexMapping("930957.CSI", "930957.INDX", "港股通中国100"),
    IndexMapping("931028.CSI", "931028.INDX", "港股通非银CNY"),
    IndexMapping("H50069.CSI", "H50069.XSHG", "上证港股通"),
    IndexMapping("931637.CSI", "931637.INDX", "港股通互联网"),
    IndexMapping("931573.CSI", "931573.INDX", "港股通科技"),
)


def parse_trade_date(value: str) -> date:
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def month_windows(start_date: date, end_date: date) -> list[tuple[date, date]]:
    if start_date > end_date:
        raise ValueError("date_from must not be after date_to")

    windows: list[tuple[date, date]] = []
    current = start_date
    while current <= end_date:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        window_end = min(end_date, next_month - timedelta(days=1))
        windows.append((current, window_end))
        current = window_end + timedelta(days=1)
    return windows


def normalize_rqdata_frame(raw: pd.DataFrame, output_code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS,
            index=pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([]), pd.DatetimeIndex([])],
                names=["trade_date", "trade_time"],
            ),
        )

    if isinstance(raw.index, pd.MultiIndex):
        if "datetime" not in raw.index.names:
            raise ValueError("RQData response is missing a datetime index level")
        trade_time = pd.to_datetime(raw.index.get_level_values("datetime"))
    elif "datetime" in raw.columns:
        trade_time = pd.to_datetime(raw["datetime"])
    else:
        trade_time = pd.to_datetime(raw.index)

    required = set(PRICE_FIELDS)
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"RQData response is missing fields: {sorted(missing)}")

    frame = pd.DataFrame(
        {
            "ts_code": output_code,
            "open": pd.to_numeric(raw["open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(raw["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(raw["low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
            "vol": pd.to_numeric(raw["volume"], errors="coerce").to_numpy(),
            "amount": pd.to_numeric(
                raw["total_turnover"], errors="coerce"
            ).to_numpy(),
        },
        index=pd.DatetimeIndex(trade_time, name="trade_time"),
    )
    frame["trade_date"] = frame.index.normalize()
    valid_ohlc = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    non_trading_dates = valid_ohlc.groupby(frame["trade_date"]).any()
    frame = frame.loc[frame["trade_date"].isin(non_trading_dates.index[non_trading_dates])]
    return frame.set_index("trade_date", append=True).reorder_levels(
        ["trade_date", "trade_time"]
    )[list(OUTPUT_COLUMNS)]


def validate_minute_frame(frame: pd.DataFrame, output_code: str) -> None:
    if frame.empty:
        raise ValueError(f"No minute data returned for {output_code}")
    if frame.index.names != ["trade_date", "trade_time"]:
        raise ValueError(f"Unexpected index schema for {output_code}: {frame.index.names}")
    if frame.columns.tolist() != list(OUTPUT_COLUMNS):
        raise ValueError(f"Unexpected column schema for {output_code}")
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate minute timestamps for {output_code}")
    if not frame["ts_code"].eq(output_code).all():
        raise ValueError(f"Unexpected ts_code values for {output_code}")

    prices = frame[["open", "high", "low", "close"]]
    if prices.isna().any().any() or not prices.gt(0).all().all():
        raise ValueError(f"Invalid OHLC values for {output_code}")
    if frame[["vol", "amount"]].isna().any().any() or not frame[
        ["vol", "amount"]
    ].ge(0).all().all():
        raise ValueError(f"Invalid volume or amount values for {output_code}")

    rows_per_date = frame.groupby(level="trade_date").size()
    unexpected = rows_per_date.loc[rows_per_date.ne(240)]
    if not unexpected.empty:
        details = ", ".join(
            f"{day:%Y-%m-%d}={rows}" for day, rows in unexpected.items()
        )
        raise ValueError(
            f"Expected 240 rows per trade date for {output_code}: {details}"
        )


def merge_minute_frames(existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, fresh) if frame is not None and not frame.empty]
    if not frames:
        return fresh
    return (
        pd.concat(frames)
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
        .sort_index()
    )


def download_start_date(
    existing: pd.DataFrame | None,
    trading_dates: list[date],
    refresh_trading_days: int,
) -> date:
    if not trading_dates:
        raise ValueError("No trading dates in requested range")
    if refresh_trading_days < 1:
        raise ValueError("refresh_trading_days must be at least 1")
    refresh_start = trading_dates[max(0, len(trading_dates) - refresh_trading_days)]
    if existing is None or existing.empty:
        return trading_dates[0]

    return refresh_start


def fetch_minutes(
    api: object,
    mapping: IndexMapping,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    frames = []
    for window_start, window_end in month_windows(start_date, end_date):
        raw = api.get_price(
            mapping.rqdata_code,
            start_date=window_start.isoformat(),
            end_date=window_end.isoformat(),
            frequency="1m",
            fields=list(PRICE_FIELDS),
            expect_df=True,
        )
        normalized = normalize_rqdata_frame(raw, mapping.output_code)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return normalize_rqdata_frame(pd.DataFrame(), mapping.output_code)
    return merge_minute_frames(None, pd.concat(frames))


def load_existing(path: Path, output_code: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    existing = pd.read_parquet(path)
    validate_minute_frame(existing, output_code)
    return existing


def write_parquet_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def previous_trading_date(api: object) -> date:
    today = pd.Timestamp.now(tz="Asia/Shanghai").date()
    return pd.Timestamp(api.get_previous_trading_date(today)).date()


def is_quota_exceeded(exc: Exception) -> bool:
    return exc.__class__.__name__ == "QuotaExceeded"


def prioritize_missing_mappings(
    mappings: tuple[IndexMapping, ...], output_root: Path
) -> tuple[IndexMapping, ...]:
    return tuple(
        sorted(
            mappings,
            key=lambda mapping: (
                (output_root / f"{mapping.output_code}.parquet").exists(),
                mapping.output_code,
            ),
        )
    )


def download_mapping(
    api: object,
    mapping: IndexMapping,
    output_root: Path,
    date_from: date,
    date_to: date,
    refresh_trading_days: int,
) -> dict[str, object]:
    output_path = output_root / f"{mapping.output_code}.parquet"
    existing = load_existing(output_path, mapping.output_code)
    trading_dates = [
        pd.Timestamp(day).date()
        for day in api.get_trading_dates(date_from.isoformat(), date_to.isoformat())
    ]
    start_date = download_start_date(
        existing, trading_dates, refresh_trading_days
    )
    fresh = fetch_minutes(api, mapping, start_date, date_to)
    merged = merge_minute_frames(existing, fresh)
    validate_minute_frame(merged, mapping.output_code)
    write_parquet_atomic(merged, output_path)
    return {
        "code": mapping.output_code,
        "rqdata_code": mapping.rqdata_code,
        "rows": len(merged),
        "date_from": merged.index.get_level_values("trade_date").min(),
        "date_to": merged.index.get_level_values("trade_date").max(),
        "download_from": start_date,
        "output_path": output_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--date-from", type=parse_trade_date, default=DEFAULT_DATE_FROM)
    parser.add_argument("--date-to", type=parse_trade_date)
    parser.add_argument("--refresh-trading-days", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.refresh_trading_days < 1:
        raise ValueError("--refresh-trading-days must be at least 1")

    import rqdatac

    try:
        rqdatac.init(lazy=False)
    except Exception as exc:
        raise RuntimeError(
            "RQData initialization failed. Configure RQDATAC2_CONF or RQDATAC_CONF."
        ) from exc

    date_to = args.date_to or previous_trading_date(rqdatac)
    if args.date_from > date_to:
        raise ValueError("--date-from must not be after --date-to")

    failures = 0
    for mapping in prioritize_missing_mappings(INDEX_MAPPINGS, args.output_root):
        try:
            result = download_mapping(
                rqdatac,
                mapping,
                args.output_root,
                args.date_from,
                date_to,
                args.refresh_trading_days,
            )
            LOGGER.info(
                "Written %(code)s from %(rqdata_code)s: %(rows)s rows, %(date_from)s to %(date_to)s (requested from %(download_from)s)",
                result,
            )
        except Exception as exc:
            failures += 1
            LOGGER.exception("Failed to download %s", mapping.output_code)
            if is_quota_exceeded(exc):
                LOGGER.error(
                    "RQData quota is exhausted; stopping before requesting the remaining indices."
                )
                break
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
