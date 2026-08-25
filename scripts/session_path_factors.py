"""Causal intraday session-path factors from minute OHLC data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SESSION_PATH_OUTPUT_COLUMNS = [
    "trade_date",
    "bar_time",
    "available_time",
    "ts_code",
    "intraday_drawdown_from_session_high",
    "intraday_rebound_from_session_low",
    "intraday_return_from_prev_close",
]


def existing_trade_dates(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_parquet(output_path, columns=["trade_date"])
    except (OSError, ValueError, KeyError):
        return set()
    return set(
        pd.to_datetime(existing["trade_date"], errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
    )


def _build_work_frame(minute: pd.DataFrame, minute_path: Path | None = None) -> pd.DataFrame:
    work = minute.reset_index()
    if "trade_date" not in work.columns or "trade_time" not in work.columns:
        location = f": {minute_path}" if minute_path is not None else ""
        raise ValueError(f"Minute file must expose trade_date and trade_time{location}")
    work["trade_date"] = pd.to_datetime(
        work["trade_date"], errors="coerce"
    ).dt.normalize()
    work["bar_time"] = pd.to_datetime(work["trade_time"], errors="coerce")
    for column in ("high", "low", "close"):
        if column not in work.columns:
            raise ValueError(f"Minute file is missing required column: {column}")
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["trade_date", "bar_time"]).sort_values(
        ["trade_date", "bar_time"], kind="mergesort"
    )
    return work.drop_duplicates(["trade_date", "bar_time"], keep="last")


def build_session_path_factor_frame_from_frame(
    minute: pd.DataFrame,
    ts_code: str,
    requested_dates: set[str] | None = None,
    minute_path: Path | None = None,
) -> pd.DataFrame:
    work = _build_work_frame(minute, minute_path)
    daily_close = work.groupby("trade_date", sort=True)["close"].last()
    work["_previous_close"] = work["trade_date"].map(daily_close.shift(1))
    session_groups = work.groupby("trade_date", sort=False)
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
    factor_columns = SESSION_PATH_OUTPUT_COLUMNS[-3:]
    numeric = result[factor_columns].to_numpy(dtype=float)
    infinite_locations = np.argwhere(np.isinf(numeric))
    if len(infinite_locations) > 0:
        details = [
            (
                f"{result.iloc[row_index]['trade_date']} "
                f"{result.iloc[row_index]['bar_time']} "
                f"{factor_columns[column_index]}"
            )
            for row_index, column_index in infinite_locations[:10]
        ]
        remaining = len(infinite_locations) - len(details)
        suffix = f"; and {remaining} more" if remaining > 0 else ""
        raise ValueError(
            f"Infinite session path factor for {ts_code}: "
            + "; ".join(details)
            + suffix
        )
    return result


def build_session_path_factor_frame(
    minute_path: Path,
    ts_code: str,
    requested_dates: set[str] | None = None,
) -> pd.DataFrame:
    if not minute_path.exists():
        raise FileNotFoundError(f"Minute file does not exist: {minute_path}")
    minute = pd.read_parquet(minute_path, columns=["high", "low", "close"])
    return build_session_path_factor_frame_from_frame(
        minute, ts_code, requested_dates, minute_path
    )


def merge_session_path_output(
    output_path: Path,
    requested: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    if output_path.exists():
        existing = pd.read_parquet(output_path).reindex(
            columns=SESSION_PATH_OUTPUT_COLUMNS
        )
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
    return combined.sort_values(
        ["trade_date", "bar_time"], kind="mergesort"
    ).reset_index(drop=True)[SESSION_PATH_OUTPUT_COLUMNS]


def process_session_path_only(
    ts_code: str,
    minute_path: Path,
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
) -> tuple[Path, int]:
    output_path = output_root / f"{ts_code}.parquet"
    existing_dates = existing_trade_dates(output_path)
    if not overwrite and date_from is not None and date_to is not None:
        expected_dates = set(
            pd.date_range(date_from, date_to, freq="B").strftime("%Y-%m-%d")
        )
        if expected_dates and expected_dates.issubset(existing_dates):
            return output_path, 0

    requested = build_session_path_factor_frame(minute_path, ts_code)
    if date_from is not None:
        requested = requested.loc[
            requested["trade_date"].ge(pd.Timestamp(date_from).strftime("%Y-%m-%d"))
        ]
    if date_to is not None:
        requested = requested.loc[
            requested["trade_date"].le(pd.Timestamp(date_to).strftime("%Y-%m-%d"))
        ]

    combined = merge_session_path_output(output_path, requested, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return output_path, len(requested)
