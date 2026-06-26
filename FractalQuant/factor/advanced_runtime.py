from __future__ import annotations

import inspect
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import advanced as advanced_module
from .base import BaseFactor


LOGGER = logging.getLogger(__name__)
POSITIVE_PRICE_COLUMNS = ("open", "high", "low", "close")


def normalize_symbol_id(value: str) -> str:
    symbol = str(value).strip()
    if symbol.lower().endswith(".parquet"):
        symbol = symbol[:-8]
    return symbol


def normalize_trade_date_arg(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 8:
        raise ValueError(f"Invalid trade date: {value}")
    return digits


def read_symbol_list_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file does not exist: {path}")

    symbols: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            symbols.append(normalize_symbol_id(line))

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)

    if not deduped:
        raise ValueError(f"No symbols found in file: {path}")

    return deduped


def normalize_minute_frame(raw_df: pd.DataFrame, ts_code_hint: str) -> pd.DataFrame:
    df = raw_df.copy()

    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename(columns={"volume": "vol"})

    if "ts_code" not in df.columns:
        df["ts_code"] = ts_code_hint
    else:
        df["ts_code"] = df["ts_code"].astype(str)

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name in {"trade_date", "trade_time"}:
        df = df.reset_index()

    for column in ("trade_date", "trade_time"):
        if column in df.columns:
            df[column] = df[column].astype(str)

    numeric_columns = [
        column
        for column in ("open", "high", "low", "close", "vol", "amount", "adj_factor")
        if column in df.columns
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    if isinstance(df.index, pd.MultiIndex):
        df = df.sort_index()
    elif "trade_date" in df.columns and "trade_time" in df.columns:
        df = df.sort_values(["trade_date", "trade_time"], kind="mergesort")
    else:
        df = df.sort_index()

    return df


def filter_raw_frame_by_date(
    df: pd.DataFrame, date_from: str | None, date_to: str | None
) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return df

    if isinstance(df.index, pd.MultiIndex) and "trade_date" in df.index.names:
        trade_dates = pd.Index(df.index.get_level_values("trade_date"))
    elif "trade_date" in df.columns:
        trade_dates = pd.Index(df["trade_date"])
    else:
        return df

    normalized_dates = (
        trade_dates.astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
    )
    mask = pd.Series(True, index=df.index)
    if date_from is not None:
        mask &= normalized_dates >= date_from
    if date_to is not None:
        mask &= normalized_dates <= date_to
    return df.loc[mask].copy()


def prepare_factor_input(df: pd.DataFrame) -> pd.DataFrame:
    factor_input = df.copy()
    for column in POSITIVE_PRICE_COLUMNS:
        if column in factor_input.columns:
            factor_input.loc[factor_input[column] <= 0, column] = np.nan
    return factor_input


@lru_cache(maxsize=2)
def build_advanced_factors(exclude_future_returns: bool = False) -> tuple[BaseFactor, ...]:
    factors: list[BaseFactor] = []
    for name, obj in advanced_module.__dict__.items():
        if not inspect.isclass(obj):
            continue
        if obj is BaseFactor or not issubclass(obj, BaseFactor):
            continue
        if obj.__module__ != advanced_module.__name__:
            continue
        if exclude_future_returns and name == "FutureReturnsFactor":
            continue
        try:
            instance = obj()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping factor %s: %s", name, exc)
            continue
        if exclude_future_returns and getattr(instance, "name", None) == "future_returns":
            continue
        factors.append(instance)
    if not factors:
        raise RuntimeError("No advanced factors could be constructed.")
    return tuple(factors)


def _calculate_factors_for_group(
    factor_input: pd.DataFrame, exclude_future_returns: bool
) -> pd.DataFrame:
    factor_series: dict[str, pd.Series] = {}

    for factor in build_advanced_factors(exclude_future_returns):
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                values = factor.calculate(factor_input)
            if isinstance(values, pd.Series):
                series = values.reindex(factor_input.index)
            else:
                series = pd.Series(values, index=factor_input.index)
            series = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            factor_series[factor.name] = series
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Factor %s failed: %s", factor.name, exc)
            factor_series[factor.name] = pd.Series(np.nan, index=factor_input.index)

    return pd.DataFrame(factor_series, index=factor_input.index)


def calculate_factor_frame(
    df: pd.DataFrame,
    *,
    day_workers: int = 1,
    exclude_future_returns: bool = False,
) -> pd.DataFrame:
    factor_input = prepare_factor_input(df)

    if "trade_date" in factor_input.columns:
        trade_days = (
            factor_input["trade_date"]
            .astype(str)
            .str.replace(r"\D", "", regex=True)
            .str.slice(0, 8)
        )
        grouped_frames = [
            group for _, group in factor_input.groupby(trade_days, sort=False)
        ]
        if len(grouped_frames) > 1:
            if day_workers > 1:
                with ProcessPoolExecutor(max_workers=day_workers) as executor:
                    futures = [
                        executor.submit(
                            _calculate_factors_for_group,
                            group,
                            exclude_future_returns,
                        )
                        for group in grouped_frames
                    ]
                    per_day_frames = [future.result() for future in futures]
            else:
                per_day_frames = [
                    _calculate_factors_for_group(group, exclude_future_returns)
                    for group in grouped_frames
                ]
            factor_df = pd.concat(per_day_frames, axis=0)
            return factor_df.reindex(factor_input.index)

    return _calculate_factors_for_group(factor_input, exclude_future_returns)


def extract_trade_dates(df: pd.DataFrame) -> list[str]:
    if isinstance(df.index, pd.MultiIndex) and "trade_date" in df.index.names:
        raw = pd.Index(df.index.get_level_values("trade_date"))
    elif "trade_date" in df.columns:
        raw = pd.Index(df["trade_date"])
    else:
        return []

    trade_dates = (
        raw.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 8)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(trade_dates)


def tick_path_for_date(tick_root: Path, trade_date: str, ts_code: str) -> Path:
    return tick_root / trade_date[:6] / trade_date / ts_code


def validate_tick_coverage(
    tick_root: Path, trade_dates: list[str], ts_code: str
) -> dict[str, object]:
    if not tick_root.exists():
        return {
            "enabled": False,
            "total": len(trade_dates),
            "present": 0,
            "missing": trade_dates[:10],
            "missing_count": len(trade_dates),
        }

    present = 0
    missing: list[str] = []
    for trade_date in trade_dates:
        if tick_path_for_date(tick_root, trade_date, ts_code).exists():
            present += 1
        else:
            missing.append(trade_date)

    return {
        "enabled": True,
        "total": len(trade_dates),
        "present": present,
        "missing": missing[:10],
        "missing_count": len(missing),
    }


def build_output_frame(df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([df, factor_df], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def process_symbol_file(
    input_path: Path,
    output_root: Path,
    tick_root: Path,
    date_from: str | None,
    date_to: str | None,
    day_workers: int,
    skip_tick_check: bool,
    overwrite: bool,
    *,
    exclude_future_returns: bool = False,
) -> dict[str, object]:
    ts_code = normalize_symbol_id(input_path.name)
    output_path = output_root / f"{ts_code}.parquet"
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "ts_code": ts_code,
            "output_path": output_path,
        }

    raw_df = pd.read_parquet(input_path)
    raw_df = filter_raw_frame_by_date(raw_df, date_from, date_to)
    minute_df = normalize_minute_frame(raw_df, ts_code)
    if minute_df.empty:
        return {
            "status": "empty",
            "ts_code": ts_code,
            "output_path": output_path,
        }

    factor_df = calculate_factor_frame(
        minute_df,
        day_workers=max(1, int(day_workers)),
        exclude_future_returns=exclude_future_returns,
    )
    result_df = build_output_frame(minute_df, factor_df)

    trade_dates = extract_trade_dates(minute_df)
    coverage = {
        "enabled": False,
        "total": len(trade_dates),
        "present": 0,
        "missing": [],
        "missing_count": 0,
    }
    if not skip_tick_check:
        coverage = validate_tick_coverage(tick_root, trade_dates, ts_code)

    output_root.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path)

    return {
        "status": "written",
        "ts_code": ts_code,
        "output_path": output_path,
        "rows": len(result_df),
        "cols": len(result_df.columns),
        "trade_dates": len(trade_dates),
        "tick_enabled": coverage["enabled"],
        "tick_total": coverage["total"],
        "tick_present": coverage["present"],
        "tick_missing_count": coverage["missing_count"],
        "tick_missing_examples": coverage["missing"],
    }


def log_result(logger: logging.Logger, result: dict[str, object]) -> None:
    ts_code = result["ts_code"]
    status = result["status"]
    output_path = result["output_path"]

    if status == "skipped":
        logger.info("Skipping existing output: %s", output_path)
        return
    if status == "empty":
        logger.info(
            "Skipping %s because no rows matched the requested date range", ts_code
        )
        return

    logger.info(
        "Wrote %s rows and %s columns for %s to %s",
        result["rows"],
        result["cols"],
        ts_code,
        output_path,
    )

    if result.get("tick_enabled"):
        total = int(result.get("tick_total", 0))
        present = int(result.get("tick_present", 0))
        missing = int(result.get("tick_missing_count", 0))
        logger.info(
            "Tick coverage for %s: %s/%s days present, %s missing",
            ts_code,
            present,
            total,
            missing,
        )
        examples = result.get("tick_missing_examples", [])
        if examples:
            logger.info("Missing tick days for %s: %s", ts_code, ", ".join(examples))
    else:
        logger.info("Tick coverage validation skipped for %s", ts_code)
