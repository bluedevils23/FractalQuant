"""Generate research-compatible tail-30-minute ETF factors.

The report's ``211m-240m`` interval is represented by the 30 one-minute
bars from 14:31 through 15:00, inclusive.  The report does not disclose the
full definitions of Bias, residual volatility, or Maxilliq, so those fields
are explicitly marked as proxies in the output names and manifest.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LOGGER = logging.getLogger("generate_tail30_factors")

DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-factors\etf_tail30_factors")
DEFAULT_UNIVERSE_FILE = Path(
    r"D:\workspace\stockdata\etf-factors\etf_universe\non_day_turnover_stock_index_universe.txt"
)
BENCHMARK_SYMBOL = "510300.SH"

RETURN_START = pd.Timedelta(hours=14, minutes=30)
TAIL_START = pd.Timedelta(hours=14, minutes=31)
TAIL_END = pd.Timedelta(hours=15)
TAIL_BAR_COUNT = 30
RESIDUAL_LOOKBACK_DAYS = 60
RESIDUAL_MIN_HISTORY_DAYS = 20

KEY_COLUMNS = ["trade_date", "available_time", "ts_code"]
DIAGNOSTIC_COLUMNS = [
    "tail30_bar_count",
    "tail30_complete",
    "tail30_start_time",
    "tail30_end_time",
    "tail30_total_amount",
    "tail30_zero_amount_bar_count",
]
REFERENCE_COLUMNS = [
    "tail30_first_open",
    "tail30_return_start_close",
    "tail30_last_close",
    "tail30_vwap",
]
FACTOR_COLUMNS = [
    "tail30_return",
    "tail30_bias_proxy",
    "tail30_daily_reverse",
    "tail30_dastd",
    "tail30_residual_volatility_proxy",
    "tail30_daily_volatility",
    "tail30_maxilliq_proxy",
    "tail30_maxilliq_signed_proxy",
    "tail30_maxilliq_w3_proxy",
    "tail30_maxilliq_w3_signed_proxy",
    "tail30_maxilliq_w5_proxy",
    "tail30_maxilliq_w5_signed_proxy",
    "tail30_daily_milliq",
]
OUTPUT_COLUMNS = KEY_COLUMNS + DIAGNOSTIC_COLUMNS + REFERENCE_COLUMNS + FACTOR_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--benchmark-symbol",
        default=BENCHMARK_SYMBOL,
        help="Optional benchmark ETF used for residual-volatility proxy; empty disables it.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def normalize_date_arg(value: str | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_universe_codes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file does not exist: {path}")
    codes: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        code = line.split("#", 1)[0].strip().upper()
        if code:
            codes.append(code)
    return list(dict.fromkeys(codes))


def normalize_minute_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if isinstance(frame.index, pd.MultiIndex) and "trade_time" in frame.index.names:
        index = pd.to_datetime(frame.index.get_level_values("trade_time"))
        frame.index = index
        frame.index.name = "trade_time"
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "trade_time"
    elif "trade_time" in frame.columns:
        frame["trade_time"] = pd.to_datetime(frame["trade_time"])
        frame = frame.set_index("trade_time")
    elif "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index("datetime")
        frame.index.name = "trade_time"
    else:
        raise ValueError("Cannot locate trade_time/datetime")

    frame = frame.rename(columns={"vol": "volume"}).sort_index()
    required = ["open", "close", "volume", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required minute columns: {missing}")
    frame = frame[~frame.index.duplicated(keep="last")].copy()
    for column in required + ["high", "low"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _date_in_range(date_text: str, date_from: str | None, date_to: str | None) -> bool:
    return (date_from is None or date_text >= date_from) and (
        date_to is None or date_text <= date_to
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _maxilliq_window(
    minute_returns: np.ndarray, amount: np.ndarray, window_size: int
) -> tuple[float, float]:
    """Return absolute and signed return/amount at the max-amount window."""
    if window_size < 1 or len(minute_returns) < window_size:
        return np.nan, np.nan
    amount_sums = np.convolve(amount, np.ones(window_size), mode="valid")
    if not np.isfinite(amount_sums).any():
        return np.nan, np.nan
    peak = int(np.nanargmax(np.where(np.isfinite(amount_sums), amount_sums, -np.inf)))
    peak_amount = float(amount_sums[peak])
    if peak_amount <= 0:
        return np.nan, np.nan
    window_return = float(np.nansum(minute_returns[peak : peak + window_size]))
    signed = _safe_ratio(window_return, peak_amount)
    return abs(signed) if np.isfinite(signed) else np.nan, signed


def _tail_index(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day + TAIL_START, day + TAIL_END, freq="min")


def _return_index(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day + RETURN_START, day + TAIL_END, freq="min")


def _build_tail_returns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for day in sorted(pd.DatetimeIndex(frame.index.normalize()).unique()):
        closes = frame.reindex(_return_index(day))["close"]
        if closes.notna().all() and (closes > 0).all():
            result[day.strftime("%Y-%m-%d")] = np.diff(
                np.log(closes.to_numpy(dtype=float))
            )
    return result


def _residual_std(
    asset_returns: np.ndarray,
    benchmark_returns: dict[str, np.ndarray],
    asset_returns_by_date: dict[str, np.ndarray],
    date_text: str,
    lookback_days: int = RESIDUAL_LOOKBACK_DAYS,
    min_history_days: int = RESIDUAL_MIN_HISTORY_DAYS,
    ordered_asset_dates: list[str] | None = None,
) -> float:
    """Estimate current residual volatility using strictly prior tail returns."""
    current_benchmark = benchmark_returns.get(date_text)
    if current_benchmark is None or len(current_benchmark) != len(asset_returns):
        return np.nan

    ordered_asset_dates = ordered_asset_dates or sorted(asset_returns_by_date)
    date_pos = bisect.bisect_left(ordered_asset_dates, date_text)
    prior_candidates = ordered_asset_dates[max(0, date_pos - lookback_days) : date_pos]
    prior_dates = [value for value in prior_candidates if value in benchmark_returns]
    if len(prior_dates) < min_history_days:
        return np.nan

    historical_asset = []
    historical_benchmark = []
    for prior_date in prior_dates:
        asset = asset_returns_by_date[prior_date]
        benchmark = benchmark_returns[prior_date]
        if len(asset) != len(benchmark):
            continue
        valid = np.isfinite(asset) & np.isfinite(benchmark)
        if valid.any():
            historical_asset.append(asset[valid])
            historical_benchmark.append(benchmark[valid])
    if len(historical_asset) < min_history_days:
        return np.nan

    x = np.concatenate(historical_benchmark)
    y = np.concatenate(historical_asset)
    if len(x) < 3 or np.nanvar(x, ddof=1) <= 0:
        return np.nan
    if np.allclose(x, y, rtol=1e-10, atol=1e-14):
        return np.nan
    beta, alpha = np.polyfit(x, y, 1)
    valid_current = np.isfinite(asset_returns) & np.isfinite(current_benchmark)
    if valid_current.sum() < 2:
        return np.nan
    residual = asset_returns[valid_current] - (
        alpha + beta * current_benchmark[valid_current]
    )
    return float(np.nanstd(residual, ddof=1)) if len(residual) >= 2 else np.nan


def calculate_tail30_factors(
    frame: pd.DataFrame,
    ts_code: str,
    next_trade_dates: dict[str, str] | None = None,
    benchmark_returns: dict[str, np.ndarray] | None = None,
    selected_dates: set[str] | None = None,
    asset_returns_by_date: dict[str, np.ndarray] | None = None,
    ordered_asset_dates: list[str] | None = None,
) -> list[dict[str, object]]:
    """Calculate one output row per complete tail-30-minute trading day."""
    frame = normalize_minute_frame(frame)
    dates = sorted(pd.DatetimeIndex(frame.index.normalize()).unique())
    next_trade_dates = next_trade_dates or {
        str(dates[i].date()): str(dates[i + 1].date()) for i in range(len(dates) - 1)
    }
    benchmark_returns = benchmark_returns or {}
    asset_returns_by_date = asset_returns_by_date or _build_tail_returns(frame)
    ordered_asset_dates = ordered_asset_dates or sorted(asset_returns_by_date)

    records: list[dict[str, object]] = []
    for day in dates:
        date_text = day.strftime("%Y-%m-%d")
        if selected_dates is not None and date_text not in selected_dates:
            continue
        window = frame.reindex(_tail_index(day))
        if window["close"].notna().sum() != TAIL_BAR_COUNT or window["open"].notna().sum() != TAIL_BAR_COUNT:
            raise ValueError(f"{ts_code} {date_text}: incomplete tail30 minute bars")
        if window["volume"].isna().any() or window["amount"].isna().any():
            raise ValueError(f"{ts_code} {date_text}: missing volume/amount")
        first_open = float(window["open"].iloc[0])
        last_close = float(window["close"].iloc[-1])
        if first_open <= 0 or last_close <= 0:
            raise ValueError(f"{ts_code} {date_text}: non-positive price")

        volume = window["volume"].to_numpy(dtype=float)
        amount = window["amount"].to_numpy(dtype=float)
        if np.any(volume < 0) or np.any(amount < 0):
            raise ValueError(f"{ts_code} {date_text}: negative volume/amount")
        total_volume = float(np.nansum(volume))
        total_amount = float(np.nansum(amount))
        vwap = _safe_ratio(total_amount, total_volume)
        return_closes = frame.reindex(_return_index(day))["close"]
        if return_closes.isna().any() or (return_closes <= 0).any():
            raise ValueError(f"{ts_code} {date_text}: missing 14:30 close for return boundary")
        return_start_close = float(return_closes.iloc[0])
        minute_returns = np.diff(np.log(return_closes.to_numpy(dtype=float)))
        tail_return = float(last_close / return_start_close - 1.0)
        dastd = float(np.nanstd(minute_returns, ddof=1))
        bias = _safe_ratio(last_close - vwap, vwap)

        amount_positive = amount > 0
        amount_peak = int(np.nanargmax(np.where(np.isfinite(amount), amount, -np.inf)))
        maxilliq_signed = (
            _safe_ratio(float(minute_returns[amount_peak]), float(amount[amount_peak]))
            if amount_positive[amount_peak]
            else np.nan
        )
        maxilliq = abs(maxilliq_signed) if np.isfinite(maxilliq_signed) else np.nan
        maxilliq_w3, maxilliq_w3_signed = _maxilliq_window(minute_returns, amount, 3)
        maxilliq_w5, maxilliq_w5_signed = _maxilliq_window(minute_returns, amount, 5)
        residual = _residual_std(
            minute_returns,
            benchmark_returns,
            asset_returns_by_date,
            date_text,
            ordered_asset_dates=ordered_asset_dates,
        )
        daily_reverse = 0.5 * tail_return + 0.5 * bias
        daily_volatility = 0.5 * dastd + 0.5 * residual
        records.append(
            {
                "trade_date": date_text,
                "available_time": (
                    pd.Timestamp(next_trade_dates[date_text]) + pd.Timedelta(hours=9, minutes=15)
                    if date_text in next_trade_dates
                    else pd.NaT
                ),
                "ts_code": ts_code,
                "tail30_bar_count": TAIL_BAR_COUNT,
                "tail30_complete": True,
                "tail30_start_time": window.index[0],
                "tail30_end_time": window.index[-1],
                "tail30_total_amount": float(np.nansum(amount)),
                "tail30_zero_amount_bar_count": int((amount <= 0).sum()),
                "tail30_first_open": first_open,
                "tail30_return_start_close": return_start_close,
                "tail30_last_close": last_close,
                "tail30_vwap": vwap,
                "tail30_return": tail_return,
                "tail30_bias_proxy": bias,
                "tail30_daily_reverse": daily_reverse,
                "tail30_dastd": dastd,
                "tail30_residual_volatility_proxy": residual,
                "tail30_daily_volatility": daily_volatility,
                "tail30_maxilliq_proxy": maxilliq,
                "tail30_maxilliq_signed_proxy": maxilliq_signed,
                "tail30_maxilliq_w3_proxy": maxilliq_w3,
                "tail30_maxilliq_w3_signed_proxy": maxilliq_w3_signed,
                "tail30_maxilliq_w5_proxy": maxilliq_w5,
                "tail30_maxilliq_w5_signed_proxy": maxilliq_w5_signed,
                "tail30_daily_milliq": maxilliq,
            }
        )
    return records


def _existing_dates(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    values = pd.read_parquet(output_path, columns=["trade_date"])["trade_date"]
    return set(pd.to_datetime(values, errors="coerce").dropna().dt.strftime("%Y-%m-%d"))


def _output_has_current_schema(output_path: Path) -> bool:
    if not output_path.exists():
        return True
    try:
        return set(OUTPUT_COLUMNS).issubset(pq.ParquetFile(output_path).schema_arrow.names)
    except (OSError, ValueError):
        return False


def merge_symbol_output(
    output_path: Path,
    requested: pd.DataFrame,
    overwrite: bool,
    replace_dates: set[str] | None = None,
) -> pd.DataFrame:
    schema_upgrade = False
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        schema_upgrade = not set(OUTPUT_COLUMNS).issubset(existing.columns)
        for column in OUTPUT_COLUMNS:
            if column not in existing:
                existing[column] = np.nan
        existing = existing[OUTPUT_COLUMNS]
    else:
        existing = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if overwrite or schema_upgrade:
        dates_to_replace = (
            set(replace_dates)
            if replace_dates is not None
            else set(requested["trade_date"].astype(str))
        )
        additions = requested
        existing = existing.loc[~existing["trade_date"].astype(str).isin(dates_to_replace)]
    else:
        existing_dates = set(existing["trade_date"].astype(str))
        additions = requested.loc[~requested["trade_date"].astype(str).isin(existing_dates)]
    result = pd.concat([existing, additions], ignore_index=True).reindex(columns=OUTPUT_COLUMNS)
    if result.empty:
        return result
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("available_time", "tail30_start_time", "tail30_end_time"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    return result.sort_values("trade_date", kind="mergesort").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def build_benchmark_returns(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    frame = normalize_minute_frame(pd.read_parquet(path))
    return _build_tail_returns(frame)


def process_symbol(
    symbol: str,
    input_path: Path,
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
    benchmark_returns: dict[str, np.ndarray],
) -> tuple[Path, int, int]:
    output_path = output_root / f"{symbol}.parquet"
    frame = normalize_minute_frame(pd.read_parquet(input_path))
    dates = sorted(pd.DatetimeIndex(frame.index.normalize()).unique())
    next_dates = {
        dates[i].strftime("%Y-%m-%d"): dates[i + 1].strftime("%Y-%m-%d")
        for i in range(len(dates) - 1)
    }
    selected_dates = {day.strftime("%Y-%m-%d") for day in dates if _date_in_range(day.strftime("%Y-%m-%d"), date_from, date_to)}
    output_has_current_schema = _output_has_current_schema(output_path)
    if not overwrite and output_has_current_schema:
        selected_dates -= _existing_dates(output_path)
    records: list[dict[str, object]] = []
    asset_returns_by_date = _build_tail_returns(frame)
    ordered_asset_dates = sorted(asset_returns_by_date)
    for date_text in sorted(selected_dates):
        try:
            single = frame.loc[date_text]
            records.extend(
                calculate_tail30_factors(
                    single,
                    symbol,
                    next_dates,
                    benchmark_returns,
                    selected_dates={date_text},
                    asset_returns_by_date=asset_returns_by_date,
                    ordered_asset_dates=ordered_asset_dates,
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("%s skipped %s: %s", symbol, date_text, exc)
    requested = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    combined = merge_symbol_output(
        output_path,
        requested,
        overwrite,
        replace_dates=selected_dates if not output_has_current_schema else None,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return output_path, len(records), len(selected_dates) - len(records)


def write_manifest(output_root: Path, input_root: Path, universe_file: Path) -> None:
    manifest = {
        "source_report_interval": "211m-240m",
        "bar_window": "14:31:00-15:00:00 inclusive (30 one-minute bars)",
        "return_boundary": "30 close-to-close returns from 14:30 close through 15:00 close",
        "availability": "next available trading day 09:15; last observed day is NaT",
        "units": {
            "volume": "ETF shares",
            "amount": "CNY",
            "tail30_vwap": "CNY per ETF share",
            "maxilliq": "log return per CNY traded",
        },
        "factor_definitions": {
            "tail30_return": "15:00 close / 14:30 close - 1",
            "tail30_bias_proxy": "15:00 close relative to sum(amount) / sum(volume) transaction VWAP",
            "tail30_daily_reverse": "0.5 * tail30_return + 0.5 * tail30_bias_proxy",
            "tail30_dastd": "sample std of 30 close-to-close log returns",
            "tail30_residual_volatility_proxy": "current tail30 residual std using alpha and beta estimated from up to 60 strictly prior trading days; minimum 20 days",
            "tail30_maxilliq_proxy": "absolute return of the maximum-amount minute divided by that minute amount",
            "tail30_maxilliq_signed_proxy": "signed return of the maximum-amount minute divided by that minute amount",
            "tail30_maxilliq_w3_proxy": "absolute three-minute log return divided by the maximum three-minute amount sum",
            "tail30_maxilliq_w3_signed_proxy": "signed three-minute log return divided by the maximum three-minute amount sum",
            "tail30_maxilliq_w5_proxy": "absolute five-minute log return divided by the maximum five-minute amount sum",
            "tail30_maxilliq_w5_signed_proxy": "signed five-minute log return divided by the maximum five-minute amount sum",
            "tail30_daily_volatility": "0.5 * tail30_dastd + 0.5 * tail30_residual_volatility_proxy",
            "tail30_daily_milliq": "tail30_maxilliq_proxy",
        },
        "limitations": "Bias, residual volatility, and Maxilliq are research-compatible proxies because the report does not disclose complete mathematical specifications.",
        "input_root": str(input_root),
        "universe_file": str(universe_file),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_logging()
    date_from = normalize_date_arg(args.date_from)
    date_to = normalize_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    if args.symbols:
        requested = list(dict.fromkeys(code.upper() for code in args.symbols))
    elif args.universe_file is not None:
        requested = load_universe_codes(args.universe_file)
    else:
        requested = []
    if not requested:
        requested = [path.stem for path in sorted(args.input_root.glob("*.parquet"))]
    if args.limit is not None:
        requested = requested[: args.limit]
    files = [(symbol, args.input_root / f"{symbol}.parquet") for symbol in requested]
    missing = [path for _, path in files if not path.exists()]
    if missing:
        LOGGER.warning("Skipping %s universe codes without minute files", len(missing))
    files = [(symbol, path) for symbol, path in files if path.exists()]
    benchmark_path = args.input_root / f"{args.benchmark_symbol}.parquet" if args.benchmark_symbol else None
    benchmark_returns = build_benchmark_returns(benchmark_path)
    write_manifest(args.output_root, args.input_root, args.universe_file)

    jobs = [(symbol, path, args.output_root, date_from, date_to, args.overwrite, benchmark_returns) for symbol, path in files]
    failures: list[tuple[str, str]] = []
    written = 0
    if max(1, args.workers) == 1:
        for job in jobs:
            try:
                output_path, row_count, skipped_dates = process_symbol(*job)
                written += int(row_count > 0)
                LOGGER.info("%s: wrote %s rows, skipped %s dates -> %s", job[0], row_count, skipped_dates, output_path)
            except Exception as exc:  # noqa: BLE001
                failures.append((job[0], str(exc)))
                LOGGER.exception("Failed to process %s", job[0])
    else:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(process_symbol, *job): job[0] for job in jobs}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    output_path, row_count, skipped_dates = future.result()
                    written += int(row_count > 0)
                    LOGGER.info("%s: wrote %s rows, skipped %s dates -> %s", symbol, row_count, skipped_dates, output_path)
                except Exception as exc:  # noqa: BLE001
                    failures.append((symbol, str(exc)))
                    LOGGER.exception("Failed to process %s", symbol)
    LOGGER.info("Completed: %s symbol files with rows, %s failures", written, len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
