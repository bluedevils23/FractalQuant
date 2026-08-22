"""Generate the seven pre-auction daily-context factors independently.

The factors in this file are calculated from the latest completed daily bar
before each target trade date.  They are deliberately kept out of the
opening-auction parquet schema so auction and daily-context features can be
selected independently in downstream research.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_auction_factors import build_historical_context  # noqa: E402


LOGGER = logging.getLogger("generate_previous_day_factors")

DEFAULT_STOCK_DAILY_PATH = Path(
    r"D:\workspace\stockdata\stock-data\行情数据\stock_daily.parquet"
)
DEFAULT_ETF_DAILY_PATH = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_STOCK_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\stock-factors\stock_daily_context_factors"
)
DEFAULT_ETF_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-factors\etf_daily_context_factors"
)
DEFAULT_BENCHMARK_TS_CODE = "510300.SH"
DAILY_FACTOR_COLUMNS = [
    "prevday_intraday_drawdown_from_session_high",
    "prevday_intraday_rebound_from_session_low",
    "prevday_intraday_return_from_prev_close",
    "prev_2d_return_rank_cs",
    "prev_20d_return_rank_cs",
    "market_above_ma20_prevclose",
    "market_momentum_2d_prevclose",
]
OUTPUT_COLUMNS = ["trade_date", "available_time", "source_trade_date", "ts_code"] + DAILY_FACTOR_COLUMNS
SYMBOL_RE = re.compile(r"^\d{6}(?:\.[A-Z]{2})?$", re.IGNORECASE)


def _read_symbols(path: Path | None, symbols: list[str] | None) -> set[str] | None:
    values = list(symbols or [])
    if path is not None:
        values.extend(path.read_text(encoding="utf-8").splitlines())
    cleaned = {value.strip().upper() for value in values if value.strip()}
    invalid = sorted(value for value in cleaned if not SYMBOL_RE.fullmatch(value))
    if invalid:
        raise ValueError(f"Invalid symbols: {invalid[:5]}")
    return cleaned or None


def _filter_codes(frame: pd.DataFrame, requested: set[str] | None) -> set[str]:
    available = set(frame["ts_code"].astype(str).str.upper().unique())
    if requested is None:
        return available
    prefixes = {value.split(".", 1)[0] for value in requested}
    return {
        code
        for code in available
        if code in requested or code.split(".", 1)[0] in prefixes
    }


def calculate_previous_day_factors(
    daily_path: Path,
    requested_codes: set[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    benchmark_ts_code: str = DEFAULT_BENCHMARK_TS_CODE,
) -> pd.DataFrame:
    """Return one row per available symbol and target date.

    ``build_historical_context`` performs the same prior-session mapping and
    full-universe cross-sectional ranking used by the former auction path.
    Keeping that mapping shared prevents the independent output from silently
    changing the historical semantics.
    """
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily file does not exist: {daily_path}")
    daily = pd.read_parquet(daily_path)
    if isinstance(daily.index, pd.MultiIndex) or {
        "trade_date",
        "ts_code",
    }.difference(daily.columns):
        daily = daily.reset_index()
    if {"trade_date", "ts_code"}.difference(daily.columns):
        raise ValueError(f"Daily file must expose trade_date and ts_code: {daily_path}")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.normalize()
    daily["ts_code"] = daily["ts_code"].astype(str).str.upper()
    dates = sorted(daily["trade_date"].dropna().unique())
    if date_from is not None:
        dates = [date for date in dates if date >= pd.Timestamp(date_from).normalize()]
    if date_to is not None:
        dates = [date for date in dates if date <= pd.Timestamp(date_to).normalize()]
    target_dates = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates]
    if not target_dates:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    requested = requested_codes
    all_codes = set(daily["ts_code"])
    benchmark = benchmark_ts_code.upper()
    benchmark_present = benchmark in all_codes
    output_codes = _filter_codes(daily, requested)
    context_codes = set(output_codes)
    if benchmark_present:
        context_codes.add(benchmark)
    contexts = build_historical_context(
        daily_path,
        target_dates,
        context_codes,
        include_daily_factor_fields=True,
    )
    rows: list[pd.DataFrame] = []
    benchmark_context = contexts.get(benchmark, pd.DataFrame())
    benchmark_by_date = (
        benchmark_context.set_index("trade_date")
        if not benchmark_context.empty
        else pd.DataFrame()
    )
    for code, context in contexts.items():
        if code not in output_codes:
            continue
        factor_frame = context[["trade_date", "source_trade_date"]].copy()
        factor_frame["ts_code"] = code
        for column in DAILY_FACTOR_COLUMNS[:5]:
            factor_frame[column] = context[column]
        factor_frame["market_above_ma20_prevclose"] = np.nan
        factor_frame["market_momentum_2d_prevclose"] = np.nan
        if not benchmark_by_date.empty:
            factor_frame["market_above_ma20_prevclose"] = factor_frame["trade_date"].map(
                benchmark_by_date["_market_above_ma20"]
            )
            factor_frame["market_momentum_2d_prevclose"] = factor_frame["trade_date"].map(
                benchmark_by_date["_prev_2d_return"]
            )
        factor_frame["available_time"] = pd.to_datetime(
            factor_frame["trade_date"] + " 09:15:00"
        )
        factor_frame = factor_frame.rename(columns={"trade_date": "target_trade_date"})
        factor_frame["trade_date"] = factor_frame.pop("target_trade_date")
        factor_frame["source_trade_date"] = pd.to_datetime(
            factor_frame["source_trade_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        rows.append(factor_frame[OUTPUT_COLUMNS])
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["ts_code", "trade_date"], kind="mergesort"
    ).reset_index(drop=True)


def _merge_output(path: Path, requested: pd.DataFrame, overwrite: bool) -> None:
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    existing = existing.reindex(columns=OUTPUT_COLUMNS)
    dates = set(requested["trade_date"].astype(str))
    if overwrite:
        existing = existing.loc[~existing["trade_date"].astype(str).isin(dates)]
        combined = pd.concat([existing, requested], ignore_index=True)
    else:
        existing_dates = set(existing["trade_date"].astype(str))
        combined = pd.concat(
            [existing, requested.loc[~requested["trade_date"].astype(str).isin(existing_dates)]],
            ignore_index=True,
        )
    combined = combined.drop_duplicates(["trade_date", "ts_code"], keep="last")
    combined = combined.sort_values("trade_date", kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)


def write_outputs(frame: pd.DataFrame, output_root: Path, overwrite: bool) -> int:
    written = 0
    for code, group in frame.groupby("ts_code", sort=True):
        _merge_output(output_root / f"{code}.parquet", group, overwrite)
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-type", choices=("stock", "etf", "both"), default="both")
    parser.add_argument("--stock-daily-path", type=Path, default=DEFAULT_STOCK_DAILY_PATH)
    parser.add_argument("--etf-daily-path", type=Path, default=DEFAULT_ETF_DAILY_PATH)
    parser.add_argument("--stock-output-root", type=Path, default=DEFAULT_STOCK_OUTPUT_ROOT)
    parser.add_argument("--etf-output-root", type=Path, default=DEFAULT_ETF_OUTPUT_ROOT)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--benchmark-ts-code", default=DEFAULT_BENCHMARK_TS_CODE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    requested = _read_symbols(args.symbols_file, args.symbols)
    jobs = (
        [("stock", args.stock_daily_path, args.stock_output_root)]
        if args.asset_type == "stock"
        else [("etf", args.etf_daily_path, args.etf_output_root)]
        if args.asset_type == "etf"
        else [
            ("stock", args.stock_daily_path, args.stock_output_root),
            ("etf", args.etf_daily_path, args.etf_output_root),
        ]
    )
    for asset_type, daily_path, output_root in jobs:
        frame = calculate_previous_day_factors(
            daily_path,
            requested_codes=requested,
            date_from=args.date_from,
            date_to=args.date_to,
            benchmark_ts_code=args.benchmark_ts_code,
        )
        count = write_outputs(frame, output_root, args.overwrite)
        LOGGER.info(
            "%s: wrote %s symbols and %s rows to %s",
            asset_type,
            count,
            len(frame),
            output_root,
        )


if __name__ == "__main__":
    main()
