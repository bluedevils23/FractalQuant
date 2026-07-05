from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_volume_weighted_price_low_turnover_backtest import (  # noqa: E402
    annualization,
    build_schedule,
    compute_buffered_weights,
    compute_weights,
    configure_logging,
    plot_equity_curves,
    save_table,
    summarize_returns,
)


LOGGER = logging.getLogger("run_stock_advanced_factor_low_turnover_backtest")

DEFAULT_FACTOR_DIR = Path(r"D:\workspace\stockdata\a-share-data\stock_advanced_factors")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stock-alphalens-reloaded\analysis_outputs\stock_advanced_factors_low_turnover_2025_2026")
DEFAULT_START_DATE = "2025-06-13"
DEFAULT_END_DATE = "2026-06-12"
DEFAULT_FACTORS = (
    "bifurcation_parameter",
    "bifurcation_diagram",
    "recurrence_plot",
    "kolmogorov_entropy",
    "recurrence_rate",
)
DEFAULT_HORIZONS = ("1m", "5m", "10m", "eod")
DEFAULT_LONG_SHORT_PCT = 0.2
DEFAULT_MIN_NAMES = 10
DEFAULT_COST_RATE = 0.0005
DEFAULT_ENTER_PCT = 0.05
DEFAULT_HOLD_PCT = 0.15
DEFAULT_EOD_SIGNAL_TIME = "14:50:00"
RETURN_PERIODS = {"1m": 1, "5m": 5, "10m": 10}
EXPECTED_MINUTES_PER_DAY = 241
EXPECTED_SESSION_CLOCKS = {
    0: "09:30",
    120: "11:30",
    121: "13:01",
    240: "15:00",
}
EXCLUDED_COLUMNS = {
    "ts_code",
    "trade_date",
    "trade_time",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "volume",
    "amount",
    "adj_factor",
    "returns",
    "log_returns",
    "future_returns",
}
OPTIONAL_COLUMNS = {"future_returns"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-turnover stock advanced factor backtests.")
    parser.add_argument("--factor-dir", type=Path, default=DEFAULT_FACTOR_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--assets", nargs="+", default=None)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--factors", nargs="*", default=list(DEFAULT_FACTORS))
    parser.add_argument("--horizons", nargs="*", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--long-short-pct", type=float, default=DEFAULT_LONG_SHORT_PCT)
    parser.add_argument("--min-names", type=int, default=DEFAULT_MIN_NAMES)
    parser.add_argument("--cost-rate", type=float, default=DEFAULT_COST_RATE)
    parser.add_argument("--enter-pct", type=float, default=DEFAULT_ENTER_PCT)
    parser.add_argument("--hold-pct", type=float, default=DEFAULT_HOLD_PCT)
    parser.add_argument("--eod-signal-time", type=str, default=DEFAULT_EOD_SIGNAL_TIME)
    return parser.parse_args()


def discover_assets(factor_dir: Path, requested_assets: list[str] | None) -> list[str]:
    if requested_assets is not None:
        assets = requested_assets
    else:
        assets = sorted(path.stem for path in factor_dir.glob("*.parquet"))
    if not assets:
        raise ValueError(f"No stock parquet files found in {factor_dir}")
    return assets


def discover_factor_columns(factor_dir: Path, assets: list[str]) -> list[str]:
    base_columns: list[str] | None = None
    base_column_set: set[str] | None = None

    for asset in assets:
        path = factor_dir / f"{asset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing factor parquet: {path}")

        columns = [name for name in pq.read_schema(path).names if name != "__index_level_0__"]
        if base_columns is None:
            base_columns = columns
            base_column_set = set(columns) - OPTIONAL_COLUMNS
            continue

        if (set(columns) - OPTIONAL_COLUMNS) != base_column_set:
            raise ValueError(
                f"Column schema mismatch for {path.name}: expected {base_columns}, got {columns}"
            )

    if base_columns is None:
        raise ValueError("No factor parquet files were inspected")

    factor_columns = [column for column in base_columns if column not in EXCLUDED_COLUMNS]
    if not factor_columns:
        raise ValueError("No candidate factor columns found after exclusions")
    return factor_columns


def discover_data_bounds(factor_dir: Path, assets: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None

    for asset in assets:
        path = factor_dir / f"{asset}.parquet"
        frame = pd.read_parquet(path, columns=["trade_time"])
        timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
        asset_min = timestamps.min()
        asset_max = timestamps.max()
        if pd.isna(asset_min) or pd.isna(asset_max):
            continue
        min_ts = asset_min if min_ts is None else min(min_ts, asset_min)
        max_ts = asset_max if max_ts is None else max(max_ts, asset_max)

    if min_ts is None or max_ts is None:
        raise ValueError("Could not resolve any trade_time bounds from the selected assets")
    return min_ts, max_ts


def resolve_analysis_window(
    factor_dir: Path,
    assets: list[str],
    start_date: str | None,
    end_date: str | None,
    lookback_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    data_min, data_max = discover_data_bounds(factor_dir, assets)

    if end_date is None:
        end_ts = data_max.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    if start_date is None:
        if lookback_days < 1:
            raise ValueError("lookback-days must be >= 1")
        start_ts = end_ts.normalize() - pd.Timedelta(days=lookback_days - 1)
    else:
        start_ts = pd.Timestamp(start_date)

    if start_ts > end_ts:
        raise ValueError("Resolved start date is after end date")
    if end_ts < data_min or start_ts > data_max:
        raise ValueError("Resolved analysis window does not overlap the selected stock data range")
    return start_ts, end_ts


def load_selected_columns(
    factor_dir: Path,
    assets: list[str],
    value_columns: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for asset in assets:
        path = factor_dir / f"{asset}.parquet"
        schema_columns = [name for name in pq.read_schema(path).names if name != "__index_level_0__"]
        parquet_columns = [column for column in ["ts_code", "trade_time", *value_columns] if column in schema_columns]
        frame = pd.read_parquet(path, columns=parquet_columns)
        if "ts_code" not in frame.columns:
            frame["ts_code"] = asset
        if "trade_time" not in frame.columns:
            frame = frame.reset_index()
        missing = [column for column in value_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        frame["datetime"] = pd.to_datetime(frame["trade_time"], errors="coerce")
        frame = frame.loc[frame["datetime"].between(start_ts, end_ts)]
        if frame.empty:
            continue
        frame["asset"] = frame["ts_code"].fillna(asset).astype("string")
        for column in value_columns:
            frame[column] = frame[column].astype("float32")
        frames.append(frame[["datetime", "asset", *value_columns]])

    if not frames:
        raise ValueError("No rows were loaded for the requested assets and time window")

    panel = pd.concat(frames, ignore_index=True)
    panel["asset"] = panel["asset"].astype("category")
    panel = panel.sort_values(["asset", "datetime"]).reset_index(drop=True)
    return panel


def build_intraday_calendar(datetimes: pd.Series) -> tuple[pd.DataFrame, int]:
    calendar = (
        pd.Series(pd.to_datetime(datetimes), name="datetime")
        .drop_duplicates()
        .sort_values()
        .to_frame()
    )
    calendar["trade_date"] = calendar["datetime"].dt.normalize()
    minute_counts = calendar.groupby("trade_date", observed=True).size()
    if minute_counts.nunique() != 1:
        raise ValueError(
            "Expected a fixed number of trading minutes per day, found "
            f"{sorted(minute_counts.unique().tolist())}"
        )
    if int(minute_counts.iloc[0]) != EXPECTED_MINUTES_PER_DAY:
        raise ValueError(
            f"Expected {EXPECTED_MINUTES_PER_DAY} trading minutes per day, found {int(minute_counts.iloc[0])}"
        )
    calendar["session_pos"] = calendar.groupby("trade_date", observed=True).cumcount().astype("int16")
    calendar["clock"] = calendar["datetime"].dt.strftime("%H:%M")
    for session_pos, expected_clock in EXPECTED_SESSION_CLOCKS.items():
        actual = calendar.loc[calendar["session_pos"].eq(session_pos), "clock"]
        if not actual.eq(expected_clock).all():
            raise ValueError(f"Expected session position {session_pos} to map to {expected_clock}")
    last_session_pos = EXPECTED_MINUTES_PER_DAY - 1
    calendar["is_eod"] = calendar["session_pos"].eq(last_session_pos)
    return calendar.drop(columns="clock"), last_session_pos


def attach_calendar(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    merged = panel.merge(calendar, on="datetime", how="left", copy=False)
    if merged["session_pos"].isna().any():
        raise ValueError("Calendar merge left missing session positions")
    return merged.sort_values(["asset", "datetime"]).reset_index(drop=True)


def compute_same_day_returns(panel: pd.DataFrame, last_session_pos: int) -> pd.DataFrame:
    asset_groups = panel.groupby("asset", observed=True)
    for label, period in RETURN_PERIODS.items():
        future_close = asset_groups["close"].shift(-period)
        future_datetime = asset_groups["datetime"].shift(-period)
        future_trade_date = asset_groups["trade_date"].shift(-period)
        expected_datetime = panel["datetime"] + pd.to_timedelta(period, unit="m")
        valid = future_trade_date.eq(panel["trade_date"]) & future_datetime.eq(expected_datetime)
        returns = (future_close / panel["close"]) - 1.0
        panel[label] = returns.where(valid).astype("float32")

    day_end_close = panel.groupby(["trade_date", "asset"], observed=True)["close"].transform("last")
    day_end_session = panel.groupby(["trade_date", "asset"], observed=True)["session_pos"].transform("last")
    valid_eod = day_end_session.eq(last_session_pos) & panel["session_pos"].lt(last_session_pos)
    eod_returns = (day_end_close / panel["close"]) - 1.0
    panel["eod"] = eod_returns.where(valid_eod).astype("float32")
    return panel


def factor_wide(panel: pd.DataFrame, factor: str, columns: pd.Index, index: pd.Index) -> pd.DataFrame:
    frame = panel.pivot(index="datetime", columns="asset", values=factor)
    return frame.reindex(index=index, columns=columns)


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def build_combo_factor(factor_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zframes = [cross_sectional_zscore(frame) for frame in factor_frames.values()]
    return sum(zframes) / len(zframes)


def build_summary_row(
    strategy: str,
    mode: str,
    horizon: str,
    cost_rate: float,
    enter_pct: float,
    hold_pct: float,
    summary: object,
    detail: pd.DataFrame,
    factor: str,
) -> dict[str, float | int | str]:
    avg_gross_return = float(detail["gross_return"].mean()) if not detail.empty else np.nan
    avg_turnover = float(detail["turnover"].mean()) if not detail.empty else np.nan
    breakeven_cost_rate = avg_gross_return / avg_turnover if detail.shape[0] and avg_turnover > 0 else np.nan
    return {
        "strategy": strategy,
        "mode": mode,
        "horizon": horizon,
        "cost_rate": cost_rate,
        "enter_pct": enter_pct if mode == "buffered" else np.nan,
        "hold_pct": hold_pct if mode == "buffered" else np.nan,
        **summary.__dict__,
        "avg_gross_return": avg_gross_return,
        "avg_net_return": float(detail["net_return"].mean()) if not detail.empty else np.nan,
        "gross_win_rate": float((detail["gross_return"] > 0).mean()) if not detail.empty else np.nan,
        "avg_position_count": float(detail["position_count"].mean()) if not detail.empty else np.nan,
        "avg_valid_count": float(detail["valid_count"].mean()) if not detail.empty else np.nan,
        "breakeven_cost_rate": breakeven_cost_rate,
        "breakeven_cost_bps": breakeven_cost_rate * 10000.0 if pd.notna(breakeven_cost_rate) else np.nan,
    }


def run_mode_backtest(
    scores_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    horizon: str,
    mode: str,
    long_short_pct: float,
    enter_pct: float,
    hold_pct: float,
    min_names: int,
    cost_rate: float,
    eod_signal_time: str,
) -> tuple[object, pd.DataFrame]:
    entry_pos, exit_pos = build_schedule(close_wide.index, horizon, eod_signal_time)
    if len(entry_pos) == 0:
        empty = pd.DataFrame(
            columns=[
                "trade_time",
                "gross_return",
                "net_return",
                "turnover",
                "cost",
                "valid_count",
                "position_count",
            ]
        )
        return summarize_returns(mode, horizon, pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), np.nan, 1.0), empty

    close_values = close_wide.to_numpy(dtype=float, copy=False)
    score_values = scores_wide.to_numpy(dtype=float, copy=False)

    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    valid_counts: list[int] = []
    position_counts: list[int] = []
    executed_times: list[pd.Timestamp] = []

    prev_weights = np.zeros(close_values.shape[1], dtype=float)
    prev_day: pd.Timestamp | None = None

    for entry_idx, exit_idx in zip(entry_pos, exit_pos):
        trade_time = pd.Timestamp(close_wide.index[entry_idx])
        trade_day = trade_time.normalize()
        if prev_day is not None and trade_day != prev_day:
            prev_weights = np.zeros_like(prev_weights)

        scores = score_values[entry_idx]
        if mode == "baseline":
            weights, valid_count = compute_weights(scores, long_short_pct, min_names)
            position_count = int(np.count_nonzero(weights))
        else:
            weights, valid_count, position_count = compute_buffered_weights(
                scores=scores,
                prev_weights=prev_weights,
                enter_pct=enter_pct,
                hold_pct=hold_pct,
                min_names=min_names,
            )

        if valid_count == 0 or position_count == 0:
            prev_day = trade_day
            continue

        entry_px = close_values[entry_idx]
        exit_px = close_values[exit_idx]
        asset_returns = exit_px / entry_px - 1.0
        valid_returns = np.isfinite(asset_returns)
        weights = np.where(valid_returns, weights, 0.0)
        position_count = int(np.count_nonzero(weights))
        if position_count == 0:
            prev_day = trade_day
            continue

        gross = float(np.nansum(weights * asset_returns))
        turnover = float(np.sum(np.abs(weights - prev_weights)))
        cost = turnover * cost_rate
        net = gross - cost

        gross_returns.append(gross)
        net_returns.append(net)
        turnover_values.append(turnover)
        cost_values.append(cost)
        valid_counts.append(valid_count)
        position_counts.append(position_count)
        executed_times.append(trade_time)

        prev_weights = weights
        prev_day = trade_day

    detail = pd.DataFrame(
        {
            "trade_time": executed_times,
            "gross_return": gross_returns,
            "net_return": net_returns,
            "turnover": turnover_values,
            "cost": cost_values,
            "valid_count": valid_counts,
            "position_count": position_counts,
        }
    )

    grouped_days = pd.Series(close_wide.index.normalize()).value_counts()
    if horizon == "eod":
        periods_per_day = 1.0
    else:
        step = RETURN_PERIODS[horizon]
        periods_per_day = float(np.median(grouped_days.to_numpy(dtype=float) / step))

    summary = summarize_returns(
        strategy=mode,
        horizon=horizon,
        net_returns=detail["net_return"],
        turnover=detail["turnover"],
        names=detail["position_count"],
        total_cost=float(detail["cost"].sum()) if not detail.empty else np.nan,
        periods_per_day=periods_per_day,
    )
    return summary, detail


def main() -> int:
    args = parse_args()
    if args.hold_pct < args.enter_pct:
        raise ValueError("--hold-pct must be >= --enter-pct")

    configure_logging()

    assets = discover_assets(args.factor_dir, args.assets)
    if args.limit is not None:
        assets = assets[: args.limit]

    discovered_factors = discover_factor_columns(args.factor_dir, assets)
    factor_names = args.factors if args.factors is not None else discovered_factors
    missing = sorted(set(factor_names) - set(discovered_factors))
    if missing:
        raise ValueError(f"Unknown factor columns requested: {missing}")

    start_ts, end_ts = resolve_analysis_window(
        args.factor_dir,
        assets,
        args.start_date,
        args.end_date,
        365,
    )

    LOGGER.info("Loading %s stock assets", len(assets))
    panel = load_selected_columns(
        args.factor_dir,
        assets,
        ["close", *factor_names],
        start_ts=start_ts,
        end_ts=end_ts,
    )
    calendar, last_session_pos = build_intraday_calendar(panel["datetime"])
    panel = attach_calendar(panel, calendar)
    panel = compute_same_day_returns(panel, last_session_pos)

    close_wide = panel.pivot(index="datetime", columns="asset", values="close").sort_index()
    factor_frames = {
        factor: factor_wide(panel, factor, close_wide.columns, close_wide.index)
        for factor in factor_names
    }
    factor_frames["combo_equal_zscore"] = build_combo_factor(factor_frames)

    summary_rows: list[dict[str, float | int | str]] = []
    detail_frames: dict[tuple[str, str], pd.DataFrame] = {}
    cost_rate = args.cost_rate

    for horizon in [h.lower() for h in args.horizons]:
        for strategy, scores_wide in factor_frames.items():
            LOGGER.info("Running %s @ %s", strategy, horizon)
            for mode in ("baseline", "buffered"):
                summary, detail = run_mode_backtest(
                    scores_wide=scores_wide,
                    close_wide=close_wide,
                    horizon=horizon,
                    mode=mode,
                    long_short_pct=args.long_short_pct,
                    enter_pct=args.enter_pct,
                    hold_pct=args.hold_pct,
                    min_names=args.min_names,
                    cost_rate=cost_rate,
                    eod_signal_time=args.eod_signal_time,
                )
                summary_rows.append(
                    build_summary_row(
                        strategy=strategy,
                        mode=mode,
                        horizon=horizon,
                        cost_rate=cost_rate,
                        enter_pct=args.enter_pct,
                        hold_pct=args.hold_pct,
                        summary=summary,
                        detail=detail,
                        factor=strategy,
                    )
                )
                detail_frames[(f"{strategy}_{mode}", horizon)] = detail

    summary_df = pd.DataFrame(summary_rows).sort_values(["horizon", "strategy", "mode"], ascending=[True, True, True])
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_table(summary_df, args.output_root / "stock_low_turnover_backtest_summary.csv")
    save_table(summary_df, args.output_root / "stock_low_turnover_backtest_summary.parquet")

    for (strategy_mode, horizon), detail in detail_frames.items():
        save_table(detail, args.output_root / f"details_{strategy_mode}_{horizon}.csv")

    plot_equity_curves(detail_frames, args.output_root / "stock_low_turnover_equity_curves.png")
    LOGGER.info("Completed. Results written to %s", args.output_root)
    LOGGER.info("Summary:\n%s", summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
