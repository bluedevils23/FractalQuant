from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_etf_factor_tradable_backtest import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    TradableBacktestResult,
    build_schedule,
    compute_weights,
    configure_logging,
    factor_wide,
    plot_equity_curves,
    save_table,
    summarize_returns,
)
from run_etf_factor_validation_backtest import (  # noqa: E402
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_UNIVERSE_FILE,
    build_wide_close,
    discover_files,
    load_panel,
)


LOGGER = logging.getLogger("run_volume_weighted_price_low_turnover_backtest")

DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_low_turnover_backtests_2020_2026")
DEFAULT_FACTOR = "volume_weighted_price"
DEFAULT_HORIZONS = ("1m", "5m", "10m", "eod")
DEFAULT_LONG_SHORT_PCT = 0.2
DEFAULT_MIN_NAMES = 10
DEFAULT_COST_RATE = 0.0003
DEFAULT_ENTER_PCT = 0.1
DEFAULT_HOLD_PCT = 0.2
DEFAULT_EOD_SIGNAL_TIME = "14:50:00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-turnover ETF backtests for volume_weighted_price.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--factor", type=str, default=DEFAULT_FACTOR)
    parser.add_argument("--horizons", nargs="*", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--long-short-pct", type=float, default=DEFAULT_LONG_SHORT_PCT)
    parser.add_argument("--min-names", type=int, default=DEFAULT_MIN_NAMES)
    parser.add_argument("--cost-rate", type=float, default=DEFAULT_COST_RATE)
    parser.add_argument("--enter-pct", type=float, default=DEFAULT_ENTER_PCT)
    parser.add_argument("--hold-pct", type=float, default=DEFAULT_HOLD_PCT)
    parser.add_argument("--eod-signal-time", type=str, default=DEFAULT_EOD_SIGNAL_TIME)
    return parser.parse_args()


def annualization(periods_per_day: float) -> float:
    return 252.0 * periods_per_day


def compute_buffered_weights(
    scores: np.ndarray,
    prev_weights: np.ndarray,
    enter_pct: float,
    hold_pct: float,
    min_names: int,
) -> tuple[np.ndarray, int, int]:
    valid = np.isfinite(scores)
    valid_count = int(valid.sum())
    weights = np.zeros_like(scores, dtype=float)
    if valid_count < min_names:
        return weights, 0, 0

    enter_count = max(int(math.floor(valid_count * enter_pct)), 1)
    hold_count = max(int(math.floor(valid_count * hold_pct)), enter_count)
    if hold_count * 2 > valid_count:
        hold_count = valid_count // 2
    if enter_count * 2 > valid_count:
        enter_count = valid_count // 2
    if enter_count < 1 or hold_count < 1:
        return weights, 0, 0

    valid_idx = np.flatnonzero(valid)
    ranked = valid_idx[np.argsort(scores[valid_idx])]

    short_enter = np.zeros_like(scores, dtype=bool)
    short_hold = np.zeros_like(scores, dtype=bool)
    long_enter = np.zeros_like(scores, dtype=bool)
    long_hold = np.zeros_like(scores, dtype=bool)

    short_enter[ranked[:enter_count]] = True
    short_hold[ranked[:hold_count]] = True
    long_enter[ranked[-enter_count:]] = True
    long_hold[ranked[-hold_count:]] = True

    prev_long = prev_weights > 0
    prev_short = prev_weights < 0

    long_mask = long_enter | (prev_long & long_hold)
    short_mask = short_enter | (prev_short & short_hold)

    long_mask &= ~short_enter
    short_mask &= ~long_enter

    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())
    if long_count < 1 or short_count < 1:
        return weights, valid_count, 0

    weights[long_mask] = 0.5 / long_count
    weights[short_mask] = -0.5 / short_count
    return weights, valid_count, long_count + short_count


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
) -> tuple[TradableBacktestResult, pd.DataFrame]:
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
        step = max(int((exit_pos[0] - entry_pos[0])) if len(entry_pos) else 1, 1)
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


def build_summary_row(
    summary: TradableBacktestResult,
    detail: pd.DataFrame,
    factor: str,
    mode: str,
    horizon: str,
    cost_rate: float,
    enter_pct: float,
    hold_pct: float,
) -> dict[str, float | int | str]:
    avg_gross_return = float(detail["gross_return"].mean()) if not detail.empty else np.nan
    avg_turnover = float(detail["turnover"].mean()) if not detail.empty else np.nan
    breakeven_cost_rate = avg_gross_return / avg_turnover if detail.shape[0] and avg_turnover > 0 else np.nan
    return {
        "factor": factor,
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


def plot_turnover(detail_frames: dict[tuple[str, str], pd.DataFrame], output_path: Path) -> None:
    horizon_list = sorted(set(horizon for _, horizon in detail_frames.keys()))
    if not horizon_list:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(horizon_list), 1, figsize=(14, 4 * len(horizon_list)), squeeze=False)
    for ax, horizon in zip(axes.ravel(), horizon_list):
        subset = {(mode, h): df for (mode, h), df in detail_frames.items() if h == horizon}
        plotted = False
        for (mode, _), detail in subset.items():
            if detail.empty:
                continue
            turnover_rolling = detail["turnover"].rolling(50, min_periods=10).mean()
            ax.plot(detail["trade_time"], turnover_rolling, label=mode)
            plotted = True
        ax.set_title(f"Rolling Turnover - {horizon}")
        if plotted:
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.hold_pct < args.enter_pct:
        raise ValueError("--hold-pct must be >= --enter-pct")

    configure_logging()

    files = discover_files(args.input_root, args.universe_file, args.limit)
    LOGGER.info("Loading %s ETF factor files", len(files))
    panel = load_panel(files, args.start_date, args.end_date)

    if args.factor not in panel.columns:
        raise ValueError(f"Factor {args.factor!r} was not found in the input panel.")

    close_wide = build_wide_close(panel)
    scores_wide = factor_wide(panel, args.factor, close_wide.columns, close_wide.index)

    detail_frames: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    for horizon in [h.lower() for h in args.horizons]:
        for mode in ("baseline", "buffered"):
            LOGGER.info("Running %s @ %s", mode, horizon)
            summary, detail = run_mode_backtest(
                scores_wide=scores_wide,
                close_wide=close_wide,
                horizon=horizon,
                mode=mode,
                long_short_pct=args.long_short_pct,
                enter_pct=args.enter_pct,
                hold_pct=args.hold_pct,
                min_names=args.min_names,
                cost_rate=args.cost_rate,
                eod_signal_time=args.eod_signal_time,
            )
            summary.strategy = f"{args.factor}_{mode}"
            summary_rows.append(
                build_summary_row(
                    summary=summary,
                    detail=detail,
                    factor=args.factor,
                    mode=mode,
                    horizon=horizon,
                    cost_rate=args.cost_rate,
                    enter_pct=args.enter_pct,
                    hold_pct=args.hold_pct,
                )
            )
            detail_frames[(mode, horizon)] = detail

    summary_df = pd.DataFrame(summary_rows).sort_values(["horizon", "mode"], ascending=[True, True])
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_table(summary_df, args.output_root / "low_turnover_backtest_summary.csv")
    save_table(summary_df, args.output_root / "low_turnover_backtest_summary.parquet")

    for (mode, horizon), detail in detail_frames.items():
        save_table(detail, args.output_root / f"details_{args.factor}_{mode}_{horizon}.csv")

    plot_equity_curves(
        {(f"{args.factor}_{mode}", horizon): detail for (mode, horizon), detail in detail_frames.items()},
        args.output_root / "low_turnover_equity_curves.png",
    )
    plot_turnover(detail_frames, args.output_root / "low_turnover_turnover_curves.png")

    LOGGER.info("Completed. Results written to %s", args.output_root)
    LOGGER.info("Summary:\n%s", summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
