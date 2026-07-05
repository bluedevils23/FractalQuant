from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_etf_factor_validation_backtest import (  # noqa: E402
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_UNIVERSE_FILE,
    HORIZON_TO_BARS,
    build_wide_close,
    discover_files,
    load_panel,
)


LOGGER = logging.getLogger("run_etf_factor_tradable_backtest")

DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_factors_v2")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_tradable_backtests_2020_2026")
DEFAULT_FACTORS = ("volume_weighted_price", "volume_spike", "liquidity_shock")
DEFAULT_HORIZONS = ("1m", "5m", "10m", "eod")
DEFAULT_LONG_SHORT_PCT = 0.2
DEFAULT_MIN_NAMES = 10
DEFAULT_COMMISSION = 0.001
DEFAULT_SLIPPAGE = 0.0005
DEFAULT_EOD_SIGNAL_TIME = "14:50:00"


@dataclass
class TradableBacktestResult:
    strategy: str
    horizon: str
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_turnover: float
    total_cost: float
    n_periods: int
    avg_names: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stricter ETF factor tradable backtests.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--factors", nargs="*", default=list(DEFAULT_FACTORS))
    parser.add_argument("--horizons", nargs="*", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--long-short-pct", type=float, default=DEFAULT_LONG_SHORT_PCT)
    parser.add_argument("--min-names", type=int, default=DEFAULT_MIN_NAMES)
    parser.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    parser.add_argument("--eod-signal-time", type=str, default=DEFAULT_EOD_SIGNAL_TIME)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def factor_wide(panel: pd.DataFrame, factor: str, columns: pd.Index, index: pd.Index) -> pd.DataFrame:
    frame = panel[factor].unstack("ts_code")
    return frame.reindex(index=index, columns=columns)


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def build_combo_factor(factor_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zframes = [cross_sectional_zscore(frame) for frame in factor_frames.values()]
    combo = sum(zframes) / len(zframes)
    return combo


def build_schedule(index: pd.DatetimeIndex, horizon: str, eod_signal_time: str) -> tuple[np.ndarray, np.ndarray]:
    signal_time = pd.Timestamp(f"2000-01-01 {eod_signal_time}").time()
    grouped = pd.Series(np.arange(len(index)), index=index).groupby(index.normalize(), sort=True)

    entry_positions: list[int] = []
    exit_positions: list[int] = []

    for _, positions in grouped:
        day_positions = positions.to_numpy(dtype=int, copy=False)
        if horizon == "eod":
            day_times = index[day_positions].time
            candidate_idx = np.where(np.array([t == signal_time for t in day_times], dtype=bool))[0]
            if len(candidate_idx) == 0:
                continue
            local_entry = int(candidate_idx[0])
            local_exit = len(day_positions) - 1
            if local_exit <= local_entry:
                continue
            entry_positions.append(int(day_positions[local_entry]))
            exit_positions.append(int(day_positions[local_exit]))
            continue

        step = HORIZON_TO_BARS[horizon]
        if step is None or len(day_positions) <= step:
            continue
        for local_entry in range(0, len(day_positions) - step, step):
            local_exit = local_entry + step
            entry_positions.append(int(day_positions[local_entry]))
            exit_positions.append(int(day_positions[local_exit]))

    return np.asarray(entry_positions, dtype=int), np.asarray(exit_positions, dtype=int)


def compute_weights(scores: np.ndarray, long_short_pct: float, min_names: int) -> tuple[np.ndarray, int]:
    valid = np.isfinite(scores)
    valid_count = int(valid.sum())
    weights = np.zeros_like(scores, dtype=float)
    if valid_count < min_names:
        return weights, 0

    side_count = max(int(math.floor(valid_count * long_short_pct)), 1)
    if side_count * 2 > valid_count:
        side_count = valid_count // 2
    if side_count < 1:
        return weights, 0

    valid_idx = np.flatnonzero(valid)
    ranked = valid_idx[np.argsort(scores[valid_idx])]
    short_idx = ranked[:side_count]
    long_idx = ranked[-side_count:]

    weights[short_idx] = -0.5 / side_count
    weights[long_idx] = 0.5 / side_count
    return weights, valid_count


def annualization(periods_per_day: float) -> float:
    return 252.0 * periods_per_day


def summarize_returns(
    strategy: str,
    horizon: str,
    net_returns: pd.Series,
    turnover: pd.Series,
    names: pd.Series,
    total_cost: float,
    periods_per_day: float,
) -> TradableBacktestResult:
    if net_returns.empty:
        return TradableBacktestResult(strategy, horizon, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    clipped = net_returns.clip(lower=-0.999999)
    log_equity = np.log1p(clipped).cumsum()
    equity = np.exp(np.clip(log_equity, -700.0, 700.0))
    total_log_return = float(np.log1p(clipped).sum())
    ann_factor = annualization(periods_per_day)

    total_return = float(np.expm1(np.clip(total_log_return, -700.0, 700.0)))
    annual_return = float(np.expm1(np.clip(total_log_return * ann_factor / len(net_returns), -700.0, 700.0)))
    annual_vol = float(net_returns.std(ddof=1) * math.sqrt(ann_factor)) if len(net_returns) > 1 else np.nan
    sharpe = float(net_returns.mean() / net_returns.std(ddof=1) * math.sqrt(ann_factor)) if len(net_returns) > 1 and net_returns.std(ddof=1) > 0 else np.nan
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    max_drawdown = float(drawdown.min())
    win_rate = float((net_returns > 0).mean())

    return TradableBacktestResult(
        strategy=strategy,
        horizon=horizon,
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        avg_turnover=float(turnover.mean()),
        total_cost=total_cost,
        n_periods=len(net_returns),
        avg_names=float(names.mean()),
    )


def run_tradable_backtest(
    scores_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    horizon: str,
    long_short_pct: float,
    min_names: int,
    cost_rate: float,
    eod_signal_time: str,
) -> tuple[TradableBacktestResult, pd.DataFrame]:
    entry_pos, exit_pos = build_schedule(close_wide.index, horizon, eod_signal_time)
    if len(entry_pos) == 0:
        empty = pd.DataFrame(columns=["trade_time", "net_return", "gross_return", "turnover", "cost", "name_count"])
        return summarize_returns("unknown", horizon, pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), np.nan, 1.0), empty

    close_values = close_wide.to_numpy(dtype=float, copy=False)
    score_values = scores_wide.to_numpy(dtype=float, copy=False)

    trade_times = close_wide.index[entry_pos]
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    name_counts: list[int] = []
    executed_times: list[pd.Timestamp] = []

    prev_weights = np.zeros(close_values.shape[1], dtype=float)
    for entry_idx, exit_idx in zip(entry_pos, exit_pos):
        scores = score_values[entry_idx]
        weights, valid_count = compute_weights(scores, long_short_pct, min_names)
        if valid_count == 0:
            continue

        entry_px = close_values[entry_idx]
        exit_px = close_values[exit_idx]
        asset_returns = exit_px / entry_px - 1.0
        valid_returns = np.isfinite(asset_returns)
        weights = np.where(valid_returns, weights, 0.0)

        gross = float(np.nansum(weights * asset_returns))
        turnover = float(np.sum(np.abs(weights - prev_weights)))
        cost = turnover * cost_rate
        net = gross - cost

        gross_returns.append(gross)
        net_returns.append(net)
        turnover_values.append(turnover)
        cost_values.append(cost)
        name_counts.append(valid_count)
        executed_times.append(pd.Timestamp(close_wide.index[entry_idx]))
        prev_weights = weights

    detail = pd.DataFrame(
        {
            "trade_time": executed_times,
            "gross_return": gross_returns,
            "net_return": net_returns,
            "turnover": turnover_values,
            "cost": cost_values,
            "name_count": name_counts,
        }
    )

    grouped_days = pd.Series(close_wide.index.normalize()).value_counts()
    if horizon == "eod":
        periods_per_day = 1.0
    else:
        step = HORIZON_TO_BARS[horizon] or 1
        periods_per_day = float(np.median(grouped_days.to_numpy(dtype=float) / step))

    summary = summarize_returns(
        strategy="unknown",
        horizon=horizon,
        net_returns=detail["net_return"],
        turnover=detail["turnover"],
        names=detail["name_count"],
        total_cost=float(detail["cost"].sum()) if not detail.empty else np.nan,
        periods_per_day=periods_per_day,
    )
    return summary, detail


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)


def plot_equity_curves(detail_frames: dict[tuple[str, str], pd.DataFrame], output_path: Path) -> None:
    if not detail_frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(set(h for _, h in detail_frames.keys())), 1, figsize=(14, 4 * len(set(h for _, h in detail_frames.keys()))), squeeze=False)
    horizon_list = sorted(set(h for _, h in detail_frames.keys()))
    for ax, horizon in zip(axes.ravel(), horizon_list):
        subset = {(s, h): df for (s, h), df in detail_frames.items() if h == horizon}
        plotted = False
        for (strategy, _), detail in subset.items():
            if detail.empty:
                continue
            equity = np.exp(np.clip(np.log1p(detail["net_return"].clip(lower=-0.999999)).cumsum(), -700.0, 700.0))
            ax.plot(detail["trade_time"], equity, label=strategy)
            plotted = True
        ax.set_title(f"Tradable Equity Curves - {horizon}")
        if plotted:
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_logging()

    files = discover_files(args.input_root, args.universe_file, args.limit)
    LOGGER.info("Loading %s ETF factor files", len(files))
    panel = load_panel(files, args.start_date, args.end_date)

    close_wide = build_wide_close(panel)
    selected_factors = [factor for factor in args.factors if factor in panel.columns]
    missing = sorted(set(args.factors) - set(selected_factors))
    if missing:
        LOGGER.warning("Missing factors in panel: %s", ", ".join(missing))
    if not selected_factors:
        raise ValueError("No requested factors were found in the input panel.")

    factor_frames = {
        factor: factor_wide(panel, factor, close_wide.columns, close_wide.index)
        for factor in selected_factors
    }
    factor_frames["combo_equal_zscore"] = build_combo_factor(factor_frames)

    detail_frames: dict[tuple[str, str], pd.DataFrame] = {}
    summary_rows: list[TradableBacktestResult] = []
    cost_rate = args.commission + args.slippage

    for horizon in [h.lower() for h in args.horizons]:
        for strategy, scores_wide in factor_frames.items():
            LOGGER.info("Running %s @ %s", strategy, horizon)
            summary, detail = run_tradable_backtest(
                scores_wide=scores_wide,
                close_wide=close_wide,
                horizon=horizon,
                long_short_pct=args.long_short_pct,
                min_names=args.min_names,
                cost_rate=cost_rate,
                eod_signal_time=args.eod_signal_time,
            )
            summary.strategy = strategy
            summary_rows.append(summary)
            detail_frames[(strategy, horizon)] = detail

    summary_df = pd.DataFrame([row.__dict__ for row in summary_rows]).sort_values(["horizon", "sharpe_ratio"], ascending=[True, False])
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_table(summary_df, args.output_root / "tradable_backtest_summary.csv")
    save_table(summary_df, args.output_root / "tradable_backtest_summary.parquet")

    for (strategy, horizon), detail in detail_frames.items():
        save_table(detail, args.output_root / f"details_{strategy}_{horizon}.csv")

    plot_equity_curves(detail_frames, args.output_root / "tradable_equity_curves.png")
    LOGGER.info("Completed. Results written to %s", args.output_root)
    LOGGER.info("Summary:\n%s", summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
