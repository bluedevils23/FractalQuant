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
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("run_etf_factor_validation_backtest")

DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_factors_v2")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_factor_backtests_2020_2026")
DEFAULT_UNIVERSE_FILE = SCRIPT_DIR / "small_etf_universe_50_latest_longest.txt"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-12-31"
DEFAULT_HORIZONS = ("1m", "5m", "10m", "eod")
HORIZON_TO_BARS = {"1m": 1, "5m": 5, "10m": 10, "eod": None}
QUANTILES = 5
TOP_N_FACTORS = 5

NON_FACTOR_COLUMNS = {"ts_code", "open", "high", "low", "close", "volume", "amount", "adj_factor"}
RETURN_COLUMNS = {"returns", "log_returns"}


@dataclass
class FactorValidationResult:
    factor: str
    horizon: str
    ic_mean: float
    ic_std: float
    icir: float
    n_days: int
    n_obs: int
    quantile_spread: float
    top_mean: float
    bottom_mean: float


@dataclass
class BacktestResult:
    factor: str
    horizon: str
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_periods: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and backtest ETF factors.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--top-n-factors", type=int, default=TOP_N_FACTORS)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_universe_codes(universe_file: Path) -> list[str]:
    if not universe_file.exists():
        raise FileNotFoundError(f"Universe file does not exist: {universe_file}")
    codes = []
    for line in universe_file.read_text(encoding="utf-8").splitlines():
        code = line.strip()
        if not code or code.startswith("#"):
            continue
        codes.append(code)
    if not codes:
        raise ValueError(f"No ETF codes found in {universe_file}")
    return codes


def discover_files(input_root: Path, universe_file: Path, limit: int | None) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    codes = load_universe_codes(universe_file)
    files = [input_root / f"{code}.parquet" for code in codes if (input_root / f"{code}.parquet").exists()]
    missing = [code for code in codes if not (input_root / f"{code}.parquet").exists()]
    if missing:
        LOGGER.warning("Skipping %s missing ETF files from universe list", len(missing))
    if not files:
        raise FileNotFoundError(f"No matching parquet files found under {input_root}")
    return files[:limit] if limit else files


def load_panel(files: list[Path], start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    frames = []
    for path in files:
        df = pd.read_parquet(path)
        if "trade_time" not in df.index.names and "trade_time" not in df.columns:
            raise ValueError(f"Missing trade_time in {path}")
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"])
            df = df.set_index("trade_time")
        else:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "trade_time"
        df = df[(df.index >= start_ts) & (df.index < end_ts)]
        if df.empty:
            continue
        if "ts_code" not in df.columns:
            df["ts_code"] = path.stem
        df["ts_code"] = df["ts_code"].astype(str)
        df = df.sort_index()
        frames.append(df)

    panel = pd.concat(frames, axis=0, ignore_index=False)
    panel = panel.sort_values(["trade_time", "ts_code"])
    panel = panel.reset_index().rename(columns={"index": "trade_time"})
    panel["trade_time"] = pd.to_datetime(panel["trade_time"])
    panel = panel.drop_duplicates(["trade_time", "ts_code"], keep="last")
    panel = panel.set_index(["trade_time", "ts_code"]).sort_index()
    return panel


def factor_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for column in df.columns:
        if column in NON_FACTOR_COLUMNS or column in RETURN_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            cols.append(column)
    return cols


def build_wide_close(panel: pd.DataFrame) -> pd.DataFrame:
    close_wide = panel["close"].unstack("ts_code").sort_index()
    close_wide.columns = close_wide.columns.astype(str)
    return close_wide


def build_forward_wide(close_wide: pd.DataFrame, horizon: str) -> pd.DataFrame:
    if horizon == "eod":
        daily_last = close_wide.groupby(close_wide.index.normalize()).transform("last")
        return daily_last / close_wide - 1.0

    bars = HORIZON_TO_BARS[horizon]
    if bars is None:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return close_wide.shift(-bars) / close_wide - 1.0


def rowwise_spearman_ic(factor_wide: pd.DataFrame, forward_wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    factor_rank = factor_wide.rank(axis=1, method="average", na_option="keep").to_numpy(dtype=float, copy=False)
    forward_rank = forward_wide.rank(axis=1, method="average", na_option="keep").to_numpy(dtype=float, copy=False)

    valid = np.isfinite(factor_rank) & np.isfinite(forward_rank)
    counts = valid.sum(axis=1).astype(float)

    x = np.where(valid, factor_rank, 0.0)
    y = np.where(valid, forward_rank, 0.0)

    sx = x.sum(axis=1)
    sy = y.sum(axis=1)
    sxx = np.square(x).sum(axis=1)
    syy = np.square(y).sum(axis=1)
    sxy = (x * y).sum(axis=1)

    counts_safe = np.where(counts > 0, counts, np.nan)
    cov = sxy - (sx * sy / counts_safe)
    varx = sxx - (sx * sx / counts_safe)
    vary = syy - (sy * sy / counts_safe)

    denom = np.sqrt(varx * vary)
    ic = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom != 0)
    ic[(counts < 2) | ~np.isfinite(ic)] = np.nan
    return ic, counts


def safe_row_mean(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=1).astype(float)
    totals = np.where(valid, values, 0.0).sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def rowwise_top_bottom_spread(factor_wide: pd.DataFrame, forward_wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor_vals = factor_wide.to_numpy(dtype=float, copy=False)
    forward_vals = forward_wide.to_numpy(dtype=float, copy=False)
    valid = np.isfinite(factor_vals) & np.isfinite(forward_vals)
    counts = valid.sum(axis=1).astype(int)

    if not len(counts):
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(np.where(valid, factor_vals, np.inf), axis=1)
    sorted_forward = np.take_along_axis(np.where(valid, forward_vals, np.nan), order, axis=1)

    pct = np.where(counts >= 10, 0.2, 0.3)
    k = np.clip(np.ceil(counts * pct).astype(int), 1, 3)
    idx = np.arange(len(counts))

    bottom_1 = sorted_forward[:, 0]
    bottom_2 = safe_row_mean(sorted_forward[:, :2])
    bottom_3 = safe_row_mean(sorted_forward[:, :3])
    top_1 = sorted_forward[idx, np.maximum(counts - 1, 0)]
    top_2 = safe_row_mean(
        np.stack(
            [
                sorted_forward[idx, np.maximum(counts - 1, 0)],
                sorted_forward[idx, np.maximum(counts - 2, 0)],
            ],
            axis=1,
        ),
    )
    top_3 = safe_row_mean(
        np.stack(
            [
                sorted_forward[idx, np.maximum(counts - 1, 0)],
                sorted_forward[idx, np.maximum(counts - 2, 0)],
                sorted_forward[idx, np.maximum(counts - 3, 0)],
            ],
            axis=1,
        ),
    )

    use_two = k == 2
    use_three = k == 3
    bottom = np.where(use_three, bottom_3, np.where(use_two, bottom_2, bottom_1))
    top = np.where(use_three, top_3, np.where(use_two, top_2, top_1))
    spread = top - bottom

    spread[~np.isfinite(spread)] = np.nan
    top[~np.isfinite(top)] = np.nan
    bottom[~np.isfinite(bottom)] = np.nan
    return spread, top, bottom


def annualization_factor(horizon: str, rows_per_day: int) -> float:
    rows_per_day = max(int(rows_per_day), 1)
    if horizon == "eod":
        return 252.0
    bars = HORIZON_TO_BARS[horizon] or 1
    return 252.0 * rows_per_day / bars


def summarize_factor_from_wide(
    factor: str,
    horizon: str,
    factor_wide: pd.DataFrame,
    forward_wide: pd.DataFrame,
    rows_per_day: int,
) -> tuple[FactorValidationResult, BacktestResult, pd.Series]:
    ic_values, ic_counts = rowwise_spearman_ic(factor_wide, forward_wide)
    spread_values, top_values, bottom_values = rowwise_top_bottom_spread(factor_wide, forward_wide)

    valid_ic = ic_values[np.isfinite(ic_values)]
    ic_mean = float(np.nanmean(valid_ic)) if len(valid_ic) else np.nan
    ic_std = float(np.nanstd(valid_ic, ddof=1)) if len(valid_ic) > 1 else np.nan
    icir = ic_mean / ic_std if len(valid_ic) > 1 and np.isfinite(ic_std) and ic_std > 0 else np.nan

    valid_spread = spread_values[np.isfinite(spread_values)]
    spread_mean = float(np.nanmean(valid_spread)) if len(valid_spread) else np.nan
    top_mean = float(np.nanmean(top_values)) if np.isfinite(top_values).any() else np.nan
    bottom_mean = float(np.nanmean(bottom_values)) if np.isfinite(bottom_values).any() else np.nan

    backtest_curve = pd.Series(spread_values, index=factor_wide.index, dtype=float).dropna()
    if backtest_curve.empty:
        backtest_result = BacktestResult(factor, horizon, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    else:
        clipped_curve = backtest_curve.clip(lower=-0.999999)
        total_log_return = float(np.log1p(clipped_curve).sum())
        total_return = float(np.expm1(np.clip(total_log_return, -700.0, 700.0)))
        ann_factor = annualization_factor(horizon, rows_per_day)
        annual_log_return = total_log_return * (ann_factor / max(len(backtest_curve), 1))
        annual_return = float(np.expm1(np.clip(annual_log_return, -700.0, 700.0)))
        annual_vol = float(backtest_curve.std(ddof=1) * math.sqrt(ann_factor)) if len(backtest_curve) > 1 else np.nan
        sharpe = (
            float(backtest_curve.mean() / backtest_curve.std(ddof=1) * math.sqrt(ann_factor))
            if len(backtest_curve) > 1 and backtest_curve.std(ddof=1) > 0
            else np.nan
        )
        equity = np.exp(np.clip(np.log1p(clipped_curve).cumsum(), -700.0, 700.0))
        drawdown = equity / equity.cummax() - 1.0
        max_drawdown = float(drawdown.min())
        win_rate = float((backtest_curve > 0).mean())
        backtest_result = BacktestResult(
            factor=factor,
            horizon=horizon,
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            n_periods=len(backtest_curve),
        )

    validation_result = FactorValidationResult(
        factor=factor,
        horizon=horizon,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        n_days=int(np.isfinite(ic_values).sum()),
        n_obs=int(np.isfinite(factor_wide.to_numpy(dtype=float, copy=False)).sum()),
        quantile_spread=spread_mean,
        top_mean=top_mean,
        bottom_mean=bottom_mean,
    )
    return validation_result, backtest_result, backtest_curve


def factor_frame_from_panel(panel: pd.DataFrame, factor: str, columns: pd.Index, index: pd.Index) -> pd.DataFrame:
    factor_wide = panel[factor].unstack("ts_code")
    return factor_wide.reindex(index=index, columns=columns)


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)


def plot_icir_rank(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(summary["horizon"].unique()), 1, figsize=(14, 4 * len(summary["horizon"].unique())), squeeze=False)
    for ax, horizon in zip(axes.ravel(), sorted(summary["horizon"].unique())):
        sub = summary[summary["horizon"] == horizon].sort_values("icir", ascending=False).head(15)
        ax.bar(sub["factor"], sub["icir"].fillna(0.0))
        ax.set_title(f"ICIR Rank - {horizon}")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(curves: dict[tuple[str, str], pd.Series], output_path: Path) -> None:
    if not curves:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    for (factor, horizon), curve in curves.items():
        ax.plot(curve.index, curve.values, label=f"{factor} / {horizon}")
    ax.set_title("Top Factor Equity Curves")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def daily_summary(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df.empty:
        return backtest_df
    grouped = backtest_df.groupby("horizon", dropna=False)
    rows = []
    for horizon, group in grouped:
        rows.append(
            {
                "horizon": horizon,
                "factors": int(group["factor"].nunique()),
                "avg_total_return": float(group["total_return"].mean()),
                "avg_annual_return": float(group["annual_return"].mean()),
                "avg_sharpe_ratio": float(group["sharpe_ratio"].mean()),
                "avg_max_drawdown": float(group["max_drawdown"].mean()),
                "avg_win_rate": float(group["win_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    configure_logging()

    files = discover_files(args.input_root, args.universe_file, args.limit)
    LOGGER.info("Loading %s ETF factor files", len(files))
    panel = load_panel(files, args.start_date, args.end_date)
    factor_cols = factor_columns(panel)
    LOGGER.info("Panel rows=%s factors=%s", len(panel), len(factor_cols))

    horizons = [h.lower() for h in args.horizons]
    close_wide = build_wide_close(panel)
    rows_per_day = int(close_wide.groupby(close_wide.index.normalize()).size().median())
    forward_wide_map = {h: build_forward_wide(close_wide, h) for h in horizons}

    validation_rows: list[FactorValidationResult] = []
    backtest_rows: list[BacktestResult] = []
    equity_curves: dict[tuple[str, str], pd.Series] = {}
    quantile_rows: list[dict[str, object]] = []

    for factor in factor_cols:
        factor_wide = factor_frame_from_panel(panel, factor, close_wide.columns, close_wide.index)
        for horizon in horizons:
            if horizon not in HORIZON_TO_BARS:
                raise ValueError(f"Unsupported horizon: {horizon}")
            forward_wide = forward_wide_map[horizon]
            try:
                validation_result, backtest_result, backtest_curve = summarize_factor_from_wide(
                    factor=factor,
                    horizon=horizon,
                    factor_wide=factor_wide,
                    forward_wide=forward_wide,
                    rows_per_day=rows_per_day,
                )
                validation_rows.append(validation_result)
                quantile_rows.append(
                    {
                        "factor": factor,
                        "horizon": horizon,
                        "quantile_spread": validation_result.quantile_spread,
                        "top_mean": validation_result.top_mean,
                        "bottom_mean": validation_result.bottom_mean,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed for %s @ %s: %s", factor, horizon, exc)

    validation_df = pd.DataFrame([r.__dict__ for r in validation_rows])
    quantile_df = pd.DataFrame(quantile_rows)

    for horizon in horizons:
        horizon_validation = validation_df[validation_df["horizon"] == horizon].copy()
        horizon_validation = horizon_validation.sort_values("icir", ascending=False)
        top_factors = horizon_validation["factor"].head(args.top_n_factors).tolist()
        forward_wide = forward_wide_map[horizon]
        for factor in top_factors:
            try:
                factor_wide = factor_frame_from_panel(panel, factor, close_wide.columns, close_wide.index)
                _, backtest_result, backtest_curve = summarize_factor_from_wide(
                    factor=factor,
                    horizon=horizon,
                    factor_wide=factor_wide,
                    forward_wide=forward_wide,
                    rows_per_day=rows_per_day,
                )
                backtest_rows.append(backtest_result)
                if len(backtest_curve) > 0:
                    equity_curves[(factor, horizon)] = backtest_curve
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Backtest failed for %s @ %s: %s", factor, horizon, exc)

    backtest_df = pd.DataFrame([r.__dict__ for r in backtest_rows])

    args.output_root.mkdir(parents=True, exist_ok=True)
    save_table(validation_df, args.output_root / "factor_ic_summary.csv")
    save_table(validation_df, args.output_root / "factor_ic_summary.parquet")
    save_table(quantile_df, args.output_root / "factor_quantile_returns.csv")
    save_table(quantile_df, args.output_root / "factor_quantile_returns.parquet")
    save_table(backtest_df, args.output_root / "factor_backtest_summary.csv")
    save_table(backtest_df, args.output_root / "factor_backtest_summary.parquet")
    save_table(daily_summary(backtest_df), args.output_root / "factor_backtest_horizon_summary.csv")

    plot_icir_rank(validation_df, args.output_root / "factor_icir_rank.png")
    plot_equity_curves(equity_curves, args.output_root / "top_factor_equity_curves.png")

    if not backtest_df.empty:
        LOGGER.info("Completed. Results written to %s", args.output_root)
        LOGGER.info("Top backtest rows:\n%s", backtest_df.head(10).to_string(index=False))
    else:
        LOGGER.warning("Completed with no backtest results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
