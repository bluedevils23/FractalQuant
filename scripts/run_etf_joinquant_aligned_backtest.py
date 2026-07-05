from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
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
    DEFAULT_INPUT_ROOT,
    DEFAULT_START_DATE,
    DEFAULT_UNIVERSE_FILE,
    build_wide_close,
    discover_files,
    load_panel,
)


LOGGER = logging.getLogger("run_etf_joinquant_aligned_backtest")

DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_joinquant_aligned_backtests_2020_2026")
DEFAULT_FACTOR = "volume_weighted_price"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_BENCHMARK_CODE = "159920.SZ"
DEFAULT_OPEN_COMMISSION = 0.0001
DEFAULT_CLOSE_COMMISSION = 0.0001
DEFAULT_SLIPPAGE = 0.0
DEFAULT_REBALANCE_INTERVAL = 1
DEFAULT_MAX_HOLDINGS = 5
DEFAULT_MIN_CANDIDATES = 5
DEFAULT_ENTER_PCT = 0.10
DEFAULT_HOLD_PCT = 0.20
DEFAULT_REQUIRE_POSITIVE_SIGNAL = False
DEFAULT_MIN_ORDER_VALUE = 5000.0
DEFAULT_REBALANCE_TOLERANCE = 0.20
DEFAULT_MIN_TRADE_SHARES = 100
DEFAULT_FLATTEN_TIME = "14:50"


@dataclass
class BacktestSummary:
    factor: str
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_turnover: float
    total_cost: float
    n_periods: int
    avg_position_count: float
    benchmark_total_return: float
    benchmark_annual_return: float
    excess_total_return: float
    excess_annual_return: float
    alpha: float
    beta: float
    final_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local JoinQuant-aligned ETF backtest.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--factor", type=str, default=DEFAULT_FACTOR)
    parser.add_argument("--benchmark-code", type=str, default=DEFAULT_BENCHMARK_CODE)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--open-commission", type=float, default=DEFAULT_OPEN_COMMISSION)
    parser.add_argument("--close-commission", type=float, default=DEFAULT_CLOSE_COMMISSION)
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    parser.add_argument("--rebalance-interval", type=int, default=DEFAULT_REBALANCE_INTERVAL)
    parser.add_argument("--max-holdings", type=int, default=DEFAULT_MAX_HOLDINGS)
    parser.add_argument("--min-candidates", type=int, default=DEFAULT_MIN_CANDIDATES)
    parser.add_argument("--enter-pct", type=float, default=DEFAULT_ENTER_PCT)
    parser.add_argument("--hold-pct", type=float, default=DEFAULT_HOLD_PCT)
    parser.add_argument("--require-positive-signal", action="store_true", default=DEFAULT_REQUIRE_POSITIVE_SIGNAL)
    parser.add_argument("--min-order-value", type=float, default=DEFAULT_MIN_ORDER_VALUE)
    parser.add_argument("--rebalance-tolerance", type=float, default=DEFAULT_REBALANCE_TOLERANCE)
    parser.add_argument("--min-trade-shares", type=int, default=DEFAULT_MIN_TRADE_SHARES)
    parser.add_argument("--flatten-time", type=str, default=DEFAULT_FLATTEN_TIME)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_single_close(input_root: Path, security: str, start_date: str, end_date: str) -> pd.Series | None:
    path = input_root / f"{security}.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if "trade_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["trade_time"])
        df = df.set_index("trade_time")
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "trade_time"

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    if df.empty or "close" not in df.columns:
        return None

    close = pd.to_numeric(df["close"], errors="coerce")
    close.name = security
    return close.sort_index()


def build_rebalance_times(interval_minutes: int, flatten_time: str) -> set[str]:
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be >= 1")

    result: set[str] = set()

    morning_hour = 9
    morning_minute = 30 + interval_minutes
    while morning_hour < 12:
        if morning_hour == 11 and morning_minute > 30:
            break
        result.add(f"{morning_hour:02d}:{morning_minute:02d}")
        morning_minute += interval_minutes
        while morning_minute >= 60:
            morning_hour += 1
            morning_minute -= 60

    afternoon_hour = 13
    afternoon_minute = interval_minutes
    while afternoon_hour < 15:
        result.add(f"{afternoon_hour:02d}:{afternoon_minute:02d}")
        afternoon_minute += interval_minutes
        while afternoon_minute >= 60:
            afternoon_hour += 1
            afternoon_minute -= 60

    result.add(flatten_time)
    return result


def floor_to_board_lot(shares: float, board_lot: int) -> int:
    return int(shares / float(board_lot)) * board_lot


def select_targets(
    scores: pd.Series,
    holdings: set[str],
    *,
    max_holdings: int,
    min_candidates: int,
    enter_pct: float,
    hold_pct: float,
    require_positive_signal: bool,
) -> list[str]:
    clean_scores = scores.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    if len(clean_scores) < min_candidates:
        return []

    filtered = clean_scores[clean_scores > 0] if require_positive_signal else clean_scores
    if filtered.empty:
        filtered = clean_scores
    if filtered.empty:
        return []

    enter_count = max(1, int(math.floor(len(filtered) * enter_pct)))
    hold_count = max(enter_count, int(math.floor(len(filtered) * hold_pct)))
    enter_count = min(enter_count, max_holdings)
    hold_count = min(max(hold_count, enter_count), len(filtered))

    enter_set = set(filtered.index[:enter_count])
    hold_set = set(filtered.index[:hold_count])
    selected = enter_set | (holdings & hold_set)

    ranked = filtered.to_dict()
    return sorted(selected, key=lambda security: ranked.get(security, -1e9), reverse=True)[:max_holdings]


def update_last_prices(last_prices: dict[str, float], price_row: pd.Series) -> None:
    clean = price_row.replace([np.inf, -np.inf], np.nan).dropna()
    for security, price in clean.items():
        if price > 0:
            last_prices[str(security)] = float(price)


def portfolio_value(positions: dict[str, int], last_prices: dict[str, float], cash: float) -> tuple[float, float]:
    holdings_value = 0.0
    for security, amount in positions.items():
        holdings_value += amount * last_prices.get(security, 0.0)
    return cash + holdings_value, holdings_value


def submit_order_target(
    *,
    trade_time: pd.Timestamp,
    security: str,
    target_amount: int,
    price: float,
    cash: float,
    positions: dict[str, int],
    min_trade_shares: int,
    min_order_value: float,
    open_rate: float,
    close_rate: float,
    portfolio_value_before: float,
    trade_logs: list[dict[str, object]],
) -> tuple[float, float, float]:
    current_amount = int(positions.get(security, 0))
    delta_amount = int(target_amount - current_amount)
    if abs(delta_amount) < min_trade_shares or price <= 0:
        return cash, 0.0, 0.0

    turnover = 0.0
    cost = 0.0
    base_value = max(portfolio_value_before, 1.0)

    if delta_amount > 0:
        additional_amount = floor_to_board_lot(delta_amount, min_trade_shares)
        if additional_amount < min_trade_shares:
            return cash, 0.0, 0.0

        max_affordable = floor_to_board_lot(cash / (price * (1.0 + open_rate)), min_trade_shares)
        buy_amount = min(additional_amount, max_affordable)
        buy_amount = floor_to_board_lot(buy_amount, min_trade_shares)
        if buy_amount < min_trade_shares:
            return cash, 0.0, 0.0

        trade_value = buy_amount * price
        if trade_value < min_order_value:
            return cash, 0.0, 0.0

        trade_cost = trade_value * open_rate
        cash -= trade_value + trade_cost
        positions[security] = current_amount + buy_amount
        turnover = trade_value / base_value
        cost = trade_cost
        trade_logs.append(
            {
                "trade_time": trade_time,
                "security": security,
                "side": "buy",
                "trade_amount": buy_amount,
                "trade_price": price,
                "trade_value": trade_value,
                "cost": trade_cost,
                "target_amount": target_amount,
            }
        )
        return cash, turnover, cost

    sell_amount = floor_to_board_lot(abs(delta_amount), min_trade_shares)
    sell_amount = min(sell_amount, floor_to_board_lot(current_amount, min_trade_shares))
    if sell_amount < min_trade_shares:
        return cash, 0.0, 0.0

    trade_value = sell_amount * price
    trade_cost = trade_value * close_rate
    cash += trade_value - trade_cost

    remaining = current_amount - sell_amount
    if remaining > 0:
        positions[security] = remaining
    else:
        positions.pop(security, None)

    turnover = trade_value / base_value
    cost = trade_cost
    trade_logs.append(
        {
            "trade_time": trade_time,
            "security": security,
            "side": "sell",
            "trade_amount": sell_amount,
            "trade_price": price,
            "trade_value": trade_value,
            "cost": trade_cost,
            "target_amount": target_amount,
        }
    )
    return cash, turnover, cost


def execute_rebalance(
    *,
    trade_time: pd.Timestamp,
    selected: list[str],
    price_row: pd.Series,
    cash: float,
    positions: dict[str, int],
    total_value: float,
    max_holdings: int,
    min_order_value: float,
    min_trade_shares: int,
    rebalance_tolerance: float,
    open_rate: float,
    close_rate: float,
    trade_logs: list[dict[str, object]],
) -> tuple[float, float, float]:
    del max_holdings
    turnover = 0.0
    total_cost = 0.0

    for security in list(positions.keys()):
        if security in selected:
            continue
        price = float(price_row.get(security, np.nan))
        if not np.isfinite(price) or price <= 0:
            continue
        cash, order_turnover, order_cost = submit_order_target(
            trade_time=trade_time,
            security=security,
            target_amount=0,
            price=price,
            cash=cash,
            positions=positions,
            min_trade_shares=min_trade_shares,
            min_order_value=min_order_value,
            open_rate=open_rate,
            close_rate=close_rate,
            portfolio_value_before=total_value,
            trade_logs=trade_logs,
        )
        turnover += order_turnover
        total_cost += order_cost

    if not selected:
        return cash, turnover, total_cost

    target_value = total_value / float(len(selected))
    if target_value < min_order_value:
        return cash, turnover, total_cost

    for security in selected:
        price = float(price_row.get(security, np.nan))
        if not np.isfinite(price) or price <= 0:
            continue

        target_amount = floor_to_board_lot(target_value / price, min_trade_shares)
        if target_amount < min_trade_shares:
            continue

        current_amount = int(positions.get(security, 0))
        current_value = current_amount * price
        deviation_ratio = abs(current_value - target_value) / max(target_value, 1.0)
        if current_amount > 0 and deviation_ratio < rebalance_tolerance:
            continue

        delta_amount = target_amount - current_amount
        if abs(delta_amount) < min_trade_shares:
            continue
        if delta_amount > 0 and delta_amount * price < min_order_value:
            continue

        cash, order_turnover, order_cost = submit_order_target(
            trade_time=trade_time,
            security=security,
            target_amount=target_amount,
            price=price,
            cash=cash,
            positions=positions,
            min_trade_shares=min_trade_shares,
            min_order_value=min_order_value,
            open_rate=open_rate,
            close_rate=close_rate,
            portfolio_value_before=total_value,
            trade_logs=trade_logs,
        )
        turnover += order_turnover
        total_cost += order_cost

    return cash, turnover, total_cost


def compute_summary(detail: pd.DataFrame, benchmark_code: str, periods_per_day: float, factor: str) -> BacktestSummary:
    if detail.empty:
        nan = float("nan")
        return BacktestSummary(
            factor=factor,
            total_return=nan,
            annual_return=nan,
            annual_volatility=nan,
            sharpe_ratio=nan,
            max_drawdown=nan,
            win_rate=nan,
            avg_turnover=nan,
            total_cost=nan,
            n_periods=0,
            avg_position_count=nan,
            benchmark_total_return=nan,
            benchmark_annual_return=nan,
            excess_total_return=nan,
            excess_annual_return=nan,
            alpha=nan,
            beta=nan,
            final_value=nan,
        )

    ann_factor = 252.0 * max(periods_per_day, 1.0)

    strategy_returns = detail["net_return"].fillna(0.0)
    benchmark_raw = detail["benchmark_return"]
    has_benchmark = bool(benchmark_raw.notna().any())
    benchmark_returns = benchmark_raw.fillna(0.0) if has_benchmark else pd.Series(np.nan, index=detail.index, dtype=float)

    clipped_strategy = strategy_returns.clip(lower=-0.999999)
    strategy_log_return = float(np.log1p(clipped_strategy).sum())
    total_return = float(np.expm1(np.clip(strategy_log_return, -700.0, 700.0)))
    annual_return = float(np.expm1(np.clip(strategy_log_return * ann_factor / len(detail), -700.0, 700.0)))
    annual_vol = float(strategy_returns.std(ddof=1) * math.sqrt(ann_factor)) if len(detail) > 1 else np.nan
    sharpe = (
        float(strategy_returns.mean() / strategy_returns.std(ddof=1) * math.sqrt(ann_factor))
        if len(detail) > 1 and strategy_returns.std(ddof=1) > 0
        else np.nan
    )

    equity = detail["equity"].to_numpy(dtype=float, copy=False)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    max_drawdown = float(np.nanmin(drawdown))
    win_rate = float((strategy_returns > 0).mean())

    if has_benchmark:
        clipped_benchmark = benchmark_returns.clip(lower=-0.999999)
        benchmark_log_return = float(np.log1p(clipped_benchmark).sum())
        benchmark_total_return = float(np.expm1(np.clip(benchmark_log_return, -700.0, 700.0)))
        benchmark_annual_return = float(np.expm1(np.clip(benchmark_log_return * ann_factor / len(detail), -700.0, 700.0)))

        excess_curve = (1.0 + strategy_returns).cumprod() / (1.0 + benchmark_returns).cumprod() - 1.0
        excess_total_return = float(excess_curve.iloc[-1])
        excess_log_return = float(np.log1p(excess_curve.iloc[-1])) if excess_curve.iloc[-1] > -1 else np.nan
        excess_annual_return = (
            float(np.expm1(np.clip(excess_log_return * ann_factor / len(detail), -700.0, 700.0)))
            if np.isfinite(excess_log_return)
            else np.nan
        )

        benchmark_var = float(benchmark_returns.var(ddof=1)) if len(detail) > 1 else np.nan
        beta = (
            float(strategy_returns.cov(benchmark_returns) / benchmark_var)
            if len(detail) > 1 and np.isfinite(benchmark_var) and benchmark_var > 0
            else np.nan
        )
        alpha = (
            float((strategy_returns.mean() - beta * benchmark_returns.mean()) * ann_factor)
            if np.isfinite(beta)
            else np.nan
        )
    else:
        benchmark_total_return = np.nan
        benchmark_annual_return = np.nan
        excess_total_return = np.nan
        excess_annual_return = np.nan
        alpha = np.nan
        beta = np.nan

    return BacktestSummary(
        factor=factor,
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        avg_turnover=float(detail["turnover"].mean()),
        total_cost=float(detail["cost"].sum()),
        n_periods=len(detail),
        avg_position_count=float(detail["position_count"].mean()),
        benchmark_total_return=benchmark_total_return,
        benchmark_annual_return=benchmark_annual_return,
        excess_total_return=excess_total_return,
        excess_annual_return=excess_annual_return,
        alpha=alpha,
        beta=beta,
        final_value=float(detail["equity"].iloc[-1]),
    )


def plot_equity(detail: pd.DataFrame, output_path: Path) -> None:
    if detail.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(detail["trade_time"], detail["equity"], label="strategy")

    if detail["benchmark_return"].notna().any():
        benchmark_equity = (1.0 + detail["benchmark_return"].fillna(0.0)).cumprod()
        benchmark_equity *= detail["equity"].iloc[0]
        ax.plot(detail["trade_time"], benchmark_equity, label="benchmark", alpha=0.8)
    ax.set_title("JoinQuant-Aligned ETF Equity Curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_logging()

    if args.hold_pct < args.enter_pct:
        raise ValueError("hold-pct must be >= enter-pct")

    files = discover_files(args.input_root, args.universe_file, args.limit)
    LOGGER.info("Loading %s ETF factor files", len(files))
    panel = load_panel(files, args.start_date, args.end_date)
    if args.factor not in panel.columns:
        raise ValueError(f"Factor not found in panel: {args.factor}")

    close_wide = build_wide_close(panel)
    factor_wide = panel[args.factor].unstack("ts_code").reindex(index=close_wide.index, columns=close_wide.columns)
    benchmark_close = close_wide.get(args.benchmark_code)
    if benchmark_close is None:
        benchmark_close = load_single_close(args.input_root, args.benchmark_code, args.start_date, args.end_date)
    if benchmark_close is None:
        LOGGER.warning("Benchmark code %s not found in loaded universe; benchmark metrics will be NaN", args.benchmark_code)
        benchmark_returns = pd.Series(np.nan, index=close_wide.index, dtype=float)
    else:
        benchmark_returns = benchmark_close.reindex(close_wide.index).ffill().pct_change().fillna(0.0)

    rebalance_times = build_rebalance_times(args.rebalance_interval, args.flatten_time)
    open_rate = args.open_commission + args.slippage
    close_rate = args.close_commission + args.slippage

    cash = float(args.initial_capital)
    positions: dict[str, int] = {}
    last_prices: dict[str, float] = {}
    trade_logs: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    last_equity = float(args.initial_capital)
    pending_selected: list[str] | None = None
    pending_selected_count = 0

    for trade_time in close_wide.index:
        price_row = close_wide.loc[trade_time]
        update_last_prices(last_prices, price_row)

        total_value, holdings_value = portfolio_value(positions, last_prices, cash)
        turnover = 0.0
        cost = 0.0
        selected_for_signal: list[str] = []

        clock = pd.Timestamp(trade_time).strftime("%H:%M")
        if pending_selected is not None:
            cash, turnover, cost = execute_rebalance(
                trade_time=pd.Timestamp(trade_time),
                selected=pending_selected,
                price_row=price_row,
                cash=cash,
                positions=positions,
                total_value=total_value,
                max_holdings=args.max_holdings,
                min_order_value=args.min_order_value,
                min_trade_shares=args.min_trade_shares,
                rebalance_tolerance=args.rebalance_tolerance,
                open_rate=open_rate,
                close_rate=close_rate,
                trade_logs=trade_logs,
            )
            pending_selected = None

        total_value, holdings_value = portfolio_value(positions, last_prices, cash)

        if clock == args.flatten_time:
            cash, turnover, cost = execute_rebalance(
                trade_time=pd.Timestamp(trade_time),
                selected=[],
                price_row=price_row,
                cash=cash,
                positions=positions,
                total_value=total_value,
                max_holdings=args.max_holdings,
                min_order_value=args.min_order_value,
                min_trade_shares=args.min_trade_shares,
                rebalance_tolerance=args.rebalance_tolerance,
                open_rate=open_rate,
                close_rate=close_rate,
                trade_logs=trade_logs,
            )
        elif clock in rebalance_times:
            scores = factor_wide.loc[trade_time]
            holdings = set(positions.keys())
            selected_for_signal = select_targets(
                scores=scores,
                holdings=holdings,
                max_holdings=args.max_holdings,
                min_candidates=args.min_candidates,
                enter_pct=args.enter_pct,
                hold_pct=args.hold_pct,
                require_positive_signal=args.require_positive_signal,
            )
            pending_selected = list(selected_for_signal)
            pending_selected_count = len(selected_for_signal)

        total_value, holdings_value = portfolio_value(positions, last_prices, cash)
        net_return = total_value / max(last_equity, 1e-12) - 1.0
        detail_rows.append(
            {
                "trade_time": pd.Timestamp(trade_time),
                "equity": total_value,
                "cash": cash,
                "holdings_value": holdings_value,
                "net_return": net_return,
                "benchmark_return": benchmark_returns.loc[trade_time] if trade_time in benchmark_returns.index else np.nan,
                "turnover": turnover,
                "cost": cost,
                "position_count": len(positions),
                "selected_count": pending_selected_count if pending_selected is not None else 0,
            }
        )
        last_equity = total_value

    detail = pd.DataFrame(detail_rows)
    if not detail.empty:
        detail["trade_time"] = pd.to_datetime(detail["trade_time"])

    grouped_days = pd.Series(close_wide.index.normalize()).value_counts()
    periods_per_day = float(np.median(grouped_days.to_numpy(dtype=float))) if len(grouped_days) else 1.0

    summary = compute_summary(detail, args.benchmark_code, periods_per_day, args.factor)
    summary.factor = args.factor
    summary_df = pd.DataFrame([asdict(summary)])

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_root / "joinquant_aligned_summary.csv", index=False)
    detail.to_csv(args.output_root / "joinquant_aligned_detail.csv", index=False)
    pd.DataFrame(trade_logs).to_csv(args.output_root / "joinquant_aligned_trades.csv", index=False)
    plot_equity(detail, args.output_root / "joinquant_aligned_equity_curve.png")

    settings = {
        "factor": args.factor,
        "benchmark_code": args.benchmark_code,
        "initial_capital": args.initial_capital,
        "open_rate": open_rate,
        "close_rate": close_rate,
        "rebalance_interval": args.rebalance_interval,
        "max_holdings": args.max_holdings,
        "min_candidates": args.min_candidates,
        "enter_pct": args.enter_pct,
        "hold_pct": args.hold_pct,
        "require_positive_signal": args.require_positive_signal,
        "min_order_value": args.min_order_value,
        "rebalance_tolerance": args.rebalance_tolerance,
        "min_trade_shares": args.min_trade_shares,
        "flatten_time": args.flatten_time,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "universe_file": str(args.universe_file),
    }
    (args.output_root / "joinquant_aligned_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Completed. Results written to %s", args.output_root)
    LOGGER.info("Summary:\n%s", summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
