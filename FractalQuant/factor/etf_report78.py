"""ETF factors motivated by Kysec *Quant Comment 78*.

The report evaluates stock-level inputs and then maps them to industry
indices/ETFs.  The local ETF source contains minute OHLCV and (for the
current files) minute amount, so this module deliberately implements only
the four technical fields that can be calculated from that source.  The
result is an ETF-level proxy, not a claim that it reproduces the Wind
stock/industry calculation.

All functions are point-in-time: a row for trade date ``d`` uses only bars
from ``d`` (and prior completed days for long momentum).  Callers should
shift the resulting row to the next trading day before execution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REPORT78_FACTOR_COLUMNS = (
    "technical_inday_volume_ratio_log",
    "technical_cvilliq",
    "technical_qua",
    "technical_long_mom",
)


def _minute_amount(frame: pd.DataFrame) -> pd.Series:
    """Return amount in yuan, or all-NaN when the source has no amount.

    ``amount`` is preferred because ETF parquet files already store traded
    notional.  We do not silently manufacture amount from volume: the report
    defines its liquidity and QUA inputs using money, and a missing amount is
    safer than a unit-mismatched substitute.
    """

    if "amount" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    return amount.where(amount > 0)


def calculate_daily_report78_technical_factors(
    minute_frame: pd.DataFrame,
) -> dict[str, float]:
    """Calculate the four report-78 technical ETF proxies for one day.

    Definitions used here are explicit approximations of the report labels:

    * ``inday_volume_ratio_log``: ``log1p(downside RV / total RV)``;
    * ``cvilliq``: coefficient of variation of minute Amihud illiquidity
      ``abs(return) / amount``;
    * ``qua``: median percentile rank of minute traded amount (minute data is
      aggregated, so this is not a true tick/single-trade percentile);
    * ``long_mom`` is populated by :func:`build_report78_daily_panel`, where
      completed daily closes are available.
    """

    if minute_frame.empty:
        return {name: np.nan for name in REPORT78_FACTOR_COLUMNS[:-1]}

    close = pd.to_numeric(minute_frame["close"], errors="coerce")
    returns = close.pct_change()
    valid_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if valid_returns.empty:
        downside_ratio_log = np.nan
    else:
        total_rv = float(np.square(valid_returns).sum())
        downside_rv = float(np.square(valid_returns[valid_returns < 0]).sum())
        downside_ratio_log = (
            float(np.log1p(downside_rv / total_rv)) if total_rv > 0 else np.nan
        )

    amount = _minute_amount(minute_frame)
    illiquidity = (valid_returns.abs() / amount.reindex(valid_returns.index)).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(illiquidity) < 2 or float(illiquidity.mean()) <= 0:
        cvilliq = np.nan
    else:
        cvilliq = float(illiquidity.std(ddof=0) / illiquidity.mean())

    valid_amount = amount.dropna()
    if valid_amount.empty:
        qua = np.nan
    else:
        qua = float(valid_amount.rank(method="average", pct=True).median())

    return {
        "technical_inday_volume_ratio_log": downside_ratio_log,
        "technical_cvilliq": cvilliq,
        "technical_qua": qua,
    }


def build_report78_daily_panel(
    minute_frame: pd.DataFrame,
    *,
    long_momentum_window: int = 20,
) -> pd.DataFrame:
    """Aggregate normalized ETF minute bars into a causal daily factor panel.

    The returned index is ``trade_date``.  The long momentum value on date
    ``d`` is the close-to-close return from ``d-window`` to ``d``; no future
    rows are used.  Empty or malformed days are retained with null factors so
    coverage gaps remain observable.
    """

    if not isinstance(minute_frame.index, pd.DatetimeIndex):
        raise TypeError("minute_frame must have a DatetimeIndex")
    if long_momentum_window < 1:
        raise ValueError("long_momentum_window must be positive")
    required = {"close"}
    missing = sorted(required - set(minute_frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = minute_frame.sort_index()
    rows: list[dict[str, float | pd.Timestamp]] = []
    daily_closes: list[float] = []
    dates: list[pd.Timestamp] = []
    for trade_date, day in frame.groupby(frame.index.normalize(), sort=True):
        values = calculate_daily_report78_technical_factors(day)
        close = pd.to_numeric(day["close"], errors="coerce").dropna()
        last_close = float(close.iloc[-1]) if not close.empty else np.nan
        dates.append(pd.Timestamp(trade_date))
        daily_closes.append(last_close)
        rows.append({"trade_date": pd.Timestamp(trade_date), **values})

    if not rows:
        return pd.DataFrame(columns=["trade_date", *REPORT78_FACTOR_COLUMNS]).set_index(
            "trade_date"
        )

    result = pd.DataFrame(rows).set_index("trade_date")
    closes = pd.Series(daily_closes, index=pd.DatetimeIndex(dates), dtype=float)
    result["technical_long_mom"] = closes.pct_change(long_momentum_window)
    return result[list(REPORT78_FACTOR_COLUMNS)]

