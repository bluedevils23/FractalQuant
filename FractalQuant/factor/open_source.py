"""Open-source microstructure factors that are reproducible from minute bars.

The research sources behind these factors use stock-level transaction data in
some cases.  This module intentionally exposes only minute OHLCV/amount
proxies that can be calculated from the local ETF data.  No order-book quote or
bar is treated as a substitute for trade-level size-flow data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


OPEN_SOURCE_FACTOR_COLUMNS = (
    "kaiyuan_ideal_amplitude_m20_q25",
    "kaiyuan_smart_money_vwap_gap_beta01_m10",
    "kaiyuan_ideal_reversal_bar_proxy_m20",
    "kaiyuan_err_m20",
    "kaiyuan_tgd_m20",
)

AUDIT_COLUMNS = (
    "trade_date",
    "available_date",
    "available_time",
    "ts_code",
    "valid_session_bars",
    "amount_available",
    "smart_money_valid_days_m10",
    "source_level",
    "tgd_cross_section_count",
)

RAW_COLUMNS = (
    "trade_date",
    "ts_code",
    "daily_close",
    "daily_return",
    "daily_amplitude",
    "daily_amount",
    "smart_money_vwap_gap_m10",
    "smart_money_valid_days_m10",
    "extreme_return_m20",
    "extreme_prior_return_m20",
    "gu",
    "gd",
    "avg_up_return",
    "avg_down_return",
    "r1",
    "r2",
    "overnight_return",
    "valid_session_bars",
    "amount_available",
)

SESSION_BAR_COUNT = 240
SMART_MONEY_WINDOW = 10
FACTOR_WINDOW = 20
SMART_MONEY_VOLUME_FRACTION = 0.20
MIN_TGD_CROSS_SECTION = 20


def normalize_minute_frame(
    raw_df: pd.DataFrame, ts_code_hint: str | None = None
) -> pd.DataFrame:
    """Normalize local minute parquet layouts to a datetime-indexed frame."""

    frame = raw_df.copy()
    if isinstance(frame.index, pd.MultiIndex):
        names = set(frame.index.names)
        if "trade_time" in names:
            frame = frame.reset_index()
        else:
            frame = frame.reset_index(drop=True)

    if "trade_time" in frame.columns:
        frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
        frame = frame.set_index("trade_time")
    elif "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.set_index("datetime")
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    else:
        raise ValueError("Cannot locate trade_time/datetime or a DatetimeIndex")

    frame.index.name = "trade_time"
    frame = frame.rename(columns={"vol": "volume"})
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required minute columns: {missing}")

    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "ts_code" not in frame.columns and ts_code_hint is not None:
        frame["ts_code"] = str(ts_code_hint)
    elif "ts_code" in frame.columns:
        frame["ts_code"] = frame["ts_code"].astype(str)

    frame = frame.sort_index()
    frame = frame.loc[~frame.index.isna()]
    return frame.loc[~frame.index.duplicated(keep="last")]


def _regular_session_frame(day: pd.DataFrame) -> pd.DataFrame:
    """Return the 240 close-labelled regular-session bars used by the reports."""

    times = day.index.strftime("%H:%M")
    mask = ((times >= "09:31") & (times <= "11:30")) | (
        (times >= "13:01") & (times <= "15:00")
    )
    return day.loc[mask].sort_index()


def _positive_frame(day: pd.DataFrame) -> pd.DataFrame:
    columns = ["open", "high", "low", "close", "volume"]
    if "amount" in day.columns:
        columns.append("amount")
    work = day[columns].copy()
    for column in ("open", "high", "low", "close"):
        work.loc[work[column] <= 0, column] = np.nan
    work.loc[work["volume"] < 0, "volume"] = np.nan
    if "amount" in work.columns:
        work.loc[work["amount"] < 0, "amount"] = np.nan
    return work


def _daily_return_series(session: pd.DataFrame) -> pd.Series:
    return session["close"].pct_change().replace([np.inf, -np.inf], np.nan)


def _time_centers(returns: pd.Series) -> tuple[float, float, float, float]:
    values = returns.to_numpy(dtype=float, copy=False)
    positions = np.arange(1, len(values) + 1, dtype=float)
    finite = np.isfinite(values)
    up = finite & (values > 0)
    down = finite & (values < 0)
    if not up.any():
        gu = np.nan
        avg_up = np.nan
    else:
        up_values = values[up]
        gu = float(np.sum(positions[up] * up_values) / np.sum(up_values))
        avg_up = float(np.mean(up_values))
    if not down.any():
        gd = np.nan
        avg_down = np.nan
    else:
        down_values = np.abs(values[down])
        gd = float(np.sum(positions[down] * down_values) / np.sum(down_values))
        avg_down = float(-np.mean(down_values))
    return gu, gd, avg_up, avg_down


def _extreme_returns(returns: pd.Series) -> tuple[float, float]:
    finite_returns = returns.replace([np.inf, -np.inf], np.nan)
    if finite_returns.notna().sum() < 2:
        return np.nan, np.nan
    median = float(finite_returns.median())
    distances = (finite_returns - median).abs()
    extreme_index = distances.idxmax()
    extreme_position = returns.index.get_loc(extreme_index)
    extreme_value = float(finite_returns.loc[extreme_index])
    if extreme_position <= 0:
        return extreme_value, np.nan
    previous_value = returns.iloc[extreme_position - 1]
    return extreme_value, float(previous_value) if np.isfinite(previous_value) else np.nan


def _smart_money_gap_for_sessions(
    sessions: list[pd.DataFrame],
    *,
    volume_fraction: float = SMART_MONEY_VOLUME_FRACTION,
) -> tuple[float, int]:
    if len(sessions) == 0:
        return np.nan, 0
    rows: list[pd.DataFrame] = []
    for session in sessions:
        if len(session) != SESSION_BAR_COUNT:
            continue
        returns = _daily_return_series(session)
        volume = pd.to_numeric(session["volume"], errors="coerce")
        close = pd.to_numeric(session["close"], errors="coerce")
        valid = returns.notna() & volume.gt(0) & close.gt(0)
        if valid.any():
            part = pd.DataFrame(
                {
                    "close": close.loc[valid],
                    "volume": volume.loc[valid],
                    "score": (returns.loc[valid].abs() / volume.loc[valid].pow(0.1)),
                }
            )
            rows.append(part.replace([np.inf, -np.inf], np.nan).dropna())
    if not rows:
        return np.nan, 0
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.loc[combined["volume"].gt(0)]
    total_volume = float(combined["volume"].sum())
    if total_volume <= 0 or combined.empty:
        return np.nan, 0
    combined = combined.sort_values("score", ascending=False, kind="mergesort")
    target = total_volume * float(volume_fraction)
    cumulative = combined["volume"].cumsum()
    selected = combined.loc[cumulative.sub(combined["volume"]).lt(target)]
    if selected.empty:
        selected = combined.iloc[[0]]
    all_vwap = float((combined["close"] * combined["volume"]).sum() / total_volume)
    smart_volume = float(selected["volume"].sum())
    if smart_volume <= 0 or all_vwap == 0:
        return np.nan, len(rows)
    smart_vwap = float(
        (selected["close"] * selected["volume"]).sum() / smart_volume
    )
    return smart_vwap / all_vwap - 1.0, len(rows)


def _rolling_cut_difference(
    values: pd.Series,
    state: pd.Series,
    window: int,
    fraction: float,
) -> pd.Series:
    result = np.full(len(values), np.nan, dtype=float)
    group_size = max(1, int(np.ceil(window * fraction)))
    value_array = values.to_numpy(dtype=float, copy=False)
    state_array = state.to_numpy(dtype=float, copy=False)
    for end in range(window - 1, len(values)):
        start = end - window + 1
        window_values = value_array[start : end + 1]
        window_state = state_array[start : end + 1]
        if not (np.isfinite(window_values).all() and np.isfinite(window_state).all()):
            continue
        order = np.argsort(window_state, kind="mergesort")
        low = order[:group_size]
        high = order[-group_size:]
        result[end] = float(np.mean(window_values[high]) - np.mean(window_values[low]))
    return pd.Series(result, index=values.index)


def build_daily_raw_features(
    minute_frame: pd.DataFrame,
    ts_code: str,
    *,
    factor_window: int = FACTOR_WINDOW,
    smart_money_window: int = SMART_MONEY_WINDOW,
) -> pd.DataFrame:
    """Build one symbol's daily raw and per-symbol rolling features."""

    frame = normalize_minute_frame(minute_frame, ts_code)
    if frame.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    daily_records: list[dict[str, Any]] = []
    sessions: list[pd.DataFrame] = []
    dates: list[pd.Timestamp] = []
    for trade_date, raw_day in frame.groupby(frame.index.normalize(), sort=True):
        day = _positive_frame(raw_day)
        session = _regular_session_frame(day)
        sessions.append(session)
        dates.append(pd.Timestamp(trade_date))
        close = day["close"].dropna()
        daily_close = float(close.iloc[-1]) if not close.empty else np.nan
        high = day["high"].max(skipna=True)
        low = day["low"].min(skipna=True)
        daily_amplitude = (
            float(high / low - 1.0) if np.isfinite(high) and np.isfinite(low) and low > 0 else np.nan
        )
        daily_amount = (
            float(day["amount"].sum(min_count=1)) if "amount" in day else np.nan
        )
        if len(session) == SESSION_BAR_COUNT:
            session_returns = _daily_return_series(session)
            gu, gd, avg_up, avg_down = _time_centers(session_returns)
            extreme_return, prior_return = _extreme_returns(session_returns)
            r1 = float(session_returns.iloc[:30].sum(min_count=1))
            r2 = float(session_returns.iloc[30:60].sum(min_count=1))
            session_open_candidates = raw_day["open"].dropna()
            session_open = (
                float(session_open_candidates.iloc[0])
                if not session_open_candidates.empty
                else np.nan
            )
            amount_available = bool("amount" in day and day["amount"].notna().any())
        else:
            gu = gd = avg_up = avg_down = np.nan
            extreme_return = prior_return = r1 = r2 = np.nan
            session_open = np.nan
            amount_available = bool("amount" in day and day["amount"].notna().any())
        daily_records.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "ts_code": str(ts_code),
                "daily_close": daily_close,
                "daily_return": np.nan,
                "daily_amplitude": daily_amplitude,
                "daily_amount": daily_amount,
                "smart_money_vwap_gap_m10": np.nan,
                "smart_money_valid_days_m10": 0,
                "extreme_return_m20": np.nan,
                "extreme_prior_return_m20": np.nan,
                "gu": gu,
                "gd": gd,
                "avg_up_return": avg_up,
                "avg_down_return": avg_down,
                "r1": r1,
                "r2": r2,
                "overnight_return": np.nan,
                "valid_session_bars": int(len(session)),
                "amount_available": amount_available,
                "_session_open": session_open,
            }
        )

    result = pd.DataFrame(daily_records).sort_values("trade_date").reset_index(drop=True)
    result["daily_return"] = result["daily_close"].div(result["daily_close"].shift(1)).sub(1.0)
    result["overnight_return"] = result["_session_open"].div(result["daily_close"].shift(1)).sub(1.0)
    raw_extreme: list[float] = []
    raw_prior: list[float] = []
    for session in sessions:
        if len(session) == SESSION_BAR_COUNT:
            raw_extreme.append(_extreme_returns(_daily_return_series(session))[0])
            raw_prior.append(_extreme_returns(_daily_return_series(session))[1])
        else:
            raw_extreme.append(np.nan)
            raw_prior.append(np.nan)
    result["_extreme_return_raw"] = raw_extreme
    result["_extreme_prior_return_raw"] = raw_prior
    result["extreme_return_m20"] = (
        result.groupby("ts_code", sort=False)["_extreme_return_raw"]
        .transform(lambda series: series.rolling(factor_window, min_periods=factor_window).mean())
    )
    result["extreme_prior_return_m20"] = (
        result.groupby("ts_code", sort=False)["_extreme_prior_return_raw"]
        .transform(lambda series: series.rolling(factor_window, min_periods=factor_window).mean())
    )

    ideal_amplitude = _rolling_cut_difference(
        result["daily_amplitude"], result["daily_close"], factor_window, 0.25
    )
    result["_ideal_amplitude"] = ideal_amplitude
    result["_ideal_reversal"] = _rolling_cut_difference(
        result["daily_return"], result["daily_amount"], factor_window, 0.5
    )

    smart_gaps: list[float] = []
    smart_days: list[int] = []
    for end in range(len(sessions)):
        start = max(0, end - smart_money_window + 1)
        gap, valid_days = _smart_money_gap_for_sessions(
            sessions[start : end + 1]
        )
        if valid_days < smart_money_window:
            gap = np.nan
        smart_gaps.append(gap)
        smart_days.append(valid_days)
    result["smart_money_vwap_gap_m10"] = smart_gaps
    result["smart_money_valid_days_m10"] = smart_days
    result["_source_level"] = np.where(
        result["amount_available"], "minute_amount_proxy", "minute_ohlcv_proxy"
    )
    result = result.drop(columns=["_session_open"])
    return result


def _cross_section_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("trade_date", sort=False)[column].rank(
        method="average", pct=True
    )


def _cross_section_residual(
    frame: pd.DataFrame,
    target: str,
    predictors: Iterable[str],
    min_count: int,
) -> tuple[pd.Series, pd.Series]:
    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0, index=frame.index, dtype="int64")
    predictors = tuple(predictors)
    for _, group in frame.groupby("trade_date", sort=False):
        columns = [target, *predictors]
        valid = group[columns].notna().all(axis=1)
        sample = group.loc[valid]
        counts.loc[group.index] = int(len(sample))
        if len(sample) < min_count:
            continue
        x = np.column_stack([np.ones(len(sample)), sample.loc[:, predictors].to_numpy(float)])
        y = sample[target].to_numpy(float)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual.loc[sample.index] = y - x @ beta
    return residual, counts


def build_open_source_factor_panel(
    raw_panel: pd.DataFrame,
    *,
    factor_window: int = FACTOR_WINDOW,
    min_tgd_cross_section: int = MIN_TGD_CROSS_SECTION,
) -> pd.DataFrame:
    """Add the five open-source factors to a combined daily raw panel."""

    required = set(RAW_COLUMNS)
    missing = sorted(required - set(raw_panel.columns))
    if missing:
        raise ValueError(f"Missing raw open-source columns: {missing}")
    frame = raw_panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame = frame.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    frame["kaiyuan_ideal_amplitude_m20_q25"] = frame["_ideal_amplitude"]
    frame["kaiyuan_ideal_reversal_bar_proxy_m20"] = frame["_ideal_reversal"]
    frame["kaiyuan_smart_money_vwap_gap_beta01_m10"] = frame["smart_money_vwap_gap_m10"]

    extreme_rank = _cross_section_rank(frame, "extreme_return_m20")
    prior_rank = _cross_section_rank(frame, "extreme_prior_return_m20")
    frame["kaiyuan_err_m20"] = (extreme_rank + prior_rank).where(
        extreme_rank.notna() & prior_rank.notna()
    )

    first_residual, first_counts = _cross_section_residual(
        frame,
        "gu",
        ("avg_up_return", "avg_down_return", "r1", "r2", "overnight_return"),
        min_tgd_cross_section,
    )
    second_input = frame.copy()
    second_input["_gu_residual"] = first_residual
    second_input["_gd_residual"] = _cross_section_residual(
        frame,
        "gd",
        ("avg_up_return", "avg_down_return", "r1", "r2", "overnight_return"),
        min_tgd_cross_section,
    )[0]
    second_residual, second_counts = _cross_section_residual(
        second_input.rename(columns={"_gd_residual": "gd_residual"}).assign(
            gu_residual=second_input["_gu_residual"]
        ),
        "gd_residual",
        ("gu_residual",),
        min_tgd_cross_section,
    )
    frame["_tgd_daily_residual"] = second_residual
    frame["tgd_cross_section_count"] = np.maximum(first_counts, second_counts)
    frame["kaiyuan_tgd_m20"] = frame.groupby("ts_code", sort=False)[
        "_tgd_daily_residual"
    ].transform(lambda s: s.rolling(factor_window, min_periods=factor_window).mean())
    frame.loc[frame["tgd_cross_section_count"] < min_tgd_cross_section, "kaiyuan_tgd_m20"] = np.nan

    source_level = frame.get("_source_level", "minute_ohlcv_proxy")
    frame["source_level"] = source_level
    frame = frame.drop(
        columns=[
            column
            for column in (
                "_ideal_amplitude",
                "_ideal_reversal",
                "_extreme_return_raw",
                "_extreme_prior_return_raw",
                "_tgd_daily_residual",
                "_source_level",
            )
            if column in frame.columns
        ]
    )
    return frame.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)


def add_availability_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add next-observed-trading-day availability metadata per symbol."""

    frame = panel.copy().sort_values(["ts_code", "trade_date"], kind="mergesort")
    frame["available_date"] = frame.groupby("ts_code", sort=False)["trade_date"].shift(-1)
    frame["available_time"] = pd.to_datetime(frame["available_date"], errors="coerce") + pd.Timedelta(
        hours=9, minutes=30
    )
    return frame


def select_output_columns(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_date",
        "available_date",
        "available_time",
        "ts_code",
        *OPEN_SOURCE_FACTOR_COLUMNS,
        "valid_session_bars",
        "amount_available",
        "smart_money_valid_days_m10",
        "source_level",
        "tgd_cross_section_count",
    ]
    result = panel.reindex(columns=columns).copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
