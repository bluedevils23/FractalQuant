"""Causal intraday factors adapted from the priority research-report set."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .intraday_strategy_p1 import P1_FACTOR_COLUMNS, build_p1_factor_frame


OPENING_PATH_BARS = 50
OPENING_POLY_BARS = 60
GFTD_N1 = 5
GFTD_N2 = 3
GFTD_N3 = 6
DELAYED_EXTREME_LAG = 2
DELAYED_EXTREME_CUTOFF = 15
ATR_ORB_MULTIPLIER = 0.4
DUAL_THRUST_MULTIPLIER = 0.2

OPENING_PATH_FACTOR_COLUMNS = [
    "opening_path_mean_drawdown_50bar",
    "opening_path_mean_reverse_drawdown_50bar",
    "opening_path_smoothness_50bar",
]
OPENING_POLY_FACTOR_COLUMNS = [
    "opening_poly_slope_60bar",
    "opening_poly_curvature_60bar",
    "opening_poly_trend_acceleration_60bar",
]
GFTD_FACTOR_COLUMNS = [
    "gftd_setup_direction_5_3",
    "gftd_setup_streak_5",
    "gftd_buy_count_6",
    "gftd_sell_count_6",
    "gftd_signal_state_5_3_6",
]
BREAKOUT_FACTOR_COLUMNS = [
    "atr10_orb_width_to_open",
    "dual_thrust_drange5_to_open",
    "distance_to_atr10_orb_upper",
    "distance_to_atr10_orb_lower",
    "distance_to_dual_thrust_upper",
    "distance_to_dual_thrust_lower",
]
TAIL_FACTOR_COLUMNS = ["prev30d_open_close_return_pearson_kurtosis"]
DELAYED_EXTREME_FACTOR_COLUMNS = [
    "distance_to_delayed_session_high_lag2",
    "distance_to_delayed_session_low_lag2",
    "delayed_session_extreme_breakout_state_lag2",
]
FACTOR_COLUMNS = (
    OPENING_PATH_FACTOR_COLUMNS
    + OPENING_POLY_FACTOR_COLUMNS
    + GFTD_FACTOR_COLUMNS
    + BREAKOUT_FACTOR_COLUMNS
    + TAIL_FACTOR_COLUMNS
    + DELAYED_EXTREME_FACTOR_COLUMNS
)
PRIORITY_PROFILE_P0 = "p0"
PRIORITY_PROFILE_P0_P1 = "p0_p1"
PRIORITY_PROFILES = (PRIORITY_PROFILE_P0, PRIORITY_PROFILE_P0_P1)
COMBINED_FACTOR_COLUMNS = FACTOR_COLUMNS + P1_FACTOR_COLUMNS
KEY_COLUMNS = ["trade_date", "trade_time", "available_time", "ts_code"]
SOURCE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "adj_factor",
]
OUTPUT_COLUMNS = KEY_COLUMNS + SOURCE_COLUMNS + FACTOR_COLUMNS
COMBINED_OUTPUT_COLUMNS = KEY_COLUMNS + SOURCE_COLUMNS + COMBINED_FACTOR_COLUMNS
DAILY_CONTEXT_COLUMNS = [
    "trade_date",
    "avg_true_range_10",
    "dual_thrust_drange_5",
    "prev30d_open_close_return_pearson_kurtosis",
]


def factor_columns_for_profile(priority_profile: str) -> list[str]:
    """Return factor columns without changing the legacy P0 registry."""
    if priority_profile == PRIORITY_PROFILE_P0:
        return list(FACTOR_COLUMNS)
    if priority_profile == PRIORITY_PROFILE_P0_P1:
        return list(COMBINED_FACTOR_COLUMNS)
    raise ValueError(
        f"Unsupported priority profile {priority_profile!r}; expected one of {PRIORITY_PROFILES}"
    )


def output_columns_for_profile(priority_profile: str) -> list[str]:
    if priority_profile == PRIORITY_PROFILE_P0:
        return list(OUTPUT_COLUMNS)
    if priority_profile == PRIORITY_PROFILE_P0_P1:
        return list(COMBINED_OUTPUT_COLUMNS)
    raise ValueError(
        f"Unsupported priority profile {priority_profile!r}; expected one of {PRIORITY_PROFILES}"
    )


def normalize_minute_frame(raw: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """Normalize a per-symbol minute parquet frame without changing bar timestamps."""
    work = raw.copy()
    if "volume" in work.columns and "vol" not in work.columns:
        work = work.rename(columns={"volume": "vol"})
    if isinstance(work.index, pd.MultiIndex) or work.index.name in {
        "trade_date",
        "trade_time",
    }:
        work = work.reset_index()
    if "trade_time" not in work.columns:
        raise ValueError("Minute frame must expose trade_time")
    if "trade_date" not in work.columns:
        work["trade_date"] = pd.to_datetime(
            work["trade_time"], errors="coerce"
        ).dt.normalize()

    missing = [
        column for column in ("open", "high", "low", "close") if column not in work
    ]
    if missing:
        raise ValueError(f"Minute frame is missing required columns: {missing}")

    work["trade_date"] = pd.to_datetime(
        work["trade_date"], errors="coerce"
    ).dt.normalize()
    work["trade_time"] = pd.to_datetime(work["trade_time"], errors="coerce")
    work = work.dropna(subset=["trade_date", "trade_time"])
    work["ts_code"] = str(ts_code).upper()

    for column in ("open", "high", "low", "close", "vol", "amount", "adj_factor"):
        if column not in work.columns:
            work[column] = 1.0 if column == "adj_factor" else np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.sort_values(["trade_date", "trade_time"], kind="mergesort")
    work = work.drop_duplicates(["trade_date", "trade_time"], keep="last")
    return work.reset_index(drop=True)


def _prepare_adjusted_minute(raw: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    work = normalize_minute_frame(raw, ts_code)
    factor = work["adj_factor"].where(work["adj_factor"].gt(0))
    for column in ("open", "high", "low", "close"):
        values = work[column].where(work[column].gt(0))
        work[f"_adj_{column}"] = values * factor
    return work


def normalize_daily_frame(raw: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """Normalize and adjust one symbol's daily history."""
    work = raw.copy()
    if isinstance(work.index, pd.MultiIndex) or work.index.name == "trade_date":
        work = work.reset_index()
    if "trade_date" not in work.columns:
        raise ValueError("Daily frame must expose trade_date")
    if "ts_code" in work.columns:
        work = work.loc[work["ts_code"].astype(str).str.upper().eq(ts_code.upper())]

    missing = [
        column for column in ("open", "high", "low", "close") if column not in work
    ]
    if missing:
        raise ValueError(f"Daily frame is missing required columns: {missing}")

    work["trade_date"] = pd.to_datetime(
        work["trade_date"], errors="coerce"
    ).dt.normalize()
    if "adj_factor" not in work.columns:
        work["adj_factor"] = 1.0
    for column in ("open", "high", "low", "close", "adj_factor"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["trade_date"])
    work = work.sort_values("trade_date", kind="mergesort")
    work = work.drop_duplicates("trade_date", keep="last").reset_index(drop=True)

    factor = work["adj_factor"].where(work["adj_factor"].gt(0))
    for column in ("open", "high", "low", "close"):
        values = work[column].where(work[column].gt(0))
        work[f"_adj_{column}"] = values * factor
    return work


def _pearson_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 4 or not np.isfinite(values).all():
        return np.nan
    if np.std(values, ddof=0) <= 0:
        return np.nan
    return float(stats.kurtosis(values, fisher=False, bias=False))


def build_daily_context(
    daily: pd.DataFrame,
    target_dates: Iterable[pd.Timestamp | str],
    ts_code: str,
) -> pd.DataFrame:
    """Build strictly historical daily context for each requested trade date."""
    targets = pd.DatetimeIndex(
        pd.to_datetime(list(target_dates), errors="coerce")
    ).normalize()
    targets = targets[~targets.isna()].unique().sort_values()
    if len(targets) == 0:
        return pd.DataFrame(columns=DAILY_CONTEXT_COLUMNS)

    if daily.empty:
        result = pd.DataFrame({"trade_date": targets})
        for column in DAILY_CONTEXT_COLUMNS[1:]:
            result[column] = np.nan
        return result

    work = normalize_daily_frame(daily, ts_code)
    if work.empty:
        result = pd.DataFrame({"trade_date": targets})
        for column in DAILY_CONTEXT_COLUMNS[1:]:
            result[column] = np.nan
        return result

    previous_close = work["_adj_close"].shift(1)
    true_high = pd.concat([work["_adj_high"], previous_close], axis=1).max(
        axis=1, skipna=False
    )
    true_low = pd.concat([work["_adj_low"], previous_close], axis=1).min(
        axis=1, skipna=False
    )
    true_range = true_high - true_low
    work["_atr10_end"] = true_range.rolling(10, min_periods=10).mean()

    highest_high = work["_adj_high"].rolling(5, min_periods=5).max()
    lowest_low = work["_adj_low"].rolling(5, min_periods=5).min()
    highest_close = work["_adj_close"].rolling(5, min_periods=5).max()
    lowest_close = work["_adj_close"].rolling(5, min_periods=5).min()
    work["_drange5_end"] = np.maximum(
        highest_high - lowest_close,
        highest_close - lowest_low,
    )

    intraday_return = work["_adj_close"] / work["_adj_open"] - 1.0
    work["_kurtosis30_end"] = intraday_return.rolling(30, min_periods=30).apply(
        _pearson_kurtosis, raw=True
    )

    history_dates = work["trade_date"].to_numpy(dtype="datetime64[ns]")
    records: list[dict[str, object]] = []
    for target in targets:
        position = int(
            np.searchsorted(history_dates, target.to_datetime64(), side="left") - 1
        )
        record: dict[str, object] = {"trade_date": target}
        if position < 0:
            record.update(
                {
                    "avg_true_range_10": np.nan,
                    "dual_thrust_drange_5": np.nan,
                    "prev30d_open_close_return_pearson_kurtosis": np.nan,
                }
            )
        else:
            row = work.iloc[position]
            record.update(
                {
                    "avg_true_range_10": row["_atr10_end"],
                    "dual_thrust_drange_5": row["_drange5_end"],
                    "prev30d_open_close_return_pearson_kurtosis": row[
                        "_kurtosis30_end"
                    ],
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=DAILY_CONTEXT_COLUMNS)


def _path_smoothness(prices: np.ndarray) -> tuple[float, float, float]:
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0 or not np.isfinite(prices).all() or np.any(prices <= 0):
        return np.nan, np.nan, np.nan
    drawdowns = np.zeros(len(prices), dtype=float)
    reverse_drawdowns = np.zeros(len(prices), dtype=float)
    for index in range(len(prices) - 1):
        future = prices[index + 1 :]
        drawdowns[index] = max(
            0.0, float(np.max((prices[index] - future) / prices[index]))
        )
        reverse_drawdowns[index] = max(
            0.0, float(np.max((future - prices[index]) / prices[index]))
        )
    mean_drawdown = float(drawdowns.mean())
    mean_reverse = float(reverse_drawdowns.mean())
    return mean_drawdown, mean_reverse, min(mean_drawdown, mean_reverse)


def _opening_indices(day: pd.DataFrame) -> np.ndarray:
    times = day["trade_time"].dt.time
    morning = times >= pd.Timestamp("09:30").time()
    morning &= times <= pd.Timestamp("11:30").time()
    return np.flatnonzero(morning.to_numpy())


def _apply_opening_factors(day: pd.DataFrame, factors: pd.DataFrame) -> None:
    opening_positions = _opening_indices(day)
    adjusted_close = day["_adj_close"].to_numpy(dtype=float)

    if len(opening_positions) >= OPENING_PATH_BARS:
        window_positions = opening_positions[:OPENING_PATH_BARS]
        values = _path_smoothness(adjusted_close[window_positions])
        available_position = int(window_positions[-1])
        factors.loc[available_position:, OPENING_PATH_FACTOR_COLUMNS] = values

    if len(opening_positions) >= OPENING_POLY_BARS:
        window_positions = opening_positions[:OPENING_POLY_BARS]
        prices = adjusted_close[window_positions]
        if np.isfinite(prices).all() and np.all(prices > 0):
            x = np.linspace(-1.0, 1.0, OPENING_POLY_BARS)
            y = np.log(prices / prices[0])
            linear_slope = float(np.polyfit(x, y, 1)[0])
            quadratic = np.polyfit(x, y, 2)
            curvature = float(2.0 * quadratic[0])
            values = (linear_slope, curvature, linear_slope * curvature)
            available_position = int(window_positions[-1])
            factors.loc[available_position:, OPENING_POLY_FACTOR_COLUMNS] = values


def calculate_gftd_state(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    n1: int = GFTD_N1,
    n2: int = GFTD_N2,
    n3: int = GFTD_N3,
) -> pd.DataFrame:
    """Calculate the GFTD setup/count state for one uninterrupted session."""
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    if not (len(close) == len(high) == len(low)):
        raise ValueError("GFTD input arrays must have equal length")
    if min(n1, n2, n3) <= 0:
        raise ValueError("GFTD parameters must be positive")

    output = np.zeros((len(close), len(GFTD_FACTOR_COLUMNS)), dtype=float)
    signed_streak = 0
    active_setup = 0
    buy_count = 0
    sell_count = 0
    last_buy_count_close = np.nan
    last_sell_count_close = np.nan

    for index in range(len(close)):
        signal = 0
        setup_event = 0
        if index >= n1 and np.isfinite(close[index]) and np.isfinite(close[index - n1]):
            comparison = int(np.sign(close[index] - close[index - n1]))
            if comparison == 0:
                signed_streak = 0
            elif signed_streak == 0 or int(np.sign(signed_streak)) != comparison:
                signed_streak = comparison
            else:
                signed_streak += comparison
            if signed_streak == -n2:
                setup_event = 1
            elif signed_streak == n2:
                setup_event = -1
        else:
            signed_streak = 0

        # A completed count wins over a new opposite setup on the same bar.
        if active_setup == 1 and index >= 2:
            first_count = buy_count == 0
            conditions = (
                np.isfinite(
                    [close[index], high[index], high[index - 1], high[index - 2]]
                ).all()
                and close[index] >= high[index - 2]
                and high[index] > high[index - 1]
                and (first_count or close[index] > last_buy_count_close)
            )
            if conditions:
                buy_count += 1
                last_buy_count_close = close[index]
                if buy_count >= n3:
                    signal = 1
        elif active_setup == -1 and index >= 2:
            first_count = sell_count == 0
            conditions = (
                np.isfinite(
                    [close[index], low[index], low[index - 1], low[index - 2]]
                ).all()
                and close[index] <= low[index - 2]
                and low[index] < low[index - 1]
                and (first_count or close[index] < last_sell_count_close)
            )
            if conditions:
                sell_count += 1
                last_sell_count_close = close[index]
                if sell_count >= n3:
                    signal = -1

        if signal == 0 and setup_event != 0:
            active_setup = setup_event
            buy_count = 0
            sell_count = 0
            last_buy_count_close = np.nan
            last_sell_count_close = np.nan

        output[index] = (
            active_setup,
            signed_streak,
            buy_count,
            sell_count,
            signal,
        )
        if signal != 0:
            active_setup = 0
            buy_count = 0
            sell_count = 0
            last_buy_count_close = np.nan
            last_sell_count_close = np.nan

    return pd.DataFrame(output, columns=GFTD_FACTOR_COLUMNS)


def _apply_gftd_factors(day: pd.DataFrame, factors: pd.DataFrame) -> None:
    hours = day["trade_time"].dt.hour
    sessions = np.where(hours < 13, "am", "pm")
    for session in ("am", "pm"):
        positions = np.flatnonzero(sessions == session)
        if len(positions) == 0:
            continue
        state = calculate_gftd_state(
            day.loc[positions, "_adj_close"].to_numpy(dtype=float),
            day.loc[positions, "_adj_high"].to_numpy(dtype=float),
            day.loc[positions, "_adj_low"].to_numpy(dtype=float),
        )
        factors.loc[positions, GFTD_FACTOR_COLUMNS] = state.to_numpy()


def _apply_breakout_factors(
    day: pd.DataFrame,
    factors: pd.DataFrame,
    context: pd.Series | None,
) -> None:
    if context is None:
        return
    day_open = day["_adj_open"].iloc[0]
    close = day["_adj_close"].to_numpy(dtype=float)
    atr = float(context["avg_true_range_10"])
    drange = float(context["dual_thrust_drange_5"])
    kurtosis = float(context["prev30d_open_close_return_pearson_kurtosis"])

    if np.isfinite(kurtosis):
        factors["prev30d_open_close_return_pearson_kurtosis"] = kurtosis
    if not np.isfinite(day_open) or day_open <= 0:
        return

    if np.isfinite(atr) and atr > 0:
        upper = day_open + ATR_ORB_MULTIPLIER * atr
        lower = day_open - ATR_ORB_MULTIPLIER * atr
        factors["atr10_orb_width_to_open"] = atr / day_open
        factors["distance_to_atr10_orb_upper"] = close / upper - 1.0
        if lower > 0:
            factors["distance_to_atr10_orb_lower"] = close / lower - 1.0

    if np.isfinite(drange) and drange > 0:
        upper = day_open + DUAL_THRUST_MULTIPLIER * drange
        lower = day_open - DUAL_THRUST_MULTIPLIER * drange
        factors["dual_thrust_drange5_to_open"] = drange / day_open
        factors["distance_to_dual_thrust_upper"] = close / upper - 1.0
        if lower > 0:
            factors["distance_to_dual_thrust_lower"] = close / lower - 1.0


def _apply_delayed_extreme_factors(day: pd.DataFrame, factors: pd.DataFrame) -> None:
    high = day["_adj_high"].to_numpy(dtype=float)
    low = day["_adj_low"].to_numpy(dtype=float)
    close = day["_adj_close"].to_numpy(dtype=float)
    cumulative_high = np.maximum.accumulate(high)
    cumulative_low = np.minimum.accumulate(low)

    delayed_high = np.full(len(day), np.nan, dtype=float)
    delayed_low = np.full(len(day), np.nan, dtype=float)
    for index in range(len(day)):
        reference = max(index - DELAYED_EXTREME_LAG, 0)
        delayed_high[index] = cumulative_high[reference]
        delayed_low[index] = cumulative_low[reference]

    valid_high = np.isfinite(close) & np.isfinite(delayed_high) & (delayed_high > 0)
    valid_low = np.isfinite(close) & np.isfinite(delayed_low) & (delayed_low > 0)
    high_distance = np.full(len(day), np.nan, dtype=float)
    low_distance = np.full(len(day), np.nan, dtype=float)
    high_distance[valid_high] = close[valid_high] / delayed_high[valid_high] - 1.0
    low_distance[valid_low] = close[valid_low] / delayed_low[valid_low] - 1.0

    state = np.zeros(len(day), dtype=float)
    eligible = np.arange(len(day)) >= DELAYED_EXTREME_CUTOFF
    eligible &= valid_high & valid_low & (delayed_high > delayed_low)
    state[eligible & (close >= delayed_high)] = 1.0
    state[eligible & (close <= delayed_low)] = -1.0
    factors[DELAYED_EXTREME_FACTOR_COLUMNS] = np.column_stack(
        [high_distance, low_distance, state]
    )


def build_intraday_strategy_factor_frame(
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    ts_code: str,
    *,
    priority_profile: str = PRIORITY_PROFILE_P0,
    minute_history: pd.DataFrame | None = None,
    active_flow_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build P0 or combined P0+P1 intraday factors for one symbol."""
    output_columns = output_columns_for_profile(priority_profile)
    factor_columns = factor_columns_for_profile(priority_profile)
    work = _prepare_adjusted_minute(minute, ts_code)
    if work.empty:
        return pd.DataFrame(columns=output_columns)

    daily_context = build_daily_context(
        daily,
        work["trade_date"].drop_duplicates().tolist(),
        ts_code,
    ).set_index("trade_date")

    per_day: list[pd.DataFrame] = []
    for trade_date, raw_day in work.groupby("trade_date", sort=True):
        day = raw_day.reset_index(drop=True)
        factors = pd.DataFrame(np.nan, index=day.index, columns=FACTOR_COLUMNS)
        _apply_opening_factors(day, factors)
        _apply_gftd_factors(day, factors)
        context = (
            daily_context.loc[trade_date] if trade_date in daily_context.index else None
        )
        if isinstance(context, pd.DataFrame):
            context = context.iloc[-1]
        _apply_breakout_factors(day, factors, context)
        _apply_delayed_extreme_factors(day, factors)

        output = day[["trade_date", "trade_time", "ts_code", *SOURCE_COLUMNS]].copy()
        # The local 1-minute files are end-labelled (13:01 first afternoon bar,
        # 15:00 final bar), so completed OHLC is available at trade_time.
        output.insert(2, "available_time", day["trade_time"].to_numpy())
        output = pd.concat([output, factors], axis=1)
        per_day.append(output[OUTPUT_COLUMNS])

    result = pd.concat(per_day, ignore_index=True)
    if priority_profile == PRIORITY_PROFILE_P0_P1:
        if active_flow_context is None:
            raise ValueError(
                "active_flow_context is required for priority profile p0_p1"
            )
        history_source = minute if minute_history is None else minute_history
        prepared_history = _prepare_adjusted_minute(history_source, ts_code)
        p1 = build_p1_factor_frame(work, prepared_history, active_flow_context)
        result[P1_FACTOR_COLUMNS] = p1.to_numpy()
        result = result[output_columns]

    numeric = result[factor_columns].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError(f"Infinite intraday strategy factor produced for {ts_code}")
    return result
