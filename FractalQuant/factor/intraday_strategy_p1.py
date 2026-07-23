"""Second-priority causal intraday-strategy factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


PATH_KNN_NEIGHBORS = 10
PATH_KNN_LOOKBACK_DAYS = 60
PATH_KNN_MIN_BARS = 10
PRIOR_ENVELOPE_BARS = 26
MIN_VOLATILITY_LOOKBACK = 7
EARLY_RANGE_BARS = 41

MARKET_FLOW_FACTOR_COLUMNS = [
    "market_active_notional_imbalance_1m",
    "market_active_notional_imbalance_cum_session",
    "asset_minus_market_active_flow_1m",
]
PATH_KNN_FACTOR_COLUMNS = [
    "path_knn_expected_next_return",
    "path_knn_up_probability",
    "path_knn_mean_distance",
    "path_knn_direction_agreement",
]
VOLATILITY_CONVERGENCE_FACTOR_COLUMNS = [
    "distance_to_prior26_upper_envelope",
    "distance_to_prior26_lower_envelope",
    "min_volatility_regime_7",
]
EARLY_RANGE_FACTOR_COLUMNS = [
    "early_range_position_41m",
    "distance_to_early_high_41m",
    "distance_to_early_low_41m",
]
P1_FACTOR_COLUMNS = (
    MARKET_FLOW_FACTOR_COLUMNS
    + PATH_KNN_FACTOR_COLUMNS
    + VOLATILITY_CONVERGENCE_FACTOR_COLUMNS
    + EARLY_RANGE_FACTOR_COLUMNS
)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0),
    )
    return result


def active_notional_imbalance(
    buy_notional: np.ndarray | pd.Series,
    sell_notional: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return stable active-buy/sell notional imbalance, using zero for no flow."""
    buy = np.asarray(buy_notional, dtype=float)
    sell = np.asarray(sell_notional, dtype=float)
    denominator = buy + sell
    result = np.zeros(np.broadcast_shapes(buy.shape, sell.shape), dtype=float)
    np.divide(
        buy - sell,
        denominator,
        out=result,
        where=np.isfinite(buy) & np.isfinite(sell) & (denominator > 0),
    )
    result[~(np.isfinite(buy) & np.isfinite(sell))] = np.nan
    return result


def aggregate_directional_notional(
    trades: pd.DataFrame,
    bar_times: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate validated B/S trades into the local minute-bar labels."""
    times = pd.DatetimeIndex(pd.to_datetime(bar_times, errors="coerce"))
    if times.isna().any() or times.has_duplicates:
        raise ValueError("Minute bar times must be unique and valid")
    required = {"event_time", "side", "price", "qty"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"Trade frame is missing required columns: {missing}")

    work = trades.copy()
    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce")
    work["side"] = work["side"].astype(str).str.strip().str.upper()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["qty"] = pd.to_numeric(work["qty"], errors="coerce")
    work = work.loc[
        work["event_time"].notna() & work["price"].gt(0) & work["qty"].gt(0)
    ].copy()
    invalid = work.loc[~work["side"].isin(["B", "S"]), "side"].unique()
    if len(invalid):
        raise ValueError(
            "Positive-notional trades contain unknown directions: "
            + ", ".join(map(str, invalid[:5]))
        )

    if work.empty:
        return pd.DataFrame(
            {
                "trade_time": times,
                "buy_notional_1m": np.zeros(len(times)),
                "sell_notional_1m": np.zeros(len(times)),
            }
        )

    # Local minute bars are end-labelled. An event inside minute m becomes
    # available at the next whole-minute label, never at the minute's start.
    work["trade_time"] = work["event_time"].dt.ceil("min")
    work["notional"] = work["price"] * work["qty"]
    grouped = work.pivot_table(
        index="trade_time",
        columns="side",
        values="notional",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(times, fill_value=0.0)
    return pd.DataFrame(
        {
            "trade_time": times,
            "buy_notional_1m": grouped.get("B", pd.Series(0.0, index=times)).to_numpy(
                dtype=float
            ),
            "sell_notional_1m": grouped.get("S", pd.Series(0.0, index=times)).to_numpy(
                dtype=float
            ),
        }
    )


def _path_knn_for_day(
    day: pd.DataFrame,
    historical_days: list[pd.DataFrame],
) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=day.index, columns=PATH_KNN_FACTOR_COLUMNS)
    if not historical_days or len(day) < PATH_KNN_MIN_BARS:
        return result

    time_keys = day["trade_time"].dt.strftime("%H:%M:%S")
    target_close = day["_adj_close"].to_numpy(dtype=float)
    target_open = float(day["_adj_open"].iloc[0])
    if not np.isfinite(target_open) or target_open <= 0:
        return result
    target_path = np.log(target_close / target_open)

    candidate_paths: list[np.ndarray] = []
    candidate_next_returns: list[np.ndarray] = []
    for historical in historical_days:
        keyed = historical.assign(
            _time_key=historical["trade_time"].dt.strftime("%H:%M:%S")
        ).set_index("_time_key")
        aligned = keyed.reindex(time_keys)
        close = aligned["_adj_close"].to_numpy(dtype=float)
        open_price = float(aligned["_adj_open"].iloc[0])
        if not np.isfinite(open_price) or open_price <= 0:
            continue
        candidate_paths.append(np.log(close / open_price))
        next_return = np.full(len(close), np.nan, dtype=float)
        valid = np.isfinite(close[:-1]) & np.isfinite(close[1:]) & (close[:-1] > 0)
        next_return[:-1][valid] = close[1:][valid] / close[:-1][valid] - 1.0
        candidate_next_returns.append(next_return)

    if len(candidate_paths) < PATH_KNN_NEIGHBORS:
        return result

    candidates = np.vstack(candidate_paths)
    next_returns = np.vstack(candidate_next_returns)
    finite = np.isfinite(candidates)
    prefix_valid = np.logical_and.accumulate(finite, axis=1)
    values = np.where(finite, candidates, 0.0)
    sum_x = np.cumsum(values, axis=1)
    sum_x2 = np.cumsum(values * values, axis=1)
    sum_xy = np.cumsum(values * target_path, axis=1)
    sum_y = np.cumsum(target_path)
    sum_y2 = np.cumsum(target_path * target_path)

    for position in range(PATH_KNN_MIN_BARS - 1, len(day) - 1):
        if not np.isfinite(target_path[: position + 1]).all():
            continue
        count = float(position + 1)
        covariance = sum_xy[:, position] - sum_x[:, position] * sum_y[position] / count
        variance_x = sum_x2[:, position] - sum_x[:, position] ** 2 / count
        variance_y = sum_y2[position] - sum_y[position] ** 2 / count
        denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
        valid = prefix_valid[:, position]
        valid &= np.isfinite(next_returns[:, position])
        valid &= denominator > 0
        if valid.sum() < PATH_KNN_NEIGHBORS:
            continue
        correlation = np.full(len(candidates), np.nan)
        correlation[valid] = covariance[valid] / denominator[valid]
        distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * correlation))
        eligible = np.flatnonzero(valid)
        nearest = eligible[
            np.argsort(distance[eligible], kind="stable")[:PATH_KNN_NEIGHBORS]
        ]
        neighbor_returns = next_returns[nearest, position]
        result.loc[position, PATH_KNN_FACTOR_COLUMNS] = (
            float(neighbor_returns.mean()),
            float((neighbor_returns > 0).mean()),
            float(distance[nearest].mean()),
            float(np.abs(np.sign(neighbor_returns).mean())),
        )
    return result


def _apply_volatility_convergence(day: pd.DataFrame, factors: pd.DataFrame) -> None:
    high = day["_adj_high"].astype(float)
    low = day["_adj_low"].astype(float)
    close = day["_adj_close"].astype(float)
    upper = (
        high.shift(1)
        .rolling(PRIOR_ENVELOPE_BARS, min_periods=PRIOR_ENVELOPE_BARS)
        .max()
    )
    lower = (
        low.shift(1).rolling(PRIOR_ENVELOPE_BARS, min_periods=PRIOR_ENVELOPE_BARS).min()
    )
    log_return = np.log(close.where(close.gt(0))).diff()
    sigma26 = log_return.rolling(
        PRIOR_ENVELOPE_BARS, min_periods=PRIOR_ENVELOPE_BARS
    ).std(ddof=0)
    min_volatility = (
        sigma26.shift(1)
        .rolling(MIN_VOLATILITY_LOOKBACK, min_periods=MIN_VOLATILITY_LOOKBACK)
        .min()
    )

    factors["distance_to_prior26_upper_envelope"] = _safe_ratio(close, upper) - 1.0
    factors["distance_to_prior26_lower_envelope"] = _safe_ratio(close, lower) - 1.0
    factors["min_volatility_regime_7"] = min_volatility.to_numpy(dtype=float)


def _apply_early_range(day: pd.DataFrame, factors: pd.DataFrame) -> None:
    times = day["trade_time"].dt.time
    opening = (times >= pd.Timestamp("09:30").time()) & (
        times <= pd.Timestamp("11:30").time()
    )
    positions = np.flatnonzero(opening.to_numpy())
    if len(positions) < EARLY_RANGE_BARS:
        return
    window = positions[:EARLY_RANGE_BARS]
    high = float(day.loc[window, "_adj_high"].max())
    low = float(day.loc[window, "_adj_low"].min())
    if not np.isfinite(high) or not np.isfinite(low) or high <= 0 or low <= 0:
        return
    available = int(window[-1])
    close = day.loc[available:, "_adj_close"].to_numpy(dtype=float)
    factors.loc[available:, "distance_to_early_high_41m"] = close / high - 1.0
    factors.loc[available:, "distance_to_early_low_41m"] = close / low - 1.0
    if high > low:
        factors.loc[available:, "early_range_position_41m"] = (
            2.0 * (close - low) / (high - low) - 1.0
        )


def _normalize_flow_context(
    context: pd.DataFrame,
    requested: pd.DataFrame,
) -> pd.DataFrame:
    required = {"trade_date", "trade_time", *MARKET_FLOW_FACTOR_COLUMNS}
    missing = sorted(required.difference(context.columns))
    if missing:
        raise ValueError(f"Active-flow context is missing required columns: {missing}")
    work = context.copy()
    work["trade_date"] = pd.to_datetime(
        work["trade_date"], errors="coerce"
    ).dt.normalize()
    work["trade_time"] = pd.to_datetime(work["trade_time"], errors="coerce")
    if work[["trade_date", "trade_time"]].isna().any().any():
        raise ValueError("Active-flow context contains invalid keys")
    if work.duplicated(["trade_date", "trade_time"]).any():
        raise ValueError("Active-flow context contains duplicate minute keys")
    keys = requested[["trade_date", "trade_time"]]
    merged = keys.merge(
        work, on=["trade_date", "trade_time"], how="left", validate="one_to_one"
    )
    if merged[MARKET_FLOW_FACTOR_COLUMNS].isna().any().any():
        raise ValueError(
            "Active-flow context does not completely cover requested minutes"
        )
    return merged


def build_p1_factor_frame(
    requested: pd.DataFrame,
    minute_history: pd.DataFrame,
    active_flow_context: pd.DataFrame,
) -> pd.DataFrame:
    """Build all 13 P1 factors for prepared minute frames."""
    factors = pd.DataFrame(np.nan, index=requested.index, columns=P1_FACTOR_COLUMNS)
    flow = _normalize_flow_context(active_flow_context, requested)
    factors[MARKET_FLOW_FACTOR_COLUMNS] = flow[MARKET_FLOW_FACTOR_COLUMNS].to_numpy()

    history_groups = {
        date: day.reset_index(drop=True)
        for date, day in minute_history.groupby("trade_date", sort=True)
    }
    history_dates = sorted(history_groups)
    for trade_date, raw_day in requested.groupby("trade_date", sort=True):
        positions = raw_day.index.to_numpy()
        day = raw_day.reset_index(drop=True)
        historical_dates = [date for date in history_dates if date < trade_date][
            -PATH_KNN_LOOKBACK_DAYS:
        ]
        historical_days = [history_groups[date] for date in historical_dates]
        path = _path_knn_for_day(day, historical_days)
        local = pd.DataFrame(np.nan, index=day.index, columns=P1_FACTOR_COLUMNS)
        local[PATH_KNN_FACTOR_COLUMNS] = path.to_numpy()
        _apply_volatility_convergence(day, local)
        _apply_early_range(day, local)
        factors.loc[
            positions,
            PATH_KNN_FACTOR_COLUMNS
            + VOLATILITY_CONVERGENCE_FACTOR_COLUMNS
            + EARLY_RANGE_FACTOR_COLUMNS,
        ] = local[
            PATH_KNN_FACTOR_COLUMNS
            + VOLATILITY_CONVERGENCE_FACTOR_COLUMNS
            + EARLY_RANGE_FACTOR_COLUMNS
        ].to_numpy()

    numeric = factors.to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Infinite P1 intraday-strategy factor produced")
    return factors
