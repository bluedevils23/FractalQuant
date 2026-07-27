from __future__ import annotations

"""DataFrame-based stock orderbook factors for the raw A-share tick CSV pipeline.

This module is intentionally scoped to the stock snapshot pipeline used by
``scripts/generate_stock_orderbook_factors.py``:
- quotes are the master snapshot index
- base order/trade events are summarized on 60s windows
- multi-window output adds causal 10s, 30s, and 300s summaries
- event features are aligned back to snapshots with ``merge_asof``

It is kept separate from the legacy single-snapshot ``factor/orderbook.py``
implementation and from the ETF minute-parquet factor pipeline.
"""

import numpy as np
import pandas as pd


ORDER_WINDOW = "60s"
TRADE_WINDOW = "60s"
OFI_WINDOW = "60s"
WINDOW_PROFILE_BASE = "base"
WINDOW_PROFILE_MULTI = "multi"
FLOW_WINDOWS = ("10s", "30s", "60s", "300s")
IMPACT_WINDOWS = ("30s", "60s", "300s")
RESILIENCE_WINDOWS = ("10s", "30s", "60s", "300s")
MLOFI_EVENT_WINDOW = 50
MLOFI_IMPACT_WINDOW = 100
MLOFI_IMPACT_MIN_HISTORY = 30
MLOFI_LEVEL_DECAY = 0.8
NEAR_TOUCH_LEVELS = 5
AMIHUD_WINDOW = "300s"
AMIHUD_MIN_RETURNS = 2
REFILL_LOOKBACK_SNAPSHOTS = 6
PRESSURE_SPREAD_FLOOR_BPS = 1.0
CONTEXT_LOOKBACK_SNAPSHOTS = 20
CONTEXT_MIN_HISTORY = 10
CONTEXT_SELECTION_WINDOW = "15min"
CONTEXT_SELECTION_MIN_PERIODS = 30
CONTEXT_SELECTION_QUANTILE = 0.98
VPIN_NUM_BUCKETS = 50
VPIN_TARGET_TRADES_PER_BUCKET = 20
VPIN_SIZE_EWMA_SPAN = 100
MARKOUT_HORIZON = "30s"
MARKOUT_ROLLING_WINDOW = "60s"
MPC_STAT_WINDOW = "300s"
PRICE_BAND_HISTORY_WINDOW = "15min"
PRICE_BAND_MIN_HISTORY = 20
LUNCH_BREAK = pd.Timedelta(hours=1, minutes=30)
# Legacy mapping below is only for the stock orderbook CSV pipeline. ETF minute factors live elsewhere.
# Existing output coverage against legacy factor/orderbook.py:
# - SpreadFactor -> spread_bps
# - OrderBookImbalanceFactor -> depth_imbalance_l5
# - OrderBookPressureFactor -> book_pressure_wap5
# - OrderBookSlopeFactor -> book_slope_diff_l5
# - OrderBookConcentrationFactor -> depth_concentration_l5
# Newly added lightweight factors in this module:
# - depth_l5_total
# - orderbook_decay_l5
# - orderbook_asymmetry_l5
# - orderbook_liquidity_l5
# - orderbook_velocity_l5
# Second batch trade-impact extensions in this module:
# - trade_size_distribution_60s
# - trade_direction_persistence_60s
# - liquidity_shock_60s
# - market_impact_60s
# - orderflow_significance_60s
# Third batch trade-window dynamics in this module:
# - volatility_adj_volume_60s
# - price_velocity_60s
# - momentum_acceleration_60s
# - volume_spike_60s
# - volume_clustering_60s
# Fourth batch trade-window structure factors in this module:
# - liquidity_depth_60s
# - price_volume_decoupling_60s
# - market_efficiency_60s
# - liquidity_migration_60s
# Fifth batch trade-window flow/liquidity factors in this module:
# - order_flow_imbalance_60s
# - liquidity_ratio_60s
# - volume_weighted_price_60s
# Sixth batch trade-window pressure factor in this module:
# - orderbook_pressure_60s
# Contextual anomaly factors, inspired by the expectation-reality and
# informative-segment selection stages in Jiao et al. (2023):
# - contextual_lob_surprise_l5
# - contextual_imbalance_surprise_l5
# - contextual_segment_anomaly_60s
# - contextual_segment_selected_60s
# Research-derived L2 additions:
# - mci_bid_l5 / mci_ask_l5
# - soir_l5_decay
# - mpc_{1m,5m}_{mean,max,skew}_5m
# - cautious_to_aggressive_buy_ratio_60s
# - trade_notional_quantile_position_60s
# - price-band trade shares and relative trade sizes


def _safe_divide(
    numerator: np.ndarray | pd.Series,
    denominator: np.ndarray | pd.Series,
    fill_value: float = np.nan,
) -> np.ndarray:
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    result = np.full(numerator_arr.shape, fill_value, dtype=float)
    valid = np.isfinite(numerator_arr) & np.isfinite(denominator_arr) & (denominator_arr != 0)
    result[valid] = numerator_arr[valid] / denominator_arr[valid]
    return result


def _imbalance(
    buy_values: np.ndarray | pd.Series,
    sell_values: np.ndarray | pd.Series,
) -> np.ndarray:
    buy_arr = np.asarray(buy_values, dtype=float)
    sell_arr = np.asarray(sell_values, dtype=float)
    total = buy_arr + sell_arr
    result = np.zeros(total.shape, dtype=float)
    valid = np.isfinite(total) & (total != 0)
    result[valid] = (buy_arr[valid] - sell_arr[valid]) / total[valid]
    return result


def _near_touch_depth(quantities: np.ndarray) -> np.ndarray:
    """Weight displayed depth by proximity to the touch: q1 + q2/2 + ... + q5/5."""
    weights = 1.0 / np.arange(1, quantities.shape[1] + 1, dtype=float)
    clean_qty = np.where(np.isfinite(quantities) & (quantities > 0), quantities, 0.0)
    return np.sum(clean_qty * weights, axis=1)


def _positive_refill_intensity(depth: np.ndarray, index: pd.Index) -> np.ndarray:
    """Measure positive depth growth against the preceding six snapshots."""
    depth_series = pd.Series(depth, index=index)
    prior_mean = (
        depth_series.shift(1)
        .rolling(REFILL_LOOKBACK_SNAPSHOTS, min_periods=REFILL_LOOKBACK_SNAPSHOTS)
        .mean()
    )
    refill = _safe_divide(depth, prior_mean.to_numpy(dtype=float)) - 1.0
    return np.where(np.isfinite(refill), np.maximum(refill, 0.0), np.nan)


def _quote_depth_event(
    prices: np.ndarray,
    quantities: np.ndarray,
    side: str,
) -> np.ndarray:
    """Calculate Cont-style queue events for one side of a levelled book."""
    previous_prices = np.vstack((np.full((1, prices.shape[1]), np.nan), prices[:-1]))
    previous_quantities = np.vstack(
        (np.full((1, quantities.shape[1]), np.nan), quantities[:-1])
    )
    valid = (
        np.isfinite(prices)
        & np.isfinite(previous_prices)
        & (prices > 0)
        & (previous_prices > 0)
        & np.isfinite(quantities)
        & np.isfinite(previous_quantities)
        & (quantities >= 0)
        & (previous_quantities >= 0)
    )
    event = np.full(prices.shape, np.nan, dtype=float)

    if side == "bid":
        event[valid & (prices > previous_prices)] = quantities[valid & (prices > previous_prices)]
        event[valid & (prices == previous_prices)] = (
            quantities[valid & (prices == previous_prices)]
            - previous_quantities[valid & (prices == previous_prices)]
        )
        event[valid & (prices < previous_prices)] = -previous_quantities[
            valid & (prices < previous_prices)
        ]
    elif side == "ask":
        event[valid & (prices < previous_prices)] = -quantities[valid & (prices < previous_prices)]
        event[valid & (prices == previous_prices)] = (
            previous_quantities[valid & (prices == previous_prices)]
            - quantities[valid & (prices == previous_prices)]
        )
        event[valid & (prices > previous_prices)] = previous_quantities[
            valid & (prices > previous_prices)
        ]
    else:
        raise ValueError(f"Unsupported book side: {side}")

    return event


def _normalize_ofi_events(
    events: np.ndarray,
    depth_scale: np.ndarray,
    index: pd.DatetimeIndex,
    window: str = OFI_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize instantaneous and rolling OFI on the continuous trading clock."""
    instantaneous = _safe_divide(events, depth_scale)
    rolling = np.full(len(events), np.nan, dtype=float)
    session_labels = _trading_session_labels(index)
    trading_index = _trading_time_index(index)

    for session in pd.unique(session_labels):
        positions = np.flatnonzero(session_labels == session)
        event_series = pd.Series(events[positions], index=trading_index[positions])
        depth_series = pd.Series(depth_scale[positions], index=trading_index[positions])
        rolling_event = event_series.rolling(window, min_periods=1).sum()
        rolling_depth = depth_series.rolling(window, min_periods=1).mean()
        rolling[positions] = _safe_divide(
            rolling_event.to_numpy(dtype=float), rolling_depth.to_numpy(dtype=float)
        )

    return instantaneous, rolling


def _calculate_normalized_ofi(
    bid_prices: np.ndarray,
    bid_qty: np.ndarray,
    ask_prices: np.ndarray,
    ask_qty: np.ndarray,
    index: pd.DatetimeIndex,
    window: str = OFI_WINDOW,
    reset_sessions: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return L1 and near-touch L5 OFI, instantaneously and over one window."""
    bid_events = _quote_depth_event(bid_prices, bid_qty, side="bid")
    ask_events = _quote_depth_event(ask_prices, ask_qty, side="ask")
    reset_starts = np.zeros(len(index), dtype=bool)
    if len(index):
        reset_starts[0] = True
        if reset_sessions:
            session_labels = _trading_session_labels(index)
            reset_starts[1:] = session_labels[1:] != session_labels[:-1]
        else:
            reset_starts[1:] = index.normalize()[1:] != index.normalize()[:-1]
    bid_events[reset_starts] = np.nan
    ask_events[reset_starts] = np.nan

    l1_events = bid_events[:, 0] + ask_events[:, 0]
    l1_depth = bid_qty[:, 0] + ask_qty[:, 0]
    l1_depth_scale = (np.concatenate(([np.nan], l1_depth[:-1])) + l1_depth) / 2.0
    l1_depth_scale[reset_starts] = np.nan
    l1_instantaneous, l1_rolling = _normalize_ofi_events(
        l1_events, l1_depth_scale, index, window
    )

    level_weights = 1.0 / np.arange(1, NEAR_TOUCH_LEVELS + 1, dtype=float)
    level_events = bid_events[:, :NEAR_TOUCH_LEVELS] + ask_events[:, :NEAR_TOUCH_LEVELS]
    valid_l1_event = np.isfinite(level_events[:, 0])
    mlofi_events = np.sum(
        np.where(np.isfinite(level_events), level_events, 0.0) * level_weights,
        axis=1,
    )
    mlofi_events[~valid_l1_event] = np.nan
    mlofi_depth = (
        _near_touch_depth(bid_qty[:, :NEAR_TOUCH_LEVELS])
        + _near_touch_depth(ask_qty[:, :NEAR_TOUCH_LEVELS])
    )
    mlofi_depth_scale = (
        np.concatenate(([np.nan], mlofi_depth[:-1])) + mlofi_depth
    ) / 2.0
    mlofi_depth_scale[reset_starts] = np.nan
    mlofi_instantaneous, mlofi_rolling = _normalize_ofi_events(
        mlofi_events, mlofi_depth_scale, index, window
    )
    return l1_instantaneous, l1_rolling, mlofi_instantaneous, mlofi_rolling


def _calculate_depth_level_ofi_slope(
    bid_prices: np.ndarray,
    bid_qty: np.ndarray,
    ask_prices: np.ndarray,
    ask_qty: np.ndarray,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    """Return the cross-level slope of instantaneous depth-normalized OFI."""
    bid_events = _quote_depth_event(bid_prices, bid_qty, side="bid")
    ask_events = _quote_depth_event(ask_prices, ask_qty, side="ask")
    level_events = bid_events[:, :NEAR_TOUCH_LEVELS] + ask_events[:, :NEAR_TOUCH_LEVELS]
    level_depth = (
        bid_qty[:, :NEAR_TOUCH_LEVELS] + ask_qty[:, :NEAR_TOUCH_LEVELS]
    )
    previous_depth = np.vstack(
        (np.full((1, NEAR_TOUCH_LEVELS), np.nan), level_depth[:-1])
    )
    depth_scale = (previous_depth + level_depth) / 2.0

    session_labels = _trading_session_labels(index)
    session_starts = np.r_[True, session_labels[1:] != session_labels[:-1]]
    level_events[session_starts] = np.nan
    depth_scale[session_starts] = np.nan
    normalized = _safe_divide(level_events, depth_scale)

    centered_levels = np.arange(1, NEAR_TOUCH_LEVELS + 1, dtype=float)
    centered_levels -= centered_levels.mean()
    denominator = float(np.dot(centered_levels, centered_levels))
    slope = np.full(len(index), np.nan, dtype=float)
    valid = np.all(np.isfinite(normalized), axis=1)
    slope[valid] = normalized[valid] @ centered_levels / denominator
    return slope


def _validate_window_profile(window_profile: str) -> None:
    if window_profile not in {WINDOW_PROFILE_BASE, WINDOW_PROFILE_MULTI}:
        raise ValueError(f"Unsupported window profile: {window_profile}")


def _trading_session_labels(index: pd.DatetimeIndex) -> np.ndarray:
    """Return day labels; afternoon is continuous with the same day's morning."""
    return index.normalize().astype("datetime64[ns]").asi8


def _trading_time_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Compress lunch while clamping any non-trading snapshots to 11:30."""
    day_start = index.normalize()
    lunch_start = day_start + pd.Timedelta(hours=11, minutes=30)
    lunch_end = day_start + pd.Timedelta(hours=13)
    during_lunch = (index >= lunch_start) & (index < lunch_end)
    after_lunch = index >= lunch_end
    compressed = index - pd.to_timedelta(
        after_lunch.astype(np.int64) * LUNCH_BREAK.value, unit="ns"
    )
    return pd.DatetimeIndex(compressed.where(~during_lunch, lunch_start))


def _calculate_ofi_level_entropy(
    bid_prices: np.ndarray,
    bid_qty: np.ndarray,
    ask_prices: np.ndarray,
    ask_qty: np.ndarray,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    bid_events = _quote_depth_event(bid_prices, bid_qty, side="bid")
    ask_events = _quote_depth_event(ask_prices, ask_qty, side="ask")
    level_magnitudes = np.abs(bid_events + ask_events)
    session_labels = _trading_session_labels(index)
    session_starts = np.r_[True, session_labels[1:] != session_labels[:-1]]
    level_magnitudes[session_starts] = np.nan
    totals = np.nansum(level_magnitudes, axis=1)
    probabilities = np.divide(
        level_magnitudes,
        totals[:, None],
        out=np.full_like(level_magnitudes, np.nan),
        where=np.isfinite(totals[:, None]) & (totals[:, None] > 0),
    )
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = -probabilities[positive] * np.log(probabilities[positive])
    entropy = np.sum(terms, axis=1) / np.log(level_magnitudes.shape[1])
    entropy[~np.isfinite(totals) | (totals <= 0)] = np.nan
    return entropy


def _calculate_book_resilience(
    depth: np.ndarray,
    index: pd.DatetimeIndex,
    window: str = "30s",
) -> np.ndarray:
    result = np.full(len(index), np.nan, dtype=float)
    session_labels = _trading_session_labels(index)
    trading_index = _trading_time_index(index)
    for session in pd.unique(session_labels):
        positions = np.flatnonzero(session_labels == session)
        series = pd.Series(depth[positions], index=trading_index[positions])
        initial = series.rolling(window, min_periods=2, closed="both").apply(
            lambda values: values[0], raw=True
        )
        trough = series.rolling(window, min_periods=2, closed="both").min()
        depletion = initial - trough
        recovered = series - trough
        values = (recovered / depletion.replace(0.0, np.nan)).clip(0.0, 1.0)
        values = values.where(depletion > 0, 1.0).where(initial.notna())
        result[positions] = values.to_numpy(dtype=float)
    return result


def _calculate_mlofi_extensions(
    bid_prices: np.ndarray,
    bid_qty: np.ndarray,
    ask_prices: np.ndarray,
    ask_qty: np.ndarray,
    mid_price: np.ndarray,
    normalized_mlofi_60s: np.ndarray,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate event-window, deep-divergence, and causal impact MLOFI factors."""
    bid_events = _quote_depth_event(bid_prices, bid_qty, side="bid")
    ask_events = _quote_depth_event(ask_prices, ask_qty, side="ask")
    level_events = bid_events[:, :NEAR_TOUCH_LEVELS] + ask_events[:, :NEAR_TOUCH_LEVELS]
    level_depth = (
        bid_qty[:, :NEAR_TOUCH_LEVELS] + ask_qty[:, :NEAR_TOUCH_LEVELS]
    ) / 2.0

    event_window_factor = np.full(len(index), np.nan, dtype=float)
    deep_divergence = np.full(len(index), np.nan, dtype=float)
    impact_beta = np.full(len(index), np.nan, dtype=float)
    level_weights = np.power(
        MLOFI_LEVEL_DECAY, np.arange(NEAR_TOUCH_LEVELS, dtype=float)
    )
    session_labels = _trading_session_labels(index)
    trading_index = _trading_time_index(index)

    for session in pd.unique(session_labels):
        positions = np.flatnonzero(session_labels == session)
        if not len(positions):
            continue
        session_events = level_events[positions].copy()
        session_events[0] = np.nan
        event_frame = pd.DataFrame(session_events, index=trading_index[positions])
        depth_frame = pd.DataFrame(level_depth[positions], index=trading_index[positions])

        time_event = event_frame.rolling(OFI_WINDOW, min_periods=1).sum()
        time_depth = depth_frame.rolling(OFI_WINDOW, min_periods=1).mean()
        normalized_time = time_event / time_depth.replace(0.0, np.nan)

        count_event = event_frame.rolling(
            MLOFI_EVENT_WINDOW, min_periods=MLOFI_EVENT_WINDOW
        ).sum()
        count_depth = depth_frame.rolling(
            MLOFI_EVENT_WINDOW, min_periods=MLOFI_EVENT_WINDOW
        ).mean()
        normalized_count = count_event / count_depth.replace(0.0, np.nan)
        event_window_factor[positions] = normalized_count.mul(
            level_weights, axis=1
        ).sum(axis=1, min_count=1)

        deep_average = normalized_time.iloc[:, 1:].mul(
            level_weights[1:], axis=1
        ).sum(axis=1, min_count=1) / level_weights[1:].sum()
        deep_divergence[positions] = (
            deep_average - normalized_time.iloc[:, 0]
        ).to_numpy(dtype=float)

        x = pd.Series(normalized_mlofi_60s[positions], index=trading_index[positions]).shift(1)
        y = (
            pd.Series(mid_price[positions], index=trading_index[positions])
            .pct_change(fill_method=None)
            .mul(10000.0)
            .shift(1)
        )
        covariance = x.rolling(
            MLOFI_IMPACT_WINDOW, min_periods=MLOFI_IMPACT_MIN_HISTORY
        ).cov(y)
        variance = x.rolling(
            MLOFI_IMPACT_WINDOW, min_periods=MLOFI_IMPACT_MIN_HISTORY
        ).var()
        impact_beta[positions] = _safe_divide(
            covariance.to_numpy(dtype=float), variance.to_numpy(dtype=float)
        )

    return event_window_factor, deep_divergence, impact_beta


def _calculate_ofi_impact_nonlinearity(
    normalized_ofi_60s: np.ndarray,
    mid_price: np.ndarray,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    """Estimate the lagged nonlinear OFI impact coefficient causally."""
    result = np.full(len(index), np.nan, dtype=float)
    session_labels = _trading_session_labels(index)
    trading_index = _trading_time_index(index)

    for session in pd.unique(session_labels):
        positions = np.flatnonzero(session_labels == session)
        x = pd.Series(
            normalized_ofi_60s[positions], index=trading_index[positions]
        ).shift(1)
        y = (
            pd.Series(mid_price[positions], index=trading_index[positions])
            .pct_change(fill_method=None)
            .mul(10000.0)
            .shift(1)
        )
        valid_pair = x.notna() & y.notna()
        x = x.where(valid_pair)
        y = y.where(valid_pair)
        nonlinear_x = x * x.abs()

        rolling = {
            "var_x": x.rolling(
                MLOFI_IMPACT_WINDOW, min_periods=MLOFI_IMPACT_MIN_HISTORY
            ).var(),
            "var_nonlinear": nonlinear_x.rolling(
                MLOFI_IMPACT_WINDOW, min_periods=MLOFI_IMPACT_MIN_HISTORY
            ).var(),
            "cov_x_nonlinear": x.rolling(
                MLOFI_IMPACT_WINDOW, min_periods=MLOFI_IMPACT_MIN_HISTORY
            ).cov(nonlinear_x),
            "cov_x_y": x.rolling(
                MLOFI_IMPACT_WINDOW, min_periods=MLOFI_IMPACT_MIN_HISTORY
            ).cov(y),
            "cov_nonlinear_y": nonlinear_x.rolling(
                MLOFI_IMPACT_WINDOW, min_periods=MLOFI_IMPACT_MIN_HISTORY
            ).cov(y),
        }
        determinant = (
            rolling["var_x"] * rolling["var_nonlinear"]
            - rolling["cov_x_nonlinear"].pow(2)
        )
        determinant_scale = (
            rolling["var_x"] * rolling["var_nonlinear"]
        ).abs()
        numerator = (
            rolling["cov_nonlinear_y"] * rolling["var_x"]
            - rolling["cov_x_y"] * rolling["cov_x_nonlinear"]
        )
        stable = determinant > determinant_scale * 1e-8
        coefficient = (numerator / determinant.where(stable)).replace(
            [np.inf, -np.inf], np.nan
        )
        result[positions] = coefficient.to_numpy(dtype=float)
    return result


def _weighted_average(prices: np.ndarray, quantities: np.ndarray) -> np.ndarray:
    valid_mask = (
        np.isfinite(prices)
        & np.isfinite(quantities)
        & (prices > 0)
        & (quantities > 0)
    )
    clean_prices = np.where(valid_mask, prices, 0.0)
    clean_qty = np.where(valid_mask, quantities, 0.0)
    notional = np.sum(clean_prices * clean_qty, axis=1)
    total_qty = np.sum(clean_qty, axis=1)
    return _safe_divide(notional, total_qty)


def _weighted_slope(distances: np.ndarray, quantities: np.ndarray) -> np.ndarray:
    valid_mask = (
        np.isfinite(distances)
        & np.isfinite(quantities)
        & (distances > 0)
        & (quantities > 0)
    )
    raw_weights = np.where(valid_mask, quantities, 0.0)
    weight_sum = raw_weights.sum(axis=1)
    normalized_qty = np.divide(
        raw_weights,
        weight_sum[:, None],
        out=np.zeros_like(raw_weights, dtype=float),
        where=weight_sum[:, None] != 0,
    )
    weights = normalized_qty
    x = np.where(valid_mask, distances, 0.0)
    y = normalized_qty

    valid_count = valid_mask.sum(axis=1)
    normalized_weight_sum = weights.sum(axis=1)

    x_bar = _safe_divide(np.sum(weights * x, axis=1), normalized_weight_sum)
    y_bar = _safe_divide(np.sum(weights * y, axis=1), normalized_weight_sum)

    x_centered = x - x_bar[:, None]
    y_centered = y - y_bar[:, None]

    covariance = np.sum(weights * x_centered * y_centered, axis=1)
    variance = np.sum(weights * x_centered * x_centered, axis=1)

    slopes = _safe_divide(covariance, variance)
    slopes[variance <= 1e-12] = np.nan
    slopes[(valid_count < 2) | ~np.isfinite(slopes)] = np.nan
    return slopes


def _coefficient_of_variation(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)
    cv = _safe_divide(std, mean)
    cv[(~np.isfinite(mean)) | (mean <= 0)] = np.nan
    return cv


def _calculate_mpc_statistics(
    mid_price: np.ndarray,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Calculate causal 1/5-minute mid-price-change statistics per session."""
    columns = [
        f"mpc_{horizon}_{stat}_5m"
        for horizon in ("1m", "5m")
        for stat in ("mean", "max", "skew")
    ]
    result = pd.DataFrame(np.nan, index=index, columns=columns)
    session_labels = _trading_session_labels(index)
    trading_index = _trading_time_index(index)

    for session in pd.unique(session_labels):
        positions = np.flatnonzero(session_labels == session)
        session_index = index[positions]
        session_trading_index = trading_index[positions]
        session_mid = mid_price[positions]
        quote_frame = pd.DataFrame(
            {"quote_time": session_trading_index, "mid_price": session_mid}
        )

        for horizon in ("1m", "5m"):
            # Use a snapshot no later than t-k; no current or future quote enters MPC.
            target_index = session_trading_index - pd.Timedelta(horizon)
            reference = pd.merge_asof(
                pd.DataFrame({"target_time": target_index}),
                quote_frame,
                left_on="target_time",
                right_on="quote_time",
                direction="backward",
            )["mid_price"].to_numpy(dtype=float)
            mpc = _safe_divide(session_mid - reference, reference)
            mpc_series = pd.Series(mpc, index=session_trading_index)
            rolling = mpc_series.rolling(MPC_STAT_WINDOW, min_periods=3)
            result.loc[session_index, f"mpc_{horizon}_mean_5m"] = rolling.mean().to_numpy(
                dtype=float
            )
            result.loc[session_index, f"mpc_{horizon}_max_5m"] = rolling.max().to_numpy(
                dtype=float
            )
            result.loc[session_index, f"mpc_{horizon}_skew_5m"] = rolling.skew().to_numpy(
                dtype=float
            )

    return result


def _size_distribution_score(values: np.ndarray) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    centered = values - np.mean(values)
    m2 = np.mean(np.square(centered))
    if m2 <= 0 or not np.isfinite(m2):
        return 0.0
    m3 = np.mean(centered * centered * centered)
    skewness = np.sqrt(n * (n - 1.0)) / (n - 2.0) * (m3 / np.power(m2, 1.5))
    if not np.isfinite(skewness):
        skewness = 0.0

    kurtosis = 0.0
    if n >= 4:
        m4 = np.mean(centered * centered * centered * centered)
        g2 = m4 / (m2 * m2) - 3.0
        kurtosis = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)
        if not np.isfinite(kurtosis):
            kurtosis = 0.0
    return float(abs(skewness) + abs(kurtosis) / 10.0)


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) != len(y) or len(x) == 0:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_var = np.mean(np.square(x_centered))
    y_var = np.mean(np.square(y_centered))
    if x_var <= 0 or y_var <= 0:
        return None
    correlation = np.mean(x_centered * y_centered) / np.sqrt(x_var * y_var)
    return float(correlation) if np.isfinite(correlation) else None


def _window_sums(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Return sums over half-open event windows using one cumulative pass."""
    cumulative = np.empty(len(values) + 1, dtype=float)
    cumulative[0] = 0.0
    np.cumsum(values, dtype=float, out=cumulative[1:])
    return cumulative[ends] - cumulative[starts]


def _causal_rolling_zscore(
    values: np.ndarray | pd.Series,
    index: pd.DatetimeIndex,
    window: str | int,
    min_periods: int,
) -> np.ndarray:
    """Standardize against strictly preceding observations, independently per day."""
    result = np.full(len(index), np.nan, dtype=float)
    value_array = np.asarray(values, dtype=float)
    trading_index = _trading_time_index(index)
    for day in pd.unique(index.normalize()):
        positions = np.flatnonzero(index.normalize() == day)
        series = pd.Series(value_array[positions], index=trading_index[positions])
        history = series.shift(1)
        mean = history.rolling(window, min_periods=min_periods).mean()
        std = history.rolling(window, min_periods=min_periods).std(ddof=0)
        std = std.where(std.isna(), std.clip(lower=1e-3))
        result[positions] = _safe_divide(
            series.to_numpy(dtype=float) - mean.to_numpy(dtype=float),
            std.to_numpy(dtype=float),
        )
    return result


def _contextual_lob_surprises(
    bid_qty: np.ndarray,
    ask_qty: np.ndarray,
    weighted_imbalance: np.ndarray,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare the current L5 book with a causal expected-book baseline.

    The rolling mean is a lightweight, training-free replacement for the paper's
    order-book generator.  It is deliberately based only on preceding snapshots,
    so it is safe for both A-share and ETF prediction datasets.
    """
    book = np.concatenate((bid_qty, ask_qty), axis=1)
    book_frame = pd.DataFrame(book, index=index)
    expected_array = np.full(book.shape, np.nan, dtype=float)
    scale_array = np.full(book.shape, np.nan, dtype=float)
    day_labels = index.normalize()
    for day in pd.unique(day_labels):
        positions = np.flatnonzero(day_labels == day)
        history = book_frame.iloc[positions].shift(1)
        expected_array[positions] = history.rolling(
            CONTEXT_LOOKBACK_SNAPSHOTS, min_periods=CONTEXT_MIN_HISTORY
        ).mean().to_numpy(dtype=float)
        scale_array[positions] = history.rolling(
            CONTEXT_LOOKBACK_SNAPSHOTS, min_periods=CONTEXT_MIN_HISTORY
        ).std(ddof=0).to_numpy(dtype=float)
    residual = book_frame.to_numpy(dtype=float) - expected_array
    scale_floor = np.maximum(np.abs(expected_array) * 0.01, 1.0)
    scale_array = np.where(np.isfinite(scale_array), np.maximum(scale_array, scale_floor), np.nan)
    valid = np.isfinite(residual) & np.isfinite(scale_array) & (scale_array > 0)
    standardized = np.full(residual.shape, np.nan, dtype=float)
    standardized[valid] = residual[valid] / scale_array[valid]
    valid_count = np.isfinite(standardized).sum(axis=1)
    squared_sum = np.nansum(np.square(standardized), axis=1)
    lob_surprise = np.full(len(index), np.nan, dtype=float)
    has_history = valid_count > 0
    lob_surprise[has_history] = np.sqrt(squared_sum[has_history] / valid_count[has_history])
    imbalance_surprise = _causal_rolling_zscore(
        weighted_imbalance,
        index,
        CONTEXT_LOOKBACK_SNAPSHOTS,
        CONTEXT_MIN_HISTORY,
    )
    return lob_surprise, imbalance_surprise


def calculate_snapshot_factors(
    quotes: pd.DataFrame,
    window_profile: str = WINDOW_PROFILE_BASE,
) -> pd.DataFrame:
    _validate_window_profile(window_profile)
    quotes = quotes.sort_index(kind="stable")
    ask_price_cols = [f"ask_price{i}" for i in range(1, 6)]
    ask_qty_cols = [f"ask_qty{i}" for i in range(1, 6)]
    bid_price_cols = [f"bid_price{i}" for i in range(1, 6)]
    bid_qty_cols = [f"bid_qty{i}" for i in range(1, 6)]

    ask_prices = quotes[ask_price_cols].to_numpy(dtype=float, copy=False)
    ask_qty = quotes[ask_qty_cols].to_numpy(dtype=float, copy=False)
    bid_prices = quotes[bid_price_cols].to_numpy(dtype=float, copy=False)
    bid_qty = quotes[bid_qty_cols].to_numpy(dtype=float, copy=False)

    ask1 = ask_prices[:, 0]
    bid1 = bid_prices[:, 0]
    ask_qty1 = ask_qty[:, 0]
    bid_qty1 = bid_qty[:, 0]

    mid_price = np.full(len(quotes), np.nan, dtype=float)
    valid_mid = np.isfinite(ask1) & np.isfinite(bid1) & (ask1 > 0) & (bid1 > 0)
    mid_price[valid_mid] = (ask1[valid_mid] + bid1[valid_mid]) / 2.0

    spread_bps = np.full(len(quotes), np.nan, dtype=float)
    valid_spread = valid_mid & (ask1 >= bid1)
    spread_bps[valid_spread] = (
        (ask1[valid_spread] - bid1[valid_spread]) / mid_price[valid_spread] * 10000.0
    )
    bid_depth_l1 = bid_qty1
    ask_depth_l1 = ask_qty1
    bid_depth_l5 = np.sum(bid_qty, axis=1)
    ask_depth_l5 = np.sum(ask_qty, axis=1)
    depth_l5_total = bid_depth_l5 + ask_depth_l5
    weighted_bid_depth_l5 = _near_touch_depth(bid_qty[:, :NEAR_TOUCH_LEVELS])
    weighted_ask_depth_l5 = _near_touch_depth(ask_qty[:, :NEAR_TOUCH_LEVELS])
    weighted_depth_imbalance_l5 = _imbalance(
        weighted_bid_depth_l5, weighted_ask_depth_l5
    )
    soir_weights = 1.0 - np.arange(NEAR_TOUCH_LEVELS, dtype=float) / NEAR_TOUCH_LEVELS
    soir_by_level = _safe_divide(
        bid_qty - ask_qty,
        bid_qty + ask_qty,
    )
    valid_soir = np.isfinite(soir_by_level)
    soir_weight_sum = np.sum(valid_soir * soir_weights, axis=1)
    soir_l5_decay = _safe_divide(
        np.nansum(soir_by_level * soir_weights, axis=1), soir_weight_sum
    )
    contextual_lob_surprise_l5, contextual_imbalance_surprise_l5 = (
        _contextual_lob_surprises(
            bid_qty, ask_qty, weighted_depth_imbalance_l5, quotes.index
        )
    )
    (
        normalized_ofi_l1,
        normalized_ofi_l1_60s,
        normalized_mlofi_l5,
        normalized_mlofi_l5_60s,
    ) = _calculate_normalized_ofi(
        bid_prices, bid_qty, ask_prices, ask_qty, quotes.index
    )
    ofi_level_entropy_l5 = _calculate_ofi_level_entropy(
        bid_prices, bid_qty, ask_prices, ask_qty, quotes.index
    )
    depth_level_ofi_slope = _calculate_depth_level_ofi_slope(
        bid_prices, bid_qty, ask_prices, ask_qty, quotes.index
    )
    (
        mlofi_event_50_l5,
        mlofi_deep_divergence_l5,
        mlofi_impact_beta,
    ) = _calculate_mlofi_extensions(
        bid_prices,
        bid_qty,
        ask_prices,
        ask_qty,
        mid_price,
        normalized_mlofi_l5_60s,
        quotes.index,
    )
    ofi_impact_nonlinearity = _calculate_ofi_impact_nonlinearity(
        normalized_ofi_l1_60s, mid_price, quotes.index
    )
    pressure_denominator = np.maximum(spread_bps, PRESSURE_SPREAD_FLOOR_BPS)

    bid_wap5 = _weighted_average(bid_prices, bid_qty)
    ask_wap5 = _weighted_average(ask_prices, ask_qty)
    liquidity_spread = ask_wap5 - bid_wap5
    bid_decay_l5 = _safe_divide(bid_qty1, bid_qty[:, -1])
    ask_decay_l5 = _safe_divide(ask_qty1, ask_qty[:, -1])
    bid_cv = _coefficient_of_variation(bid_qty)
    ask_cv = _coefficient_of_variation(ask_qty)

    mid_denominator = np.where(np.isfinite(mid_price) & (mid_price > 0), mid_price, np.nan)
    bid_distances = np.abs((bid_prices - mid_price[:, None]) / mid_denominator[:, None]) * 10000.0
    ask_distances = np.abs((ask_prices - mid_price[:, None]) / mid_denominator[:, None]) * 10000.0
    bid_slope = _weighted_slope(bid_distances, bid_qty)
    ask_slope = _weighted_slope(ask_distances, ask_qty)

    result = pd.DataFrame(index=quotes.index)
    result["mid_price"] = mid_price
    result["spread_bps"] = spread_bps
    result["depth_imbalance_l1"] = _imbalance(bid_depth_l1, ask_depth_l1)
    result["depth_imbalance_l5"] = _imbalance(bid_depth_l5, ask_depth_l5)
    result["normalized_ofi_l1"] = normalized_ofi_l1
    result["normalized_ofi_l1_60s"] = normalized_ofi_l1_60s
    result["ofi_spread_scaled_impact"] = (
        normalized_ofi_l1_60s * spread_bps / 2.0
    )
    result["depth_level_ofi_slope"] = depth_level_ofi_slope
    result["ofi_impact_nonlinearity"] = ofi_impact_nonlinearity
    result["ofi_level_entropy_l5"] = ofi_level_entropy_l5
    result["normalized_mlofi_l5"] = normalized_mlofi_l5
    result["normalized_mlofi_l5_60s"] = normalized_mlofi_l5_60s
    result["mlofi_event_50_l5"] = mlofi_event_50_l5
    result["mlofi_deep_divergence_l5"] = mlofi_deep_divergence_l5
    result["mlofi_impact_beta"] = mlofi_impact_beta
    result["weighted_depth_imbalance_l5"] = weighted_depth_imbalance_l5
    result["soir_l5_decay"] = soir_l5_decay
    result["weighted_depth_pressure_l5"] = np.clip(
        _safe_divide(weighted_depth_imbalance_l5, pressure_denominator), -1.0, 1.0
    )
    result["weighted_imbalance_velocity_l5"] = (
        pd.Series(weighted_depth_imbalance_l5, index=quotes.index).diff(5)
    )
    result["contextual_lob_surprise_l5"] = contextual_lob_surprise_l5
    result["contextual_imbalance_surprise_l5"] = contextual_imbalance_surprise_l5
    result["bid_refill_intensity_l5"] = _positive_refill_intensity(
        weighted_bid_depth_l5, quotes.index
    )
    result["ask_refill_intensity_l5"] = _positive_refill_intensity(
        weighted_ask_depth_l5, quotes.index
    )
    result["bid_ask_qty_ratio_l1"] = _safe_divide(bid_qty1, ask_qty1)
    result["depth_l5_total"] = depth_l5_total
    bid_resilience_30s = _calculate_book_resilience(bid_depth_l5, quotes.index)
    ask_resilience_30s = _calculate_book_resilience(ask_depth_l5, quotes.index)
    result["bid_resilience_30s"] = bid_resilience_30s
    result["ask_resilience_30s"] = ask_resilience_30s
    result["resilience_imbalance_30s"] = bid_resilience_30s - ask_resilience_30s
    result["orderbook_decay_l5"] = (bid_decay_l5 + ask_decay_l5) / 2.0
    result["orderbook_asymmetry_l5"] = np.abs(bid_cv - ask_cv)
    result["depth_concentration_l5"] = (
        _safe_divide(bid_qty1, bid_depth_l5) + _safe_divide(ask_qty1, ask_depth_l5)
    ) / 2.0
    result["orderbook_liquidity_l5"] = _safe_divide(depth_l5_total, liquidity_spread)
    bid_notional_l5 = np.sum(
        np.where(
            np.isfinite(bid_prices)
            & np.isfinite(bid_qty)
            & (bid_prices > 0)
            & (bid_qty > 0),
            bid_prices * bid_qty,
            0.0,
        ),
        axis=1,
    )
    ask_notional_l5 = np.sum(
        np.where(
            np.isfinite(ask_prices)
            & np.isfinite(ask_qty)
            & (ask_prices > 0)
            & (ask_qty > 0),
            ask_prices * ask_qty,
            0.0,
        ),
        axis=1,
    )
    result["book_pressure_wap5"] = _imbalance(bid_notional_l5, ask_notional_l5)
    result["book_slope_diff_l5"] = bid_slope - ask_slope
    result["mci_bid_l5"] = _safe_divide(mid_price - bid_wap5, bid_notional_l5)
    result["mci_ask_l5"] = _safe_divide(ask_wap5 - mid_price, ask_notional_l5)
    result = result.join(_calculate_mpc_statistics(mid_price, quotes.index))

    if window_profile == WINDOW_PROFILE_MULTI:
        for window in FLOW_WINDOWS:
            if window == OFI_WINDOW:
                continue
            (
                _,
                normalized_ofi,
                _,
                normalized_mlofi,
            ) = _calculate_normalized_ofi(
                bid_prices,
                bid_qty,
                ask_prices,
                ask_qty,
                quotes.index,
                window,
                reset_sessions=True,
            )
            result[f"normalized_ofi_l1_{window}"] = normalized_ofi
            result[f"normalized_mlofi_l5_{window}"] = normalized_mlofi
            result[f"ofi_spread_scaled_impact_{window}"] = (
                normalized_ofi * spread_bps / 2.0
            )

        for window in RESILIENCE_WINDOWS:
            if window == "30s":
                continue
            bid_resilience = _calculate_book_resilience(
                bid_depth_l5, quotes.index, window
            )
            ask_resilience = _calculate_book_resilience(
                ask_depth_l5, quotes.index, window
            )
            result[f"bid_resilience_{window}"] = bid_resilience
            result[f"ask_resilience_{window}"] = ask_resilience
            result[f"resilience_imbalance_{window}"] = (
                bid_resilience - ask_resilience
            )
    return result.replace([np.inf, -np.inf], np.nan)


def _rolling_event_sums_at_quotes(
    events: pd.DataFrame,
    quote_index: pd.DatetimeIndex,
    value_columns: list[str],
    window: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(0.0, index=quote_index, columns=value_columns)

    ordered = events.sort_values("event_time", kind="stable")
    values = ordered[value_columns].to_numpy(dtype=float, copy=False)
    result = np.zeros((len(quote_index), len(value_columns)), dtype=float)
    event_times = pd.DatetimeIndex(pd.to_datetime(ordered["event_time"]))
    event_sessions = _trading_session_labels(event_times)
    quote_sessions = _trading_session_labels(quote_index)
    event_trading_times = _trading_time_index(event_times)
    quote_trading_times = _trading_time_index(quote_index)
    window_delta = pd.Timedelta(window).to_timedelta64()

    for session in pd.unique(quote_sessions):
        quote_positions = np.flatnonzero(quote_sessions == session)
        event_positions = np.flatnonzero(event_sessions == session)
        if not len(event_positions):
            continue
        session_event_times = event_trading_times[event_positions].to_numpy(
            dtype="datetime64[ns]", copy=False
        )
        session_quote_times = quote_trading_times[quote_positions].to_numpy(
            dtype="datetime64[ns]", copy=False
        )
        starts = np.searchsorted(
            session_event_times, session_quote_times - window_delta, side="right"
        )
        ends = np.searchsorted(session_event_times, session_quote_times, side="right")
        cumulative = np.vstack(
            (
                np.zeros((1, len(value_columns))),
                np.cumsum(values[event_positions], axis=0),
            )
        )
        result[quote_positions] = cumulative[ends] - cumulative[starts]
    return pd.DataFrame(result, index=quote_index, columns=value_columns)


def _calculate_trade_impact_factors(
    trades: pd.DataFrame,
    quote_index: pd.DatetimeIndex,
    window: str,
) -> pd.DataFrame:
    factor_bases = [
        "trade_size_distribution",
        "trade_direction_persistence",
        "liquidity_shock",
        "market_impact",
        "orderflow_significance",
        "volatility_adj_volume",
        "price_velocity",
        "momentum_acceleration",
        "volume_spike",
        "volume_clustering",
        "liquidity_depth",
        "price_volume_decoupling",
        "market_efficiency",
        "liquidity_migration",
        "order_flow_imbalance",
        "liquidity_ratio",
        "volume_weighted_price",
        "orderbook_pressure",
    ]
    factor_columns = [f"{base}_{window}" for base in factor_bases]
    result_array = np.full((len(quote_index), len(factor_columns)), np.nan, dtype=float)
    if trades.empty:
        return pd.DataFrame(result_array, index=quote_index, columns=factor_columns)

    window_delta = pd.Timedelta(window).to_timedelta64()
    trade_times = _trading_time_index(
        pd.DatetimeIndex(pd.to_datetime(trades["event_time"]))
    ).to_numpy(dtype="datetime64[ns]", copy=False)
    quote_times = _trading_time_index(quote_index).to_numpy(
        dtype="datetime64[ns]", copy=False
    )
    start_idx = np.searchsorted(trade_times, quote_times - window_delta, side="left")
    end_idx = np.searchsorted(trade_times, quote_times, side="right")

    prices = trades["price"].to_numpy(dtype=float, copy=False)
    quantities = trades["qty"].to_numpy(dtype=float, copy=False)
    notionals = trades["notional"].to_numpy(dtype=float, copy=False)
    directions = np.where(
        trades["side"].to_numpy(dtype=str, copy=False) == "B",
        1.0,
        -1.0,
    )
    trade_count = end_idx - start_idx
    valid = trade_count >= 10

    qty_sum = _window_sums(quantities, start_idx, end_idx)
    notional_sum = _window_sums(notionals, start_idx, end_idx)
    signed_qty_sum = _window_sums(quantities * directions, start_idx, end_idx)
    signed_notional_sum = _window_sums(notionals * directions, start_idx, end_idx)
    buy_count = _window_sums(directions > 0, start_idx, end_idx)

    valid_qty = valid & (qty_sum > 0)
    result_array[valid_qty, 14] = signed_qty_sum[valid_qty] / qty_sum[valid_qty]
    valid_notional = valid & (notional_sum > 0)
    result_array[valid_notional, 17] = (
        signed_notional_sum[valid_notional] / notional_sum[valid_notional]
    )
    p_hat = np.zeros(len(quote_index), dtype=float)
    p_hat[valid] = buy_count[valid] / trade_count[valid]
    result_array[valid, 4] = (
        (p_hat[valid] - 0.5) / np.sqrt(0.25 / trade_count[valid])
    )

    returns = np.diff(prices)
    pair_end = np.clip(end_idx - 1, 0, len(returns))
    pair_start = np.minimum(np.clip(start_idx, 0, len(returns)), pair_end)
    pair_count = pair_end - pair_start
    returns_sum = _window_sums(returns, pair_start, pair_end)
    returns_square_sum = _window_sums(np.square(returns), pair_start, pair_end)
    returns_mean = np.zeros(len(quote_index), dtype=float)
    returns_variance = np.zeros(len(quote_index), dtype=float)
    has_returns = pair_count > 0
    returns_mean[has_returns] = returns_sum[has_returns] / pair_count[has_returns]
    returns_variance[has_returns] = np.maximum(
        returns_square_sum[has_returns] / pair_count[has_returns]
        - np.square(returns_mean[has_returns]),
        0.0,
    )
    for idx in np.flatnonzero(has_returns & (returns_variance == 0)):
        window_returns = returns[pair_start[idx] : pair_end[idx]]
        returns_mean[idx] = window_returns.mean()
        returns_variance[idx] = window_returns.var()
    returns_std = np.sqrt(returns_variance)

    persistence_sum = _window_sums(
        (directions[1:] == directions[:-1]).astype(float),
        pair_start,
        pair_end,
    )
    result_array[valid, 1] = (
        persistence_sum[valid] / pair_count[valid] * 2.0 - 1.0
    )

    price_impacts = np.abs(returns) * quantities[1:]
    impact_history_end = np.clip(end_idx - 2, 0, len(price_impacts))
    impact_history_start = np.minimum(pair_start, impact_history_end)
    impact_history_count = impact_history_end - impact_history_start
    impact_history_sum = _window_sums(
        price_impacts, impact_history_start, impact_history_end
    )
    impact_history_square_sum = _window_sums(
        np.square(price_impacts), impact_history_start, impact_history_end
    )
    impact_history_mean = np.zeros(len(quote_index), dtype=float)
    impact_history_variance = np.zeros(len(quote_index), dtype=float)
    has_impact_history = valid & (impact_history_count > 0)
    impact_history_mean[has_impact_history] = (
        impact_history_sum[has_impact_history]
        / impact_history_count[has_impact_history]
    )
    impact_history_variance[has_impact_history] = np.maximum(
        impact_history_square_sum[has_impact_history]
        / impact_history_count[has_impact_history]
        - np.square(impact_history_mean[has_impact_history]),
        0.0,
    )
    for idx in np.flatnonzero(
        has_impact_history & (impact_history_variance == 0)
    ):
        history_impacts = price_impacts[
            impact_history_start[idx] : impact_history_end[idx]
        ]
        impact_history_mean[idx] = history_impacts.mean()
        impact_history_variance[idx] = history_impacts.var()
    latest_impact = np.zeros(len(quote_index), dtype=float)
    latest_impact[valid] = price_impacts[end_idx[valid] - 2]
    positive_impact_std = has_impact_history & (impact_history_variance > 0)
    result_array[positive_impact_std, 2] = (
        latest_impact[positive_impact_std]
        - impact_history_mean[positive_impact_std]
    ) / np.sqrt(impact_history_variance[positive_impact_std])
    constant_impact = (
        has_impact_history
        & (impact_history_variance == 0)
        & (latest_impact == impact_history_mean)
    )
    result_array[constant_impact, 2] = 0.0

    first_prices = prices[np.minimum(start_idx, len(prices) - 1)]

    enough_returns = valid & (pair_count >= 19)
    positive_returns_std = enough_returns & (returns_std > 0)
    result_array[positive_returns_std, 6] = (
        returns_mean[positive_returns_std] / returns_std[positive_returns_std]
    )
    qty_after_first_sum = _window_sums(
        quantities, np.minimum(start_idx + 1, len(quantities)), end_idx
    )
    result_array[positive_returns_std, 15] = (
        qty_after_first_sum[positive_returns_std]
        / pair_count[positive_returns_std]
        / (returns_std[positive_returns_std] * 100.0)
    )
    last_prices = prices[np.maximum(end_idx - 1, 0)]
    valid_vwap = (
        enough_returns
        & (qty_sum > 0)
        & np.isfinite(last_prices)
        & (last_prices > 0)
    )
    window_vwap = np.zeros(len(quote_index), dtype=float)
    window_vwap[valid_vwap] = notional_sum[valid_vwap] / qty_sum[valid_vwap]
    result_array[valid_vwap, 16] = (
        window_vwap[valid_vwap] - last_prices[valid_vwap]
    ) / last_prices[valid_vwap]

    enough_volume = valid & (trade_count >= 20)
    prior_qty_sum = _window_sums(
        quantities, start_idx, np.maximum(end_idx - 1, 0)
    )
    prior_qty_mean = np.zeros(len(quote_index), dtype=float)
    prior_qty_mean[enough_volume] = (
        prior_qty_sum[enough_volume] / (trade_count[enough_volume] - 1)
    )
    valid_volume_spike = enough_volume & (prior_qty_mean > 0)
    result_array[valid_volume_spike, 8] = np.maximum(
        0.0,
        quantities[end_idx[valid_volume_spike] - 1]
        / prior_qty_mean[valid_volume_spike]
        - 2.0,
    )

    momentum = np.diff(returns)
    momentum_end = np.clip(end_idx - 2, 0, len(momentum))
    momentum_start = np.minimum(np.clip(start_idx, 0, len(momentum)), momentum_end)
    momentum_count = momentum_end - momentum_start
    momentum_sum = _window_sums(momentum, momentum_start, momentum_end)
    momentum_square_sum = _window_sums(
        np.square(momentum), momentum_start, momentum_end
    )
    enough_momentum = valid & (momentum_count >= 10)
    momentum_mean = np.zeros(len(quote_index), dtype=float)
    momentum_mean[enough_momentum] = (
        momentum_sum[enough_momentum] / momentum_count[enough_momentum]
    )
    momentum_variance = np.zeros(len(quote_index), dtype=float)
    momentum_variance[enough_momentum] = np.maximum(
        momentum_square_sum[enough_momentum] / momentum_count[enough_momentum]
        - np.square(momentum_mean[enough_momentum]),
        0.0,
    )
    for idx in np.flatnonzero(enough_momentum & (momentum_variance == 0)):
        window_momentum = momentum[momentum_start[idx] : momentum_end[idx]]
        momentum_mean[idx] = window_momentum.mean()
        momentum_variance[idx] = window_momentum.var()
    positive_momentum_std = enough_momentum & (momentum_variance > 0)
    result_array[positive_momentum_std, 7] = (
        momentum_mean[positive_momentum_std]
        / np.sqrt(momentum_variance[positive_momentum_std])
    )

    enough_structure = valid & (pair_count >= 10)
    price_change_bps = np.abs(returns) / prices[:-1] * 10000.0
    price_change_bps_sum = _window_sums(
        price_change_bps, pair_start, pair_end
    )
    mean_price_change_bps = np.zeros(len(quote_index), dtype=float)
    mean_price_change_bps[enough_structure] = (
        price_change_bps_sum[enough_structure]
        / pair_count[enough_structure]
    )
    valid_liquidity_depth = enough_structure & (mean_price_change_bps > 0)
    result_array[valid_liquidity_depth, 10] = (
        qty_after_first_sum[valid_liquidity_depth]
        / pair_count[valid_liquidity_depth]
        / mean_price_change_bps[valid_liquidity_depth]
    )

    log_qty_returns = np.diff(np.log(quantities + 1.0))
    log_qty_sum = _window_sums(log_qty_returns, pair_start, pair_end)
    log_qty_square_sum = _window_sums(
        np.square(log_qty_returns), pair_start, pair_end
    )
    returns_log_qty_sum = _window_sums(
        returns * log_qty_returns, pair_start, pair_end
    )
    log_qty_mean = np.zeros(len(quote_index), dtype=float)
    log_qty_mean[enough_structure] = (
        log_qty_sum[enough_structure] / pair_count[enough_structure]
    )
    log_qty_variance = np.zeros(len(quote_index), dtype=float)
    log_qty_variance[enough_structure] = np.maximum(
        log_qty_square_sum[enough_structure] / pair_count[enough_structure]
        - np.square(log_qty_mean[enough_structure]),
        0.0,
    )
    returns_log_qty_covariance = np.zeros(len(quote_index), dtype=float)
    returns_log_qty_covariance[enough_structure] = (
        returns_log_qty_sum[enough_structure] / pair_count[enough_structure]
        - returns_mean[enough_structure] * log_qty_mean[enough_structure]
    )
    valid_decoupling = (
        enough_structure & (returns_variance > 0) & (log_qty_variance > 0)
    )
    correlation = np.zeros(len(quote_index), dtype=float)
    correlation[valid_decoupling] = (
        returns_log_qty_covariance[valid_decoupling]
        / np.sqrt(
            returns_variance[valid_decoupling]
            * log_qty_variance[valid_decoupling]
        )
    )
    finite_correlation = valid_decoupling & np.isfinite(correlation)
    result_array[finite_correlation, 11] = 1.0 - np.abs(
        correlation[finite_correlation]
    )

    path_length = _window_sums(np.abs(returns), pair_start, pair_end)
    valid_efficiency = enough_structure & (path_length > 0)
    result_array[valid_efficiency, 12] = (
        np.abs(last_prices[valid_efficiency] - first_prices[valid_efficiency])
        / path_length[valid_efficiency]
    )

    half = trade_count // 2
    midpoint = start_idx + half
    first_half_sum = _window_sums(quantities, start_idx, midpoint)
    second_half_sum = _window_sums(quantities, midpoint, end_idx)
    first_half_mean = np.zeros(len(quote_index), dtype=float)
    second_half_mean = np.zeros(len(quote_index), dtype=float)
    first_half_mean[valid] = first_half_sum[valid] / half[valid]
    second_half_mean[valid] = (
        second_half_sum[valid] / (trade_count[valid] - half[valid])
    )
    valid_migration = valid & (first_half_mean > 0)
    result_array[valid_migration, 13] = (
        second_half_mean[valid_migration] - first_half_mean[valid_migration]
    ) / first_half_mean[valid_migration]

    for idx in np.flatnonzero(valid):
        start = start_idx[idx]
        end = end_idx[idx]
        window_prices = prices[start:end]
        window_notionals = notionals[start:end]
        result_array[idx, 0] = _size_distribution_score(window_notionals)

        cumulative_signed_millions = np.cumsum(
            window_notionals * directions[start:end]
        ) / 1_000_000.0
        x_centered = cumulative_signed_millions - cumulative_signed_millions.mean()
        x_variance = np.mean(np.square(x_centered))
        if x_variance > 0:
            relative_price_bps = (
                window_prices / window_prices[0] - 1.0
            ) * 10000.0
            slope = np.mean(
                x_centered * (relative_price_bps - relative_price_bps.mean())
            ) / x_variance
            if np.isfinite(slope):
                result_array[idx, 3] = slope

        if pair_count[idx] >= 19:
            chunk_size = 5
            chunk_count = pair_count[idx] // chunk_size
            if chunk_count > 0:
                trimmed_returns = returns[
                    pair_end[idx] - chunk_count * chunk_size : pair_end[idx]
                ]
                avg_vol = np.mean(
                    np.std(
                        trimmed_returns.reshape(chunk_count, chunk_size),
                        axis=1,
                    )
                )
                mean_volume = qty_sum[idx] / trade_count[idx]
                if avg_vol > 0 and mean_volume > 0:
                    result_array[idx, 5] = (
                        quantities[end - 1]
                        * (returns_std[idx] / avg_vol)
                        / mean_volume
                    )

        if trade_count[idx] >= 30:
            window_qty = quantities[start:end]
            avg_volume = qty_sum[idx] / trade_count[idx]
            high_volume = window_qty > avg_volume
            high_count = int(high_volume[:-1].sum())
            if high_count > 0:
                high_to_high = int(
                    np.logical_and(high_volume[:-1], high_volume[1:]).sum()
                )
                result_array[idx, 9] = high_to_high / high_count * 2.0 - 1.0

    return pd.DataFrame(result_array, index=quote_index, columns=factor_columns)


def calculate_order_flow_factors(
    orders: pd.DataFrame,
    quote_index: pd.DatetimeIndex,
    window: str = ORDER_WINDOW,
    window_profile: str = WINDOW_PROFILE_BASE,
) -> pd.DataFrame:
    _validate_window_profile(window_profile)
    base_factor_columns = [
        "order_count_imbalance_60s",
        "order_qty_imbalance_60s",
        "order_notional_imbalance_60s",
    ]
    factor_columns = list(base_factor_columns)
    if window_profile == WINDOW_PROFILE_MULTI:
        factor_columns.extend(
            f"order_{metric}_imbalance_{multi_window}"
            for multi_window in FLOW_WINDOWS
            if multi_window != window
            for metric in ("count", "qty", "notional")
        )
    if orders.empty:
        return pd.DataFrame(0.0, index=quote_index, columns=factor_columns)

    order_metrics = orders.copy()
    order_metrics["buy_count"] = (order_metrics["side"] == "B").astype(float)
    order_metrics["sell_count"] = (order_metrics["side"] == "S").astype(float)
    order_metrics["buy_qty"] = np.where(order_metrics["side"] == "B", order_metrics["qty"], 0.0)
    order_metrics["sell_qty"] = np.where(order_metrics["side"] == "S", order_metrics["qty"], 0.0)
    order_metrics["buy_notional"] = np.where(
        order_metrics["side"] == "B", order_metrics["notional"], 0.0
    )
    order_metrics["sell_notional"] = np.where(
        order_metrics["side"] == "S", order_metrics["notional"], 0.0
    )

    value_columns = [
        "buy_count",
        "sell_count",
        "buy_qty",
        "sell_qty",
        "buy_notional",
        "sell_notional",
    ]
    rolling = _rolling_event_sums_at_quotes(
        order_metrics, quote_index, value_columns, window
    )
    rolling["order_count_imbalance_60s"] = _imbalance(rolling["buy_count"], rolling["sell_count"])
    rolling["order_qty_imbalance_60s"] = _imbalance(rolling["buy_qty"], rolling["sell_qty"])
    rolling["order_notional_imbalance_60s"] = _imbalance(
        rolling["buy_notional"], rolling["sell_notional"]
    )
    result = rolling[base_factor_columns].fillna(0.0)

    if window_profile == WINDOW_PROFILE_MULTI:
        for multi_window in FLOW_WINDOWS:
            if multi_window == window:
                continue
            windowed = _rolling_event_sums_at_quotes(
                order_metrics, quote_index, value_columns, multi_window
            )
            result[f"order_count_imbalance_{multi_window}"] = _imbalance(
                windowed["buy_count"], windowed["sell_count"]
            )
            result[f"order_qty_imbalance_{multi_window}"] = _imbalance(
                windowed["buy_qty"], windowed["sell_qty"]
            )
            result[f"order_notional_imbalance_{multi_window}"] = _imbalance(
                windowed["buy_notional"], windowed["sell_notional"]
            )
    return result.fillna(0.0)


def calculate_vpin_factor(
    trades: pd.DataFrame,
    quote_index: pd.DatetimeIndex,
    bucket_volume: float | None = None,
    num_buckets: int = VPIN_NUM_BUCKETS,
) -> pd.Series:
    """Causal VPIN from completed volume buckets, reset each trade date.

    When ``bucket_volume`` is omitted, each new bucket is sized from the causal
    EWMA trade size available when that bucket starts. This avoids a fixed share
    threshold that has incompatible meanings across stocks and ETFs.
    """
    result = pd.Series(np.nan, index=quote_index, name="vpin_50bucket", dtype=float)
    if trades.empty:
        return result
    if (bucket_volume is not None and bucket_volume <= 0) or num_buckets <= 0:
        raise ValueError("bucket_volume and num_buckets must be positive")

    event_times = pd.DatetimeIndex(pd.to_datetime(trades["event_time"]))
    event_sessions = _trading_session_labels(event_times)
    quote_sessions = _trading_session_labels(quote_index)
    sides = trades["side"].to_numpy(dtype=str, copy=False)
    quantities = trades["qty"].to_numpy(dtype=float, copy=False)

    for session in pd.unique(event_sessions):
        event_positions = np.flatnonzero(event_sessions == session)
        quote_positions = np.flatnonzero(quote_sessions == session)
        if not len(quote_positions):
            continue
        completed_times: list[pd.Timestamp] = []
        bucket_imbalances: list[float] = []
        buy_volume = sell_volume = filled = 0.0
        adaptive_target: float | None = bucket_volume
        ewma_trade_size: float | None = None
        ewma_alpha = 2.0 / (VPIN_SIZE_EWMA_SPAN + 1.0)
        for position in event_positions:
            remaining = quantities[position]
            if not np.isfinite(remaining) or remaining <= 0:
                continue
            trade_size = remaining
            while remaining > 0:
                if adaptive_target is None:
                    reference_size = (
                        ewma_trade_size if ewma_trade_size is not None else trade_size
                    )
                    adaptive_target = max(
                        reference_size * VPIN_TARGET_TRADES_PER_BUCKET, 1.0
                    )
                allocation = min(remaining, adaptive_target - filled)
                if sides[position] == "B":
                    buy_volume += allocation
                else:
                    sell_volume += allocation
                filled += allocation
                remaining -= allocation
                if filled >= adaptive_target - 1e-12:
                    completed_times.append(event_times[position])
                    bucket_imbalances.append(
                        abs(buy_volume - sell_volume) / adaptive_target
                    )
                    buy_volume = sell_volume = filled = 0.0
                    adaptive_target = bucket_volume
            ewma_trade_size = (
                trade_size
                if ewma_trade_size is None
                else ewma_alpha * trade_size + (1.0 - ewma_alpha) * ewma_trade_size
            )

        if len(bucket_imbalances) < num_buckets:
            continue
        bucket_series = pd.Series(bucket_imbalances, index=completed_times)
        bucket_vpin = bucket_series.rolling(
            num_buckets, min_periods=num_buckets
        ).mean()
        bucket_frame = bucket_vpin.rename("vpin").rename_axis("time").reset_index()
        aligned = pd.merge_asof(
            pd.DataFrame({"time": quote_index[quote_positions]}),
            bucket_frame,
            on="time",
            direction="backward",
        )
        result.iloc[quote_positions] = aligned["vpin"].to_numpy(dtype=float)
    return result


def calculate_amihud_illiquidity_factor(
    trades: pd.DataFrame,
    quote_index: pd.DatetimeIndex,
    window: str = AMIHUD_WINDOW,
    min_returns: int = AMIHUD_MIN_RETURNS,
) -> pd.Series:
    """Mean absolute trade return per traded notional over a causal window."""
    result = pd.Series(
        np.nan, index=quote_index, name="amihud_illiquidity_5m", dtype=float
    )
    if trades.empty:
        return result
    if min_returns <= 0:
        raise ValueError("min_returns must be positive")

    ordered = trades.sort_values("event_time", kind="stable")
    event_times = pd.DatetimeIndex(pd.to_datetime(ordered["event_time"]))
    trading_times = _trading_time_index(event_times)
    sessions = _trading_session_labels(event_times)
    prices = ordered["price"].to_numpy(dtype=float, copy=False)
    notionals = ordered["notional"].to_numpy(dtype=float, copy=False)
    previous_prices = np.full(len(ordered), np.nan, dtype=float)
    previous_times = np.full(
        len(ordered), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
    for session in pd.unique(sessions):
        positions = np.flatnonzero(sessions == session)
        if len(positions) > 1:
            previous_prices[positions[1:]] = prices[positions[:-1]]
            previous_times[positions[1:]] = trading_times[positions[:-1]].to_numpy(
                dtype="datetime64[ns]", copy=False
            )

    event_time_values = trading_times.to_numpy(dtype="datetime64[ns]", copy=False)
    within_window = (
        event_time_values - previous_times
        <= pd.Timedelta(window).to_timedelta64()
    )
    valid = (
        np.isfinite(prices)
        & (prices > 0)
        & np.isfinite(previous_prices)
        & (previous_prices > 0)
        & np.isfinite(notionals)
        & (notionals > 0)
        & within_window
    )
    components = np.zeros(len(ordered), dtype=float)
    components[valid] = (
        np.abs(prices[valid] / previous_prices[valid] - 1.0) / notionals[valid]
    )
    metrics = pd.DataFrame(
        {
            "event_time": event_times,
            "amihud_sum": components,
            "amihud_count": valid.astype(float),
        }
    )
    rolling = _rolling_event_sums_at_quotes(
        metrics, quote_index, ["amihud_sum", "amihud_count"], window
    )
    enough_history = rolling["amihud_count"] >= min_returns
    result.loc[enough_history] = (
        rolling.loc[enough_history, "amihud_sum"]
        / rolling.loc[enough_history, "amihud_count"]
    )
    return result


def _calculate_trade_report_factors(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    window: str = TRADE_WINDOW,
) -> pd.DataFrame:
    """Calculate causal trade-structure factors proposed by the report review."""
    columns = [
        "cautious_to_aggressive_buy_ratio_60s",
        "trade_notional_quantile_position_60s",
        "price_band_high_trade_count_share_60s",
        "price_band_low_trade_count_share_60s",
        "price_band_high_trade_size_rel_60s",
        "price_band_low_trade_size_rel_60s",
    ]
    result_array = np.full((len(quotes), len(columns)), np.nan, dtype=float)
    if trades.empty:
        return pd.DataFrame(result_array, index=quotes.index, columns=columns)

    trade_frame = trades.sort_values("event_time", kind="stable").copy()
    trade_times = pd.DatetimeIndex(pd.to_datetime(trade_frame["event_time"]))
    trading_times = _trading_time_index(trade_times)
    prices = trade_frame["price"].to_numpy(dtype=float, copy=False)
    quantities = trade_frame["qty"].to_numpy(dtype=float, copy=False)
    notionals = trade_frame["notional"].to_numpy(dtype=float, copy=False)
    trade_sessions = _trading_session_labels(trade_times)
    aggressive_buy_qty = np.zeros(len(trade_frame), dtype=float)
    cautious_buy_qty = np.zeros(len(trade_frame), dtype=float)
    if {"ask_price1", "bid_price1"} <= set(quotes.columns):
        quote_reference = pd.DataFrame(
            {
                "quote_time": quotes.index,
                "previous_ask1": quotes["ask_price1"].to_numpy(
                    dtype=float, copy=False
                ),
                "previous_bid1": quotes["bid_price1"].to_numpy(
                    dtype=float, copy=False
                ),
                "quote_session": _trading_session_labels(quotes.index),
            }
        )
        matched_quotes = pd.merge_asof(
            trade_frame[["event_time"]],
            quote_reference,
            left_on="event_time",
            right_on="quote_time",
            direction="backward",
            allow_exact_matches=False,
        )
        same_session = matched_quotes["quote_session"].to_numpy() == trade_sessions
        prior_ask = matched_quotes["previous_ask1"].to_numpy(dtype=float)
        prior_bid = matched_quotes["previous_bid1"].to_numpy(dtype=float)
        aggressive_buy_qty = np.where(
            same_session & np.isfinite(prior_ask) & (prices >= prior_ask), quantities, 0.0
        )
        cautious_buy_qty = np.where(
            same_session & np.isfinite(prior_bid) & (prices <= prior_bid), quantities, 0.0
        )

    low_band = np.full(len(trade_frame), np.nan, dtype=float)
    high_band = np.full(len(trade_frame), np.nan, dtype=float)
    for session in pd.unique(trade_sessions):
        positions = np.flatnonzero(trade_sessions == session)
        session_prices = pd.Series(prices[positions], index=trading_times[positions])
        history = session_prices.rolling(
            PRICE_BAND_HISTORY_WINDOW,
            min_periods=PRICE_BAND_MIN_HISTORY,
            closed="left",
        )
        low_band[positions] = history.quantile(0.2).to_numpy(dtype=float)
        high_band[positions] = history.quantile(0.8).to_numpy(dtype=float)

    window_delta = pd.Timedelta(window).to_timedelta64()
    quote_times = _trading_time_index(quotes.index).to_numpy(
        dtype="datetime64[ns]", copy=False
    )
    trade_times_array = trading_times.to_numpy(dtype="datetime64[ns]", copy=False)
    start_idx = np.searchsorted(trade_times_array, quote_times - window_delta, side="left")
    end_idx = np.searchsorted(trade_times_array, quote_times, side="right")

    aggressive = _window_sums(aggressive_buy_qty, start_idx, end_idx)
    cautious = _window_sums(cautious_buy_qty, start_idx, end_idx)
    valid_aggressive = aggressive > 0
    result_array[valid_aggressive, 0] = (
        cautious[valid_aggressive] / aggressive[valid_aggressive]
    )

    valid_bands = np.isfinite(low_band) & np.isfinite(high_band)
    high_mask = valid_bands & (prices >= high_band)
    low_mask = valid_bands & (prices <= low_band)
    valid_count = _window_sums(valid_bands, start_idx, end_idx)
    high_count = _window_sums(high_mask, start_idx, end_idx)
    low_count = _window_sums(low_mask, start_idx, end_idx)
    has_valid_bands = valid_count > 0
    result_array[has_valid_bands, 2] = (
        high_count[has_valid_bands] / valid_count[has_valid_bands]
    )
    result_array[has_valid_bands, 3] = (
        low_count[has_valid_bands] / valid_count[has_valid_bands]
    )

    valid_qty = _window_sums(
        np.where(valid_bands, quantities, 0.0), start_idx, end_idx
    )
    high_qty = _window_sums(
        np.where(high_mask, quantities, 0.0), start_idx, end_idx
    )
    low_qty = _window_sums(
        np.where(low_mask, quantities, 0.0), start_idx, end_idx
    )
    valid_high_size = has_valid_bands & (valid_qty > 0) & (high_count > 0)
    valid_low_size = has_valid_bands & (valid_qty > 0) & (low_count > 0)
    result_array[valid_high_size, 4] = (
        high_qty[valid_high_size]
        * valid_count[valid_high_size]
        / (high_count[valid_high_size] * valid_qty[valid_high_size])
    )
    result_array[valid_low_size, 5] = (
        low_qty[valid_low_size]
        * valid_count[valid_low_size]
        / (low_count[valid_low_size] * valid_qty[valid_low_size])
    )

    for idx, (start, end) in enumerate(zip(start_idx, end_idx, strict=False)):
        if end - start > 10:
            trimmed_notional = np.sort(notionals[start:end])[:-10]
            lower = trimmed_notional.min()
            upper = trimmed_notional.max()
            if upper > lower:
                result_array[idx, 1] = (np.quantile(trimmed_notional, 0.1) - lower) / (
                    upper - lower
                )

    return pd.DataFrame(result_array, index=quotes.index, columns=columns)


def _calculate_adverse_selection_markout(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    horizon: str = MARKOUT_HORIZON,
    rolling_window: str = MARKOUT_ROLLING_WINDOW,
) -> pd.Series:
    """Average signed trade markout, exposed only once its horizon has matured."""
    result = pd.Series(
        np.nan, index=quotes.index, name="adverse_selection_markout_30s", dtype=float
    )
    if trades.empty:
        return result

    horizon_delta = pd.Timedelta(horizon)
    rolling_delta = pd.Timedelta(rolling_window)
    trade_times = pd.DatetimeIndex(pd.to_datetime(trades["event_time"]))
    trading_trade_times = _trading_time_index(trade_times)
    trade_sessions = _trading_session_labels(trade_times)
    quote_sessions = _trading_session_labels(quotes.index)
    trading_quote_times = _trading_time_index(quotes.index)
    directions = np.where(trades["side"].to_numpy(dtype=str) == "B", 1.0, -1.0)
    mid = quotes["mid_price"].to_numpy(dtype=float, copy=False)

    for session in pd.unique(trade_sessions):
        trade_positions = np.flatnonzero(trade_sessions == session)
        quote_positions = np.flatnonzero(quote_sessions == session)
        if not len(quote_positions):
            continue
        session_quote_times = trading_quote_times[quote_positions]
        session_trade_times = trading_trade_times[trade_positions]
        initial_offsets = session_quote_times.searchsorted(session_trade_times, side="right") - 1
        mature_offsets = session_quote_times.searchsorted(
            session_trade_times + horizon_delta, side="left"
        )
        valid = (initial_offsets >= 0) & (mature_offsets < len(session_quote_times))
        if not np.any(valid):
            continue

        valid_trade_positions = trade_positions[valid]
        valid_initial_offsets = initial_offsets[valid]
        valid_mature_offsets = mature_offsets[valid]
        initial_mid = mid[quote_positions[valid_initial_offsets]]
        mature_mid = mid[quote_positions[valid_mature_offsets]]
        finite = np.isfinite(initial_mid) & (initial_mid > 0) & np.isfinite(mature_mid)
        if not np.any(finite):
            continue

        valid_trade_positions = valid_trade_positions[finite]
        valid_mature_offsets = valid_mature_offsets[finite]
        initial_mid = initial_mid[finite]
        mature_mid = mature_mid[finite]
        maturity_times = session_quote_times[valid_mature_offsets]
        markouts = (
            directions[valid_trade_positions]
            * (mature_mid - initial_mid)
            / initial_mid
            * 10000.0
        )

        maturity_ns = pd.DatetimeIndex(maturity_times).astype("datetime64[ns]").asi8
        order = np.argsort(maturity_ns, kind="stable")
        maturity_ns = maturity_ns[order]
        markout_values = np.asarray(markouts, dtype=float)[order]
        quote_ns = session_quote_times.astype("datetime64[ns]").asi8
        starts = np.searchsorted(maturity_ns, quote_ns - rolling_delta.value, side="left")
        ends = np.searchsorted(maturity_ns, quote_ns, side="right")
        cumulative = np.r_[0.0, np.cumsum(markout_values)]
        counts = ends - starts
        valid = counts > 0
        session_result = np.full(len(quote_positions), np.nan, dtype=float)
        session_result[valid] = (
            cumulative[ends[valid]] - cumulative[starts[valid]]
        ) / counts[valid]
        result.iloc[quote_positions] = session_result
    return result


def _calculate_trade_window_summary(
    trade_metrics: pd.DataFrame,
    quotes: pd.DataFrame,
    window: str,
) -> pd.DataFrame:
    """Aggregate simple signed trade flow over one causal event-time window."""
    rolling = _rolling_event_sums_at_quotes(
        trade_metrics,
        quotes.index,
        ["buy_count", "sell_count", "buy_qty", "sell_qty", "qty", "notional"],
        window,
    )
    suffix = window
    result = pd.DataFrame(index=quotes.index)
    result[f"trade_count_imbalance_{suffix}"] = _imbalance(
        rolling["buy_count"], rolling["sell_count"]
    )
    result[f"trade_qty_imbalance_{suffix}"] = _imbalance(
        rolling["buy_qty"], rolling["sell_qty"]
    )
    trade_vwap = _safe_divide(rolling["notional"], rolling["qty"])
    result[f"trade_vwap_gap_{suffix}"] = _safe_divide(
        trade_vwap - quotes["mid_price"].to_numpy(dtype=float, copy=False),
        quotes["mid_price"].to_numpy(dtype=float, copy=False),
    )
    result[f"trade_count_imbalance_{suffix}"] = result[
        f"trade_count_imbalance_{suffix}"
    ].fillna(0.0)
    result[f"trade_qty_imbalance_{suffix}"] = result[
        f"trade_qty_imbalance_{suffix}"
    ].fillna(0.0)
    return result


def calculate_trade_flow_factors(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    window: str = TRADE_WINDOW,
    window_profile: str = WINDOW_PROFILE_BASE,
) -> pd.DataFrame:
    _validate_window_profile(window_profile)
    factor_columns = [
        "trade_count_imbalance_60s",
        "trade_qty_imbalance_60s",
        "trade_vwap_gap_60s",
        "trade_size_distribution_60s",
        "trade_direction_persistence_60s",
        "liquidity_shock_60s",
        "market_impact_60s",
        "amihud_illiquidity_5m",
        "orderflow_significance_60s",
        "volatility_adj_volume_60s",
        "price_velocity_60s",
        "momentum_acceleration_60s",
        "volume_spike_60s",
        "volume_clustering_60s",
        "liquidity_depth_60s",
        "price_volume_decoupling_60s",
        "market_efficiency_60s",
        "liquidity_migration_60s",
        "order_flow_imbalance_60s",
        "liquidity_ratio_60s",
        "volume_weighted_price_60s",
        "orderbook_pressure_60s",
        "vpin_50bucket",
        "adverse_selection_markout_30s",
        "cautious_to_aggressive_buy_ratio_60s",
        "trade_notional_quantile_position_60s",
        "price_band_high_trade_count_share_60s",
        "price_band_low_trade_count_share_60s",
        "price_band_high_trade_size_rel_60s",
        "price_band_low_trade_size_rel_60s",
    ]
    multi_summary_columns = [
        f"trade_{metric}_{multi_window}"
        for multi_window in FLOW_WINDOWS
        if multi_window != window
        for metric in ("count_imbalance", "qty_imbalance", "vwap_gap")
    ]
    multi_impact_columns = [
        f"{base}_{multi_window}"
        for multi_window in IMPACT_WINDOWS
        if multi_window != window
        for base in (
            "trade_size_distribution",
            "trade_direction_persistence",
            "liquidity_shock",
            "market_impact",
            "orderflow_significance",
            "volatility_adj_volume",
            "price_velocity",
            "momentum_acceleration",
            "volume_spike",
            "volume_clustering",
            "liquidity_depth",
            "price_volume_decoupling",
            "market_efficiency",
            "liquidity_migration",
            "order_flow_imbalance",
            "liquidity_ratio",
            "volume_weighted_price",
            "orderbook_pressure",
        )
    ]
    if trades.empty:
        empty = pd.DataFrame(0.0, index=quotes.index, columns=factor_columns[:2])
        empty["trade_vwap_gap_60s"] = np.nan
        for column in factor_columns[3:]:
            empty[column] = np.nan
        if window_profile == WINDOW_PROFILE_MULTI:
            for column in multi_summary_columns:
                empty[column] = 0.0 if "imbalance" in column else np.nan
            for column in multi_impact_columns:
                empty[column] = np.nan
        return empty

    trade_metrics = trades.copy()
    trade_metrics["buy_count"] = (trade_metrics["side"] == "B").astype(float)
    trade_metrics["sell_count"] = (trade_metrics["side"] == "S").astype(float)
    trade_metrics["buy_qty"] = np.where(trade_metrics["side"] == "B", trade_metrics["qty"], 0.0)
    trade_metrics["sell_qty"] = np.where(trade_metrics["side"] == "S", trade_metrics["qty"], 0.0)

    result = _calculate_trade_window_summary(trade_metrics, quotes, window)
    advanced = _calculate_trade_impact_factors(trades, quotes.index, window)
    result = pd.concat([result, advanced], axis=1)
    result["vpin_50bucket"] = calculate_vpin_factor(trades, quotes.index)
    result["amihud_illiquidity_5m"] = calculate_amihud_illiquidity_factor(
        trades, quotes.index
    )
    result["adverse_selection_markout_30s"] = _calculate_adverse_selection_markout(
        trades, quotes
    )
    result = pd.concat([result, _calculate_trade_report_factors(trades, quotes)], axis=1)
    if window_profile == WINDOW_PROFILE_MULTI:
        for multi_window in FLOW_WINDOWS:
            if multi_window != window:
                result = result.join(
                    _calculate_trade_window_summary(
                        trade_metrics, quotes, multi_window
                    )
                )
        for multi_window in IMPACT_WINDOWS:
            if multi_window != window:
                result = result.join(
                    _calculate_trade_impact_factors(
                        trades, quotes.index, multi_window
                    )
                )
    result.index = quotes.index
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_contextual_orderflow_factors(
    snapshot_factors: pd.DataFrame,
    trade_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Select unusual order-flow segments conditional on the current LOB state.

    A segment is unusual when the actual L5 book differs from its causal expected
    book and/or its signed traded quantity differs from its own recent history.
    The selection threshold is the preceding 15-minute 98th percentile, mirroring
    the paper's top-mu signal selection without using future observations.
    """
    index = snapshot_factors.index
    flow_surprise = _causal_rolling_zscore(
        trade_factors["trade_qty_imbalance_60s"].to_numpy(dtype=float, copy=False),
        index,
        CONTEXT_SELECTION_WINDOW,
        CONTEXT_SELECTION_MIN_PERIODS,
    )
    components = np.column_stack(
        [
            snapshot_factors["contextual_lob_surprise_l5"].to_numpy(dtype=float, copy=False),
            snapshot_factors["contextual_imbalance_surprise_l5"].to_numpy(dtype=float, copy=False),
            flow_surprise,
        ]
    )
    available = np.isfinite(components).any(axis=1)
    score = np.sqrt(np.nansum(np.square(components), axis=1))
    score[~available] = np.nan

    threshold = np.full(len(index), np.nan, dtype=float)
    day_labels = index.normalize()
    trading_index = _trading_time_index(index)
    for day in pd.unique(day_labels):
        positions = np.flatnonzero(day_labels == day)
        history = pd.Series(score[positions], index=trading_index[positions]).shift(1)
        threshold[positions] = history.rolling(
            CONTEXT_SELECTION_WINDOW,
            min_periods=CONTEXT_SELECTION_MIN_PERIODS,
        ).quantile(CONTEXT_SELECTION_QUANTILE).to_numpy(dtype=float)
    selected = np.isfinite(score) & np.isfinite(threshold) & (score >= threshold)

    result = pd.DataFrame(index=index)
    result["contextual_flow_surprise_60s"] = flow_surprise
    result["contextual_segment_anomaly_60s"] = score
    result["contextual_segment_selected_60s"] = selected.astype(float)
    result["contextual_selected_flow_imbalance_60s"] = np.where(
        selected,
        trade_factors["trade_qty_imbalance_60s"].to_numpy(dtype=float, copy=False),
        0.0,
    )
    result["contextual_selected_lob_surprise_60s"] = np.where(
        selected,
        snapshot_factors["contextual_lob_surprise_l5"].to_numpy(dtype=float, copy=False),
        0.0,
    )
    return result


def build_stock_orderbook_factor_frame(
    quotes: pd.DataFrame,
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    window_profile: str = WINDOW_PROFILE_BASE,
) -> pd.DataFrame:
    _validate_window_profile(window_profile)
    quotes = quotes.sort_index(kind="stable")
    quote_factors = calculate_snapshot_factors(quotes, window_profile)
    elapsed_seconds = _trading_time_index(quote_factors.index).to_series().diff(5).dt.total_seconds()
    quote_factors["orderbook_velocity_l5"] = _safe_divide(
        quote_factors["depth_imbalance_l5"].diff(5), elapsed_seconds
    )
    quote_input = quotes.join(quote_factors)
    order_factors = calculate_order_flow_factors(
        orders, quote_input.index, window_profile=window_profile
    )
    trade_factors = calculate_trade_flow_factors(
        trades, quote_input, window_profile=window_profile
    )
    contextual_factors = calculate_contextual_orderflow_factors(quote_factors, trade_factors)
    return pd.concat([quote_factors, order_factors, trade_factors, contextual_factors], axis=1)
