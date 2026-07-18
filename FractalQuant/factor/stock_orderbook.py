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
    """Normalize instantaneous and rolling OFI without crossing sessions."""
    instantaneous = _safe_divide(events, depth_scale)
    rolling = np.full(len(events), np.nan, dtype=float)
    session_labels = _trading_session_labels(index)

    for session in pd.unique(session_labels):
        positions = np.flatnonzero(session_labels == session)
        event_series = pd.Series(events[positions], index=index[positions])
        depth_series = pd.Series(depth_scale[positions], index=index[positions])
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


def _validate_window_profile(window_profile: str) -> None:
    if window_profile not in {WINDOW_PROFILE_BASE, WINDOW_PROFILE_MULTI}:
        raise ValueError(f"Unsupported window profile: {window_profile}")


def _trading_session_labels(index: pd.DatetimeIndex) -> np.ndarray:
    afternoon = (index.hour >= 13).astype(np.int64)
    normalized_ns = index.normalize().astype("datetime64[ns]").asi8
    return normalized_ns * 2 + afternoon


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
    for session in pd.unique(session_labels):
        positions = np.flatnonzero(session_labels == session)
        series = pd.Series(depth[positions], index=index[positions])
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

    for session in pd.unique(session_labels):
        positions = np.flatnonzero(session_labels == session)
        if not len(positions):
            continue
        session_events = level_events[positions].copy()
        session_events[0] = np.nan
        event_frame = pd.DataFrame(session_events, index=index[positions])
        depth_frame = pd.DataFrame(level_depth[positions], index=index[positions])

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

        x = pd.Series(normalized_mlofi_60s[positions], index=index[positions]).shift(1)
        y = (
            pd.Series(mid_price[positions], index=index[positions])
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


def _causal_rolling_zscore(
    values: np.ndarray | pd.Series,
    index: pd.DatetimeIndex,
    window: str | int,
    min_periods: int,
) -> np.ndarray:
    """Standardize against strictly preceding observations, independently per day."""
    result = np.full(len(index), np.nan, dtype=float)
    value_array = np.asarray(values, dtype=float)
    for day in pd.unique(index.normalize()):
        positions = np.flatnonzero(index.normalize() == day)
        series = pd.Series(value_array[positions], index=index[positions])
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
    result["ofi_level_entropy_l5"] = ofi_level_entropy_l5
    result["normalized_mlofi_l5"] = normalized_mlofi_l5
    result["normalized_mlofi_l5_60s"] = normalized_mlofi_l5_60s
    result["mlofi_event_50_l5"] = mlofi_event_50_l5
    result["mlofi_deep_divergence_l5"] = mlofi_deep_divergence_l5
    result["mlofi_impact_beta"] = mlofi_impact_beta
    result["weighted_depth_imbalance_l5"] = weighted_depth_imbalance_l5
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
    window_delta = pd.Timedelta(window).to_timedelta64()

    for session in pd.unique(quote_sessions):
        quote_positions = np.flatnonzero(quote_sessions == session)
        event_positions = np.flatnonzero(event_sessions == session)
        if not len(event_positions):
            continue
        session_event_times = event_times[event_positions].to_numpy(
            dtype="datetime64[ns]", copy=False
        )
        session_quote_times = quote_index[quote_positions].to_numpy(
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
    trade_times = trades["event_time"].to_numpy(dtype="datetime64[ns]", copy=False)
    quote_times = quote_index.to_numpy(dtype="datetime64[ns]", copy=False)
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

    for idx, (start, end) in enumerate(zip(start_idx, end_idx, strict=False)):
        if end - start < 10:
            continue

        window_prices = prices[start:end]
        window_qty = quantities[start:end]
        window_notionals = notionals[start:end]
        window_directions = directions[start:end]

        size_distribution = _size_distribution_score(window_notionals)

        persistence_pairs = window_directions[1:] == window_directions[:-1]
        if len(persistence_pairs) > 0:
            persistence = float(persistence_pairs.mean() * 2.0 - 1.0)
        else:
            persistence = np.nan

        liquidity_shock = np.nan
        price_impacts = np.abs(np.diff(window_prices)) * window_qty[1:]
        if len(price_impacts) >= 2:
            history_impacts = price_impacts[:-1]
            history_std = history_impacts.std()
            if history_std > 0:
                liquidity_shock = float(
                    (price_impacts[-1] - history_impacts.mean()) / history_std
                )
            elif len(history_impacts) > 0:
                liquidity_shock = float(0.0 if price_impacts[-1] == history_impacts.mean() else np.nan)

        market_impact = np.nan
        cumulative_signed_millions = np.cumsum(
            window_notionals * window_directions
        ) / 1_000_000.0
        relative_price_bps = (window_prices / window_prices[0] - 1.0) * 10000.0
        x = cumulative_signed_millions
        x_centered = x - x.mean()
        x_var = np.mean(np.square(x_centered))
        if x_var > 0:
            y = relative_price_bps
            slope = np.mean(x_centered * (y - y.mean())) / x_var
            if np.isfinite(slope):
                market_impact = float(slope)

        buy_count = float((window_directions > 0).sum())
        trade_count = len(window_directions)
        p_hat = buy_count / trade_count
        significance = float((p_hat - 0.5) / np.sqrt(0.25 / trade_count))

        volatility_adj_volume = np.nan
        price_velocity = np.nan
        momentum_acceleration = np.nan
        volume_spike = np.nan
        volume_clustering = np.nan
        liquidity_depth = np.nan
        price_volume_decoupling = np.nan
        market_efficiency = np.nan
        liquidity_migration = np.nan
        order_flow_imbalance = np.nan
        liquidity_ratio = np.nan
        volume_weighted_price = np.nan
        orderbook_pressure = np.nan

        signed_qty = window_qty * window_directions
        total_qty = window_qty.sum()
        if total_qty > 0:
            order_flow_imbalance = float(signed_qty.sum() / total_qty)
        total_notional = window_notionals.sum()
        if total_notional > 0:
            orderbook_pressure = float(
                np.sum(window_notionals * window_directions) / total_notional
            )

        returns = np.diff(window_prices)
        if len(returns) >= 19:
            returns_std = returns.std()
            if returns_std > 0:
                price_velocity = float(returns.mean() / returns_std)
                liquidity_ratio = float(window_qty[1:].mean() / (returns_std * 100.0))

            chunk_size = 5
            chunk_count = len(returns) // chunk_size
            if chunk_count > 0:
                trimmed_returns = returns[-chunk_count * chunk_size :]
                avg_vol = np.mean(
                    np.std(trimmed_returns.reshape(chunk_count, chunk_size), axis=1)
                )
                current_volume = window_qty[-1]
                mean_volume = window_qty.mean()
                current_vol = returns.std()
                if avg_vol > 0 and mean_volume > 0:
                    volatility_adj_volume = float(
                        current_volume * (current_vol / avg_vol) / mean_volume
                    )

            last_trade_price = window_prices[-1]
            if total_qty > 0 and np.isfinite(last_trade_price) and last_trade_price > 0:
                window_vwap = window_notionals.sum() / total_qty
                volume_weighted_price = float((window_vwap - last_trade_price) / last_trade_price)

        if len(window_qty) >= 20:
            avg_prior_volume = window_qty[:-1].mean()
            if avg_prior_volume > 0:
                volume_spike = float(max(0.0, window_qty[-1] / avg_prior_volume - 2.0))

        momentum = np.diff(returns)
        if len(momentum) >= 10:
            momentum_std = momentum.std()
            if momentum_std > 0:
                momentum_acceleration = float(momentum.mean() / momentum_std)

        if len(window_qty) >= 30:
            avg_volume = window_qty.mean()
            high_volume = window_qty > avg_volume
            high_count = int(high_volume[:-1].sum())
            if high_count > 0:
                high_to_high = int(np.logical_and(high_volume[:-1], high_volume[1:]).sum())
                volume_clustering = float(high_to_high / high_count * 2.0 - 1.0)

        if len(returns) >= 10:
            price_changes = np.abs(returns)
            trade_vols = window_qty[1:]

            price_changes_bps = price_changes / window_prices[:-1] * 10000.0
            mean_price_change_bps = price_changes_bps.mean()
            if mean_price_change_bps > 0:
                liquidity_depth = float(trade_vols.mean() / mean_price_change_bps)

            vol_returns = np.diff(np.log(window_qty + 1.0))
            if (
                len(vol_returns) == len(returns)
                and np.std(returns) > 0
                and np.std(vol_returns) > 0
            ):
                correlation = _safe_correlation(returns, vol_returns)
                if correlation is not None:
                    price_volume_decoupling = float(1.0 - abs(correlation))

            path_length = np.abs(returns).sum()
            if path_length > 0:
                market_efficiency = float(
                    abs(window_prices[-1] - window_prices[0]) / path_length
                )

        if len(window_qty) >= 10:
            half = len(window_qty) // 2
            if half > 0 and half < len(window_qty):
                first_half = window_qty[:half].mean()
                second_half = window_qty[half:].mean()
                if first_half > 0:
                    liquidity_migration = float((second_half - first_half) / first_half)

        result_array[idx] = [
            size_distribution,
            persistence,
            liquidity_shock,
            market_impact,
            significance,
            volatility_adj_volume,
            price_velocity,
            momentum_acceleration,
            volume_spike,
            volume_clustering,
            liquidity_depth,
            price_volume_decoupling,
            market_efficiency,
            liquidity_migration,
            order_flow_imbalance,
            liquidity_ratio,
            volume_weighted_price,
            orderbook_pressure,
        ]

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
    """Causal VPIN from completed volume buckets, reset each session.

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
    trade_sessions = _trading_session_labels(trade_times)
    quote_sessions = _trading_session_labels(quotes.index)
    directions = np.where(trades["side"].to_numpy(dtype=str) == "B", 1.0, -1.0)
    mid = quotes["mid_price"].to_numpy(dtype=float, copy=False)

    for session in pd.unique(trade_sessions):
        trade_positions = np.flatnonzero(trade_sessions == session)
        quote_positions = np.flatnonzero(quote_sessions == session)
        if not len(quote_positions):
            continue
        session_quote_times = quotes.index[quote_positions]
        session_trade_times = trade_times[trade_positions]
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
    result["adverse_selection_markout_30s"] = _calculate_adverse_selection_markout(
        trades, quotes
    )
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
    for day in pd.unique(day_labels):
        positions = np.flatnonzero(day_labels == day)
        history = pd.Series(score[positions], index=index[positions]).shift(1)
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
    quote_factors = calculate_snapshot_factors(quotes, window_profile)
    elapsed_seconds = quote_factors.index.to_series().diff(5).dt.total_seconds()
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
