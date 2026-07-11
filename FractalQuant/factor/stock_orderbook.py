from __future__ import annotations

"""DataFrame-based stock orderbook factors for the raw A-share tick CSV pipeline.

This module is intentionally scoped to the stock snapshot pipeline used by
``scripts/generate_stock_orderbook_factors.py``:
- quotes are the master snapshot index
- order/trade events are summarized on 60s windows
- event features are aligned back to snapshots with ``merge_asof``

It is kept separate from the legacy single-snapshot ``factor/orderbook.py``
implementation and from the ETF minute-parquet factor pipeline.
"""

import numpy as np
import pandas as pd
from scipy import stats


ORDER_WINDOW = "60s"
TRADE_WINDOW = "60s"
NEAR_TOUCH_LEVELS = 5
REFILL_LOOKBACK_SNAPSHOTS = 6
PRESSURE_SPREAD_FLOOR_BPS = 1.0
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


def _safe_skew(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values) == 0:
        return 0.0
    skewness = stats.skew(values, bias=False)
    return float(0.0 if not np.isfinite(skewness) else skewness)


def _safe_kurtosis(values: np.ndarray) -> float:
    if len(values) < 4 or np.std(values) == 0:
        return 0.0
    kurtosis = stats.kurtosis(values, fisher=True, bias=False)
    return float(0.0 if not np.isfinite(kurtosis) else kurtosis)


def calculate_snapshot_factors(quotes: pd.DataFrame) -> pd.DataFrame:
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
    result["weighted_depth_imbalance_l5"] = weighted_depth_imbalance_l5
    result["weighted_depth_pressure_l5"] = np.clip(
        _safe_divide(weighted_depth_imbalance_l5, pressure_denominator), -1.0, 1.0
    )
    result["weighted_imbalance_velocity_l5"] = (
        pd.Series(weighted_depth_imbalance_l5, index=quotes.index).diff(5)
    )
    result["bid_refill_intensity_l5"] = _positive_refill_intensity(
        weighted_bid_depth_l5, quotes.index
    )
    result["ask_refill_intensity_l5"] = _positive_refill_intensity(
        weighted_ask_depth_l5, quotes.index
    )
    result["bid_ask_qty_ratio_l1"] = _safe_divide(bid_qty1, ask_qty1)
    result["depth_l5_total"] = depth_l5_total
    result["orderbook_decay_l5"] = (bid_decay_l5 + ask_decay_l5) / 2.0
    result["orderbook_asymmetry_l5"] = np.abs(bid_cv - ask_cv)
    result["depth_concentration_l5"] = (
        _safe_divide(bid_qty1, bid_depth_l5) + _safe_divide(ask_qty1, ask_depth_l5)
    ) / 2.0
    result["orderbook_liquidity_l5"] = _safe_divide(depth_l5_total, liquidity_spread)
    result["book_pressure_wap5"] = _safe_divide(bid_wap5 - ask_wap5, mid_price)
    result["book_slope_diff_l5"] = bid_slope - ask_slope
    return result.replace([np.inf, -np.inf], np.nan)


def _align_event_metrics(
    event_metrics: pd.DataFrame,
    quote_index: pd.DatetimeIndex,
    factor_columns: list[str],
    tolerance: str | None = None,
) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame(0.0, index=quote_index, columns=factor_columns)

    quote_frame = pd.DataFrame({"trade_time": quote_index})
    event_frame = event_metrics.reset_index().rename(columns={"event_time": "trade_time"})
    merge_kwargs: dict[str, object] = {
        "on": "trade_time",
        "direction": "backward",
    }
    # 限制向后对齐的时间容差：若快照之前在 tolerance 时间内没有任何事件，
    # 则该窗口应视为空，而不是沿用很久以前的过期滚动值。
    if tolerance is not None:
        merge_kwargs["tolerance"] = pd.Timedelta(tolerance)
    merged = pd.merge_asof(
        quote_frame.sort_values("trade_time"),
        event_frame.sort_values("trade_time"),
        **merge_kwargs,
    )
    merged = merged.set_index("trade_time")
    merged = merged.reindex(quote_index)
    return merged[factor_columns]


def _calculate_trade_impact_factors(
    trades: pd.DataFrame,
    quote_index: pd.DatetimeIndex,
    window: str,
) -> pd.DataFrame:
    factor_columns = [
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
    ]
    result = pd.DataFrame(np.nan, index=quote_index, columns=factor_columns)
    if trades.empty:
        return result

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

        size_distribution = abs(_safe_skew(window_notionals)) + _safe_kurtosis(window_notionals) / 10.0

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
        price_std = window_prices.std()
        if price_std > 0:
            signed_qty = window_qty * window_directions
            market_impact = float(signed_qty[-5:].sum() / (price_std * 100.0))

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

        returns = np.diff(window_prices)
        if len(returns) >= 19:
            returns_std = returns.std()
            if returns_std > 0:
                price_velocity = float(returns.mean() / returns_std)
                liquidity_ratio = float(window_qty[1:].mean() / (returns_std * 100.0))

            up_count = int((returns > 0).sum())
            down_count = int((returns < 0).sum())
            total_directional = up_count + down_count
            if total_directional > 0:
                orderbook_pressure = float((up_count - down_count) / total_directional)

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

            trade_vols = window_qty[1:]
            buy_mask = returns > 0
            sell_mask = returns < 0
            buy_pressure = np.sum(returns[buy_mask] * trade_vols[buy_mask])
            sell_pressure = np.sum(np.abs(returns[sell_mask] * trade_vols[sell_mask]))
            total_pressure = buy_pressure + sell_pressure
            if total_pressure > 0:
                order_flow_imbalance = float((buy_pressure - sell_pressure) / total_pressure)

            total_qty = window_qty.sum()
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

            if np.std(price_changes) > 0 and np.std(trade_vols) > 0:
                slope = stats.linregress(price_changes, trade_vols).slope
                mean_trade_vol = trade_vols.mean()
                if mean_trade_vol > 0:
                    liquidity_depth = float(slope / mean_trade_vol)

            vol_returns = np.diff(np.log(window_qty + 1.0))
            if (
                len(vol_returns) == len(returns)
                and np.std(returns) > 0
                and np.std(vol_returns) > 0
            ):
                correlation = np.corrcoef(returns, vol_returns)[0, 1]
                if np.isfinite(correlation):
                    price_volume_decoupling = float(abs(correlation))

            abs_returns = np.abs(returns)
            if np.std(abs_returns) > 0:
                slope = stats.linregress(np.arange(len(abs_returns)), abs_returns).slope
                if np.isfinite(slope):
                    market_efficiency = float(1.0 / (1.0 + abs(slope)))

        if len(window_qty) >= 10:
            half = len(window_qty) // 2
            if half > 0 and half < len(window_qty):
                first_half = window_qty[:half].mean()
                second_half = window_qty[half:].mean()
                if first_half > 0:
                    liquidity_migration = float((second_half - first_half) / first_half)

        result.iloc[idx] = [
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

    return result


def calculate_order_flow_factors(
    orders: pd.DataFrame,
    quote_index: pd.DatetimeIndex,
    window: str = ORDER_WINDOW,
) -> pd.DataFrame:
    factor_columns = [
        "order_count_imbalance_60s",
        "order_qty_imbalance_60s",
        "order_notional_imbalance_60s",
    ]
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

    rolling = (
        order_metrics.set_index("event_time")[
            [
                "buy_count",
                "sell_count",
                "buy_qty",
                "sell_qty",
                "buy_notional",
                "sell_notional",
            ]
        ]
        .rolling(window, min_periods=1)
        .sum()
    )
    rolling.index.name = "event_time"
    rolling["order_count_imbalance_60s"] = _imbalance(rolling["buy_count"], rolling["sell_count"])
    rolling["order_qty_imbalance_60s"] = _imbalance(rolling["buy_qty"], rolling["sell_qty"])
    rolling["order_notional_imbalance_60s"] = _imbalance(
        rolling["buy_notional"], rolling["sell_notional"]
    )
    aligned = _align_event_metrics(
        rolling[factor_columns], quote_index, factor_columns, tolerance=window
    )
    # 容差外（窗口内无委托）回填为0：无订单即失衡为0。
    return aligned.fillna(0.0)


def calculate_trade_flow_factors(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    window: str = TRADE_WINDOW,
) -> pd.DataFrame:
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
    ]
    if trades.empty:
        empty = pd.DataFrame(0.0, index=quotes.index, columns=factor_columns[:2])
        empty["trade_vwap_gap_60s"] = np.nan
        for column in factor_columns[3:]:
            empty[column] = np.nan
        return empty

    trade_metrics = trades.copy()
    trade_metrics["buy_count"] = (trade_metrics["side"] == "B").astype(float)
    trade_metrics["sell_count"] = (trade_metrics["side"] == "S").astype(float)
    trade_metrics["buy_qty"] = np.where(trade_metrics["side"] == "B", trade_metrics["qty"], 0.0)
    trade_metrics["sell_qty"] = np.where(trade_metrics["side"] == "S", trade_metrics["qty"], 0.0)

    rolling = (
        trade_metrics.set_index("event_time")[
            ["buy_count", "sell_count", "buy_qty", "sell_qty", "qty", "notional"]
        ]
        .rolling(window, min_periods=1)
        .sum()
    )
    rolling.index.name = "event_time"
    rolling["trade_count_imbalance_60s"] = _imbalance(rolling["buy_count"], rolling["sell_count"])
    rolling["trade_qty_imbalance_60s"] = _imbalance(rolling["buy_qty"], rolling["sell_qty"])
    rolling["trade_vwap_60s"] = _safe_divide(rolling["notional"], rolling["qty"])

    aligned = _align_event_metrics(
        rolling[["trade_count_imbalance_60s", "trade_qty_imbalance_60s", "trade_vwap_60s"]],
        quotes.index,
        ["trade_count_imbalance_60s", "trade_qty_imbalance_60s", "trade_vwap_60s"],
        tolerance=window,
    )
    result = aligned.rename(columns={"trade_vwap_60s": "trade_vwap_gap_60s"})
    # 容差外（窗口内无成交）失衡回填0；vwap无成交保持NaN。
    result["trade_count_imbalance_60s"] = result["trade_count_imbalance_60s"].fillna(0.0)
    result["trade_qty_imbalance_60s"] = result["trade_qty_imbalance_60s"].fillna(0.0)
    result["trade_vwap_gap_60s"] = _safe_divide(
        result["trade_vwap_gap_60s"].to_numpy(dtype=float, copy=False)
        - quotes["mid_price"].to_numpy(dtype=float, copy=False),
        quotes["mid_price"].to_numpy(dtype=float, copy=False),
    )
    advanced = _calculate_trade_impact_factors(trades, quotes.index, window)
    result = pd.concat([result, advanced], axis=1)
    result.index = quotes.index
    return result.replace([np.inf, -np.inf], np.nan)


def build_stock_orderbook_factor_frame(
    quotes: pd.DataFrame,
    orders: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    quote_factors = calculate_snapshot_factors(quotes)
    quote_factors["orderbook_velocity_l5"] = (
        quote_factors["depth_imbalance_l5"] - quote_factors["depth_imbalance_l5"].shift(5)
    )
    quote_input = quotes.join(quote_factors)
    order_factors = calculate_order_flow_factors(orders, quote_input.index)
    trade_factors = calculate_trade_flow_factors(trades, quote_input)
    return pd.concat([quote_factors, order_factors, trade_factors], axis=1)
