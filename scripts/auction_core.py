"""Pure single-session auction factor calculation; no QMT client or file I/O."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

if __package__:
    from scripts.auction_schema import (
        KEY_COLUMNS,
        DIAGNOSTIC_COLUMNS,
        REFERENCE_COLUMNS,
        CORE_FACTOR_COLUMNS,
        EVENT_FACTOR_COLUMNS,
        TRANSACTION_ORDER_RATIO_COLUMNS,
        PATH_FACTOR_COLUMNS,
        ROBUST_IMBALANCE_FACTOR_COLUMNS,
        PARTICIPATION_FACTOR_COLUMNS,
        PRIORITY_REPORT_FACTOR_COLUMNS,
        REPORT_SUPPLEMENT_FACTOR_COLUMNS,
        REPORT_SMOOTHED_SOURCE_COLUMNS,
        REPORT_SMOOTHED_FACTOR_COLUMNS,
        CONTEXT_SUPPLEMENT_FACTOR_COLUMNS,
        VOLUME_RATIO_FACTOR_COLUMNS,
        TURNOVER_FACTOR_COLUMNS,
        SUPPLEMENT_REFERENCE_COLUMNS,
        SUPPLEMENT_OUTPUT_COLUMNS,
        FACTOR_COLUMNS,
        OUTPUT_COLUMNS,
        EVENT_COLUMNS,
        LARGE_ORDER_LOOKBACK_DAYS,
        LARGE_ORDER_QUANTILE,
        HISTORICAL_AMOUNT_LOOKBACK_DAYS,
        SHORT_DAILY_AMOUNT_LOOKBACK_DAYS,
        MINUTES_PER_TRADING_DAY,
        STAGE2_TWAP_MIN_COVERAGE,
        IMBALANCE_RELATIVE_FLOOR,
        IMBALANCE_FISHER_CLIP,
        REPORT_SMOOTHED_LOOKBACK_DAYS,
        REPORT_SMOOTHED_MIN_PERIODS,
        RAW_COLUMN_MAP,
        PRICE_COLUMNS,
        QUANTITY_COLUMNS,
    )
else:
    from auction_schema import (
        KEY_COLUMNS,
        DIAGNOSTIC_COLUMNS,
        REFERENCE_COLUMNS,
        CORE_FACTOR_COLUMNS,
        EVENT_FACTOR_COLUMNS,
        TRANSACTION_ORDER_RATIO_COLUMNS,
        PATH_FACTOR_COLUMNS,
        ROBUST_IMBALANCE_FACTOR_COLUMNS,
        PARTICIPATION_FACTOR_COLUMNS,
        PRIORITY_REPORT_FACTOR_COLUMNS,
        REPORT_SUPPLEMENT_FACTOR_COLUMNS,
        REPORT_SMOOTHED_SOURCE_COLUMNS,
        REPORT_SMOOTHED_FACTOR_COLUMNS,
        CONTEXT_SUPPLEMENT_FACTOR_COLUMNS,
        VOLUME_RATIO_FACTOR_COLUMNS,
        TURNOVER_FACTOR_COLUMNS,
        SUPPLEMENT_REFERENCE_COLUMNS,
        SUPPLEMENT_OUTPUT_COLUMNS,
        FACTOR_COLUMNS,
        OUTPUT_COLUMNS,
        EVENT_COLUMNS,
        LARGE_ORDER_LOOKBACK_DAYS,
        LARGE_ORDER_QUANTILE,
        HISTORICAL_AMOUNT_LOOKBACK_DAYS,
        SHORT_DAILY_AMOUNT_LOOKBACK_DAYS,
        MINUTES_PER_TRADING_DAY,
        STAGE2_TWAP_MIN_COVERAGE,
        IMBALANCE_RELATIVE_FLOOR,
        IMBALANCE_FISHER_CLIP,
        REPORT_SMOOTHED_LOOKBACK_DAYS,
        REPORT_SMOOTHED_MIN_PERIODS,
        RAW_COLUMN_MAP,
        PRICE_COLUMNS,
        QUANTITY_COLUMNS,
    )

LOGGER = logging.getLogger("generate_auction_factors")

def _without_qmt_state_anchors(frame: pd.DataFrame) -> pd.DataFrame:
    if "_qmt_state_anchor" not in frame.columns:
        return frame
    return frame.loc[~frame["_qmt_state_anchor"].eq(True)]


def _safe_return(end_value: float, start_value: float) -> float:
    if not np.isfinite(end_value) or not np.isfinite(start_value) or start_value <= 0:
        return np.nan
    return float(end_value / start_value - 1.0)


def _stable_previous_close(series: pd.Series) -> float:
    """Return a robust quote-window fallback for a constant previous close.

    Quote feeds occasionally contain an isolated boundary/sentinel value.  The
    median identifies the stable price without changing the raw cached rows.
    """
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values) & (values > 0)]
    if values.empty:
        return np.nan
    numeric = values.to_numpy(dtype=float)
    median = float(np.median(numeric))
    distances = np.abs(numeric - median)
    closest = np.flatnonzero(np.isclose(distances, distances.min(), atol=1e-12))
    # If only two conflicting snapshots exist, prefer the later one; boundary
    # sentinels in the local CSV precede the stable auction quote.
    return float(values.iloc[closest[-1]])


def _resolve_previous_close(
    quotes: pd.DataFrame,
    expected_previous_close: float | None = None,
    *,
    ts_code: str | None = None,
    trade_date: str | None = None,
) -> float:
    """Resolve a stable raw auction previous close with daily validation.

    A daily reference detects materially inconsistent quote clusters, while a
    close quote value is retained to preserve the feed's finer price precision.
    When daily history is unavailable (for example, warm-up sessions), the
    quote-window median is used as the fallback.
    """
    raw_quote_values = (
        quotes["previous_close"]
        if "previous_close" in quotes.columns
        else pd.Series(dtype=float)
    )
    quote_values = pd.to_numeric(raw_quote_values, errors="coerce")
    quote_values = quote_values[np.isfinite(quote_values) & (quote_values > 0)]
    expected = pd.to_numeric(pd.Series([expected_previous_close]), errors="coerce").iloc[0]
    if np.isfinite(expected) and expected > 0:
        stable = _stable_previous_close(quote_values)
        if np.isfinite(stable) and stable > 0:
            stable_matches = bool(
                np.isclose(stable, float(expected), rtol=5e-3, atol=1e-4)
            )
            matches = np.isclose(
                quote_values.to_numpy(dtype=float),
                float(expected),
                rtol=5e-3,
                atol=1e-4,
            )
            if not stable_matches:
                LOGGER.warning(
                    "%s %s quote previous_close conflict; daily benchmark %.10g, "
                    "stable quote %.10g (%s/%s quote values match)",
                    ts_code or "unknown",
                    trade_date or "unknown-date",
                    float(expected),
                    float(stable),
                    int(matches.sum()),
                    len(matches),
                )
            # Preserve the stable quote precision when it is consistent with
            # the daily benchmark; use the benchmark to reject isolated or
            # materially inconsistent quote clusters.
            if stable_matches:
                return float(stable)
            return float(expected)
        return float(expected)
    return _stable_previous_close(quote_values)


def _context_value_map(
    context: pd.DataFrame | None, column: str
) -> dict[str, float]:
    """Build a date-keyed lookup for numeric context values."""
    if context is None or context.empty or column not in context.columns:
        return {}
    dates = pd.to_datetime(context["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    values = pd.to_numeric(context[column], errors="coerce")
    result: dict[str, float] = {}
    for date, value in zip(dates, values, strict=False):
        if pd.notna(date) and np.isfinite(value) and value > 0:
            result[str(date)] = float(value)
    return result


def _calculate_indicative_price(frame: pd.DataFrame) -> pd.Series:
    ask = frame["ask_price1"]
    bid = frame["bid_price1"]
    valid = ask.notna() & bid.notna() & (ask > 0) & (bid > 0)
    return ((ask + bid) / 2.0).where(valid)


def _calculate_l3_imbalance(frame: pd.DataFrame) -> pd.Series:
    bid_qty = frame[[f"bid_qty{level}" for level in range(2, 4)]].sum(axis=1)
    ask_qty = frame[[f"ask_qty{level}" for level in range(2, 4)]].sum(axis=1)
    total = bid_qty + ask_qty
    return ((bid_qty - ask_qty) / total).where(total > 0)


def _calculate_relative_spread(frame: pd.DataFrame) -> pd.Series:
    """
    Calculate relative spread during call auction.

    During the opening auction (09:15-09:25), bid_price1 == ask_price1 (virtual
    matched price), and bid_price2/ask_price2 have no data (0% valid rate in
    production). Therefore, the spread is always zero and this factor carries no
    signal. Return NaN to explicitly mark it as unavailable for auction data.
    """
    return pd.Series(np.nan, index=frame.index)


def _time_weighted_mean(
    stage: pd.DataFrame,
    column: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> tuple[float, float]:
    duration_seconds = float((end_time - start_time).total_seconds())
    if stage.empty or duration_seconds <= 0:
        return np.nan, 0.0

    ordered = (
        stage.loc[
            stage["trade_time"].ge(start_time) & stage["trade_time"].lt(end_time),
            ["trade_time", column],
        ]
        .sort_values("trade_time", kind="mergesort")
        .drop_duplicates("trade_time", keep="last")
    )
    if ordered.empty:
        return np.nan, 0.0

    next_times = ordered["trade_time"].shift(-1).fillna(end_time)
    weights = (next_times - ordered["trade_time"]).dt.total_seconds().clip(lower=0)
    values = pd.to_numeric(ordered[column], errors="coerce")
    valid = np.isfinite(values) & weights.gt(0)
    covered_seconds = float(weights.loc[valid].sum())
    coverage_ratio = covered_seconds / duration_seconds
    if covered_seconds <= 0 or coverage_ratio < STAGE2_TWAP_MIN_COVERAGE:
        return np.nan, coverage_ratio
    weighted_mean = float(np.average(values.loc[valid], weights=weights.loc[valid]))
    return weighted_mean, coverage_ratio


def _apply_stage2_twap_factors(
    row: dict[str, object],
    stage2: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> None:
    price_twap, price_coverage = _time_weighted_mean(
        stage2, "indicative_price", start_time, end_time
    )
    imbalance_twap, imbalance_coverage = _time_weighted_mean(
        stage2, "l3_imbalance", start_time, end_time
    )
    spread_twap, spread_coverage = _time_weighted_mean(
        stage2, "relative_spread", start_time, end_time
    )
    # During call auction, relative_spread is unavailable (always NaN), so we
    # exclude spread_coverage from the overall coverage_ratio calculation.
    row["auction_stage2_twap_coverage_ratio"] = float(
        min(price_coverage, imbalance_coverage)
    )
    row["auction_stage2_twap_price"] = price_twap
    row["auction_l3_imbalance_twap_stage2"] = imbalance_twap
    row["auction_relative_spread_twap_stage2"] = spread_twap

    open_price = row["auction_open_price"]
    if np.isfinite(open_price) and np.isfinite(price_twap) and price_twap > 0:
        row["auction_final_vs_stage2_twap"] = float(open_price / price_twap - 1.0)


def _stage2_slope(stage2: pd.DataFrame, previous_close: float) -> float:
    valid = stage2.dropna(subset=["trade_time", "indicative_price"])
    if len(valid) < 3 or not np.isfinite(previous_close) or previous_close <= 0:
        return np.nan
    elapsed_minutes = (
        valid["trade_time"] - valid["trade_time"].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float) / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_log_price = np.log(
            valid["indicative_price"].to_numpy(dtype=float) / previous_close
        )
    finite = np.isfinite(elapsed_minutes) & np.isfinite(relative_log_price)
    finite_elapsed = elapsed_minutes[finite]
    finite_log_price = relative_log_price[finite]
    if (
        len(finite_elapsed) < 3
        or np.unique(finite_elapsed).size < 3
        or np.ptp(finite_elapsed) <= 0
    ):
        return np.nan
    slope = float(np.polyfit(finite_elapsed, finite_log_price, 1)[0] * 10000.0)
    return slope if np.isfinite(slope) else np.nan


def _stage2_efficiency(stage2: pd.DataFrame) -> float:
    prices = stage2["indicative_price"].dropna().to_numpy(dtype=float)
    if len(prices) < 2:
        return np.nan
    total_variation = float(np.abs(np.diff(prices)).sum())
    if total_variation == 0:
        return 0.0
    return float(abs(prices[-1] - prices[0]) / total_variation)


def _apply_stage2_path_factors(
    row: dict[str, object], stage2: pd.DataFrame, previous_close: float
) -> None:
    prices = stage2["indicative_price"].dropna().to_numpy(dtype=float)
    if prices.size == 0:
        return
    update_prices = (
        _without_qmt_state_anchors(stage2)["indicative_price"]
        .dropna()
        .to_numpy(dtype=float)
    )
    if np.isfinite(previous_close) and previous_close > 0:
        mean_prices = update_prices if update_prices.size else prices
        row["auction_stage2_mid_mean_return"] = float(
            mean_prices.mean() / previous_close - 1.0
        )
        row["auction_stage2_mid_max_return"] = float(
            prices.max() / previous_close - 1.0
        )
        row["auction_stage2_mid_min_return"] = float(
            prices.min() / previous_close - 1.0
        )
    if prices.size < 2:
        return

    changes = np.diff(prices)
    if np.isfinite(previous_close) and previous_close > 0:
        row["auction_stage2_total_variation_bps"] = float(
            np.abs(changes).sum() / previous_close * 10000.0
        )
    update_changes = np.diff(update_prices)
    if len(update_changes) > 0:
        row["auction_stage2_up_step_ratio"] = float(
            np.count_nonzero(update_changes > 0) / len(update_changes)
        )
    nonzero_directions = np.sign(changes[changes != 0])
    row["auction_stage2_reversal_count"] = int(
        np.count_nonzero(nonzero_directions[1:] != nonzero_directions[:-1])
    )


def _indicative_price_range_ratio(frame: pd.DataFrame) -> float:
    prices = pd.to_numeric(frame["indicative_price"], errors="coerce").to_numpy(
        dtype=float
    )
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size == 0:
        return np.nan
    minimum = float(prices.min())
    return float((prices.max() - minimum) / minimum)


def _finite_indicative_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prices = pd.to_numeric(frame["indicative_price"], errors="coerce")
    return frame.loc[np.isfinite(prices) & prices.gt(0)]


def _l3_buy_share(endpoint: pd.Series) -> float:
    imbalance = pd.to_numeric(
        pd.Series([endpoint["l3_imbalance"]]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(imbalance):
        return np.nan
    return float((imbalance + 1.0) / 2.0)


def _stage2_tail_buy_share(
    stage2: pd.DataFrame,
    nominal_end_time: pd.Timestamp,
) -> float:
    """Return the mean L3 buy share from the final Stage2 quote window."""
    if stage2.empty:
        return np.nan
    ordered = (
        stage2.copy()
        .sort_values("trade_time", kind="mergesort")
        .drop_duplicates("trade_time", keep="last")
    )
    prices = pd.to_numeric(ordered["indicative_price"], errors="coerce")
    imbalance = pd.to_numeric(ordered["l3_imbalance"], errors="coerce")
    valid = ordered.loc[
        np.isfinite(prices)
        & prices.gt(0)
        & np.isfinite(imbalance)
        & ordered["trade_time"].lt(nominal_end_time)
    ]
    if valid.empty:
        return np.nan

    tail_start = nominal_end_time - pd.Timedelta(seconds=60)
    tail = valid.loc[valid["trade_time"].ge(tail_start)]
    selected = tail if len(tail) >= 3 else valid.tail(min(3, len(valid)))
    shares = (pd.to_numeric(selected["l3_imbalance"], errors="coerce") + 1.0) / 2.0
    shares = shares[np.isfinite(shares)]
    return float(shares.mean()) if not shares.empty else np.nan


def _stage2_imbalance_endpoints(
    stage2: pd.DataFrame,
) -> tuple[float, float]:
    """Return smoothed Stage2 start/end L3 imbalance values."""
    if stage2.empty:
        return np.nan, np.nan
    ordered = (
        stage2.copy()
        .sort_values("trade_time", kind="mergesort")
        .drop_duplicates("trade_time", keep="last")
    )
    prices = pd.to_numeric(ordered["indicative_price"], errors="coerce")
    imbalance = pd.to_numeric(ordered["l3_imbalance"], errors="coerce")
    valid = ordered.loc[np.isfinite(prices) & prices.gt(0) & np.isfinite(imbalance)]
    values = imbalance.loc[valid.index].to_numpy(dtype=float)
    if values.size < 2:
        return np.nan, np.nan
    window = max(1, int(np.ceil(values.size * 0.1)))
    return float(values[:window].mean()), float(values[-window:].mean())


def _apply_report_supplement_factors(
    row: dict[str, object],
    valid_price: pd.DataFrame,
    stage1_valid: pd.DataFrame,
    stage2_valid: pd.DataFrame,
    previous_close: float,
    nominal_end_time: pd.Timestamp,
) -> None:
    valid_price = _finite_indicative_price_frame(valid_price)
    stage1_valid = _finite_indicative_price_frame(stage1_valid)
    stage2_valid = _finite_indicative_price_frame(stage2_valid)
    snapshot_updates = _without_qmt_state_anchors(valid_price)
    row["auction_snapshot_count_total"] = int(len(snapshot_updates))
    row["auction_range_ratio"] = _indicative_price_range_ratio(valid_price)
    row["auction_stage1_range_ratio"] = _indicative_price_range_ratio(stage1_valid)
    row["auction_stage2_range_ratio"] = _indicative_price_range_ratio(stage2_valid)

    if not valid_price.empty:
        prices = valid_price["indicative_price"].to_numpy(dtype=float)
        final = valid_price.iloc[-1]
        changes = np.diff(snapshot_updates["indicative_price"].to_numpy(dtype=float))
        if len(changes) > 0:
            row["auction_up_step_ratio"] = float(
                np.count_nonzero(changes > 0) / len(changes)
            )
            row["auction_down_step_ratio"] = float(
                np.count_nonzero(changes < 0) / len(changes)
            )
        else:
            row["auction_up_step_ratio"] = 0.0
            row["auction_down_step_ratio"] = 0.0
        row["auction_final_to_full_max"] = _safe_return(
            float(final["indicative_price"]), float(prices.max())
        )
        last_minute = valid_price.loc[
            valid_price["trade_time"].ge(nominal_end_time - pd.Timedelta(minutes=1))
        ]
        if len(last_minute) >= 2:
            row["auction_last60s_price_return"] = _safe_return(
                float(last_minute.iloc[-1]["indicative_price"]),
                float(last_minute.iloc[0]["indicative_price"]),
            )

    stage1_final = None
    stage1_buy_share = np.nan
    if not stage1_valid.empty:
        stage1_final = stage1_valid.iloc[-1]
        stage1_prices = stage1_valid["indicative_price"].to_numpy(dtype=float)
        row["auction_stage1_end_time"] = pd.Timestamp(stage1_final["trade_time"])
        row["auction_stage1_end_return_from_prev_close"] = _safe_return(
            float(stage1_final["indicative_price"]), previous_close
        )
        row["auction_stage1_max_return_from_prev_close"] = _safe_return(
            float(stage1_prices.max()), previous_close
        )
        row["auction_stage1_min_return_from_prev_close"] = _safe_return(
            float(stage1_prices.min()), previous_close
        )
        stage1_buy_share = _l3_buy_share(stage1_final)
        row["auction_l3_buy_share_stage1_end"] = stage1_buy_share

        open_price = row["auction_open_price"]
        if np.isfinite(open_price):
            stage1_max = float(stage1_prices.max())
            stage1_min = float(stage1_prices.min())
            row["auction_open_pullback_from_stage1_max"] = float(
                1.0 - open_price / stage1_max
            )
            row["auction_open_rebound_from_stage1_min"] = _safe_return(
                open_price, stage1_min
            )

    if not stage2_valid.empty:
        stage2_final = stage2_valid.iloc[-1]
        row["auction_stage2_end_time"] = pd.Timestamp(stage2_final["trade_time"])
        if stage1_final is not None:
            row["auction_stage2_end_return_from_stage1_end"] = _safe_return(
                float(stage2_final["indicative_price"]),
                float(stage1_final["indicative_price"]),
            )
        stage2_buy_share = _l3_buy_share(stage2_final)
        if np.isfinite(stage1_buy_share) and np.isfinite(stage2_buy_share):
            row["auction_l3_buy_share_change_stage2"] = float(
                stage2_buy_share - stage1_buy_share
            )

    row["auction_l3_buy_share_final"] = _stage2_tail_buy_share(
        stage2_valid, nominal_end_time
    )
    if not valid_price.empty and not np.isfinite(final["l3_imbalance"]):
        row["auction_l3_buy_share_final"] = np.nan


def _relative_imbalance_change(end_value: float, start_value: float) -> float:
    if not np.isfinite(end_value) or not np.isfinite(start_value):
        return np.nan
    denominator = max(abs(float(start_value)), IMBALANCE_RELATIVE_FLOOR)
    return float((end_value - start_value) / denominator)


def _fisher_imbalance_change(end_value: float, start_value: float) -> float:
    if not np.isfinite(end_value) or not np.isfinite(start_value):
        return np.nan
    end_clipped = np.clip(end_value, -IMBALANCE_FISHER_CLIP, IMBALANCE_FISHER_CLIP)
    start_clipped = np.clip(start_value, -IMBALANCE_FISHER_CLIP, IMBALANCE_FISHER_CLIP)
    return float(np.arctanh(end_clipped) - np.arctanh(start_clipped))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _signed_imbalance(bid_value: float, ask_value: float) -> float:
    total = bid_value + ask_value
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float((bid_value - ask_value) / total)


def _event_slice(
    events: pd.DataFrame,
    event_type: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    return events.loc[
        events["event_type"].eq(event_type)
        & events["trade_time"].ge(start_time)
        & events["trade_time"].lt(end_time)
    ]


def _side_sum(frame: pd.DataFrame, side: str, column: str) -> float:
    return float(frame.loc[frame["side"].eq(side), column].sum())


def _apply_event_factors(
    row: dict[str, object],
    trade_day: pd.Timestamp,
    events: pd.DataFrame | None,
    reconstruction_ok: bool,
) -> None:
    row["auction_event_reconstruction_ok"] = bool(reconstruction_ok)
    if events is None or events.empty:
        if reconstruction_ok:
            row["auction_cancel_imbalance_stage1"] = 0.0
            row["auction_late_cancel_notional_share"] = 0.0
        return

    start_time = trade_day + pd.Timedelta(hours=9, minutes=15)
    late_cancel_time = trade_day + pd.Timedelta(hours=9, minutes=19)
    split_time = trade_day + pd.Timedelta(hours=9, minutes=20)
    late_add_time = trade_day + pd.Timedelta(hours=9, minutes=24)
    end_time = trade_day + pd.Timedelta(hours=9, minutes=25)

    stage1_adds = _event_slice(events, "A", start_time, split_time)
    stage1_cancels = _event_slice(events, "C", start_time, split_time)
    stage2_adds = _event_slice(events, "A", split_time, end_time)
    row["auction_add_count_stage1"] = int(len(stage1_adds))
    row["auction_cancel_count_stage1"] = int(len(stage1_cancels))
    row["auction_add_count_stage2"] = int(len(stage2_adds))

    stage1_add_notional = float(stage1_adds["notional"].sum())
    stage1_cancel_notional = float(stage1_cancels["notional"].sum())
    stage2_add_notional = float(stage2_adds["notional"].sum())
    row["auction_stage1_add_notional"] = stage1_add_notional
    row["auction_stage1_cancel_notional"] = stage1_cancel_notional
    row["auction_stage2_add_notional"] = stage2_add_notional
    row["auction_submitted_volume"] = float(
        stage1_adds["quantity"].sum() + stage2_adds["quantity"].sum()
    )

    # Apply transaction/order ratio factors
    _apply_transaction_order_ratio_factors(
        row, trade_day, events, stage1_adds, stage1_cancels, stage2_adds,
        reconstruction_ok,
    )

    if not reconstruction_ok:
        return

    add_qty = {side: _side_sum(stage1_adds, side, "quantity") for side in ("B", "S")}
    cancel_qty = {
        side: _side_sum(stage1_cancels, side, "quantity") for side in ("B", "S")
    }
    add_notional = {
        side: _side_sum(stage1_adds, side, "notional") for side in ("B", "S")
    }
    cancel_notional = {
        side: _side_sum(stage1_cancels, side, "notional") for side in ("B", "S")
    }
    stage2_notional = {
        side: _side_sum(stage2_adds, side, "notional") for side in ("B", "S")
    }

    row["auction_bid_cancel_qty_ratio_stage1"] = _safe_ratio(
        cancel_qty["B"], add_qty["B"]
    )
    row["auction_ask_cancel_qty_ratio_stage1"] = _safe_ratio(
        cancel_qty["S"], add_qty["S"]
    )
    row["auction_cancel_notional_ratio_stage1"] = _safe_ratio(
        stage1_cancel_notional, stage1_add_notional
    )
    if stage1_cancel_notional > 0:
        row["auction_cancel_imbalance_stage1"] = float(
            (cancel_notional["S"] - cancel_notional["B"]) / stage1_cancel_notional
        )
        late_cancels = _event_slice(events, "C", late_cancel_time, split_time)
        row["auction_late_cancel_notional_share"] = float(
            late_cancels["notional"].sum() / stage1_cancel_notional
        )
    else:
        row["auction_cancel_imbalance_stage1"] = 0.0
        row["auction_late_cancel_notional_share"] = 0.0

    row["auction_stage2_add_imbalance"] = _signed_imbalance(
        stage2_notional["B"], stage2_notional["S"]
    )
    remaining = {
        side: max(add_notional[side] - cancel_notional[side], 0.0)
        for side in ("B", "S")
    }
    row["auction_stage2_commitment_ratio"] = _safe_ratio(
        stage2_add_notional,
        remaining["B"] + remaining["S"] + stage2_add_notional,
    )
    if stage2_add_notional > 0:
        late_adds = _event_slice(events, "A", late_add_time, end_time)
        row["auction_stage2_last60s_add_share"] = float(
            late_adds["notional"].sum() / stage2_add_notional
        )

    initial_imbalance = _signed_imbalance(add_notional["B"], add_notional["S"])
    surviving_imbalance = _signed_imbalance(remaining["B"], remaining["S"])
    if np.isfinite(initial_imbalance) and np.isfinite(surviving_imbalance):
        row["auction_fake_pressure_proxy"] = float(
            initial_imbalance - surviving_imbalance
        )


def _apply_transaction_order_ratio_factors(
    row: dict[str, object],
    trade_day: pd.Timestamp,
    events: pd.DataFrame,
    stage1_adds: pd.DataFrame,
    stage1_cancels: pd.DataFrame,
    stage2_adds: pd.DataFrame,
    reconstruction_ok: bool = True,
) -> None:
    """Calculate transaction/order ratio factors based on events data."""

    if not reconstruction_ok:
        return

    # Get matched volume (final transaction volume)
    matched_volume = row.get("auction_matched_volume", np.nan)

    # Initialize all factors to NaN
    row["auction_transaction_submitted_ratio"] = np.nan
    row["auction_transaction_net_order_ratio"] = np.nan
    row["auction_stage1_net_order_qty"] = np.nan
    row["auction_stage2_order_qty"] = np.nan
    row["auction_buy_order_imbalance"] = np.nan
    row["auction_stage1_order_participation"] = np.nan
    row["auction_stage2_order_participation"] = np.nan

    # Calculate order quantities by side
    stage1_buy_qty = _side_sum(stage1_adds, "B", "quantity")
    stage1_sell_qty = _side_sum(stage1_adds, "S", "quantity")
    stage1_buy_cancel_qty = _side_sum(stage1_cancels, "B", "quantity")
    stage1_sell_cancel_qty = _side_sum(stage1_cancels, "S", "quantity")
    stage2_buy_qty = _side_sum(stage2_adds, "B", "quantity")
    stage2_sell_qty = _side_sum(stage2_adds, "S", "quantity")

    # Calculate total order volumes
    total_add_qty = float(stage1_adds["quantity"].sum() + stage2_adds["quantity"].sum())
    stage1_net_buy = stage1_buy_qty - stage1_buy_cancel_qty
    stage1_net_sell = stage1_sell_qty - stage1_sell_cancel_qty
    stage1_net_qty = stage1_net_buy + stage1_net_sell
    stage2_total_qty = stage2_buy_qty + stage2_sell_qty
    net_order_qty = stage1_net_qty + stage2_total_qty

    # Store intermediate values
    row["auction_stage1_net_order_qty"] = float(stage1_net_qty) if stage1_net_qty > 0 else np.nan
    row["auction_stage2_order_qty"] = float(stage2_total_qty) if stage2_total_qty > 0 else np.nan

    # Calculate transaction/submitted ratio
    if total_add_qty > 0 and np.isfinite(matched_volume):
        row["auction_transaction_submitted_ratio"] = float(matched_volume / total_add_qty)

    # Calculate transaction/net order ratio (after cancellations)
    if net_order_qty > 0 and np.isfinite(matched_volume):
        row["auction_transaction_net_order_ratio"] = float(matched_volume / net_order_qty)

    # Calculate buy order imbalance
    total_buy = stage1_buy_qty - stage1_buy_cancel_qty + stage2_buy_qty
    total_sell = stage1_sell_qty - stage1_sell_cancel_qty + stage2_sell_qty
    total_orders = total_buy + total_sell
    if total_orders > 0:
        row["auction_buy_order_imbalance"] = float((total_buy - total_sell) / total_orders)

    # Calculate order participation by stage
    if total_add_qty > 0:
        stage1_add_qty = float(stage1_adds["quantity"].sum())
        row["auction_stage1_order_participation"] = float(stage1_add_qty / total_add_qty)
        row["auction_stage2_order_participation"] = float(stage2_total_qty / total_add_qty)


def _apply_matched_volume_participation(row: dict[str, object]) -> None:
    if not bool(row["auction_event_reconstruction_ok"]):
        return
    matched_volume = row["auction_matched_volume"]
    submitted_volume = row["auction_submitted_volume"]
    row["auction_matched_volume_to_submitted_ratio"] = _safe_ratio(
        matched_volume, submitted_volume
    )


def _apply_stage_reversal(row: dict[str, object]) -> None:
    stage1_return = row["auction_return_stage1"]
    stage2_return = row["auction_return_stage2"]
    if not np.isfinite(stage1_return) or not np.isfinite(stage2_return):
        return
    if stage1_return * stage2_return < 0:
        row["auction_stage_reversal_strength_bps"] = float(
            np.sign(stage2_return)
            * min(abs(stage1_return), abs(stage2_return))
            * 10000.0
        )
    else:
        row["auction_stage_reversal_strength_bps"] = 0.0


def _empty_output_row(trade_date: str, ts_code: str) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in OUTPUT_COLUMNS}
    row.update(
        {
            "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
            "available_time": pd.NaT,
            "ts_code": ts_code,
            "auction_stage1_end_time": pd.NaT,
            "auction_stage2_end_time": pd.NaT,
            "auction_has_match": False,
            "auction_match_source": "none",
            "snapshot_count_stage1": 0,
            "snapshot_count_stage2": 0,
            "auction_event_reconstruction_ok": False,
            "auction_add_count_stage1": 0,
            "auction_cancel_count_stage1": 0,
            "auction_add_count_stage2": 0,
            "auction_large_order_history_days": 0,
        }
    )
    return row


def calculate_daily_auction_factors(
    quotes: pd.DataFrame,
    ts_code: str,
    events: pd.DataFrame | None = None,
    event_reconstruction_ok: bool = False,
    match_override: dict[str, object] | None = None,
    match_source: str | None = None,
    expected_previous_close: float | None = None,
) -> dict[str, object]:
    if quotes.empty:
        raise ValueError(f"Empty quote frame for {ts_code}")

    trade_day = pd.Timestamp(quotes["trade_time"].iloc[0]).normalize()
    trade_date = trade_day.strftime("%Y-%m-%d")
    row = _empty_output_row(trade_date, ts_code)
    _apply_event_factors(row, trade_day, events, event_reconstruction_ok)

    start_time = trade_day + pd.Timedelta(hours=9, minutes=15)
    split_time = trade_day + pd.Timedelta(hours=9, minutes=20)
    nominal_end_time = trade_day + pd.Timedelta(hours=9, minutes=25)
    match_deadline = trade_day + pd.Timedelta(hours=9, minutes=30)

    match_mask = (
        quotes["trade_time"].ge(nominal_end_time)
        & quotes["trade_time"].lt(match_deadline)
        & quotes["open_price"].notna()
        & quotes["trade_volume"].gt(0)
        & quotes["trade_amount"].gt(0)
    )
    match_rows = quotes.loc[match_mask]
    has_match = not match_rows.empty
    match_row = match_rows.iloc[0] if has_match else None
    if match_override is not None:
        override_price = pd.to_numeric(match_override.get("open_price"), errors="coerce")
        override_volume = pd.to_numeric(match_override.get("trade_volume"), errors="coerce")
        override_amount = pd.to_numeric(match_override.get("trade_amount"), errors="coerce")
        if (
            np.isfinite(override_price)
            and override_price > 0
            and np.isfinite(override_volume)
            and override_volume > 0
            and np.isfinite(override_amount)
            and override_amount > 0
        ):
            match_row = pd.Series(
                {
                    "trade_time": match_override.get("trade_time", nominal_end_time),
                    "open_price": float(override_price),
                    "trade_volume": float(override_volume),
                    "trade_amount": float(override_amount),
                }
            )
            has_match = True
    auction = quotes.loc[
        quotes["trade_time"].ge(start_time)
        & quotes["trade_time"].lt(nominal_end_time)
    ].copy()
    auction["indicative_price"] = _calculate_indicative_price(auction)
    auction["l3_imbalance"] = _calculate_l3_imbalance(auction)
    auction["relative_spread"] = _calculate_relative_spread(auction)

    valid_price = auction.dropna(subset=["indicative_price"])
    stage1 = auction.loc[auction["trade_time"].lt(split_time)]
    stage2 = auction.loc[auction["trade_time"].ge(split_time)]
    stage1_valid = stage1.dropna(subset=["indicative_price"])
    stage2_valid = stage2.dropna(subset=["indicative_price"])

    previous_close = _resolve_previous_close(
        _without_qmt_state_anchors(quotes),
        expected_previous_close,
        ts_code=ts_code,
        trade_date=trade_date,
    )
    row["previous_close"] = previous_close
    row["auction_has_match"] = has_match
    if has_match:
        row["auction_match_source"] = match_source or "quote"
    row["snapshot_count_stage1"] = int(len(_without_qmt_state_anchors(stage1_valid)))
    row["snapshot_count_stage2"] = int(len(_without_qmt_state_anchors(stage2_valid)))

    if match_row is not None:
        row["available_time"] = pd.Timestamp(match_row["trade_time"])
        row["auction_open_price"] = float(match_row["open_price"])
        row["auction_amount"] = float(match_row["trade_amount"])
        row["auction_matched_volume"] = float(match_row["trade_volume"])
        row["auction_overnight_return"] = _safe_return(
            float(match_row["open_price"]), previous_close
        )
        if event_reconstruction_ok and events is not None:
            stage1_events = _event_slice(events, "A", start_time, split_time)
            stage1_cancels = _event_slice(events, "C", start_time, split_time)
            stage2_events = _event_slice(events, "A", split_time, nominal_end_time)
            _apply_transaction_order_ratio_factors(
                row, trade_day, events, stage1_events, stage1_cancels,
                stage2_events, True,
            )
    elif not auction.empty or (events is not None and not events.empty):
        row["available_time"] = nominal_end_time

    _apply_report_supplement_factors(
        row,
        valid_price,
        stage1_valid,
        stage2_valid,
        previous_close,
        nominal_end_time,
    )
    _apply_matched_volume_participation(row)
    _apply_stage2_twap_factors(row, stage2, split_time, nominal_end_time)

    if not valid_price.empty:
        final = valid_price.iloc[-1]
        row["auction_final_indicative_price"] = float(final["indicative_price"])

    if len(stage1_valid) >= 2:
        stage1_first = stage1_valid.iloc[0]
        stage1_final = stage1_valid.iloc[-1]
        row["auction_return_stage1"] = _safe_return(
            float(stage1_final["indicative_price"]),
            float(stage1_first["indicative_price"]),
        )
        first_imbalance = stage1_first["l3_imbalance"]
        final_imbalance = stage1_final["l3_imbalance"]
        if np.isfinite(first_imbalance) and np.isfinite(final_imbalance):
            row["auction_imbalance_change_stage1"] = float(
                final_imbalance - first_imbalance
            )
            row["auction_imbalance_relative_change_stage1"] = _relative_imbalance_change(
                final_imbalance, first_imbalance
            )
            row["auction_imbalance_fisher_change_stage1"] = _fisher_imbalance_change(
                final_imbalance, first_imbalance
            )

    if len(stage2_valid) >= 2:
        stage2_first = stage2_valid.iloc[0]
        stage2_final = stage2_valid.iloc[-1]
        row["auction_return_stage2"] = _safe_return(
            float(stage2_final["indicative_price"]),
            float(stage2_first["indicative_price"]),
        )
        first_imbalance, final_imbalance = _stage2_imbalance_endpoints(stage2_valid)
        if np.isfinite(first_imbalance) and np.isfinite(final_imbalance):
            row["auction_imbalance_change_stage2"] = float(
                final_imbalance - first_imbalance
            )
            row["auction_imbalance_relative_change_stage2"] = _relative_imbalance_change(
                final_imbalance, first_imbalance
            )
            row["auction_imbalance_fisher_change_stage2"] = _fisher_imbalance_change(
                final_imbalance, first_imbalance
            )

    stage1_updates = _without_qmt_state_anchors(stage1)
    stage2_updates = _without_qmt_state_anchors(stage2)
    stage1_imbalance = (stage1_updates if not stage1_updates.empty else stage1)[
        "l3_imbalance"
    ].dropna()
    stage2_imbalance = (stage2_updates if not stage2_updates.empty else stage2)[
        "l3_imbalance"
    ].dropna()
    if not stage1_imbalance.empty and not stage2_imbalance.empty:
        row["auction_commitment_shift"] = float(
            stage2_imbalance.median() - stage1_imbalance.median()
        )

    row["auction_stage2_slope_bps_per_min"] = _stage2_slope(
        _without_qmt_state_anchors(stage2_valid), previous_close
    )
    if not stage2_valid.empty and np.isfinite(previous_close) and previous_close > 0:
        row["auction_stage2_range_bps"] = float(
            (
                stage2_valid["indicative_price"].max()
                - stage2_valid["indicative_price"].min()
            )
            / previous_close
            * 10000.0
        )
    row["auction_stage2_efficiency_ratio"] = _stage2_efficiency(stage2_valid)
    _apply_stage2_path_factors(row, stage2_valid, previous_close)

    if not valid_price.empty:
        final = valid_price.iloc[-1]
        unmatched_bid = float(final["bid_qty2"] + final["bid_qty3"])
        unmatched_ask = float(final["ask_qty2"] + final["ask_qty3"])
        unmatched_total = unmatched_bid + unmatched_ask
        if not np.isfinite(unmatched_bid) or not np.isfinite(unmatched_ask):
            row["auction_unmatched_imbalance"] = np.nan
        elif unmatched_total > 0:
            row["auction_unmatched_imbalance"] = float(
                (unmatched_bid - unmatched_ask) / unmatched_total
            )
        else:
            row["auction_unmatched_imbalance"] = np.nan
    _apply_stage_reversal(row)
    return row
