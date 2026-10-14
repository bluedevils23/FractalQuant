"""Auction output schema and calculation constants (no I/O)."""

KEY_COLUMNS = ["trade_date", "available_time", "ts_code"]
DIAGNOSTIC_COLUMNS = [
    "auction_has_match",
    "auction_match_source",
    "snapshot_count_stage1",
    "snapshot_count_stage2",
    "auction_event_reconstruction_ok",
    "auction_add_count_stage1",
    "auction_cancel_count_stage1",
    "auction_add_count_stage2",
    "auction_large_order_history_days",
    "auction_stage2_twap_coverage_ratio",
    "benchmark_auction_has_match",
]
REFERENCE_COLUMNS = [
    "previous_close",
    "auction_open_price",
    "auction_final_indicative_price",
    "auction_amount",
    "auction_matched_volume",
    "auction_stage1_add_notional",
    "auction_stage1_cancel_notional",
    "auction_stage2_add_notional",
    "auction_large_order_threshold",
    "previous_20d_average_daily_amount",
    "previous_5d_average_daily_amount",
    "auction_stage2_twap_price",
    "auction_stage1_end_time",
    "auction_stage2_end_time",
    "previous_day_volume_shares",
    "previous_day_high",
    "previous_7d_close_max",
    "previous_day_float_market_cap_cny",
    "auction_limit_up_price",
    "auction_limit_down_price",
    "auction_submitted_volume",
    "benchmark_ts_code",
    "benchmark_available_time",
]
CORE_FACTOR_COLUMNS = [
    "auction_overnight_return",
    "auction_return_stage1",
    "auction_return_stage2",
    "auction_amount_ratio_5d",
    "auction_imbalance_change_stage1",
    "auction_imbalance_change_stage2",
    "auction_commitment_shift",
    "auction_stage2_slope_bps_per_min",
    "auction_stage2_range_bps",
    "auction_stage2_efficiency_ratio",
    "auction_matched_volume_ratio_5d",
    "auction_unmatched_imbalance",
]
EVENT_FACTOR_COLUMNS = [
    "auction_bid_cancel_qty_ratio_stage1",
    "auction_ask_cancel_qty_ratio_stage1",
    "auction_cancel_notional_ratio_stage1",
    "auction_cancel_imbalance_stage1",
    "auction_late_cancel_notional_share",
    "auction_large_order_cancel_ratio_stage1",
    "auction_large_cancel_imbalance_stage1",
    "auction_stage2_add_imbalance",
    "auction_stage2_commitment_ratio",
    "auction_stage2_last60s_add_share",
    "auction_fake_pressure_proxy",
    "auction_stage_reversal_strength_bps",
]
TRANSACTION_ORDER_RATIO_COLUMNS = [
    "auction_transaction_submitted_ratio",
    "auction_transaction_net_order_ratio",
    "auction_stage1_net_order_qty",
    "auction_stage2_order_qty",
    "auction_buy_order_imbalance",
    "auction_stage1_order_participation",
    "auction_stage2_order_participation",
]
PATH_FACTOR_COLUMNS = [
    "auction_stage2_mid_mean_return",
    "auction_stage2_mid_max_return",
    "auction_stage2_mid_min_return",
    "auction_stage2_total_variation_bps",
    "auction_stage2_up_step_ratio",
    "auction_stage2_reversal_count",
]
ROBUST_IMBALANCE_FACTOR_COLUMNS = [
    "auction_imbalance_relative_change_stage1",
    "auction_imbalance_relative_change_stage2",
    "auction_imbalance_fisher_change_stage1",
    "auction_imbalance_fisher_change_stage2",
]
PARTICIPATION_FACTOR_COLUMNS = [
    "auction_amount_to_prev5d_adv_240",
    "auction_amount_to_prev20d_adv",
    "auction_amount_zscore_20d",
    "auction_matched_volume_to_submitted_ratio",
]
PRIORITY_REPORT_FACTOR_COLUMNS = [
    "auction_final_vs_stage2_twap",
    "auction_l3_imbalance_twap_stage2",
    "auction_relative_spread_twap_stage2",
    "market_return_from_prev_close",
    "auction_gap_excess_benchmark",
    "auction_stage2_excess_return_benchmark",
]
REPORT_SUPPLEMENT_FACTOR_COLUMNS = [
    "auction_range_ratio",
    "auction_stage1_range_ratio",
    "auction_stage2_range_ratio",
    "auction_stage1_end_return_from_prev_close",
    "auction_stage2_end_return_from_stage1_end",
    "auction_up_step_ratio",
    "auction_down_step_ratio",
    "auction_snapshot_count_total",
    "auction_l3_buy_share_final",
    "auction_l3_buy_share_stage1_end",
    "auction_l3_buy_share_change_stage2",
    "auction_stage1_max_return_from_prev_close",
    "auction_stage1_min_return_from_prev_close",
    "auction_open_pullback_from_stage1_max",
    "auction_open_rebound_from_stage1_min",
    "auction_last60s_price_return",
    "auction_final_to_full_max",
]
REPORT_SMOOTHED_SOURCE_COLUMNS = [
    "auction_range_ratio",
    "auction_stage1_range_ratio",
    "auction_stage2_range_ratio",
    "auction_overnight_return",
    "auction_stage1_end_return_from_prev_close",
    "auction_stage2_end_return_from_stage1_end",
    "auction_up_step_ratio",
    "auction_down_step_ratio",
    "auction_snapshot_count_total",
    "auction_matched_volume",
    "auction_l3_buy_share_final",
    "auction_l3_buy_share_stage1_end",
    "auction_l3_buy_share_change_stage2",
]
REPORT_SMOOTHED_FACTOR_COLUMNS = [
    f"{column}_mean_20d" for column in REPORT_SMOOTHED_SOURCE_COLUMNS
]
CONTEXT_SUPPLEMENT_FACTOR_COLUMNS = [
    "auction_volume_to_prevday_volume",
    "auction_amount_to_float_mcap_prevclose",
    "auction_open_to_prev_high",
    "auction_open_to_prev7d_close_max",
    "auction_stage1_touched_limit_up",
    "auction_stage1_touched_limit_down",
    "auction_stage1_limit_up_distance_bps",
    "auction_stage1_limit_down_distance_bps",
]
VOLUME_RATIO_FACTOR_COLUMNS = [
    "auction_volume_ratio_5d",
    "auction_volume_ratio_20d",
    "auction_volume_ratio_5d_zscore",
]
TURNOVER_FACTOR_COLUMNS = [
    "auction_turnover_rate",
    "auction_turnover_rate_free",
]
SUPPLEMENT_REFERENCE_COLUMNS = [
    "auction_stage1_end_time",
    "auction_stage2_end_time",
    "previous_day_volume_shares",
    "previous_day_high",
    "previous_7d_close_max",
    "previous_day_float_market_cap_cny",
    "previous_day_float_share",
    "previous_day_free_share",
    "auction_limit_up_price",
    "auction_limit_down_price",
]
SUPPLEMENT_OUTPUT_COLUMNS = (
    SUPPLEMENT_REFERENCE_COLUMNS
    + REPORT_SUPPLEMENT_FACTOR_COLUMNS
    + CONTEXT_SUPPLEMENT_FACTOR_COLUMNS
)
FACTOR_COLUMNS = (
    CORE_FACTOR_COLUMNS
    + EVENT_FACTOR_COLUMNS
    + TRANSACTION_ORDER_RATIO_COLUMNS
    + PATH_FACTOR_COLUMNS
    + ROBUST_IMBALANCE_FACTOR_COLUMNS
    + PARTICIPATION_FACTOR_COLUMNS
    + PRIORITY_REPORT_FACTOR_COLUMNS
    + REPORT_SUPPLEMENT_FACTOR_COLUMNS
    + CONTEXT_SUPPLEMENT_FACTOR_COLUMNS
    + REPORT_SMOOTHED_FACTOR_COLUMNS
    + VOLUME_RATIO_FACTOR_COLUMNS
    + TURNOVER_FACTOR_COLUMNS
)
OUTPUT_COLUMNS = KEY_COLUMNS + DIAGNOSTIC_COLUMNS + REFERENCE_COLUMNS + FACTOR_COLUMNS
EVENT_COLUMNS = [
    "trade_time",
    "event_type",
    "side",
    "order_id",
    "price",
    "quantity",
    "notional",
    "original_notional",
]
LARGE_ORDER_LOOKBACK_DAYS = 20
LARGE_ORDER_QUANTILE = 0.90
HISTORICAL_AMOUNT_LOOKBACK_DAYS = 20
SHORT_DAILY_AMOUNT_LOOKBACK_DAYS = 5
MINUTES_PER_TRADING_DAY = 240
STAGE2_TWAP_MIN_COVERAGE = 0.80
IMBALANCE_RELATIVE_FLOOR = 0.05
IMBALANCE_FISHER_CLIP = 1.0 - 1e-6
REPORT_SMOOTHED_LOOKBACK_DAYS = 20
REPORT_SMOOTHED_MIN_PERIODS = 5

RAW_COLUMN_MAP = {
    "自然日": "raw_trade_date",
    "时间": "raw_time",
    "成交价": "trade_price",
    "成交量": "trade_volume",
    "成交额": "trade_amount",
    "开盘价": "open_price",
    "前收盘": "previous_close",
    **{f"申卖价{level}": f"ask_price{level}" for level in range(1, 4)},
    **{f"申卖量{level}": f"ask_qty{level}" for level in range(1, 4)},
    **{f"申买价{level}": f"bid_price{level}" for level in range(1, 4)},
    **{f"申买量{level}": f"bid_qty{level}" for level in range(1, 4)},
}
PRICE_COLUMNS = [
    "trade_price",
    "open_price",
    "previous_close",
    *[f"ask_price{level}" for level in range(1, 4)],
    *[f"bid_price{level}" for level in range(1, 4)],
]
QUANTITY_COLUMNS = [
    "trade_volume",
    *[f"ask_qty{level}" for level in range(1, 4)],
    *[f"bid_qty{level}" for level in range(1, 4)],
]
