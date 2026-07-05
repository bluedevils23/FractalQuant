"""
JoinQuant ETF strategy driven by the volume_weighted_price factor.

How to use on JoinQuant:
1. Create a minute-level ETF strategy.
2. Paste this whole file into the strategy editor.
3. Adjust g.universe_name / g.custom_universe in initialize() if needed.

Notes:
- JoinQuant standard ETF accounts are long-only, so this is a long-only adaptation
  of the cross-sectional factor research.
- The factor definition matches the repo implementation:
  volume_weighted_price = (rolling_vwap - last_close) / last_close
- Signals are generated and executed within the same minute callback.
"""

import math

import numpy as np
import pandas as pd


HK_ETF_CODES = [
    "513180.SH", "513330.SH", "513130.SH", "159605.SZ", "513060.SH", "513090.SH",
    "513050.SH", "513120.SH", "159607.SZ", "159920.SZ", "159740.SZ", "513010.SH",
    "510900.SH", "159892.SZ", "159792.SZ", "513770.SH", "513980.SH", "513580.SH",
    "513380.SH", "159742.SZ", "159688.SZ", "513200.SH", "513660.SH", "159741.SZ",
    "513550.SH", "159699.SZ", "513600.SH", "513020.SH", "513700.SH", "513260.SH",
    "513890.SH", "159776.SZ", "159747.SZ", "159506.SZ", "159636.SZ", "513860.SH",
    "513810.SH", "513160.SH", "159750.SZ", "513690.SH", "159735.SZ", "159850.SZ",
    "513960.SH", "513530.SH", "513970.SH", "513320.SH", "513900.SH", "159691.SZ",
    "513070.SH", "513280.SH", "513150.SH", "513040.SH", "159954.SZ", "159718.SZ",
    "513140.SH", "513990.SH", "159726.SZ", "159751.SZ", "159960.SZ", "159823.SZ",
    "513560.SH", "513230.SH", "159788.SZ", "513590.SH", "159712.SZ", "159711.SZ",
]

OVERSEAS_ETF_CODES = [
    "159941.SZ", "513100.SH", "513300.SH", "159632.SZ", "513500.SH", "513360.SH",
    "513310.SH", "159509.SZ", "513110.SH", "159612.SZ", "159501.SZ", "513520.SH",
    "159866.SZ", "159696.SZ", "513220.SH", "513030.SH", "159513.SZ", "159660.SZ",
    "513000.SH", "513080.SH", "513390.SH", "513650.SH", "513880.SH", "159659.SZ",
    "513290.SH", "159822.SZ", "159655.SZ", "513800.SH", "159687.SZ",
]


def report_status(context):
    positions = [
        security
        for security, position in context.portfolio.positions.items()
        if position.total_amount > 0
    ]
    log.info(
        "universe=%s holdings=%s total_value=%.2f cash=%.2f",
        g.universe_name,
        positions,
        context.portfolio.total_value,
        context.portfolio.available_cash,
    )


def rebalance_daily(context):
    rebalance(context, None)


def initialize(context):
    set_slippage(FixedSlippage(0), type='fund')
    set_option("use_real_price", True)
    try:
        set_option("avoid_future_data", True)
    except Exception:
        pass

    log.set_level("order", "error")
    log.set_level("system", "error")

    # The backtest side was highly cost-sensitive, so default costs stay explicit.
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0001,
            close_commission=0.0001,
            close_today_commission=0,
            min_commission=0,
        ),
        type="fund",
    )
    g.open_commission_rate = 0.0001

    g.universe_name = "hk"  # "hk", "overseas", or "custom"
    g.custom_universe = []
    g.etf_pool = resolve_universe(g.universe_name, g.custom_universe)
    set_benchmark(build_equal_weight_benchmark(g.etf_pool))

    g.signal_window = 50
    g.rebalance_interval = 1
    g.max_holdings = 5
    g.min_candidates = 5
    g.enter_pct = 0.10
    g.hold_pct = 0.20
    g.require_positive_signal = False
    g.min_order_value = 5000
    g.rebalance_tolerance = 0.20
    g.min_trade_shares = 100
    g.use_intraday_rebalance = True
    g.flatten_time = "14:50"
    g.use_limit_orders = False
    g.buy_order_price_offset = 0.0
    g.sell_order_price_offset = 0.0
    g.min_price_tick = 0.001
    g.last_rebalance_minute = None
    g.last_flatten_date = None
    g.rebalance_times = build_rebalance_times(g.rebalance_interval)

    run_daily(report_status, time="14:55")


def resolve_universe(universe_name, custom_universe):
    if universe_name == "hk":
        raw_codes = HK_ETF_CODES
    elif universe_name == "overseas":
        raw_codes = OVERSEAS_ETF_CODES
    elif universe_name == "custom":
        raw_codes = custom_universe
    else:
        raise ValueError("unknown universe_name: %s" % universe_name)
    return [to_joinquant_code(code) for code in raw_codes]


def build_equal_weight_benchmark(securities):
    if not securities:
        raise ValueError("benchmark universe is empty")

    valid = []
    invalid = []
    try:
        fund_universe = get_all_securities(["fund"]).index
        fund_set = set(str(security) for security in fund_universe)
        for security in securities:
            if security in fund_set:
                valid.append(security)
            else:
                invalid.append(security)
    except Exception:
        valid = list(securities)

    if invalid:
        log.info("skip invalid benchmark securities=%s", invalid)
    if not valid:
        raise ValueError("no valid benchmark securities found")

    weight = 1.0 / float(len(valid))
    return {security: weight for security in valid}


def to_joinquant_code(code):
    if code.endswith(".SH"):
        return code[:-3] + ".XSHG"
    if code.endswith(".SZ"):
        return code[:-3] + ".XSHE"
    return code


def build_rebalance_times(interval_minutes):
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be >= 1")

    result = set()

    morning_hour = 9
    morning_minute = 30 + interval_minutes
    while morning_hour < 12:
        if morning_hour == 11 and morning_minute > 30:
            break
        result.add("%02d:%02d" % (morning_hour, morning_minute))
        morning_minute += interval_minutes
        while morning_minute >= 60:
            morning_hour += 1
            morning_minute -= 60

    afternoon_hour = 13
    afternoon_minute = interval_minutes
    while afternoon_hour < 15 or (afternoon_hour == 15 and afternoon_minute == 0):
        if afternoon_hour == 15:
            break
        result.add("%02d:%02d" % (afternoon_hour, afternoon_minute))
        afternoon_minute += interval_minutes
        while afternoon_minute >= 60:
            afternoon_hour += 1
            afternoon_minute -= 60

    result.add("14:50")
    return result


def select_targets(context, scores):
    if len(scores) < g.min_candidates:
        return []

    filtered_scores = [(security, score) for security, score in scores if score > 0] if g.require_positive_signal else list(scores)
    if len(filtered_scores) < 1:
        filtered_scores = list(scores)
    if len(filtered_scores) < 1:
        return []

    enter_count = max(1, int(math.floor(len(filtered_scores) * g.enter_pct)))
    hold_count = max(enter_count, int(math.floor(len(filtered_scores) * g.hold_pct)))
    enter_count = min(enter_count, g.max_holdings)
    hold_count = min(max(hold_count, enter_count), len(filtered_scores))

    enter_set = set(security for security, _ in filtered_scores[:enter_count])
    hold_set = set(security for security, _ in filtered_scores[:hold_count])
    holdings = set(
        security
        for security, position in context.portfolio.positions.items()
        if position.total_amount > 0
    )

    selected = enter_set | (holdings & hold_set)
    ranked = {security: score for security, score in filtered_scores}
    selected = sorted(selected, key=lambda security: ranked.get(security, -1e9), reverse=True)
    return selected[: g.max_holdings]


def floor_to_board_lot(shares):
    return int(shares / 100.0) * 100


def affordable_buy_amount(available_cash, price):
    if available_cash <= 0 or price <= 0:
        return 0
    budget_shares = available_cash / (price * (1.0 + g.open_commission_rate))
    return floor_to_board_lot(budget_shares)


def build_order_style(current_price, target_amount, current_amount):
    if not g.use_limit_orders:
        return None
    offset = g.buy_order_price_offset if target_amount >= current_amount else g.sell_order_price_offset
    limit_price = max(current_price + offset, g.min_price_tick)
    limit_price = round(limit_price, 3)
    return LimitOrderStyle(limit_price)


def execute_trades(context, selected):
    selected = list(selected)
    current_data = get_current_data()
    available_cash = float(context.portfolio.available_cash)
    holding_positions = {
        security: position
        for security, position in context.portfolio.positions.items()
        if position.total_amount > 0
    }
    holding_securities = list(holding_positions.keys())

    for security in holding_securities:
        if security not in selected:
            position = holding_positions[security]
            if position.closeable_amount >= g.min_trade_shares:
                current = current_data[security]
                current_price = float(current.last_price) if current.last_price else 0.0
                style = build_order_style(current_price, 0, int(position.total_amount)) if current_price > 0 else None
                if style is None:
                    order_target(security, 0)
                else:
                    order_target(security, 0, style=style)

    if not selected:
        return

    target_value = context.portfolio.total_value / float(len(selected))
    if target_value < g.min_order_value:
        return

    for security in selected:
        current = current_data[security]
        current_price = float(current.last_price) if current.last_price else 0.0
        if current_price <= 0:
            continue

        target_amount = floor_to_board_lot(target_value / current_price)
        if target_amount < g.min_trade_shares:
            continue

        position = holding_positions.get(security)
        current_value = 0.0 if position is None else float(position.value)
        current_amount = 0 if position is None else int(position.total_amount)
        style = build_order_style(current_price, target_amount, current_amount)

        if position is None:
            target_amount = min(target_amount, affordable_buy_amount(available_cash, current_price))
            if target_amount < g.min_trade_shares:
                continue
            if style is None:
                order_target(security, target_amount)
            else:
                order_target(security, target_amount, style=style)
            available_cash -= target_amount * current_price * (1.0 + g.open_commission_rate)
            continue

        deviation_ratio = abs(current_value - target_value) / max(target_value, 1.0)
        if deviation_ratio < g.rebalance_tolerance:
            continue

        trade_amount = target_amount - current_amount
        if abs(trade_amount) < g.min_trade_shares:
            continue

        if trade_amount > 0:
            max_additional = affordable_buy_amount(available_cash, current_price)
            trade_amount = min(trade_amount, max_additional)
            trade_amount = floor_to_board_lot(trade_amount)
            if trade_amount < g.min_trade_shares:
                continue
            target_amount = current_amount + trade_amount
            if trade_amount * current_price >= g.min_order_value:
                if style is None:
                    order_target(security, target_amount)
                else:
                    order_target(security, target_amount, style=style)
                available_cash -= trade_amount * current_price * (1.0 + g.open_commission_rate)
            continue

        sell_amount = abs(trade_amount)
        if sell_amount >= g.min_trade_shares and position.closeable_amount >= sell_amount:
            if style is None:
                order_target(security, target_amount)
            else:
                order_target(security, target_amount, style=style)


def flatten_positions(context):
    current_data = get_current_data()
    for security, position in context.portfolio.positions.items():
        if position.total_amount <= 0:
            continue
        if position.closeable_amount < g.min_trade_shares:
            continue
        current = current_data[security]
        current_price = float(current.last_price) if current.last_price else 0.0
        style = build_order_style(current_price, 0, int(position.total_amount)) if current_price > 0 else None
        if style is None:
            order_target(security, 0)
        else:
            order_target(security, 0, style=style)


def compute_scores(context):
    current_data = get_current_data()
    trade_day = context.current_dt.strftime("%Y-%m-%d")
    scores = []

    for security in g.etf_pool:
        current = current_data[security]
        if current.paused:
            continue

        hist = attribute_history(
            security,
            count=g.signal_window,
            unit="1m",
            fields=("close", "volume"),
            skip_paused=True,
            df=True,
        )
        if hist is None or len(hist) < g.signal_window:
            continue

        hist = hist[hist.index.strftime("%Y-%m-%d") == trade_day]
        hist = hist.dropna()
        if len(hist) < g.signal_window:
            continue

        volume_sum = float(hist["volume"].sum())
        last_close = float(hist["close"].iloc[-1])
        if volume_sum <= 0 or last_close <= 0:
            continue

        rolling_vwap = float((hist["close"] * hist["volume"]).sum() / volume_sum)
        score = (rolling_vwap - last_close) / last_close
        if not np.isfinite(score):
            continue
        scores.append((security, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def rebalance(context, data):
    scores = compute_scores(context)
    selected = select_targets(context, scores)
    log.info(
        "rebalance dt=%s score_count=%d selected=%s",
        context.current_dt.strftime("%Y-%m-%d %H:%M"),
        len(scores),
        selected,
    )
    execute_trades(context, selected)


def handle_data(context, data):
    if not g.use_intraday_rebalance:
        return
    trade_day = context.current_dt.strftime("%Y-%m-%d")
    minute_key = context.current_dt.strftime("%Y-%m-%d %H:%M")
    clock = context.current_dt.strftime("%H:%M")
    if clock == g.flatten_time:
        if g.last_flatten_date != trade_day:
            g.last_flatten_date = trade_day
            flatten_positions(context)
        return
    if clock not in g.rebalance_times:
        return
    if g.last_rebalance_minute == minute_key:
        return
    g.last_rebalance_minute = minute_key
    rebalance(context, data)
