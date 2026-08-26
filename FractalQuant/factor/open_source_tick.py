"""Transaction-tick factors from the open-source microstructure research.

The first open-source factor batch uses minute OHLCV bars.  This module is a
separate transaction-data path for the paper's minute single-trade-notional
signals (QUA, MTS, MTE and SR), tick ideal-reversal siblings, full-day
order-direction memory diagnostics, and a strict order-notional money-flow
family.  The latter only uses a matched original order amount and never
substitutes quote or OHLCV information for missing lifecycle fields.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        """Fallback decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator


LOGGER = logging.getLogger(__name__)


TICK_FACTOR_COLUMNS = (
    "kaiyuan_trade_notional_quantile_position_m20",
    "kaiyuan_trade_notional_q90_q10_ratio_m20",
    "kaiyuan_mts_m20",
    "kaiyuan_mte_m20",
    "kaiyuan_sr_l020_m20",
    "kaiyuan_ideal_reversal_tick_notional_m20",
    "kaiyuan_ideal_reversal_tick_volume_m20",
    "kaiyuan_trade_notional_return_corr_m20",
    "kaiyuan_order_lms_m20",
    "kaiyuan_order_memo_m20",
    "kaiyuan_order_island_mean_m20",
    "kaiyuan_order_island_std_m20",
    "kaiyuan_large_flow_s3_m20",
    "kaiyuan_small_flow_s3_m20",
    "kaiyuan_large_flow_s3_resid_ret20_cs_m20",
    "kaiyuan_small_flow_s3_resid_ret20_cs_m20",
    "kaiyuan_nir_ge20k_m20",
    "kaiyuan_nir_mod_ge20k_cs_m20",
    "kaiyuan_cnir_cs_m20",
    "kaiyuan_act_m20",
    "kaiyuan_act_pos_highret_m20_l010",
    "kaiyuan_act_neg_lowret_m20_l010",
    "kaiyuan_flow_large_high_amp_resid_m20",
    "kaiyuan_flow_small_high_amp_resid_m20",
    "kaiyuan_flow_large_open30_resid_m20",
    "kaiyuan_flow_small_open30_resid_m20",
    "kaiyuan_evl_m20",
    "kaiyuan_evm_m20",
    "kaiyuan_evs_m20",
    "kaiyuan_extra_amount_adj_m20",
    "kaiyuan_large_amount_adj_m20",
    "kaiyuan_medium_amount_adj_m20",
    "kaiyuan_small_amount_adj_m20",
)

RAW_TICK_COLUMNS = (
    "trade_date",
    "ts_code",
    "daily_qua",
    "daily_q90_q10_ratio",
    "daily_mts",
    "daily_mte",
    "daily_sr_l020",
    "daily_return",
    "daily_close",
    "daily_mean_trade_notional",
    "daily_mean_trade_volume",
    "valid_trade_minutes",
    "trade_count",
    "daily_amount",
    "amount_available",
)

ORDER_RAW_COLUMNS = (
    "trade_date",
    "ts_code",
    "daily_order_lms",
    "daily_order_memo",
    "daily_order_island_mean",
    "daily_order_island_std",
    "valid_order_events",
    "order_side_coverage",
    "order_id_available",
)

FLOW_RAW_COLUMNS = (
    "flow_direction_coverage",
    "active_order_id_coverage",
    "active_order_match_amount_coverage",
    "flow_classified_amount",
    "flow_unmatched_amount",
    "flow_classification_valid",
    "flow_bucket_schema",
    "daily_flow_small_buy",
    "daily_flow_small_sell",
    "daily_flow_medium_buy",
    "daily_flow_medium_sell",
    "daily_flow_large_buy",
    "daily_flow_large_sell",
    "daily_flow_extra_buy",
    "daily_flow_extra_sell",
    "daily_flow_ge20k_buy",
    "daily_flow_ge20k_sell",
    "daily_flow_power_buy",
    "daily_flow_power_sell",
    "daily_flow_total_buy",
    "daily_flow_total_sell",
    "daily_flow_large_high_amp_buy",
    "daily_flow_large_high_amp_sell",
    "daily_flow_small_high_amp_buy",
    "daily_flow_small_high_amp_sell",
    "daily_flow_large_open30_buy",
    "daily_flow_large_open30_sell",
    "daily_flow_small_open30_buy",
    "daily_flow_small_open30_sell",
    "conditional_flow_high_amp_count",
    "conditional_flow_open30_count",
)

AUDIT_COLUMNS = (
    "trade_date",
    "available_date",
    "available_time",
    "ts_code",
    "valid_trade_minutes",
    "trade_count",
    "daily_amount",
    "amount_available",
    "source_level",
    "valid_order_events",
    "order_side_coverage",
    "order_id_available",
    "flow_direction_coverage",
    "active_order_id_coverage",
    "active_order_match_amount_coverage",
    "flow_classified_amount",
    "flow_unmatched_amount",
    "flow_classification_valid",
    "flow_bucket_schema",
    "flow_cross_section_count",
    "flow_s3_resid_r2",
    "flow_nir_mod_r2",
    "flow_cnir_r2",
    "conditional_flow_high_amp_count",
    "conditional_flow_open30_count",
    "flow_evl_r2",
    "flow_evm_r2",
    "flow_evs_r2",
)

SOURCE_LEVEL = "tick_transaction"
SOURCE_LEVEL_ORDER = "tick_transaction+order"
SOURCE_LEVEL_ORDER_ONLY = "tick_order"
MIN_VALID_TRADE_MINUTES = 120
ROLLING_WINDOW = 20
SR_TOP_FRACTION = 0.20
REVERSAL_FRACTION = 0.50
FLOW_MATCH_MIN_COVERAGE = 0.95
FLOW_DIRECTION_MIN_COVERAGE = 0.99
SMALL_ORDER_MAX = 40_000.0
MEDIUM_ORDER_MAX = 200_000.0
LARGE_ORDER_MAX = 1_000_000.0
NIR_ORDER_THRESHOLD = 20_000.0
FLOW_BUCKET_SCHEMA = "order_notional_lt40k_40k_200k_200k_1m_ge1m_nir_ge20k"

_RAW_POSITIONAL_COLUMNS = [
    "symbol",
    "exchange_code",
    "raw_trade_date",
    "raw_time",
    "trade_no",
    "trade_type",
    "order_id",
    "bs_flag",
    "price",
    "quantity",
    "sell_order_id",
    "buy_order_id",
    "extra",
]

_ORDER_POSITIONAL_COLUMNS = [
    "symbol",
    "exchange_code",
    "raw_trade_date",
    "raw_time",
    "order_sequence",
    "order_id",
    "order_type",
    "side",
    "price",
    "quantity",
    "extra",
]

_COLUMN_ALIASES = {
    "vol": "quantity",
    "volume": "quantity",
    "trade_volume": "quantity",
    "成交数量": "quantity",
    "成交量": "quantity",
    "成交价格": "price",
    "成交价": "price",
    "成交代码": "trade_code",
    "成交类型": "trade_code",
    "trade_code": "trade_code",
    "trade_type": "trade_code",
    "BS标志": "bs_flag",
    "bs_flag": "bs_flag",
    "万得代码": "ts_code",
    "交易所代码": "exchange_code",
    "成交编号": "trade_no",
    "委托代码": "order_id",
    "叫卖序号": "sell_order_id",
    "叫买序号": "buy_order_id",
    "卖方委托号": "sell_order_id",
    "买方委托号": "buy_order_id",
    "datetime": "trade_time",
    "time": "raw_time",
    "时间": "raw_time",
    "自然日": "raw_trade_date",
    "date": "raw_trade_date",
    "trade_date": "trade_date",
    "symbol": "ts_code",
    "ts_code": "ts_code",
}

_ORDER_COLUMN_ALIASES = {
    "委托数量": "quantity",
    "委托价格": "price",
    "委托代码": "side",
    "委托类型": "order_type",
    "交易所委托号": "order_id",
    "委托编号": "order_sequence",
    "委托序号": "order_sequence",
    "万得代码": "ts_code",
    "交易所代码": "exchange_code",
    "order_id": "order_id",
    "order_type": "order_type",
    "side": "side",
    "price": "price",
    "quantity": "quantity",
    "volume": "quantity",
    "trade_time": "trade_time",
    "datetime": "trade_time",
    "时间": "raw_time",
    "自然日": "raw_trade_date",
    "trade_date": "raw_trade_date",
    "symbol": "ts_code",
    "ts_code": "ts_code",
}


def _parse_trade_time(date: pd.Series, raw_time: pd.Series) -> pd.Series:
    date_text = date.astype("string").str.replace(r"\D", "", regex=True).str.zfill(8)
    time_text = raw_time.astype("string").str.replace(r"\D", "", regex=True).str.zfill(9)
    return pd.to_datetime(
        date_text + time_text, format="%Y%m%d%H%M%S%f", errors="coerce"
    )


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = str(column).strip()
        renamed[column] = _COLUMN_ALIASES.get(key, key)
    return frame.rename(columns=renamed)


def normalize_tick_transactions(
    raw_df: pd.DataFrame, ts_code_hint: str | None = None
) -> pd.DataFrame:
    """Normalize canonical/parquet or the local Chinese tick schema.

    Prices in the raw ``E:\\逐笔数据`` CSV files are integer ten-thousandths;
    a conservative magnitude check converts those values to yuan.  Rows with
    zero price/quantity or cancellation records are excluded.
    """

    frame = _rename_columns(raw_df.copy())
    if "trade_time" not in frame.columns:
        if {"raw_trade_date", "raw_time"}.issubset(frame.columns):
            frame["trade_time"] = _parse_trade_time(
                frame["raw_trade_date"], frame["raw_time"]
            )
        elif {"trade_date", "raw_time"}.issubset(frame.columns):
            frame["trade_time"] = _parse_trade_time(
                frame["trade_date"], frame["raw_time"]
            )
        else:
            raise ValueError("Cannot locate trade_time or raw date/time columns")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    missing = sorted({"price", "quantity"} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required tick columns: {missing}")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    for column in ("buy_order_id", "sell_order_id", "order_id"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "bs_flag" in frame.columns:
        frame["bs_flag"] = frame["bs_flag"].astype("string").str.strip().str.upper()
    if frame["price"].dropna().median() > 1000:
        frame["price"] = frame["price"] / 10000.0

    if "trade_code" in frame.columns:
        code = frame["trade_code"].astype("string").str.strip().str.upper()
        frame = frame.loc[~code.isin(["C", "D", "A"])].copy()
    frame = frame.loc[
        frame["trade_time"].notna()
        & frame["price"].gt(0)
        & frame["quantity"].gt(0)
    ].copy()
    frame["amount"] = frame["price"] * frame["quantity"]
    if "ts_code" not in frame.columns:
        frame["ts_code"] = ts_code_hint
    frame["ts_code"] = frame["ts_code"].astype("string")
    clock = frame["trade_time"].dt.strftime("%H:%M:%S")
    session_mask = (
        (clock.ge("09:30:00") & clock.lt("11:30:00"))
        | (clock.ge("13:00:00") & clock.lt("15:00:00"))
    )
    frame = frame.loc[session_mask].copy()
    frame["trade_date"] = frame["trade_time"].dt.normalize()
    return frame.sort_values("trade_time", kind="mergesort").reset_index(drop=True)


def load_tick_file(path: Path, ts_code_hint: str | None = None) -> pd.DataFrame:
    """Read one local CSV/parquet transaction file into canonical columns."""

    if path.suffix.lower() == ".parquet":
        return normalize_tick_transactions(pd.read_parquet(path), ts_code_hint)
    try:
        header = pd.read_csv(path, encoding="gb18030", nrows=0).columns.tolist()
        known = {"trade_time", "price", "quantity", "volume", "成交价格", "成交数量"}
        if not known.intersection(map(str, header)) and len(header) >= 10:
            positional_names = _RAW_POSITIONAL_COLUMNS + [
                f"extra_{i}" for i in range(max(0, len(header) - len(_RAW_POSITIONAL_COLUMNS)))
            ]
            frame = pd.read_csv(
                path,
                encoding="gb18030",
                header=0,
                names=positional_names[: len(header)],
                low_memory=False,
            )
        else:
            frame = pd.read_csv(path, encoding="gb18030", low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="gbk", low_memory=False)
    return normalize_tick_transactions(frame, ts_code_hint)


def normalize_tick_orders(
    raw_df: pd.DataFrame, ts_code_hint: str | None = None
) -> pd.DataFrame:
    """Normalize full-day order events without inferring missing lifecycle data."""

    frame = raw_df.copy()
    renamed = {}
    for column in frame.columns:
        key = str(column).strip()
        renamed[column] = _ORDER_COLUMN_ALIASES.get(key, key)
    frame = frame.rename(columns=renamed)
    if "trade_time" not in frame.columns:
        if {"raw_trade_date", "raw_time"}.issubset(frame.columns):
            frame["trade_time"] = _parse_trade_time(
                frame["raw_trade_date"], frame["raw_time"]
            )
        else:
            raise ValueError("Cannot locate order trade_time or raw date/time columns")
    missing = sorted({"side", "quantity"} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required order columns: {missing}")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    if "price" in frame.columns:
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        if frame["price"].dropna().median() > 1000:
            frame["price"] = frame["price"] / 10000.0
    frame["side"] = frame["side"].astype("string").str.strip().str.upper()
    if "order_type" in frame.columns:
        frame["order_type"] = frame["order_type"].astype("string").str.strip().str.upper()
        # The local archive uses 0 for an order add.  Other event types are
        # retained only when an explicit canonical input does not provide it.
        frame = frame.loc[frame["order_type"].isin(["0", "B", "S", "ADD", "A", ""])].copy()
    frame = frame.loc[
        frame["trade_time"].notna()
        & frame["quantity"].gt(0)
        & frame["side"].isin(["B", "S"])
    ].copy()
    if "ts_code" not in frame.columns:
        frame["ts_code"] = ts_code_hint
    frame["ts_code"] = frame["ts_code"].astype("string")
    clock = frame["trade_time"].dt.strftime("%H:%M:%S")
    session_mask = (
        (clock.ge("09:30:00") & clock.lt("11:30:00"))
        | (clock.ge("13:00:00") & clock.lt("15:00:00"))
    )
    frame = frame.loc[session_mask].copy()
    frame["trade_date"] = frame["trade_time"].dt.normalize()
    if "order_id" in frame.columns:
        frame["order_id"] = pd.to_numeric(frame["order_id"], errors="coerce")
    return frame.sort_values("trade_time", kind="mergesort").reset_index(drop=True)


def load_order_file(path: Path, ts_code_hint: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return normalize_tick_orders(pd.read_parquet(path), ts_code_hint)
    header = pd.read_csv(path, encoding="gb18030", nrows=0).columns.tolist()
    known = {"trade_time", "side", "quantity", "委托代码", "委托数量"}
    if not known.intersection(map(str, header)) and len(header) >= 8:
        positional_names = _ORDER_POSITIONAL_COLUMNS + [
            f"extra_{i}" for i in range(max(0, len(header) - len(_ORDER_POSITIONAL_COLUMNS)))
        ]
        frame = pd.read_csv(
            path,
            encoding="gb18030",
            header=0,
            names=positional_names[: len(header)],
            low_memory=False,
        )
    else:
        frame = pd.read_csv(path, encoding="gb18030", low_memory=False)
    return normalize_tick_orders(frame, ts_code_hint)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def _daily_metrics(day: pd.DataFrame, symbol: str) -> dict[str, object]:
    minute = day.set_index("trade_time").resample("1min").agg(
        minute_close=("price", "last"),
        minute_amount=("amount", "sum"),
        trade_count=("price", "size"),
    ).dropna(subset=["minute_close"])
    if minute.empty:
        return {"trade_date": day["trade_date"].iloc[0], "ts_code": symbol}
    minute["single_trade_amount"] = minute["minute_amount"] / minute["trade_count"]
    minute["minute_return"] = minute["minute_close"].pct_change()
    valid_minutes = int(len(minute))
    total_count = int(minute["trade_count"].sum())
    total_amount = float(minute["minute_amount"].sum())
    result: dict[str, object] = {
        "trade_date": day["trade_date"].iloc[0],
        "ts_code": symbol,
        "valid_trade_minutes": valid_minutes,
        "trade_count": total_count,
        "daily_amount": total_amount,
        "amount_available": True,
        "daily_return": float(day["price"].iloc[-1] / day["price"].iloc[0] - 1.0),
        "daily_close": float(day["price"].iloc[-1]),
        "daily_mean_trade_notional": float(day["amount"].mean()),
        "daily_mean_trade_volume": float(day["quantity"].mean()),
    }
    if valid_minutes < MIN_VALID_TRADE_MINUTES:
        return result

    amounts = minute["single_trade_amount"].replace([np.inf, -np.inf], np.nan).dropna()
    clean = amounts.sort_values().iloc[:-10] if len(amounts) > 10 else amounts
    if len(clean) >= 3:
        amin, amax = float(clean.min()), float(clean.max())
        if amax > amin:
            result["daily_qua"] = float((clean.quantile(0.10) - amin) / (amax - amin))
    q10, q90 = amounts.quantile([0.10, 0.90]) if len(amounts) >= 3 else (np.nan, np.nan)
    if pd.notna(q10) and q10 > 0:
        result["daily_q90_q10_ratio"] = float(q90 / q10)
    result["daily_mts"] = _safe_corr(minute["single_trade_amount"], minute["minute_amount"])
    result["daily_mte"] = _safe_corr(minute["single_trade_amount"], minute["minute_close"])
    ranked = minute.dropna(subset=["single_trade_amount", "minute_return"]).sort_values(
        "single_trade_amount", ascending=False
    )
    top_n = max(1, int(np.ceil(len(ranked) * SR_TOP_FRACTION)))
    if len(ranked) >= 3:
        result["daily_sr_l020"] = float(ranked.head(top_n)["minute_return"].sum())
    return result


def build_daily_tick_raw_features(
    transactions: pd.DataFrame, ts_code: str | None = None
) -> pd.DataFrame:
    frame = normalize_tick_transactions(transactions, ts_code)
    if frame.empty:
        return pd.DataFrame(columns=RAW_TICK_COLUMNS)
    symbol = str(frame["ts_code"].dropna().iloc[0]) if frame["ts_code"].notna().any() else str(ts_code)
    rows = [_daily_metrics(day, symbol) for _, day in frame.groupby("trade_date", sort=True)]
    result = pd.DataFrame(rows)
    for column in RAW_TICK_COLUMNS:
        if column not in result:
            result[column] = np.nan
    return result[list(RAW_TICK_COLUMNS)].sort_values("trade_date").reset_index(drop=True)


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
        valid = np.isfinite(window_values) & np.isfinite(window_state)
        if valid.sum() < window:
            continue
        order = np.argsort(window_state, kind="mergesort")
        low = order[:group_size]
        high = order[-group_size:]
        result[end] = float(np.mean(window_values[high]) - np.mean(window_values[low]))
    return pd.Series(result, index=values.index)


def _rolling_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    result = np.full(len(left), np.nan, dtype=float)
    left_array = left.to_numpy(dtype=float, copy=False)
    right_array = right.to_numpy(dtype=float, copy=False)
    for end in range(window - 1, len(left)):
        start = end - window + 1
        x = left_array[start : end + 1]
        y = right_array[start : end + 1]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < window or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
            continue
        result[end] = float(np.corrcoef(x[valid], y[valid])[0, 1])
    return pd.Series(result, index=left.index)


def _order_daily_metrics(day: pd.DataFrame, symbol: str) -> dict[str, object]:
    result: dict[str, object] = {
        "trade_date": day["trade_date"].iloc[0],
        "ts_code": symbol,
        "valid_order_events": int(len(day)),
        "order_side_coverage": float(day["side"].isin(["B", "S"]).mean()),
        "order_id_available": bool("order_id" in day.columns and day["order_id"].notna().any()),
    }
    signs = np.where(day["side"].eq("B"), 1.0, -1.0)
    if len(signs) < 20:
        return result
    max_lag = min(100, len(signs) - 2)
    correlations: list[float] = []
    for lag in range(1, max_lag + 1):
        left = signs[lag:]
        right = signs[:-lag]
        if np.unique(left).size < 2 or np.unique(right).size < 2:
            continue
        correlations.append(float(np.corrcoef(left, right)[0, 1]))
    if len(correlations) >= 3:
        lags = np.arange(1, len(correlations) + 1, dtype=float)
        x = np.column_stack([np.ones(len(lags)), np.log(lags)])
        beta, *_ = np.linalg.lstsq(x, np.asarray(correlations), rcond=None)
        result["daily_order_lms"] = float(beta[0])
        result["daily_order_memo"] = float(
            0.5 * pd.Series(correlations).skew() + 0.5 * pd.Series(correlations).kurt()
        )
    runs: list[int] = []
    current_sign = signs[0]
    run_length = 1
    for sign in signs[1:]:
        if sign == current_sign:
            run_length += 1
        else:
            runs.append(run_length)
            current_sign = sign
            run_length = 1
    runs.append(run_length)
    result["daily_order_island_mean"] = float(np.mean(runs))
    result["daily_order_island_std"] = float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0
    return result


def build_daily_order_raw_features(
    orders: pd.DataFrame, ts_code: str | None = None
) -> pd.DataFrame:
    frame = normalize_tick_orders(orders, ts_code)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "ts_code",
                "daily_order_lms",
                "daily_order_memo",
                "daily_order_island_mean",
                "daily_order_island_std",
                "valid_order_events",
                "order_side_coverage",
                "order_id_available",
            ]
        )
    symbol = str(frame["ts_code"].dropna().iloc[0]) if frame["ts_code"].notna().any() else str(ts_code)
    rows = [
        _order_daily_metrics(day, symbol)
        for _, day in frame.groupby("trade_date", sort=True)
    ]
    result = pd.DataFrame(rows)
    columns = [
        "trade_date",
        "ts_code",
        "daily_order_lms",
        "daily_order_memo",
        "daily_order_island_mean",
        "daily_order_island_std",
        "valid_order_events",
        "order_side_coverage",
        "order_id_available",
    ]
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result[columns].sort_values("trade_date").reset_index(drop=True)


def _empty_flow_row(trade_date: object, symbol: str) -> dict[str, object]:
    result: dict[str, object] = {
        "trade_date": trade_date,
        "ts_code": symbol,
        "flow_direction_coverage": np.nan,
        "active_order_id_coverage": np.nan,
        "active_order_match_amount_coverage": np.nan,
        "flow_classified_amount": np.nan,
        "flow_unmatched_amount": np.nan,
        "flow_classification_valid": False,
        "flow_bucket_schema": FLOW_BUCKET_SCHEMA,
    }
    result.update({column: np.nan for column in FLOW_RAW_COLUMNS if column not in result})
    return result


def _flow_daily_metrics(
    trades: pd.DataFrame, orders: pd.DataFrame, symbol: str
) -> dict[str, object]:
    trade_date = trades["trade_date"].iloc[0]
    result = _empty_flow_row(trade_date, symbol)
    if trades.empty or orders.empty:
        return result

    amount = pd.to_numeric(trades["amount"], errors="coerce")
    valid_amount = amount.gt(0) & amount.notna()
    total_amount = float(amount.where(valid_amount, 0.0).sum())
    if total_amount <= 0 or "bs_flag" not in trades.columns:
        return result

    side = trades["bs_flag"].astype("string").str.strip().str.upper()
    directional = side.isin(["B", "S"]) & valid_amount
    directional_amount = float(amount.where(directional, 0.0).sum())
    result["flow_direction_coverage"] = directional_amount / total_amount
    if directional_amount <= 0:
        return result

    buy_id = (
        pd.to_numeric(trades["buy_order_id"], errors="coerce")
        if "buy_order_id" in trades.columns
        else pd.Series(np.nan, index=trades.index)
    )
    sell_id = (
        pd.to_numeric(trades["sell_order_id"], errors="coerce")
        if "sell_order_id" in trades.columns
        else pd.Series(np.nan, index=trades.index)
    )
    active_id = buy_id.where(side.eq("B"), sell_id.where(side.eq("S")))
    id_amount = float(amount.where(directional & active_id.notna(), 0.0).sum())
    result["active_order_id_coverage"] = id_amount / directional_amount

    order_frame = orders.copy()
    if "order_id" not in order_frame.columns or "price" not in order_frame.columns:
        return result
    order_frame["order_id"] = pd.to_numeric(order_frame["order_id"], errors="coerce")
    order_frame["price"] = pd.to_numeric(order_frame["price"], errors="coerce")
    order_frame["quantity"] = pd.to_numeric(order_frame["quantity"], errors="coerce")
    order_frame = order_frame.loc[
        order_frame["order_id"].gt(0)
        & order_frame["price"].gt(0)
        & order_frame["quantity"].gt(0)
    ].copy()
    if order_frame.empty:
        return result
    duplicate_ids = order_frame.loc[
        order_frame["order_id"].duplicated(keep=False), "order_id"
    ]
    if not duplicate_ids.empty:
        order_frame = order_frame.loc[
            ~order_frame["order_id"].isin(set(duplicate_ids))
        ].copy()
    if order_frame.empty:
        return result
    order_notional_map = (
        order_frame.assign(order_notional=order_frame["price"] * order_frame["quantity"])
        .set_index("order_id")["order_notional"]
        .to_dict()
    )
    matched_notional = active_id.map(order_notional_map)
    matched = directional & matched_notional.gt(0) & matched_notional.notna()
    matched_amount = float(amount.where(matched, 0.0).sum())
    result["active_order_match_amount_coverage"] = matched_amount / directional_amount
    result["flow_classified_amount"] = matched_amount
    result["flow_unmatched_amount"] = max(0.0, directional_amount - matched_amount)
    if (
        result["flow_direction_coverage"] < FLOW_DIRECTION_MIN_COVERAGE
        or result["active_order_match_amount_coverage"] < FLOW_MATCH_MIN_COVERAGE
    ):
        return result

    order_value = matched_notional
    buckets = {
        "small": order_value.lt(SMALL_ORDER_MAX),
        "medium": order_value.ge(SMALL_ORDER_MAX)
        & order_value.lt(MEDIUM_ORDER_MAX),
        "large": order_value.ge(MEDIUM_ORDER_MAX)
        & order_value.lt(LARGE_ORDER_MAX),
        "extra": order_value.ge(LARGE_ORDER_MAX),
        "ge20k": order_value.ge(NIR_ORDER_THRESHOLD),
        "power": order_value.ge(SMALL_ORDER_MAX),
    }
    for name, mask in buckets.items():
        for direction, sign in (("buy", "B"), ("sell", "S")):
            result[f"daily_flow_{name}_{direction}"] = float(
                amount.where(matched & mask & side.eq(sign), 0.0).sum()
            )
    result["daily_flow_total_buy"] = float(
        amount.where(matched & side.eq("B"), 0.0).sum()
    )
    result["daily_flow_total_sell"] = float(
        amount.where(matched & side.eq("S"), 0.0).sum()
    )
    matched_rows = trades.loc[matched].copy()
    matched_rows["_matched_notional"] = order_value.loc[matched]
    matched_rows["_amount"] = amount.loc[matched]
    if not matched_rows.empty and "trade_time" in matched_rows.columns:
        matched_rows["_minute"] = matched_rows["trade_time"].dt.floor("min")
        minute_stats = matched_rows.groupby("_minute", sort=True).agg(
            minute_high=("price", "max"),
            minute_low=("price", "min"),
        )
        minute_stats["amplitude"] = minute_stats["minute_high"] / minute_stats["minute_low"] - 1.0
        minute_stats = minute_stats.replace([np.inf, -np.inf], np.nan).dropna(subset=["amplitude"])
        high_amp_minutes = set()
        if len(minute_stats) >= 2:
            cutoff = float(minute_stats["amplitude"].quantile(0.50))
            high_amp_minutes = set(minute_stats.index[minute_stats["amplitude"].ge(cutoff)])
        open30_mask = (
            matched_rows["trade_time"].dt.strftime("%H:%M:%S").ge("09:30:00")
            & matched_rows["trade_time"].dt.strftime("%H:%M:%S").lt("10:00:00")
        )
        high_amp_mask = matched_rows["_minute"].isin(high_amp_minutes)
        for name, condition in (
            ("high_amp", high_amp_mask),
            ("open30", open30_mask),
        ):
            result[f"conditional_flow_{name}_count"] = float(condition.sum())
            for bucket, bucket_mask in (
                ("large", matched_rows["_matched_notional"].ge(NIR_ORDER_THRESHOLD)),
                ("small", matched_rows["_matched_notional"].lt(SMALL_ORDER_MAX)),
            ):
                for direction, sign in (("buy", "B"), ("sell", "S")):
                    result[f"daily_flow_{bucket}_{name}_{direction}"] = float(
                        matched_rows.loc[condition & bucket_mask & matched_rows["bs_flag"].eq(sign), "_amount"].sum()
                    )
    result["flow_classification_valid"] = True
    return result


def build_daily_flow_raw_features(
    transactions: pd.DataFrame,
    orders: pd.DataFrame,
    ts_code: str | None = None,
) -> pd.DataFrame:
    """Build strict order-notional flow inputs from matched tick data.

    The original order notional classifies a fill; the fill's own notional is
    what is accumulated into buy/sell flow.  This keeps split fills from
    multiplying an order's notional and preserves nulls for unmatched data.
    """

    trade_frame = normalize_tick_transactions(transactions, ts_code)
    order_frame = normalize_tick_orders(orders, ts_code)
    if trade_frame.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", *FLOW_RAW_COLUMNS])
    symbol = (
        str(trade_frame["ts_code"].dropna().iloc[0])
        if trade_frame["ts_code"].notna().any()
        else str(ts_code)
    )
    order_frame = order_frame.copy()
    rows = []
    for trade_date, day in trade_frame.groupby("trade_date", sort=True):
        same_day_orders = order_frame.loc[order_frame["trade_date"].eq(trade_date)]
        rows.append(_flow_daily_metrics(day, same_day_orders, symbol))
    result = pd.DataFrame(rows)
    columns = ["trade_date", "ts_code", *FLOW_RAW_COLUMNS]
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result[columns].sort_values("trade_date").reset_index(drop=True)


@jit(nopython=True, cache=True)
def _rolling_flow_ratio_kernel(
    buy_values: np.ndarray,
    sell_values: np.ndarray,
    window: int,
    signed_denominator: bool,
) -> np.ndarray:
    """Numba-optimized kernel for rolling flow ratio computation."""
    n = len(buy_values)
    result = np.full(n, np.nan, dtype=np.float64)
    for end in range(window - 1, n):
        start = end - window + 1
        b = buy_values[start : end + 1]
        s = sell_values[start : end + 1]
        # Check all finite
        all_finite = True
        for i in range(len(b)):
            if not np.isfinite(b[i]) or not np.isfinite(s[i]):
                all_finite = False
                break
        if not all_finite:
            continue
        net_sum = 0.0
        if signed_denominator:
            abs_net_sum = 0.0
            for i in range(len(b)):
                net = b[i] - s[i]
                net_sum += net
                abs_net_sum += abs(net)
            if abs_net_sum > 0:
                result[end] = net_sum / abs_net_sum
        else:
            total_sum = 0.0
            for i in range(len(b)):
                net_sum += b[i] - s[i]
                total_sum += b[i] + s[i]
            if total_sum > 0:
                result[end] = net_sum / total_sum
    return result


@jit(nopython=True, cache=True)
def _rolling_conditional_act_kernel(
    buy_values: np.ndarray,
    sell_values: np.ndarray,
    return_values: np.ndarray,
    window: int,
    fraction: float,
) -> np.ndarray:
    """Numba-optimized kernel for rolling conditional ACT computation."""
    n = len(buy_values)
    result = np.full(n, np.nan, dtype=np.float64)
    group_size = max(1, int(np.ceil(window * fraction)))
    for end in range(window - 1, n):
        start = end - window + 1
        b = buy_values[start : end + 1]
        s = sell_values[start : end + 1]
        r = return_values[start : end + 1]
        # Check all finite
        all_finite = True
        for i in range(len(b)):
            if not (np.isfinite(b[i]) and np.isfinite(s[i]) and np.isfinite(r[i])):
                all_finite = False
                break
        if not all_finite:
            continue
        # Use argpartition for O(n) top-k instead of full sort
        if group_size >= len(r):
            selected = np.arange(len(r))
        else:
            # argpartition not available in numba nopython, use argsort
            selected = np.argsort(r)[-group_size:]
        numerator = 0.0
        denominator = 0.0
        for i in selected:
            numerator += b[i] - s[i]
            denominator += b[i] + s[i]
        if denominator > 0:
            result[end] = numerator / denominator
    return result


def _rolling_flow_ratio(
    buy: pd.Series,
    sell: pd.Series,
    window: int,
    *,
    signed_denominator: bool = False,
) -> pd.Series:
    buy_values = pd.to_numeric(buy, errors="coerce").to_numpy(dtype=float)
    sell_values = pd.to_numeric(sell, errors="coerce").to_numpy(dtype=float)

    if HAS_NUMBA:
        result = _rolling_flow_ratio_kernel(buy_values, sell_values, window, signed_denominator)
        return pd.Series(result, index=buy.index)

    # Fallback: manual loop (pandas rolling.apply doesn't work well for multi-column logic)
    result = np.full(len(buy_values), np.nan, dtype=float)
    for end in range(window - 1, len(buy_values)):
        start = end - window + 1
        b = buy_values[start : end + 1]
        s = sell_values[start : end + 1]
        if not (np.isfinite(b).all() and np.isfinite(s).all()):
            continue
        net = b - s
        net_sum = net.sum()
        if signed_denominator:
            denominator = np.abs(net).sum()
        else:
            denominator = (b + s).sum()
        if denominator > 0:
            result[end] = float(net_sum / denominator)
    return pd.Series(result, index=buy.index)


def _rolling_conditional_act(
    frame: pd.DataFrame,
    buy: pd.Series,
    sell: pd.Series,
    returns: pd.Series,
    window: int,
    fraction: float,
) -> pd.Series:
    b_values = pd.to_numeric(buy, errors="coerce").to_numpy(dtype=float)
    s_values = pd.to_numeric(sell, errors="coerce").to_numpy(dtype=float)
    r_values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)

    if HAS_NUMBA:
        result = _rolling_conditional_act_kernel(b_values, s_values, r_values, window, fraction)
        return pd.Series(result, index=frame.index)

    # Fallback: manual loop
    result = np.full(len(frame), np.nan, dtype=float)
    group_size = max(1, int(np.ceil(window * fraction)))
    for end in range(window - 1, len(frame)):
        start = end - window + 1
        b = b_values[start : end + 1]
        s = s_values[start : end + 1]
        r = r_values[start : end + 1]
        if not (np.isfinite(b).all() and np.isfinite(s).all() and np.isfinite(r).all()):
            continue
        selected = np.argsort(r)[-group_size:]
        numerator = (b[selected] - s[selected]).sum()
        denominator = (b[selected] + s[selected]).sum()
        if denominator > 0:
            result[end] = float(numerator / denominator)
    return pd.Series(result, index=frame.index)


def _cross_section_residual(
    frame: pd.DataFrame,
    value_column: str,
    return_column: str,
    *,
    min_count: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0.0, index=frame.index, dtype=float)
    r2_values = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("trade_date", sort=False):
        valid = (
            pd.to_numeric(group[value_column], errors="coerce").notna()
            & pd.to_numeric(group[return_column], errors="coerce").notna()
        )
        if int(valid.sum()) < min_count:
            continue
        x = group.loc[valid, return_column].to_numpy(dtype=float)
        y = group.loc[valid, value_column].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        error = y - fitted
        residual.loc[group.index[valid]] = error
        counts.loc[group.index] = float(valid.sum())
        ss_total = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - float(np.square(error).sum()) / ss_total if ss_total > 0 else np.nan
        r2_values.loc[group.index] = r2
    return residual, counts, r2_values


def _cross_section_log_flow_residual(
    frame: pd.DataFrame,
    buy_column: str,
    sell_column: str,
    return_column: str,
    *,
    min_count: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Residualize same-day log buy/sell imbalance against close return."""

    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0.0, index=frame.index, dtype=float)
    r2_values = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("trade_date", sort=False):
        buy = pd.to_numeric(group[buy_column], errors="coerce")
        sell = pd.to_numeric(group[sell_column], errors="coerce")
        ret = pd.to_numeric(group[return_column], errors="coerce")
        valid = buy.gt(0) & sell.gt(0) & ret.notna()
        if int(valid.sum()) < min_count:
            continue
        x = ret.loc[valid].to_numpy(dtype=float)
        y = np.log((buy.loc[valid] / sell.loc[valid]).to_numpy(dtype=float))
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        error = y - design @ beta
        indices = group.index[valid]
        residual.loc[indices] = error
        counts.loc[group.index] = float(valid.sum())
        total = float(np.square(y - y.mean()).sum())
        r2_values.loc[group.index] = 1.0 - float(np.square(error).sum()) / total if total > 0 else np.nan
    return residual, counts, r2_values


def _cross_section_multi_residual(
    frame: pd.DataFrame,
    value_column: str,
    predictor_columns: list[str],
    *,
    min_count: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Residualize a daily cross-section on one or more ratio exposures."""

    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0.0, index=frame.index, dtype=float)
    r2_values = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("trade_date", sort=False):
        values = group[[value_column, *predictor_columns]].apply(pd.to_numeric, errors="coerce")
        valid = values.notna().all(axis=1)
        if int(valid.sum()) < max(min_count, len(predictor_columns) + 2):
            continue
        y = values.loc[valid, value_column].to_numpy(dtype=float)
        x = values.loc[valid, predictor_columns].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        error = y - design @ beta
        indices = group.index[valid]
        residual.loc[indices] = error
        counts.loc[group.index] = float(valid.sum())
        total = float(np.square(y - y.mean()).sum())
        r2_values.loc[group.index] = 1.0 - float(np.square(error).sum()) / total if total > 0 else np.nan
    return residual, counts, r2_values


def _cross_section_modified_flow(
    frame: pd.DataFrame,
    buy_column: str,
    sell_column: str,
    return_column: str,
    *,
    min_count: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    modified_buy = pd.Series(np.nan, index=frame.index, dtype=float)
    modified_sell = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0.0, index=frame.index, dtype=float)
    r2_values = pd.Series(np.nan, index=frame.index, dtype=float)
    daily_ratio = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("trade_date", sort=False):
        b = pd.to_numeric(group[buy_column], errors="coerce")
        s = pd.to_numeric(group[sell_column], errors="coerce")
        ret = pd.to_numeric(group[return_column], errors="coerce")
        valid = b.gt(0) & s.gt(0) & ret.notna()
        if int(valid.sum()) < min_count:
            continue
        x = ret.loc[valid].to_numpy(dtype=float)
        y = np.log((b.loc[valid] / s.loc[valid]).to_numpy(dtype=float))
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        error = y - fitted
        total = (b.loc[valid] + s.loc[valid]).to_numpy(dtype=float)
        share = 1.0 / (1.0 + np.exp(-np.clip(error, -60.0, 60.0)))
        b_hat = share * total
        s_hat = (1.0 - share) * total
        indices = group.index[valid]
        modified_buy.loc[indices] = b_hat
        modified_sell.loc[indices] = s_hat
        daily_ratio.loc[indices] = (b_hat - s_hat) / total
        counts.loc[group.index] = float(valid.sum())
        ss_total = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - float(np.square(error).sum()) / ss_total if ss_total > 0 else np.nan
        r2_values.loc[group.index] = r2
    return modified_buy, modified_sell, counts, r2_values, daily_ratio


def build_open_source_tick_factor_panel(
    raw_panel: pd.DataFrame, window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """Apply full-history rolling means per symbol; no same-day look-ahead."""

    if raw_panel.empty:
        return pd.DataFrame(columns=[*AUDIT_COLUMNS, *TICK_FACTOR_COLUMNS])
    frame = raw_panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame = frame.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    for column in ("daily_close", *FLOW_RAW_COLUMNS):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["close_to_close_return"] = frame.groupby("ts_code", sort=False)[
        "daily_close"
    ].pct_change()
    frame["ret20"] = frame.groupby("ts_code", sort=False)["daily_close"].transform(
        lambda values: values / values.shift(window) - 1.0
    )
    for raw_name, factor_name in {
        "daily_qua": TICK_FACTOR_COLUMNS[0],
        "daily_q90_q10_ratio": TICK_FACTOR_COLUMNS[1],
        "daily_mts": TICK_FACTOR_COLUMNS[2],
        "daily_mte": TICK_FACTOR_COLUMNS[3],
        "daily_sr_l020": TICK_FACTOR_COLUMNS[4],
    }.items():
        frame[factor_name] = frame.groupby("ts_code", sort=False)[raw_name].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
    frame["kaiyuan_ideal_reversal_tick_notional_m20"] = np.nan
    frame["kaiyuan_ideal_reversal_tick_volume_m20"] = np.nan
    frame["kaiyuan_trade_notional_return_corr_m20"] = np.nan
    for factor_name in TICK_FACTOR_COLUMNS[12:]:
        frame[factor_name] = np.nan

    # Merged groupby loop: compute all per-symbol rolling factors in one pass
    for _, group in frame.groupby("ts_code", sort=False):
        indices = group.index
        # Block 1: ideal reversal and correlation factors
        frame.loc[indices, "kaiyuan_ideal_reversal_tick_notional_m20"] = (
            _rolling_cut_difference(
                group["daily_return"],
                group["daily_mean_trade_notional"],
                window,
                REVERSAL_FRACTION,
            ).to_numpy()
        )
        frame.loc[indices, "kaiyuan_ideal_reversal_tick_volume_m20"] = (
            _rolling_cut_difference(
                group["daily_return"],
                group["daily_mean_trade_volume"],
                window,
                REVERSAL_FRACTION,
            ).to_numpy()
        )
        frame.loc[indices, "kaiyuan_trade_notional_return_corr_m20"] = _rolling_corr(
            group["daily_return"], group["daily_mean_trade_notional"], window
        ).to_numpy()

        # Block 2: flow ratio factors (originally second loop)
        frame.loc[indices, "kaiyuan_large_flow_s3_m20"] = _rolling_flow_ratio(
            group["daily_flow_large_buy"],
            group["daily_flow_large_sell"],
            window,
            signed_denominator=True,
        ).to_numpy()
        frame.loc[indices, "kaiyuan_small_flow_s3_m20"] = _rolling_flow_ratio(
            group["daily_flow_small_buy"],
            group["daily_flow_small_sell"],
            window,
            signed_denominator=True,
        ).to_numpy()
        frame.loc[indices, "kaiyuan_nir_ge20k_m20"] = _rolling_flow_ratio(
            group["daily_flow_ge20k_buy"],
            group["daily_flow_ge20k_sell"],
            window,
        ).to_numpy()
        frame.loc[indices, "kaiyuan_act_m20"] = _rolling_flow_ratio(
            group["daily_flow_total_buy"],
            group["daily_flow_total_sell"],
            window,
        ).to_numpy()
        frame.loc[indices, "kaiyuan_act_pos_highret_m20_l010"] = (
            _rolling_conditional_act(
                group,
                group["daily_flow_medium_buy"] + group["daily_flow_large_buy"],
                group["daily_flow_medium_sell"] + group["daily_flow_large_sell"],
                group["close_to_close_return"],
                window,
                0.10,
            ).to_numpy()
        )
        frame.loc[indices, "kaiyuan_act_neg_lowret_m20_l010"] = (
            _rolling_conditional_act(
                group,
                group["daily_flow_small_buy"],
                group["daily_flow_small_sell"],
                -group["close_to_close_return"],
                window,
                0.10,
            ).to_numpy()
        )

        # Block 3: modified flow factors (originally third loop, executed after cross-section ops below)
        # These depend on _mod_ge20k_buy/sell and _mod_power_buy/sell computed later
        # Will be filled in a separate minimal loop after cross-section operations
    for column in (
        "daily_order_lms",
        "daily_order_memo",
        "daily_order_island_mean",
        "daily_order_island_std",
        "valid_order_events",
        "order_side_coverage",
        "order_id_available",
    ):
        if column not in frame.columns:
            frame[column] = np.nan
    for raw_name, factor_name in {
        "daily_order_lms": "kaiyuan_order_lms_m20",
        "daily_order_memo": "kaiyuan_order_memo_m20",
        "daily_order_island_mean": "kaiyuan_order_island_mean_m20",
        "daily_order_island_std": "kaiyuan_order_island_std_m20",
    }.items():
        frame[factor_name] = frame.groupby("ts_code", sort=False)[raw_name].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )

    conditional_specs = (
        ("large_high_amp", "daily_flow_large_high_amp_buy", "daily_flow_large_high_amp_sell", "kaiyuan_flow_large_high_amp_resid_m20"),
        ("small_high_amp", "daily_flow_small_high_amp_buy", "daily_flow_small_high_amp_sell", "kaiyuan_flow_small_high_amp_resid_m20"),
        ("large_open30", "daily_flow_large_open30_buy", "daily_flow_large_open30_sell", "kaiyuan_flow_large_open30_resid_m20"),
        ("small_open30", "daily_flow_small_open30_buy", "daily_flow_small_open30_sell", "kaiyuan_flow_small_open30_resid_m20"),
    )
    conditional_counts: list[pd.Series] = []
    for name, buy_column, sell_column, factor_name in conditional_specs:
        residual, count, r2 = _cross_section_log_flow_residual(
            frame, buy_column, sell_column, "close_to_close_return"
        )
        frame[f"_{name}_residual"] = residual
        frame[f"_{name}_r2"] = r2
        conditional_counts.append(count)
        frame[factor_name] = frame.groupby("ts_code", sort=False)[f"_{name}_residual"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
    frame["conditional_flow_high_amp_count"] = conditional_counts[0]
    frame["conditional_flow_open30_count"] = conditional_counts[2]

    category_names = ("extra", "large", "medium", "small")
    category_ratios: dict[str, str] = {}
    rolling_total_amount = frame.groupby("ts_code", sort=False)["daily_amount"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(window, min_periods=window).sum()
    )
    for category in category_names:
        category_amount = (
            pd.to_numeric(frame[f"daily_flow_{category}_buy"], errors="coerce")
            + pd.to_numeric(frame[f"daily_flow_{category}_sell"], errors="coerce")
        )
        rolling_category = category_amount.groupby(frame["ts_code"], sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).sum()
        )
        ratio_column = f"_flow_{category}_ratio_m20"
        frame[ratio_column] = rolling_category / rolling_total_amount.where(rolling_total_amount.gt(0))
        category_ratios[category] = ratio_column

    evl_resid, evl_count, evl_r2 = _cross_section_multi_residual(
        frame, category_ratios["extra"], [category_ratios["large"], category_ratios["medium"], category_ratios["small"]]
    )
    evm_resid, evm_count, evm_r2 = _cross_section_multi_residual(
        frame, category_ratios["medium"], [category_ratios["extra"], category_ratios["large"], category_ratios["small"]]
    )
    evs_resid, evs_count, evs_r2 = _cross_section_multi_residual(
        frame, category_ratios["small"], [category_ratios["extra"], category_ratios["large"], category_ratios["medium"]]
    )
    frame["kaiyuan_evl_m20"] = evl_resid
    frame["kaiyuan_evm_m20"] = evm_resid
    frame["kaiyuan_evs_m20"] = evs_resid
    frame["flow_evl_r2"] = evl_r2
    frame["flow_evm_r2"] = evm_r2
    frame["flow_evs_r2"] = evs_r2
    frame["flow_cross_section_count"] = np.maximum(
        frame.get("flow_cross_section_count", pd.Series(0.0, index=frame.index)),
        np.maximum.reduce([evl_count.to_numpy(), evm_count.to_numpy(), evs_count.to_numpy()]),
    )
    for category, factor_name in {
        "extra": "kaiyuan_extra_amount_adj_m20",
        "large": "kaiyuan_large_amount_adj_m20",
        "medium": "kaiyuan_medium_amount_adj_m20",
        "small": "kaiyuan_small_amount_adj_m20",
    }.items():
        frame[factor_name] = frame[category_ratios[category]] * rolling_total_amount

    large_resid, large_count, large_r2 = _cross_section_residual(
        frame, "kaiyuan_large_flow_s3_m20", "ret20"
    )
    small_resid, small_count, _ = _cross_section_residual(
        frame, "kaiyuan_small_flow_s3_m20", "ret20"
    )
    frame["kaiyuan_large_flow_s3_resid_ret20_cs_m20"] = large_resid
    frame["kaiyuan_small_flow_s3_resid_ret20_cs_m20"] = small_resid

    ge20_buy, ge20_sell, mod_count, mod_r2, _ = _cross_section_modified_flow(
        frame,
        "daily_flow_ge20k_buy",
        "daily_flow_ge20k_sell",
        "close_to_close_return",
    )
    power_buy, power_sell, cnir_count, cnir_r2, _ = _cross_section_modified_flow(
        frame,
        "daily_flow_power_buy",
        "daily_flow_power_sell",
        "close_to_close_return",
    )
    frame["_mod_ge20k_buy"] = ge20_buy
    frame["_mod_ge20k_sell"] = ge20_sell
    frame["_mod_power_buy"] = power_buy
    frame["_mod_power_sell"] = power_sell

    # Separate minimal loop for modified flow factors: these depend on cross-section
    # residuals computed above (_mod_ge20k_*, _mod_power_*), so cannot be merged
    # into the main per-symbol loop earlier
    for _, group in frame.groupby("ts_code", sort=False):
        indices = group.index
        frame.loc[indices, "kaiyuan_nir_mod_ge20k_cs_m20"] = _rolling_flow_ratio(
            group["_mod_ge20k_buy"],
            group["_mod_ge20k_sell"],
            window,
        ).to_numpy()
        frame.loc[indices, "kaiyuan_cnir_cs_m20"] = _rolling_flow_ratio(
            group["_mod_power_buy"],
            group["_mod_power_sell"],
            window,
        ).to_numpy()

    frame["flow_cross_section_count"] = np.maximum.reduce(
        [
            mod_count.to_numpy(dtype=float),
            cnir_count.to_numpy(dtype=float),
            large_count.to_numpy(dtype=float),
            small_count.to_numpy(dtype=float),
            evl_count.to_numpy(dtype=float),
            evm_count.to_numpy(dtype=float),
            evs_count.to_numpy(dtype=float),
        ]
    )
    max_cross_section = float(frame["flow_cross_section_count"].max())
    if max_cross_section < 10:
        LOGGER.warning(
            "Cross-section sample count for flow residual factors is very low "
            "(max = %.0f). These factors may be unreliable. Consider running with "
            "more symbols or a longer date range.",
            max_cross_section,
        )

    frame["flow_s3_resid_r2"] = large_r2
    frame["flow_nir_mod_r2"] = mod_r2
    frame["flow_cnir_r2"] = cnir_r2
    has_trade = frame["valid_trade_minutes"].notna()
    has_order = frame["order_id_available"].fillna(False).astype(bool)
    frame["source_level"] = np.select(
        [has_trade & has_order, has_trade, has_order],
        [SOURCE_LEVEL_ORDER, SOURCE_LEVEL, SOURCE_LEVEL_ORDER_ONLY],
        default=SOURCE_LEVEL,
    )
    base_columns = [
        "trade_date",
        "ts_code",
        "valid_trade_minutes",
        "trade_count",
        "daily_amount",
        "amount_available",
        "source_level",
        "valid_order_events",
        "order_side_coverage",
        "order_id_available",
        *FLOW_RAW_COLUMNS,
        "flow_cross_section_count",
        "flow_s3_resid_r2",
        "flow_nir_mod_r2",
        "flow_cnir_r2",
        "flow_evl_r2",
        "flow_evm_r2",
        "flow_evs_r2",
        *TICK_FACTOR_COLUMNS,
    ]
    return frame[base_columns].sort_values(
        ["ts_code", "trade_date"], kind="mergesort"
    ).reset_index(drop=True)


def add_availability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    result = result.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    result["available_date"] = result.groupby("ts_code", sort=False)["trade_date"].shift(-1)
    result["available_time"] = result["available_date"] + pd.Timedelta(hours=9, minutes=30)
    columns = ["trade_date", "available_date", "available_time", "ts_code"]
    columns += [c for c in result.columns if c not in columns]
    return result[columns]


def select_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = [*AUDIT_COLUMNS, *TICK_FACTOR_COLUMNS]
    result = frame.copy()
    for column in ordered:
        if column not in result:
            result[column] = np.nan
    result = result[ordered].replace([np.inf, -np.inf], np.nan)
    return result.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
