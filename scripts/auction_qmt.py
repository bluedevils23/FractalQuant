"""QMT in-memory auction adapters. Inputs are shares/CNY, timestamps Shanghai local."""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

def _filter_qmt_heartbeat_frames(
    frame: pd.DataFrame, *, keep_stage_anchors: bool = False
) -> pd.DataFrame:
    """Drop unchanged factor-relevant states, optionally retaining stage endpoints."""
    ordered = (
        frame.sort_values("trade_time", kind="mergesort")
        .drop_duplicates("trade_time", keep="last")
        .reset_index(drop=True)
    )
    if ordered.empty or "amount" not in ordered.columns:
        return ordered

    day_key = (
        ordered["trade_date"]
        if "trade_date" in ordered.columns
        else ordered["trade_time"].dt.normalize()
    )
    price = pd.to_numeric(ordered["last_price"], errors="coerce")
    amount = pd.to_numeric(ordered["amount"], errors="coerce")
    previous_price = price.groupby(day_key, sort=False).shift(1)
    previous_amount = amount.groupby(day_key, sort=False).shift(1)
    trade_same = price.eq(previous_price) & amount.eq(previous_amount)
    state_columns = [
        "last_price",
        "amount",
        "previous_close",
        *[f"ask_price{level}" for level in range(1, 4)],
        *[f"ask_vol{level}" for level in range(1, 4)],
        *[f"bid_price{level}" for level in range(1, 4)],
        *[f"bid_vol{level}" for level in range(1, 4)],
    ]
    previous = ordered[state_columns].groupby(day_key, sort=False).shift(1)
    same_state = (
        ordered[state_columns].eq(previous)
        | (ordered[state_columns].isna() & previous.isna())
    ).all(axis=1)
    # QMT often publishes 0/0 before the opening match. Keep L3 book changes.
    heartbeat = trade_same & same_state
    keep = ~heartbeat
    if keep_stage_anchors:
        stage2 = ordered["trade_time"].dt.time.ge(pd.Timestamp("09:20").time())
        stages = ordered.groupby([day_key, stage2], sort=False)
        keep = (
            keep | stages.cumcount().eq(0) | stages.cumcount(ascending=False).eq(0)
        )
        valid = ordered["ask_price1"].gt(0) & ordered["bid_price1"].gt(0)
        valid_stages = ordered.loc[valid].groupby(
            [day_key.loc[valid], stage2.loc[valid]], sort=False
        )
        valid_endpoints = (
            valid_stages.cumcount().eq(0)
            | valid_stages.cumcount(ascending=False).eq(0)
        )
        keep = keep | valid_endpoints.reindex(ordered.index, fill_value=False)
        tail = ordered["trade_time"].dt.time.ge(pd.Timestamp("09:24").time())
        tail_first = (
            ordered.loc[tail].groupby(day_key.loc[tail], sort=False).cumcount().eq(0)
        )
        keep = keep | tail_first.reindex(ordered.index, fill_value=False)
        # These observed endpoints restore state duration, not quote updates.
        ordered["_qmt_state_anchor"] = heartbeat
    return ordered.loc[keep].reset_index(drop=True)


def parse_qmt_timestamp(value: Any) -> pd.Timestamp | pd.Series:
    """Parse XtData epoch-ms, epoch-s, compact, or datetime timestamps."""
    scalar = not isinstance(value, pd.Series)
    if isinstance(value, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(value):
            result = pd.to_datetime(value, errors="coerce")
            if getattr(result.dt, "tz", None) is not None:
                result = result.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
            return result
        numeric = pd.to_numeric(value, errors="coerce")
        result = pd.to_datetime(value, errors="coerce")
        compact_mask = numeric.abs().between(10**13, 10**15)
        if compact_mask.any():
            result.loc[compact_mask] = pd.to_datetime(
                numeric.loc[compact_mask].astype("int64").astype(str),
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )
        ms_mask = numeric.abs().gt(10**11) & ~compact_mask
        if ms_mask.any():
            result.loc[ms_mask] = (
                pd.to_datetime(numeric.loc[ms_mask], unit="ms", errors="coerce", utc=True)
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
            )
        sec_mask = numeric.abs().gt(10**8) & ~compact_mask & ~ms_mask
        if sec_mask.any():
            result.loc[sec_mask] = (
                pd.to_datetime(numeric.loc[sec_mask], unit="s", errors="coerce", utc=True)
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
            )
        return result.iloc[0] if scalar else result
    if value is None:
        return pd.NaT
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        result = pd.Timestamp(value)
        if result.tzinfo is not None:
            result = result.tz_convert("Asia/Shanghai").tz_localize(None)
        return result
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pd.to_datetime(value, errors="coerce")
    if 10**13 <= abs(number) <= 10**15:
        return pd.to_datetime(str(int(number)), format="%Y%m%d%H%M%S", errors="coerce")
    if abs(number) > 10**11:
        return pd.to_datetime(number, unit="ms", errors="coerce", utc=True).tz_convert("Asia/Shanghai").tz_localize(None)
    if abs(number) > 10**8:
        return pd.to_datetime(number, unit="s", errors="coerce", utc=True).tz_convert("Asia/Shanghai").tz_localize(None)
    return pd.to_datetime(value, errors="coerce")


def normalize_qmt_tick_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an in-memory QMT tick frame for auction factor calculation.

    The input quantities must already be shares and ``amount`` must be CNY.
    This is the same boundary contract used by :func:`load_qmt_quote_frame`,
    but it also accepts the field aliases returned by XtData.  The function
    keeps state anchors for time and state calculations; the factor calculator
    excludes them from snapshot and step-count statistics.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["trade_time"])
    work = frame.copy()
    for name in frame.index.names:
        if name is not None and name not in work.columns:
            work[name] = frame.index.get_level_values(name)
    work = work.reset_index(drop=True)
    if "trade_time" not in work:
        for candidate in ("time", "timestamp"):
            if candidate in work:
                work["trade_time"] = work[candidate]
                break
    if "trade_time" not in work:
        raise ValueError("QMT tick frame must expose trade_time")
    work["trade_time"] = parse_qmt_timestamp(work["trade_time"])
    work = work.dropna(subset=["trade_time"])
    clock = work["trade_time"].dt.time
    work = work.loc[
        (clock >= pd.Timestamp("09:15").time())
        & (clock < pd.Timestamp("09:25").time())
    ].copy()

    def source_column(*names: str) -> str | None:
        return next((name for name in names if name in work.columns), None)

    aliases = {
        "last_price": source_column("last_price", "lastPrice"),
        "previous_close": source_column("previous_close", "lastClose"),
        "amount": source_column("amount", "trade_amount"),
    }
    for side in ("ask", "bid"):
        for level in range(1, 4):
            aliases[f"{side}_price{level}"] = source_column(
                f"{side}_price{level}", f"{side}Price{level}"
            )
            aliases[f"{side}_vol{level}"] = source_column(
                f"{side}_vol{level}", f"{side}Vol{level}",
                f"{side}_qty{level}", f"{side}Qty{level}"
            )
    required_heartbeat = [
        aliases["last_price"], aliases["previous_close"], aliases["amount"],
        *[aliases[f"{side}_{field}{level}"]
          for side in ("ask", "bid")
          for field in ("price", "vol")
          for level in range(1, 4)],
    ]
    if all(column is not None for column in required_heartbeat):
        heartbeat_frame = work.rename(
            columns={source: target for target, source in aliases.items() if source is not None}
        )
        for column in aliases:
            heartbeat_frame[column] = pd.to_numeric(heartbeat_frame[column], errors="coerce")
        work = _filter_qmt_heartbeat_frames(heartbeat_frame, keep_stage_anchors=True)

    result = pd.DataFrame({"trade_time": work["trade_time"]})
    if "_qmt_state_anchor" in work:
        result["_qmt_state_anchor"] = work["_qmt_state_anchor"].astype(bool)
    for target, source in aliases.items():
        if source is None:
            result[target] = np.nan
        else:
            column = target if target in work.columns else source
            result[target] = pd.to_numeric(work[column], errors="coerce")
    result["trade_price"] = result["last_price"]
    result["trade_volume"] = 0.0
    result["trade_amount"] = 0.0
    result["open_price"] = np.nan
    for level in range(1, 4):
        result[f"ask_qty{level}"] = result[f"ask_vol{level}"]
        result[f"bid_qty{level}"] = result[f"bid_vol{level}"]
    columns = ["trade_time"]
    if "_qmt_state_anchor" in result:
        columns.append("_qmt_state_anchor")
    columns += ["trade_price", "trade_volume", "trade_amount", "open_price", "previous_close"]
    for level in range(1, 4):
        columns += [f"ask_price{level}", f"ask_qty{level}", f"bid_price{level}", f"bid_qty{level}"]
    return result[columns].sort_values("trade_time", kind="mergesort").drop_duplicates(
        "trade_time", keep="last"
    ).reset_index(drop=True)



def _match_candidates(frame: pd.DataFrame | None, *, minute: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    for name in frame.index.names:
        if name is not None and name not in work.columns:
            work[name] = frame.index.get_level_values(name)
    work = work.reset_index(drop=True)
    aliases = {
        "trade_time": ("trade_time", "time", "timestamp"),
        "price": ("open",) if minute else ("last_price", "lastPrice", "trade_price"),
        "quantity": ("vol", "volume") if minute else ("volume", "trade_volume"),
        "turnover": ("amount", "trade_amount"),
    }
    for target, names in aliases.items():
        source = next((name for name in names if name in work), None)
        if source is None:
            return pd.DataFrame()
        work[target] = (parse_qmt_timestamp(work[source]) if target == "trade_time"
                        else pd.to_numeric(work[source], errors="coerce"))
    clock = work["trade_time"].dt.strftime("%H:%M:%S")
    time_mask = clock.eq("09:30:00") if minute else clock.between("09:25:00", "09:25:59")
    return work.loc[
        time_mask & work["price"].gt(0) & work["quantity"].gt(0) & work["turnover"].gt(0)
    ].sort_values("trade_time", kind="mergesort")


def _match_record(row: pd.Series, *, minute: bool) -> dict[str, object]:
    volume, amount = float(row["quantity"]), float(row["turnover"])
    result = {
        "trade_time": row["trade_time"].normalize() + pd.Timedelta(hours=9, minutes=25),
        "open_price": float(row["price"]),
        "trade_volume": volume,
        "trade_amount": amount,
        "raw_volume": volume,
        "raw_amount": amount,
        "source": "qmt_0930_minute" if minute else "qmt_tick_0925",
    }
    if not minute:
        result["provider_raw_volume_lots"] = row.get("provider_raw_volume_lots", volume / 100.0)
    return result


def select_last_0925_match(ticks: pd.DataFrame) -> dict[str, object] | None:
    """Last valid 09:25 tick; quantities must already be shares."""
    valid = _match_candidates(ticks, minute=False)
    return None if valid.empty else _match_record(valid.iloc[-1], minute=False)


def select_qmt_auction_match(
    tick_frame: pd.DataFrame,
    minute_frame: pd.DataFrame | None,
    target_date: pd.Timestamp,
    *,
    allow_0930_fallback: bool = True,
) -> dict[str, object] | None:
    """Live policy: last 09:25 tick, optionally first 09:30 bar of the same day.

    The minute bar's synthetic 09:25 timestamp does not imply 09:25 availability.
    """
    day = pd.Timestamp(target_date).normalize()
    for frame, minute in ((tick_frame, False), (minute_frame, True)):
        if minute and not allow_0930_fallback:
            break
        valid = _match_candidates(frame, minute=minute)
        if valid.empty:
            continue
        valid = valid.loc[valid["trade_time"].dt.normalize().eq(day)]
        if not valid.empty:
            return _match_record(valid.iloc[0 if minute else -1], minute=minute)
    return None
