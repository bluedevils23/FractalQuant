"""Open-source APM/OVP factors using minute bars and a mapped benchmark.

The implementation keeps benchmark alignment explicit.  APM is the paper's
20-day within-symbol regression of asset and benchmark session returns; the
resulting statistic is cross-sectionally residualized against Ret20.  OVP and
AVP sum daily segment returns (paper p.8); cross-sectional OLS has no
intercept (p.3). The mapped ETF benchmark is an adaptation of the stock model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from functools import lru_cache

from exchange_calendars.exchange_calendar_xshg import XSHGExchangeCalendar

from .open_source import normalize_minute_frame, add_availability_columns as _add_availability


APM_FACTOR_COLUMNS = (
    "kaiyuan_apmnew_m20",
    "kaiyuan_apmraw_m20",
    "kaiyuan_ovp_m20",
    "kaiyuan_avp_m20",
)

APM_RAW_COLUMNS = (
    "trade_date",
    "ts_code",
    "reference_index_code",
    "overnight_return",
    "am_return",
    "pm_return",
    "ret20",
    "apmnew_stat_raw",
    "apmraw_stat_raw",
    "valid_session_bars",
    "reference_valid_session_bars",
    "mapping_available",
)

AUDIT_COLUMNS = (
    "trade_date",
    "available_date",
    "available_time",
    "ts_code",
    "reference_index_code",
    "mapping_available",
    "valid_session_bars",
    "reference_valid_session_bars",
    "source_level",
    "apm_cross_section_count",
    "apm_cross_section_r2",
    *APM_FACTOR_COLUMNS,
)

WINDOW = 20
SOURCE_LEVEL = "minute_ohlcv_proxy+mapped_index"


@lru_cache(maxsize=128)
def trading_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Use exchange sessions, never a compressed sequence of observed dates."""
    calendar = XSHGExchangeCalendar(start=start - pd.Timedelta(days=7), end=end + pd.Timedelta(days=7))
    sessions = calendar.sessions.tz_localize(None)
    return sessions[(sessions >= start) & (sessions <= end)]


def _daily_blocks(frame: pd.DataFrame, adjusted: bool) -> pd.DataFrame:
    rows = []
    for date, day in frame.groupby(frame.index.normalize(), sort=True):
        clock = day.index.strftime("%H:%M")
        prices = day[["open", "high", "low", "close"]]
        good = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        good &= day.index == day.index.floor("min")
        am = (clock >= "09:31") & (clock <= "11:30")
        pm = (clock >= "13:01") & (clock <= "15:00")
        am_ok = int((good & am).sum()) == 120
        pm_ok = int((good & pm).sum()) == 120

        def price(hhmm: str, column: str) -> float:
            values = day.loc[clock == hhmm, column]
            value = float(values.iloc[0]) if len(values) else np.nan
            return value if np.isfinite(value) and value > 0 else np.nan

        # Local bars are close-labelled. 09:30 is the auction observation;
        # 13:01 open is the start of the first afternoon minute, not 13:00.
        opening = price("09:30" if (clock == "09:30").any() else "09:31", "open")
        close = price("15:00", "close")
        am_close = price("11:30", "close")
        pm_open = price("13:01", "open")
        factor = 1.0
        if adjusted:
            factors = day.get("adj_factor", pd.Series(np.nan, index=day.index))
            valid = np.isfinite(factors) & factors.gt(0)
            factor = float(factors.iloc[0]) if valid.all() and factors.nunique() == 1 else np.nan
        rows.append({
            "trade_date": date,
            "open": opening * factor,
            "close": close * factor,
            "am": am_close / opening - 1.0 if am_ok else np.nan,
            "pm": close / pm_open - 1.0 if pm_ok else np.nan,
            "bars": int((good & (am | pm)).sum()),
        })
    return pd.DataFrame(rows).set_index("trade_date")


def build_daily_apm_raw_features(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    ts_code: str,
    reference_index_code: str,
) -> pd.DataFrame:
    """Build separate daily series before alignment; retain missing sessions."""
    asset_frame = normalize_minute_frame(asset, ts_code)
    benchmark_frame = normalize_minute_frame(benchmark, reference_index_code)
    if asset_frame.empty or benchmark_frame.empty:
        return pd.DataFrame(columns=APM_RAW_COLUMNS)
    dates = trading_dates(asset_frame.index.min().normalize(), asset_frame.index.max().normalize())
    asset_daily = _daily_blocks(asset_frame, adjusted=True).reindex(dates)
    benchmark_daily = _daily_blocks(benchmark_frame, adjusted=False).reindex(dates)
    result = pd.DataFrame({"trade_date": dates, "ts_code": ts_code,
                           "reference_index_code": reference_index_code})
    for daily, prefix in ((asset_daily, ""), (benchmark_daily, "_benchmark_")):
        result[prefix + "overnight_return"] = (daily["open"] / daily["close"].shift(1) - 1.0).to_numpy()
        result[prefix + "am_return"] = daily["am"].to_numpy()
        result[prefix + "pm_return"] = daily["pm"].to_numpy()
    result["_asset_close"] = asset_daily["close"].to_numpy()
    result["valid_session_bars"] = asset_daily["bars"].fillna(0).to_numpy(dtype=int)
    result["reference_valid_session_bars"] = benchmark_daily["bars"].fillna(0).to_numpy(dtype=int)
    result["mapping_available"] = True
    for column in APM_RAW_COLUMNS:
        if column not in result:
            result[column] = np.nan
    return result


def _rolling_apm_stat(
    group: pd.DataFrame,
    first_asset_column: str,
    first_benchmark_column: str,
    second_asset_column: str,
    second_benchmark_column: str,
    window: int,
) -> pd.Series:
    output = np.full(len(group), np.nan, dtype=float)
    first_asset = pd.to_numeric(group[first_asset_column], errors="coerce").to_numpy(dtype=float)
    first_benchmark = pd.to_numeric(group[first_benchmark_column], errors="coerce").to_numpy(dtype=float)
    second_asset = pd.to_numeric(group[second_asset_column], errors="coerce").to_numpy(dtype=float)
    second_benchmark = pd.to_numeric(group[second_benchmark_column], errors="coerce").to_numpy(dtype=float)
    for end in range(window - 1, len(group)):
        start = end - window + 1
        y_first = first_asset[start : end + 1]
        x_first = first_benchmark[start : end + 1]
        y_second = second_asset[start : end + 1]
        x_second = second_benchmark[start : end + 1]
        if not (np.isfinite(y_first).all() and np.isfinite(x_first).all() and np.isfinite(y_second).all() and np.isfinite(x_second).all()):
            continue
        x_all = np.r_[x_first, x_second]
        y_all = np.r_[y_first, y_second]
        design = np.column_stack([np.ones(2 * window), x_all])
        beta, *_ = np.linalg.lstsq(design, y_all, rcond=None)
        residual = y_all - design @ beta
        delta = residual[:window] - residual[window:]
        scale = float(np.std(delta, ddof=1))
        if scale > 0:
            output[end] = float(delta.mean() / scale * np.sqrt(window))
    return pd.Series(output, index=group.index)


def _cross_section_residual(frame: pd.DataFrame, value: str, ret20: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    residual = pd.Series(np.nan, index=frame.index, dtype=float)
    counts = pd.Series(0.0, index=frame.index, dtype=float)
    r2 = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("trade_date", sort=False):
        valid = np.isfinite(group[[value, ret20]].apply(pd.to_numeric, errors="coerce")).all(axis=1)
        counts.loc[group.index] = float(valid.sum())
        if int(valid.sum()) < 20:
            continue
        x = group.loc[valid, ret20].to_numpy(dtype=float)
        y = group.loc[valid, value].to_numpy(dtype=float)
        design = x[:, None]  # Paper p.3: stat_j = b * Ret20_j + epsilon_j.
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        error = y - design @ beta
        residual.loc[group.index[valid]] = error
        counts.loc[group.index] = float(valid.sum())
        total = float(np.square(y).sum())  # Uncentered R2 for no-intercept OLS.
        r2.loc[group.index] = 1.0 - float(np.square(error).sum()) / total if total > 0 else np.nan
    return residual, counts, r2


def build_open_source_apm_panel(raw_panel: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    if raw_panel.empty:
        return pd.DataFrame(columns=[*AUDIT_COLUMNS])
    frame = raw_panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    if frame["trade_date"].isna().any() or frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("APM requires unique, valid (ts_code, trade_date) rows")
    groups = []
    for symbol, group in frame.groupby("ts_code", sort=True):
        dates = trading_dates(group["trade_date"].min(), group["trade_date"].max())
        group = group.set_index("trade_date").reindex(dates).rename_axis("trade_date").reset_index()
        group["ts_code"] = symbol
        groups.append(group)
    frame = pd.concat(groups, ignore_index=True)
    if "_asset_close" not in frame.columns:
        frame["_asset_close"] = pd.to_numeric(frame.get("daily_close"), errors="coerce")
    frame["ret20"] = frame.groupby("ts_code", sort=False)["_asset_close"].transform(lambda values: values / values.shift(window) - 1.0)
    frame["apmnew_stat_raw"] = np.nan
    frame["apmraw_stat_raw"] = np.nan
    for _, group in frame.groupby("ts_code", sort=False):
        frame.loc[group.index, "apmnew_stat_raw"] = _rolling_apm_stat(
            group,
            "overnight_return",
            "_benchmark_overnight_return",
            "pm_return",
            "_benchmark_pm_return",
            window,
        ).to_numpy()
        frame.loc[group.index, "apmraw_stat_raw"] = _rolling_apm_stat(
            group,
            "am_return",
            "_benchmark_am_return",
            "pm_return",
            "_benchmark_pm_return",
            window,
        ).to_numpy()
    frame["_ovp_daily"] = frame["overnight_return"] - frame["pm_return"]
    frame["_avp_daily"] = frame["am_return"] - frame["pm_return"]
    for raw, factor in (("_ovp_daily", "kaiyuan_ovp_m20"), ("_avp_daily", "kaiyuan_avp_m20")):
        frame[factor] = frame.groupby("ts_code", sort=False)[raw].transform(lambda values: values.rolling(window, min_periods=window).sum())
    apmnew_resid, counts, r2 = _cross_section_residual(frame, "apmnew_stat_raw", "ret20")
    apmraw_resid, _, _ = _cross_section_residual(frame, "apmraw_stat_raw", "ret20")
    frame["kaiyuan_apmnew_m20"] = apmnew_resid
    frame["kaiyuan_apmraw_m20"] = apmraw_resid
    frame["apm_cross_section_count"] = counts
    frame["apm_cross_section_r2"] = r2
    frame["source_level"] = SOURCE_LEVEL
    result = frame.rename(columns={"reference_index_code": "reference_index_code"})
    for column in AUDIT_COLUMNS:
        if column not in result:
            result[column] = np.nan
    return result[list(AUDIT_COLUMNS)].sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def add_availability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = _add_availability(frame).reset_index(drop=True)
    return result[list(AUDIT_COLUMNS)]
