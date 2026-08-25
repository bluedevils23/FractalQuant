"""Open-source APM/OVP factors using minute bars and a mapped benchmark.

The implementation keeps benchmark alignment explicit.  APM is the paper's
20-day within-symbol regression of asset and benchmark session returns; the
resulting statistic is cross-sectionally residualized against Ret20.  OVP and
AVP are the transparent W-cut cumulative-return variants from the same paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .open_source import normalize_minute_frame


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


def _session_day(frame: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    day = frame.loc[frame.index.normalize() == date]
    return day.sort_index()


def _close_at(day: pd.DataFrame, hhmm: str) -> float:
    rows = day.loc[day.index.strftime("%H:%M") == hhmm, "close"]
    return float(rows.iloc[-1]) if not rows.empty and pd.notna(rows.iloc[-1]) else np.nan


def _open_at(day: pd.DataFrame, hhmm: str) -> float:
    rows = day.loc[day.index.strftime("%H:%M") == hhmm, "open"]
    return float(rows.iloc[0]) if not rows.empty and pd.notna(rows.iloc[0]) else np.nan


def build_daily_apm_raw_features(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    ts_code: str,
    reference_index_code: str,
) -> pd.DataFrame:
    """Build aligned daily block returns for one asset and its index."""

    asset_frame = normalize_minute_frame(asset, ts_code)
    benchmark_frame = normalize_minute_frame(benchmark, reference_index_code)
    dates = sorted(set(asset_frame.index.normalize()) & set(benchmark_frame.index.normalize()))
    rows: list[dict[str, object]] = []
    previous_asset_close = np.nan
    previous_benchmark_close = np.nan
    for date in dates:
        asset_day = _session_day(asset_frame, date)
        benchmark_day = _session_day(benchmark_frame, date)
        asset_regular = asset_day.loc[
            ((asset_day.index.strftime("%H:%M") >= "09:31") & (asset_day.index.strftime("%H:%M") <= "11:30"))
            | ((asset_day.index.strftime("%H:%M") >= "13:01") & (asset_day.index.strftime("%H:%M") <= "15:00"))
        ]
        benchmark_regular = benchmark_day.loc[
            ((benchmark_day.index.strftime("%H:%M") >= "09:31") & (benchmark_day.index.strftime("%H:%M") <= "11:30"))
            | ((benchmark_day.index.strftime("%H:%M") >= "13:01") & (benchmark_day.index.strftime("%H:%M") <= "15:00"))
        ]
        asset_open = _open_at(asset_day, "09:30")
        benchmark_open = _open_at(benchmark_day, "09:30")
        asset_close = _close_at(asset_day, "15:00")
        benchmark_close = _close_at(benchmark_day, "15:00")
        if not np.isfinite(previous_asset_close):
            previous_asset_close = _close_at(asset_frame.loc[asset_frame.index.normalize() < date], "15:00")
        if not np.isfinite(previous_benchmark_close):
            previous_benchmark_close = _close_at(benchmark_frame.loc[benchmark_frame.index.normalize() < date], "15:00")
        am_end = _close_at(asset_day, "11:30")
        pm_start = _open_at(asset_day, "13:00")
        pm_end = asset_close
        bench_am_end = _close_at(benchmark_day, "11:30")
        bench_pm_start = _open_at(benchmark_day, "13:00")
        bench_pm_end = benchmark_close
        row: dict[str, object] = {
            "trade_date": date,
            "ts_code": ts_code,
            "reference_index_code": reference_index_code,
            "overnight_return": asset_open / previous_asset_close - 1.0 if np.isfinite(asset_open) and np.isfinite(previous_asset_close) and previous_asset_close > 0 else np.nan,
            "am_return": am_end / asset_open - 1.0 if np.isfinite(am_end) and np.isfinite(asset_open) and asset_open > 0 else np.nan,
            "pm_return": pm_end / pm_start - 1.0 if np.isfinite(pm_end) and np.isfinite(pm_start) and pm_start > 0 else np.nan,
            "_benchmark_overnight_return": benchmark_open / previous_benchmark_close - 1.0 if np.isfinite(benchmark_open) and np.isfinite(previous_benchmark_close) and previous_benchmark_close > 0 else np.nan,
            "_benchmark_am_return": bench_am_end / benchmark_open - 1.0 if np.isfinite(bench_am_end) and np.isfinite(benchmark_open) and benchmark_open > 0 else np.nan,
            "_benchmark_pm_return": bench_pm_end / bench_pm_start - 1.0 if np.isfinite(bench_pm_end) and np.isfinite(bench_pm_start) and bench_pm_start > 0 else np.nan,
            "valid_session_bars": int(len(asset_regular)),
            "reference_valid_session_bars": int(len(benchmark_regular)),
            "mapping_available": True,
            "_asset_close": asset_close,
        }
        rows.append(row)
        if np.isfinite(asset_close):
            previous_asset_close = asset_close
        if np.isfinite(benchmark_close):
            previous_benchmark_close = benchmark_close
    if not rows:
        return pd.DataFrame(columns=APM_RAW_COLUMNS)
    result = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
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
        valid = group[[value, ret20]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        if int(valid.sum()) < 20:
            continue
        x = group.loc[valid, ret20].to_numpy(dtype=float)
        y = group.loc[valid, value].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        error = y - design @ beta
        residual.loc[group.index[valid]] = error
        counts.loc[group.index] = float(valid.sum())
        total = float(np.square(y - y.mean()).sum())
        r2.loc[group.index] = 1.0 - float(np.square(error).sum()) / total if total > 0 else np.nan
    return residual, counts, r2


def build_open_source_apm_panel(raw_panel: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    if raw_panel.empty:
        return pd.DataFrame(columns=[*AUDIT_COLUMNS])
    frame = raw_panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame = frame.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
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
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    result = result.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    result["available_date"] = result.groupby("ts_code", sort=False)["trade_date"].shift(-1)
    result["available_time"] = result["available_date"] + pd.Timedelta(hours=9, minutes=30)
    return result[list(AUDIT_COLUMNS)]
