from __future__ import annotations

"""Generate ETF minute CrossMarket factors using mapped reference indices."""

import argparse
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

for import_root in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from factor.advanced_runtime import (  # noqa: E402
    normalize_trade_date_arg,
    read_symbol_list_file,
)
from factor.crossmarket import (  # noqa: E402
    ArbitrageOpportunityFactor,
    CointegrationFactor,
    CrossMarketCoherenceFactor,
    CrossMarketCopulaFactor,
    CrossMarketCorrelationFactor,
    CrossMarketDynamicCorrelationFactor,
    CrossMarketEntropyFactor,
    CrossMarketGrangerFactor,
    CrossMarketInformationFlowFactor,
    CrossMarketJointDistributionFactor,
    CrossMarketMultiscaleCorrelationFactor,
    CrossMarketPhaseSynchronizationFactor,
    CrossMarketVolatilityFactor,
    MarketLinkageFactor,
    MarketRegimeSwitchFactor,
    RelativeStrengthFactor,
)
from scripts.generate_etf_minute_factors import (  # noqa: E402
    normalize_minute_frame,
)


LOGGER = logging.getLogger("generate_etf_crossmarket_factors")

DEFAULT_MAPPING_PATH = Path(
    r"D:\workspace\stockdata\etf-data\exchange_fund_index_mapping.csv"
)
DEFAULT_ETF_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_ETF_DAILY_PATH = Path(
    r"D:\workspace\stockdata\etf-data\etf_daily.parquet"
)
DEFAULT_INDEX_ROOT = Path(
    r"D:\workspace\stockdata\指数数据\index_1min_rqdata"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_crossmarket_factors"
)
REQUIRED_MAPPING_COLUMNS = ("fund_code", "reference_index_code")
REFERENCE_DATA_ALIASES = {
    "931573CNY00.CSI": "931573.CSI",
}


@dataclass(frozen=True)
class MappingRecord:
    fund_code: str
    reference_index_code: str
    uses_data_alias: bool = field(default=False, compare=False)


def build_crossmarket_factors() -> tuple[object, ...]:
    return (
        CrossMarketCorrelationFactor(),
        ArbitrageOpportunityFactor(),
        MarketLinkageFactor(),
        RelativeStrengthFactor(),
        CointegrationFactor(),
        CrossMarketVolatilityFactor(),
        MarketRegimeSwitchFactor(),
        CrossMarketEntropyFactor(),
        CrossMarketCoherenceFactor(),
        CrossMarketGrangerFactor(),
        CrossMarketJointDistributionFactor(),
        CrossMarketCopulaFactor(),
        CrossMarketPhaseSynchronizationFactor(),
        CrossMarketInformationFlowFactor(),
        CrossMarketMultiscaleCorrelationFactor(),
        CrossMarketDynamicCorrelationFactor(),
    )


FACTOR_COLUMNS = tuple(
    factor.name for factor in build_crossmarket_factors()
)
ETF_INDEX_RELATIVE_VALUE_COLUMNS = (
    "etf_index_basis_1m",
    "etf_basis_zscore_20m",
    "etf_fair_value_gap_proxy_1m",
    "basis_reversion_signal_5m",
    "basis_reversion_signal_20m",
    "etf_index_intraday_price_gap",
    "etf_index_intraday_price_gap_zscore_60m",
    "etf_index_return_gap_10m",
    "etf_index_return_gap_30m",
    "etf_index_beta_residual_return",
    "etf_index_leadlag_corr",
    "etf_index_tracking_error",
    "etf_index_realized_vol_ratio",
    "etf_index_volume_shock_gap",
    "etf_index_momentum_divergence",
)
RELATED_ETF_FACTOR_COLUMNS = (
    "same_index_relative_return_1m",
    "same_index_relative_liquidity_1m",
    "same_index_intraday_cumulative_return_gap",
    "same_index_relative_amount_shock_60m",
)
DAILY_STATE_COLUMNS = ("prev_nav", "prev_total_size", "prev_total_share")
ALL_FACTOR_COLUMNS = (
    FACTOR_COLUMNS
    + ETF_INDEX_RELATIVE_VALUE_COLUMNS
    + RELATED_ETF_FACTOR_COLUMNS
)
ROLLING_WINDOW = 60
ROLLING_MIN_PERIODS = 30
VOLATILITY_WINDOW = 30
BASIS_ZSCORE_WINDOW = 20
TOTAL_SIZE_AMOUNT_MULTIPLIER = 10_000.0
REFERENCE_HISTORY_DAYS = 45


def normalize_fund_code(value: object) -> str:
    code = str(value).strip()
    if code.lower().endswith(".parquet"):
        code = code[:-8]
    code = code.split(".", 1)[0]
    if not code.isdigit():
        raise ValueError(f"Invalid ETF fund code: {value}")
    return code.zfill(6)


def normalize_reference_code(value: object) -> str:
    code = str(value).strip().upper()
    if not code or code in {"NAN", "NONE"}:
        raise ValueError("Reference index code is empty")
    return code


def canonical_reference_code(value: object) -> str:
    reference_code = normalize_reference_code(value)
    return REFERENCE_DATA_ALIASES.get(reference_code, reference_code)


def load_mapping_records(mapping_path: Path) -> dict[str, MappingRecord]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file does not exist: {mapping_path}")

    mapping = pd.read_csv(mapping_path, dtype=str)
    missing = [
        column
        for column in REQUIRED_MAPPING_COLUMNS
        if column not in mapping.columns
    ]
    if missing:
        raise ValueError(f"Mapping file is missing columns: {missing}")

    records: dict[str, MappingRecord] = {}
    for row in mapping.loc[:, REQUIRED_MAPPING_COLUMNS].itertuples(
        index=False
    ):
        try:
            fund_code = normalize_fund_code(row.fund_code)
            source_reference_code = normalize_reference_code(
                row.reference_index_code
            )
        except ValueError:
            continue

        reference_code = canonical_reference_code(source_reference_code)

        record = MappingRecord(
            fund_code,
            reference_code,
            uses_data_alias=source_reference_code != reference_code,
        )
        existing = records.get(fund_code)
        if existing is not None and existing != record:
            raise ValueError(
                f"Conflicting reference mappings for ETF {fund_code}: "
                f"{existing.reference_index_code} and {reference_code}"
            )
        records[fund_code] = record

    if not records:
        raise ValueError(f"No valid mapping records found in: {mapping_path}")
    return records


def discover_etf_files(etf_root: Path) -> dict[str, Path]:
    if not etf_root.exists():
        raise FileNotFoundError(f"ETF minute root does not exist: {etf_root}")

    files: dict[str, Path] = {}
    for path in sorted(etf_root.glob("*.parquet")):
        fund_code = normalize_fund_code(path.name)
        if fund_code in files:
            raise ValueError(
                f"Multiple ETF minute files share fund code {fund_code}"
            )
        files[fund_code] = path
    if not files:
        raise FileNotFoundError(
            f"No ETF minute parquet files found in: {etf_root}"
        )
    return files


def build_index_file_lookup(
    index_root: Path,
) -> tuple[dict[str, Path], dict[str, tuple[Path, ...]]]:
    if not index_root.exists():
        raise FileNotFoundError(
            f"Index minute root does not exist: {index_root}"
        )

    by_full_code: dict[str, Path] = {}
    paths_by_stem: dict[str, list[Path]] = {}
    for path in sorted(index_root.glob("*.parquet")):
        full_code = path.stem.upper()
        by_full_code[full_code] = path
        code_stem = full_code.rsplit(".", 1)[0]
        paths_by_stem.setdefault(code_stem, []).append(path)

    if not by_full_code:
        raise FileNotFoundError(
            f"No index minute parquet files found in: {index_root}"
        )
    return (
        by_full_code,
        {
            code: tuple(paths)
            for code, paths in paths_by_stem.items()
        },
    )


def resolve_reference_path(
    reference_code: str,
    by_full_code: dict[str, Path],
    paths_by_stem: dict[str, tuple[Path, ...]],
) -> tuple[Path | None, str]:
    normalized = normalize_reference_code(reference_code)
    exact = by_full_code.get(normalized)
    if exact is not None:
        return exact, "exact"

    for candidate_code, match_type in (
        (normalized, "stem"),
        (REFERENCE_DATA_ALIASES.get(normalized), "data_alias"),
    ):
        if candidate_code is None:
            continue
        candidate_exact = by_full_code.get(candidate_code)
        if candidate_exact is not None:
            return candidate_exact, match_type
        code_stem = candidate_code.rsplit(".", 1)[0]
        candidates = paths_by_stem.get(code_stem, ())
        if len(candidates) == 1:
            return candidates[0], match_type
    return None, "missing"


def filter_date_range(
    frame: pd.DataFrame,
    date_from: str | None,
    date_to: str | None,
) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return frame

    trade_dates = frame.index.strftime("%Y%m%d")
    mask = np.ones(len(frame), dtype=bool)
    if date_from is not None:
        mask &= trade_dates >= date_from
    if date_to is not None:
        mask &= trade_dates <= date_to
    return frame.loc[mask].copy()


def normalize_reference_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    if isinstance(frame.index, pd.MultiIndex) and (
        "trade_time" in frame.index.names
    ):
        frame.index = pd.to_datetime(
            frame.index.get_level_values("trade_time")
        )
    elif isinstance(frame.index, pd.MultiIndex):
        fallback_index = None
        for level in range(frame.index.nlevels - 1, -1, -1):
            candidate = pd.to_datetime(
                frame.index.get_level_values(level), errors="coerce"
            )
            if candidate.isna().all():
                continue
            fallback_index = candidate
            if ((candidate.normalize() != candidate) & candidate.notna()).any():
                frame.index = candidate
                break
        else:
            if fallback_index is not None:
                frame.index = fallback_index
            else:
                raise ValueError(
                    "Cannot locate trade_time/datetime index or column."
                )
    elif "trade_time" in frame.columns:
        frame.index = pd.to_datetime(frame.pop("trade_time"))
    elif "datetime" in frame.columns:
        frame.index = pd.to_datetime(frame.pop("datetime"))
    else:
        raise ValueError(
            "Cannot locate trade_time/datetime index or column."
        )

    required = ("open", "close", "volume", "amount")
    if "vol" in frame.columns and "volume" not in frame.columns:
        frame = frame.rename(columns={"vol": "volume"})
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Reference minute frame is missing columns: {missing}")
    frame.index.name = "trade_time"
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_index()
    return frame.loc[
        ~frame.index.duplicated(keep="last"), list(required)
    ]


def read_reference_frame(
    reference_path: Path,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp,
) -> pd.DataFrame:
    history_start = date_from.normalize() - pd.Timedelta(
        days=REFERENCE_HISTORY_DAYS
    )
    filters = [
        ("trade_date", ">=", history_start),
        ("trade_date", "<=", date_to.normalize()),
    ]
    try:
        raw_frame = pd.read_parquet(
            reference_path,
            columns=["open", "close", "vol", "amount"],
            filters=filters,
        )
    except (TypeError, ValueError, NotImplementedError):
        raw_frame = pd.read_parquet(
            reference_path, columns=["open", "close", "vol", "amount"]
        )
    return normalize_reference_frame(raw_frame)


def load_etf_daily_histories(
    etf_daily_path: Path,
    member_paths: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    if not etf_daily_path.exists():
        raise FileNotFoundError(
            f"ETF daily parquet does not exist: {etf_daily_path}"
        )
    symbols = sorted({path.stem.upper() for path in member_paths.values()})
    raw = pd.read_parquet(
        etf_daily_path,
        columns=["nav", "total_size", "total_share"],
        filters=[("ts_code", "in", symbols)],
    ).reset_index()
    required = ("trade_date", "ts_code", "nav", "total_size", "total_share")
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"ETF daily frame is missing columns: {missing}")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce")
    raw["_fund_code"] = raw["ts_code"].map(normalize_fund_code)
    for column in ("nav", "total_size", "total_share"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    histories: dict[str, pd.DataFrame] = {}
    for fund_code in member_paths:
        history = raw.loc[
            raw["_fund_code"].eq(fund_code),
            ["trade_date", "nav", "total_size", "total_share"],
        ].dropna(subset=["trade_date"])
        history = history.sort_values("trade_date", kind="mergesort")
        history = history.drop_duplicates("trade_date", keep="last")
        histories[fund_code] = history.set_index("trade_date")
    return histories


def align_previous_daily_state(
    minute_index: pd.DatetimeIndex,
    daily_history: pd.DataFrame | None,
    trade_calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan,
        index=minute_index,
        columns=DAILY_STATE_COLUMNS,
        dtype=float,
    )
    if daily_history is None or daily_history.empty:
        return result

    history = daily_history.sort_index()
    history_dates = pd.DatetimeIndex(history.index).normalize().unique()
    calendar = pd.DatetimeIndex(
        minute_index if trade_calendar is None else trade_calendar
    ).normalize().unique().sort_values()
    previous_trade_days = dict(zip(calendar[1:], calendar[:-1]))
    for trade_day in minute_index.normalize().unique():
        previous_trade_day = previous_trade_days.get(trade_day)
        if previous_trade_day is None or previous_trade_day not in history_dates:
            continue
        source = history.loc[previous_trade_day]
        day_mask = minute_index.normalize() == trade_day
        result.loc[day_mask, "prev_nav"] = source["nav"]
        result.loc[day_mask, "prev_total_size"] = source["total_size"]
        result.loc[day_mask, "prev_total_share"] = source["total_share"]
    return result


def _log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    valid_close = pd.to_numeric(close, errors="coerce").where(
        lambda values: values.gt(0)
    )
    return np.log(valid_close).diff(periods)


def _simple_return(close: pd.Series, periods: int = 1) -> pd.Series:
    valid_close = pd.to_numeric(close, errors="coerce").where(
        lambda values: values.gt(0)
    )
    return valid_close.pct_change(periods, fill_method=None)


def _rolling_zscore(values: pd.Series) -> pd.Series:
    mean = values.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean()
    std = values.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).std()
    return values.sub(mean).div(std.where(std.gt(0)))


def _basis_zscore_20m(basis: pd.Series) -> pd.Series:
    mean = basis.rolling(
        BASIS_ZSCORE_WINDOW, min_periods=BASIS_ZSCORE_WINDOW
    ).mean()
    std = basis.rolling(
        BASIS_ZSCORE_WINDOW, min_periods=BASIS_ZSCORE_WINDOW
    ).std()
    return basis.sub(mean).div(std.where(std.gt(0)))


def _previous_reference_close(
    reference_frame: pd.DataFrame,
    trade_day: pd.Timestamp,
) -> float:
    prior_close = pd.to_numeric(
        reference_frame.loc[
            reference_frame.index.normalize() < trade_day, "close"
        ],
        errors="coerce",
    ).where(lambda values: values.gt(0)).dropna()
    if prior_close.empty:
        return np.nan
    return float(prior_close.iloc[-1])


def _calculate_day_relative_value_factors(
    etf_day: pd.DataFrame,
    reference_day: pd.DataFrame,
    daily_state_day: pd.DataFrame,
    reference_previous_close: float,
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan,
        index=etf_day.index,
        columns=ETF_INDEX_RELATIVE_VALUE_COLUMNS,
        dtype=float,
    )
    reference = reference_day.reindex(etf_day.index)
    etf_close = pd.to_numeric(etf_day["close"], errors="coerce")
    reference_close = pd.to_numeric(reference["close"], errors="coerce")
    etf_return = _log_return(etf_close)
    reference_return = _log_return(reference_close)

    basis_1m = _simple_return(etf_close).sub(
        _simple_return(reference_close)
    )
    result["etf_index_basis_1m"] = basis_1m
    result["etf_basis_zscore_20m"] = _basis_zscore_20m(basis_1m)

    return_gaps: dict[int, pd.Series] = {}
    for horizon in (5, 10, 30):
        gap = _log_return(etf_close, horizon).sub(
            _log_return(reference_close, horizon)
        )
        if horizon != 5:
            result[f"etf_index_return_gap_{horizon}m"] = gap
        return_gaps[horizon] = gap

    historical_etf_return = etf_return.shift(1)
    historical_reference_return = reference_return.shift(1)
    valid_history = (
        historical_etf_return.notna() & historical_reference_return.notna()
    )
    beta_covariance = historical_etf_return.where(valid_history).rolling(
        ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
    ).cov(historical_reference_return.where(valid_history))
    beta_variance = historical_reference_return.where(valid_history).rolling(
        ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
    ).var()
    beta = beta_covariance.div(beta_variance.where(beta_variance.gt(0)))
    result["etf_index_beta_residual_return"] = etf_return.sub(
        beta.mul(reference_return)
    )
    result["etf_index_leadlag_corr"] = etf_return.rolling(
        ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
    ).corr(reference_return.shift(1))

    active_return = etf_return.sub(reference_return)
    result["etf_index_tracking_error"] = active_return.rolling(
        VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW
    ).std()
    etf_volatility = etf_return.rolling(
        VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW
    ).std()
    reference_volatility = reference_return.rolling(
        VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW
    ).std()
    result["etf_index_realized_vol_ratio"] = etf_volatility.div(
        reference_volatility.where(reference_volatility.gt(0))
    )

    etf_amount = np.log1p(pd.to_numeric(etf_day["amount"], errors="coerce"))
    reference_amount = np.log1p(
        pd.to_numeric(reference["amount"], errors="coerce")
    )
    result["etf_index_volume_shock_gap"] = _rolling_zscore(etf_amount).sub(
        _rolling_zscore(reference_amount)
    )
    result["etf_index_momentum_divergence"] = return_gaps[5].sub(
        return_gaps[30]
    )

    etf_open = pd.to_numeric(etf_day["open"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    reference_open = pd.to_numeric(reference["open"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    common_open = etf_open.notna() & reference_open.notna()
    if common_open.any():
        base_time = common_open[common_open].index[0]
        intraday_proxy = etf_open.loc[base_time] * reference_close.div(
            reference_open.loc[base_time]
        )
        intraday_gap = etf_close.div(
            intraday_proxy.where(intraday_proxy.gt(0))
        ).sub(1.0)
        result["etf_index_intraday_price_gap"] = intraday_gap
        result["etf_index_intraday_price_gap_zscore_60m"] = (
            _rolling_zscore(intraday_gap)
        )

    previous_nav = pd.to_numeric(
        daily_state_day["prev_nav"], errors="coerce"
    ).where(lambda values: values.gt(0)).dropna()
    if previous_nav.size and np.isfinite(reference_previous_close):
        fair_value = previous_nav.iloc[0] * reference_close.div(
            reference_previous_close
        )
        fair_value_gap = etf_close.div(
            fair_value.where(fair_value.gt(0))
        ).sub(1.0)
        result["etf_fair_value_gap_proxy_1m"] = fair_value_gap
        result["basis_reversion_signal_5m"] = -fair_value_gap.diff(5)
        result["basis_reversion_signal_20m"] = -fair_value_gap.diff(20)
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_related_etf_factor_frames(
    member_frames: dict[str, pd.DataFrame],
    daily_state_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    if daily_state_frames is None:
        daily_state_frames = {
            code: pd.DataFrame(
                np.nan, index=frame.index, columns=DAILY_STATE_COLUMNS
            )
            for code, frame in member_frames.items()
        }
    results = {
        code: pd.DataFrame(
            np.nan,
            index=frame.index,
            columns=RELATED_ETF_FACTOR_COLUMNS,
            dtype=float,
        )
        for code, frame in member_frames.items()
    }
    if len(member_frames) < 2:
        return results

    for day_frames in _group_frames_by_trade_day(member_frames).values():
        return_panel = pd.concat(
            {
                code: _simple_return(frame["close"])
                for code, frame in day_frames.items()
            },
            axis=1,
        ).sort_index()
        cumulative_return_panel = pd.concat(
            {
                code: _day_cumulative_return(frame)
                for code, frame in day_frames.items()
            },
            axis=1,
        ).reindex(return_panel.index)
        amount_panel = pd.concat(
            {
                code: pd.to_numeric(frame["amount"], errors="coerce")
                for code, frame in day_frames.items()
            },
            axis=1,
        ).reindex(return_panel.index)
        liquidity_panel = pd.concat(
            {
                code: pd.to_numeric(frame["amount"], errors="coerce").div(
                    pd.to_numeric(
                        daily_state_frames[code]["prev_total_size"],
                        errors="coerce",
                    ).mul(TOTAL_SIZE_AMOUNT_MULTIPLIER)
                )
                for code, frame in day_frames.items()
            },
            axis=1,
        ).reindex(return_panel.index)
        amount_shock_panel = pd.concat(
            {
                code: _rolling_zscore(
                    np.log1p(
                        pd.to_numeric(frame["amount"], errors="coerce")
                    )
                )
                for code, frame in day_frames.items()
            },
            axis=1,
        ).reindex(return_panel.index)
        lagged_amount_weights = amount_panel.shift(1).where(
            lambda values: values.gt(0)
        )

        for fund_code, frame in day_frames.items():
            peer_return = return_panel.drop(columns=fund_code).mean(
                axis=1, skipna=True
            )
            peer_liquidity = liquidity_panel.drop(columns=fund_code).mean(
                axis=1, skipna=True
            )
            peer_weights = lagged_amount_weights.drop(columns=fund_code)
            peer_cumulative_returns = cumulative_return_panel.drop(
                columns=fund_code
            )
            cumulative_weights = peer_weights.where(
                peer_cumulative_returns.notna()
            )
            peer_cumulative_return = peer_cumulative_returns.mul(
                cumulative_weights
            ).sum(axis=1, min_count=1).div(
                cumulative_weights.sum(axis=1, min_count=1).where(
                    lambda values: values.gt(0)
                )
            )
            peer_amount_shocks = amount_shock_panel.drop(columns=fund_code)
            shock_weights = peer_weights.where(peer_amount_shocks.notna())
            peer_amount_shock = peer_amount_shocks.mul(shock_weights).sum(
                axis=1, min_count=1
            ).div(
                shock_weights.sum(axis=1, min_count=1).where(
                    lambda values: values.gt(0)
                )
            )
            related = pd.DataFrame(
                {
                    "same_index_relative_return_1m": return_panel[fund_code].sub(
                        peer_return
                    ),
                    "same_index_relative_liquidity_1m": liquidity_panel[
                        fund_code
                    ].sub(
                        peer_liquidity
                    ),
                    "same_index_intraday_cumulative_return_gap": (
                        cumulative_return_panel[fund_code].sub(
                            peer_cumulative_return
                        )
                    ),
                    "same_index_relative_amount_shock_60m": (
                        amount_shock_panel[fund_code].sub(
                            peer_amount_shock
                        )
                    ),
                },
                index=return_panel.index,
            )
            results[fund_code].loc[frame.index, RELATED_ETF_FACTOR_COLUMNS] = (
                related.reindex(frame.index)
            )
    return {
        code: frame.replace([np.inf, -np.inf], np.nan)
        for code, frame in results.items()
    }


def _group_frames_by_trade_day(
    member_frames: dict[str, pd.DataFrame],
) -> dict[pd.Timestamp, dict[str, pd.DataFrame]]:
    grouped: dict[pd.Timestamp, dict[str, pd.DataFrame]] = defaultdict(dict)
    for fund_code, frame in member_frames.items():
        for trade_day, day_frame in frame.groupby(frame.index.normalize(), sort=False):
            grouped[trade_day][fund_code] = day_frame
    return grouped


def _day_cumulative_return(frame: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_price = pd.to_numeric(frame["open"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    if not open_price.notna().any():
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return close.div(open_price.dropna().iloc[0]).sub(1.0)


def _calculate_day_factors(
    etf_day: pd.DataFrame,
    reference_day: pd.DataFrame,
    daily_state_day: pd.DataFrame,
    reference_previous_close: float,
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan,
        index=etf_day.index,
        columns=ALL_FACTOR_COLUMNS,
        dtype=float,
    )
    aligned_reference = reference_day.reindex(etf_day.index)
    result.loc[:, ETF_INDEX_RELATIVE_VALUE_COLUMNS] = (
        _calculate_day_relative_value_factors(
            etf_day,
            reference_day,
            daily_state_day,
            reference_previous_close,
        )
    )
    current_close = pd.to_numeric(etf_day["close"], errors="coerce")
    reference_close = pd.to_numeric(
        aligned_reference["close"], errors="coerce"
    )
    valid = (
        current_close.notna()
        & reference_close.notna()
        & current_close.gt(0)
        & reference_close.gt(0)
    )
    if valid.sum() < 50:
        return result

    current_input = etf_day.loc[valid]
    reference_input = aligned_reference.loc[valid]
    for factor in build_crossmarket_factors():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = factor.calculate(current_input, reference_input)
        series = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        result.loc[series.index, factor.name] = series
    return result


def calculate_crossmarket_factor_frame(
    etf_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    daily_state_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if "close" not in etf_frame.columns:
        raise ValueError("ETF minute frame is missing close")
    if "close" not in reference_frame.columns:
        raise ValueError("Reference minute frame is missing close")

    result = pd.DataFrame(
        np.nan,
        index=etf_frame.index,
        columns=ALL_FACTOR_COLUMNS,
        dtype=float,
    )
    if daily_state_frame is None:
        daily_state_frame = pd.DataFrame(
            np.nan,
            index=etf_frame.index,
            columns=DAILY_STATE_COLUMNS,
            dtype=float,
        )
    for trade_day, etf_day in etf_frame.groupby(
        etf_frame.index.normalize(), sort=False
    ):
        reference_day = reference_frame.loc[
            reference_frame.index.normalize() == trade_day
        ]
        if reference_day.empty:
            continue
        result.loc[etf_day.index] = _calculate_day_factors(
            etf_day,
            reference_day,
            daily_state_frame.reindex(etf_day.index),
            _previous_reference_close(reference_frame, trade_day),
        )
    return result.replace([np.inf, -np.inf], np.nan)


def build_output_frame(
    etf_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    mapping_reference_code: str,
    reference_ts_code: str,
    factor_frame: pd.DataFrame,
    daily_state_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "reference_index_code": mapping_reference_code,
            "reference_ts_code": reference_ts_code,
            "reference_close": reference_frame["close"].reindex(
                etf_frame.index
            ),
        },
        index=etf_frame.index,
    )
    if daily_state_frame is None:
        daily_state_frame = pd.DataFrame(
            np.nan,
            index=etf_frame.index,
            columns=DAILY_STATE_COLUMNS,
            dtype=float,
        )
    result = pd.concat(
        [
            etf_frame,
            metadata,
            daily_state_frame.reindex(etf_frame.index),
            factor_frame,
        ],
        axis=1,
    )
    return result.replace([np.inf, -np.inf], np.nan)


def restrict_to_reference_minutes(
    etf_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.DataFrame:
    return etf_frame.loc[etf_frame.index.intersection(reference_frame.index)]


def no_overlap_result(
    etf_path: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, object]:
    if overwrite and output_path.exists():
        output_path.unlink()
    return {
        "status": "no_overlap",
        "etf_code": etf_path.stem,
        "output_path": output_path,
    }


def process_mapping_record(
    etf_path: Path,
    reference_path: Path,
    mapping_reference_code: str,
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
    etf_daily_path: Path | None = None,
) -> dict[str, object]:
    output_path = output_root / etf_path.name
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "etf_code": etf_path.stem,
            "output_path": output_path,
        }

    raw_etf_frame = normalize_minute_frame(pd.read_parquet(etf_path))
    etf_frame = filter_date_range(raw_etf_frame, date_from, date_to)
    if etf_frame.empty:
        return {
            "status": "empty",
            "etf_code": etf_path.stem,
            "output_path": output_path,
        }
    reference_frame = read_reference_frame(
        reference_path,
        etf_frame.index.min(),
        etf_frame.index.max(),
    )
    etf_frame = restrict_to_reference_minutes(etf_frame, reference_frame)
    if etf_frame.empty:
        return no_overlap_result(etf_path, output_path, overwrite)
    fund_code = normalize_fund_code(etf_path.name)
    daily_history = None
    if etf_daily_path is not None:
        daily_history = load_etf_daily_histories(
            etf_daily_path, {fund_code: etf_path}
        )[fund_code]
    daily_state_frame = align_previous_daily_state(
        etf_frame.index, daily_history, raw_etf_frame.index
    )

    factor_frame = calculate_crossmarket_factor_frame(
        etf_frame, reference_frame, daily_state_frame
    )
    result = build_output_frame(
        etf_frame,
        reference_frame,
        mapping_reference_code,
        reference_path.stem,
        factor_frame,
        daily_state_frame,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    factor_values = result.loc[:, ALL_FACTOR_COLUMNS]
    return {
        "status": "written",
        "etf_code": etf_path.stem,
        "reference_code": reference_path.stem,
        "output_path": output_path,
        "rows": len(result),
        "cols": len(result.columns),
        "factor_non_null": int(factor_values.notna().sum().sum()),
    }


def process_mapping_group(
    member_paths: dict[str, Path],
    target_codes: tuple[str, ...],
    reference_path: Path,
    mapping_reference_code: str,
    etf_daily_path: Path,
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
) -> list[dict[str, object]]:
    writable_targets: list[str] = []
    results: list[dict[str, object]] = []
    for fund_code in target_codes:
        output_path = output_root / member_paths[fund_code].name
        if output_path.exists() and not overwrite:
            results.append(
                {
                    "status": "skipped",
                    "etf_code": member_paths[fund_code].stem,
                    "output_path": output_path,
                }
            )
        else:
            writable_targets.append(fund_code)
    if not writable_targets:
        return results

    raw_member_frames = {
        fund_code: normalize_minute_frame(pd.read_parquet(path))
        for fund_code, path in member_paths.items()
    }
    member_frames = {
        fund_code: filter_date_range(frame, date_from, date_to)
        for fund_code, frame in raw_member_frames.items()
    }
    member_frames = {
        fund_code: frame
        for fund_code, frame in member_frames.items()
        if not frame.empty
    }
    if not member_frames:
        return results + [
            {
                "status": "empty",
                "etf_code": member_paths[fund_code].stem,
                "output_path": output_root / member_paths[fund_code].name,
            }
            for fund_code in writable_targets
        ]

    reference_frame = read_reference_frame(
        reference_path,
        min(frame.index.min() for frame in member_frames.values()),
        max(frame.index.max() for frame in member_frames.values()),
    )
    member_frames = {
        fund_code: restrict_to_reference_minutes(frame, reference_frame)
        for fund_code, frame in member_frames.items()
    }
    member_frames = {
        fund_code: frame
        for fund_code, frame in member_frames.items()
        if not frame.empty
    }
    if not member_frames:
        return results + [
            no_overlap_result(
                member_paths[fund_code],
                output_root / member_paths[fund_code].name,
                overwrite,
            )
            for fund_code in writable_targets
        ]
    daily_histories = load_etf_daily_histories(
        etf_daily_path,
        {
            fund_code: member_paths[fund_code]
            for fund_code in member_frames
        },
    )
    daily_state_frames = {
        fund_code: align_previous_daily_state(
            frame.index,
            daily_histories.get(fund_code),
            raw_member_frames[fund_code].index,
        )
        for fund_code, frame in member_frames.items()
    }
    related_frames = calculate_related_etf_factor_frames(
        member_frames, daily_state_frames
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for fund_code in writable_targets:
        etf_frame = member_frames.get(fund_code)
        output_path = output_root / member_paths[fund_code].name
        if etf_frame is None:
            results.append(
                no_overlap_result(
                    member_paths[fund_code], output_path, overwrite
                )
            )
            continue
        factor_frame = calculate_crossmarket_factor_frame(
            etf_frame,
            reference_frame,
            daily_state_frames[fund_code],
        )
        factor_frame.loc[:, RELATED_ETF_FACTOR_COLUMNS] = related_frames[
            fund_code
        ].reindex(factor_frame.index)
        result = build_output_frame(
            etf_frame,
            reference_frame,
            mapping_reference_code,
            reference_path.stem,
            factor_frame,
            daily_state_frames[fund_code],
        )
        result.to_parquet(output_path)
        results.append(
            {
                "status": "written",
                "etf_code": member_paths[fund_code].stem,
                "reference_code": reference_path.stem,
                "output_path": output_path,
                "rows": len(result),
                "cols": len(result.columns),
                "factor_non_null": int(
                    result.loc[:, ALL_FACTOR_COLUMNS].notna().sum().sum()
                ),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ETF CrossMarket minute factors from mapped local "
            "reference-index minute data."
        )
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--etf-root", type=Path, default=DEFAULT_ETF_ROOT)
    parser.add_argument(
        "--etf-daily", type=Path, default=DEFAULT_ETF_DAILY_PATH
    )
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional ETF symbols such as 159008.SZ or 510300.SH.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional text file with one ETF symbol per line.",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _requested_fund_codes(args: argparse.Namespace) -> list[str] | None:
    requested: list[str] = []
    if args.symbols_file is not None:
        requested.extend(read_symbol_list_file(args.symbols_file))
    if args.symbols:
        requested.extend(args.symbols)
    if not requested:
        return None
    return list(dict.fromkeys(normalize_fund_code(code) for code in requested))


def main() -> int:
    args = parse_args()
    configure_logging()
    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")

    mappings = load_mapping_records(args.mapping)
    etf_files = discover_etf_files(args.etf_root)
    by_full_code, paths_by_stem = build_index_file_lookup(args.index_root)
    requested = _requested_fund_codes(args)
    selected_codes = requested or sorted(mappings)

    if requested:
        missing_mapping = [code for code in requested if code not in mappings]
        missing_etf = [code for code in requested if code not in etf_files]
        if missing_mapping:
            raise FileNotFoundError(
                "ETF codes missing from mapping: "
                + ", ".join(missing_mapping)
            )
        if missing_etf:
            raise FileNotFoundError(
                "ETF minute files missing: " + ", ".join(missing_etf)
            )

    selected_records: list[tuple[str, Path, MappingRecord, Path]] = []
    missing_references: list[tuple[str, str]] = []
    alias_count = 0
    for fund_code in selected_codes:
        mapping = mappings.get(fund_code)
        etf_path = etf_files.get(fund_code)
        if mapping is None or etf_path is None:
            continue
        reference_path, match_type = resolve_reference_path(
            mapping.reference_index_code,
            by_full_code,
            paths_by_stem,
        )
        if reference_path is None:
            missing_references.append(
                (fund_code, mapping.reference_index_code)
            )
            continue
        alias_count += int(
            match_type == "stem" or mapping.uses_data_alias
        )
        selected_records.append((fund_code, etf_path, mapping, reference_path))

    if args.limit is not None:
        selected_records = selected_records[: args.limit]

    selected_by_reference: dict[
        str, list[tuple[str, Path, MappingRecord, Path]]
    ] = defaultdict(list)
    for record in selected_records:
        selected_by_reference[record[2].reference_index_code].append(record)

    jobs: list[
        tuple[
            dict[str, Path],
            tuple[str, ...],
            Path,
            str,
            Path,
            Path,
            str | None,
            str | None,
            bool,
        ]
    ] = []
    for reference_code, targets in selected_by_reference.items():
        member_paths = {
            fund_code: etf_path
            for fund_code, mapping in mappings.items()
            if mapping.reference_index_code == reference_code
            and (etf_path := etf_files.get(fund_code)) is not None
        }
        if not member_paths:
            continue
        jobs.append(
            (
                member_paths,
                tuple(fund_code for fund_code, _, _, _ in targets),
                targets[0][3],
                reference_code,
                args.etf_daily,
                args.output_root,
                date_from,
                date_to,
                args.overwrite,
            )
        )

    LOGGER.info(
        "Prepared %s ETF/reference groups; %s selected ETFs use reference "
        "code aliases",
        len(jobs),
        alias_count,
    )
    if missing_references:
        examples = ", ".join(
            f"{fund}->{reference}"
            for fund, reference in missing_references[:10]
        )
        LOGGER.warning(
            "Excluding %s ETF(s) across %s reference group(s) without local "
            "index minute files; "
            "examples: %s",
            len(missing_references),
            len({reference for _, reference in missing_references}),
            examples,
        )
    if not jobs:
        LOGGER.error("No ETF/reference jobs can be processed")
        return 1

    failures: list[tuple[str, str]] = []
    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        results = []
        for job in jobs:
            try:
                results.extend(process_mapping_group(*job))
            except Exception as exc:  # noqa: BLE001
                failures.append((job[3], str(exc)))
                LOGGER.exception("Failed to process reference %s", job[3])
    else:
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(process_mapping_group, *job): job[3]
                for job in jobs
            }
            for future in as_completed(future_map):
                reference_code = future_map[future]
                try:
                    results.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append((reference_code, str(exc)))
                    LOGGER.exception(
                        "Failed to process reference %s", reference_code
                    )

    for result in results:
        if result["status"] == "written":
            LOGGER.info(
                "Wrote %s rows, %s columns and %s non-null factor values "
                "for %s against %s to %s",
                result["rows"],
                result["cols"],
                result["factor_non_null"],
                result["etf_code"],
                result["reference_code"],
                result["output_path"],
            )
        else:
            LOGGER.info(
                "%s %s: %s",
                result["status"].capitalize(),
                result["etf_code"],
                result["output_path"],
            )

    if failures:
        LOGGER.error("Completed with %s failures", len(failures))
        for etf_code, reason in failures[:10]:
            LOGGER.error("  %s -> %s", etf_code, reason)
        return 1

    LOGGER.info("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
