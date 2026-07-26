from __future__ import annotations

"""Generate ETF minute CrossMarket factors using mapped reference indices."""

import argparse
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
DEFAULT_INDEX_ROOT = Path(r"D:\workspace\stockdata\index-data\index_1min")
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_crossmarket_factors"
)
REQUIRED_MAPPING_COLUMNS = ("fund_code", "reference_index_code")


@dataclass(frozen=True)
class MappingRecord:
    fund_code: str
    reference_index_code: str


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
    "etf_index_return_gap_1m",
    "etf_index_return_gap_5m",
    "etf_index_return_gap_10m",
    "etf_index_return_gap_30m",
    "etf_index_beta_residual_return",
    "etf_index_leadlag_corr",
    "etf_index_tracking_error",
    "etf_index_realized_vol_ratio",
    "etf_index_volume_shock_gap",
    "etf_index_momentum_divergence",
    "etf_fair_value_premium",
    "etf_fair_value_premium_zscore",
    "premium_mean_reversion_speed",
    "premium_change_1m",
    "premium_change_5m",
)
RELATED_ETF_FACTOR_COLUMNS = (
    "related_etf_price_spread",
    "related_etf_liquidity_gap",
)
ALL_FACTOR_COLUMNS = (
    FACTOR_COLUMNS
    + ETF_INDEX_RELATIVE_VALUE_COLUMNS
    + RELATED_ETF_FACTOR_COLUMNS
)
ROLLING_WINDOW = 60
ROLLING_MIN_PERIODS = 30
VOLATILITY_WINDOW = 30


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
            reference_code = normalize_reference_code(
                row.reference_index_code
            )
        except ValueError:
            continue

        record = MappingRecord(fund_code, reference_code)
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

    code_stem = normalized.rsplit(".", 1)[0]
    candidates = paths_by_stem.get(code_stem, ())
    if len(candidates) == 1:
        return candidates[0], "stem"
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
    filters = [
        ("trade_date", ">=", date_from.normalize()),
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


def _log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    valid_close = pd.to_numeric(close, errors="coerce").where(
        lambda values: values.gt(0)
    )
    return np.log(valid_close).diff(periods)


def _rolling_zscore(values: pd.Series) -> pd.Series:
    mean = values.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean()
    std = values.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).std()
    return values.sub(mean).div(std.where(std.gt(0)))


def _mean_reversion_speed(premium: pd.Series) -> pd.Series:
    lagged = premium.shift(1)
    covariance = premium.rolling(
        ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
    ).cov(lagged)
    variance = lagged.rolling(
        ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS
    ).var()
    phi = covariance.div(variance.where(variance.gt(0)))
    return -np.log(phi.where(phi.gt(0) & phi.lt(1)))


def _calculate_day_relative_value_factors(
    etf_day: pd.DataFrame,
    reference_day: pd.DataFrame,
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

    return_gaps: dict[int, pd.Series] = {}
    for horizon in (1, 5, 10, 30):
        gap = _log_return(etf_close, horizon).sub(
            _log_return(reference_close, horizon)
        )
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
    if etf_open.notna().any() and reference_open.notna().any():
        fair_value = etf_open.dropna().iloc[0] * reference_close.div(
            reference_open.dropna().iloc[0]
        )
        premium = etf_close.div(fair_value.where(fair_value.gt(0))).sub(1.0)
        result["etf_fair_value_premium"] = premium
        result["etf_fair_value_premium_zscore"] = _rolling_zscore(premium)
        result["premium_mean_reversion_speed"] = _mean_reversion_speed(
            premium
        )
        result["premium_change_1m"] = premium.diff(1)
        result["premium_change_5m"] = premium.diff(5)
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_related_etf_factor_frames(
    member_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
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
        cumulative_returns = pd.concat(
            {
                code: _day_cumulative_return(frame)
                for code, frame in day_frames.items()
            },
            axis=1,
        ).sort_index()
        amount_panel = pd.concat(
            {
                code: pd.to_numeric(frame["amount"], errors="coerce")
                for code, frame in day_frames.items()
            },
            axis=1,
        ).reindex(cumulative_returns.index)
        liquidity_shocks = pd.concat(
            {
                code: _rolling_zscore(
                    np.log1p(pd.to_numeric(frame["amount"], errors="coerce"))
                )
                for code, frame in day_frames.items()
            },
            axis=1,
        ).reindex(cumulative_returns.index)
        weights = amount_panel.shift(1).where(lambda values: values.gt(0))

        for fund_code, frame in day_frames.items():
            peer_weights = weights.drop(columns=fund_code)
            peer_returns = cumulative_returns.drop(columns=fund_code)
            peer_liquidity_shocks = liquidity_shocks.drop(columns=fund_code)
            return_weights = peer_weights.where(peer_returns.notna())
            liquidity_weights = peer_weights.where(
                peer_liquidity_shocks.notna()
            )
            peer_return = peer_returns.mul(return_weights).sum(
                axis=1, min_count=1
            ).div(return_weights.sum(axis=1, min_count=1).where(
                lambda values: values.gt(0)
            ))
            peer_liquidity = peer_liquidity_shocks.mul(liquidity_weights).sum(
                axis=1, min_count=1
            ).div(liquidity_weights.sum(axis=1, min_count=1).where(
                lambda values: values.gt(0)
            ))
            related = pd.DataFrame(
                {
                    "related_etf_price_spread": cumulative_returns[fund_code].sub(
                        peer_return
                    ),
                    "related_etf_liquidity_gap": liquidity_shocks[fund_code].sub(
                        peer_liquidity
                    ),
                },
                index=cumulative_returns.index,
            )
            results[fund_code].loc[frame.index, RELATED_ETF_FACTOR_COLUMNS] = (
                related.reindex(frame.index)
            )
    return results


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
) -> pd.DataFrame:
    result = pd.DataFrame(
        np.nan,
        index=etf_day.index,
        columns=ALL_FACTOR_COLUMNS,
        dtype=float,
    )
    aligned_reference = reference_day.reindex(etf_day.index)
    result.loc[:, ETF_INDEX_RELATIVE_VALUE_COLUMNS] = (
        _calculate_day_relative_value_factors(etf_day, reference_day)
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
    for trade_day, etf_day in etf_frame.groupby(
        etf_frame.index.normalize(), sort=False
    ):
        reference_day = reference_frame.loc[
            reference_frame.index.normalize() == trade_day
        ]
        if reference_day.empty:
            continue
        result.loc[etf_day.index] = _calculate_day_factors(
            etf_day, reference_day
        )
    return result.replace([np.inf, -np.inf], np.nan)


def build_output_frame(
    etf_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    mapping_reference_code: str,
    reference_ts_code: str,
    factor_frame: pd.DataFrame,
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
    result = pd.concat([etf_frame, metadata, factor_frame], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def process_mapping_record(
    etf_path: Path,
    reference_path: Path,
    mapping_reference_code: str,
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
) -> dict[str, object]:
    output_path = output_root / etf_path.name
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "etf_code": etf_path.stem,
            "output_path": output_path,
        }

    etf_frame = filter_date_range(
        normalize_minute_frame(pd.read_parquet(etf_path)),
        date_from,
        date_to,
    )
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

    factor_frame = calculate_crossmarket_factor_frame(
        etf_frame, reference_frame
    )
    result = build_output_frame(
        etf_frame,
        reference_frame,
        mapping_reference_code,
        reference_path.stem,
        factor_frame,
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

    member_frames = {
        fund_code: filter_date_range(
            normalize_minute_frame(pd.read_parquet(path)), date_from, date_to
        )
        for fund_code, path in member_paths.items()
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
    related_frames = calculate_related_etf_factor_frames(member_frames)
    output_root.mkdir(parents=True, exist_ok=True)
    for fund_code in writable_targets:
        etf_frame = member_frames.get(fund_code)
        output_path = output_root / member_paths[fund_code].name
        if etf_frame is None:
            results.append(
                {
                    "status": "empty",
                    "etf_code": member_paths[fund_code].stem,
                    "output_path": output_path,
                }
            )
            continue
        factor_frame = calculate_crossmarket_factor_frame(
            etf_frame, reference_frame
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
        alias_count += int(match_type == "stem")
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
                args.output_root,
                date_from,
                date_to,
                args.overwrite,
            )
        )

    LOGGER.info(
        "Prepared %s ETF/reference groups; %s selected ETFs use code-stem aliases",
        len(jobs),
        alias_count,
    )
    if missing_references:
        examples = ", ".join(
            f"{fund}->{reference}"
            for fund, reference in missing_references[:10]
        )
        LOGGER.warning(
            "Skipping %s mappings without local index minute files; "
            "examples: %s",
            len(missing_references),
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
