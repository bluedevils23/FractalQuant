from __future__ import annotations

"""Append cross-ETF OFI factors to existing ETF CrossMarket parquet files."""

import argparse
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from scripts.generate_etf_crossmarket_factors import (  # noqa: E402
    DEFAULT_MAPPING_PATH,
    load_mapping_records,
    normalize_fund_code,
)


LOGGER = logging.getLogger("generate_etf_crossmarket_ofi_factors")

DEFAULT_ORDERBOOK_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_orderbook_factors"
)
DEFAULT_CROSSMARKET_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_1min_crossmarket_factors"
)
OFI_INPUT_COLUMNS = (
    "normalized_ofi_l1_60s",
    "normalized_mlofi_l5_60s",
)
ORDERBOOK_READ_COLUMNS = ("trade_time", "amount", *OFI_INPUT_COLUMNS)
ROLLING_WINDOW = 60
MIN_PERIODS = 30
FACTOR_COLUMNS = tuple(
    f"{factor}_{suffix}"
    for suffix in ("l1_60s", "mlofi_l5_60s")
    for factor in (
        "idiosyncratic_ofi",
        "market_ofi_beta",
        "lead_market_ofi",
        "sector_ofi_dispersion",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append leave-one-out, same-index ETF OFI factors to existing "
            "CrossMarket parquet files."
        )
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument(
        "--orderbook-root", type=Path, default=DEFAULT_ORDERBOOK_ROOT
    )
    parser.add_argument(
        "--crossmarket-root", type=Path, default=DEFAULT_CROSSMARKET_ROOT
    )
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--enable-orderbook-factors",
        action="store_true",
        help=(
            "Run the orderbook-dependent OFI generation. Disabled by default "
            "until ETF orderbook coverage is available."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def discover_symbol_files(
    root: Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> dict[str, Path]:
    if not root.exists():
        if allow_missing:
            LOGGER.warning("%s root does not exist: %s", label, root)
            return {}
        raise FileNotFoundError(f"{label} root does not exist: {root}")

    files: dict[str, Path] = {}
    for path in sorted(root.glob("*.parquet")):
        fund_code = normalize_fund_code(path.name)
        if fund_code in files:
            raise ValueError(
                f"Multiple {label} files share fund code {fund_code}"
            )
        files[fund_code] = path
    return files


def _timestamps_from_frame(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.MultiIndex) and "trade_time" in frame.index.names:
        timestamps = pd.to_datetime(frame.index.get_level_values("trade_time"))
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.to_datetime(frame.index)
    elif "trade_time" in frame.columns:
        timestamps = pd.to_datetime(frame["trade_time"])
    elif "datetime" in frame.columns:
        timestamps = pd.to_datetime(frame["datetime"])
    else:
        raise ValueError("Cannot locate trade_time/datetime index or column")
    return pd.DatetimeIndex(timestamps, name="trade_time")


def normalize_orderbook_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column
        for column in ("amount", *OFI_INPUT_COLUMNS)
        if column not in raw_frame.columns
    ]
    if missing:
        raise ValueError(f"Orderbook frame is missing columns: {missing}")

    result = raw_frame.loc[:, ["amount", *OFI_INPUT_COLUMNS]].copy()
    result.index = _timestamps_from_frame(raw_frame)
    result = result.sort_index()
    if result.index.has_duplicates:
        raise ValueError("Orderbook frame has duplicate minute timestamps")
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result[column].replace([np.inf, -np.inf], np.nan)
    return result


def filter_date_range(
    frame: pd.DataFrame,
    date_from: str | None,
    date_to: str | None,
) -> pd.DataFrame:
    if date_from is None and date_to is None:
        return frame
    dates = frame.index.strftime("%Y%m%d")
    mask = np.ones(len(frame), dtype=bool)
    if date_from is not None:
        mask &= dates >= date_from
    if date_to is not None:
        mask &= dates <= date_to
    return frame.loc[mask].copy()


def _daily_rolling_beta(
    target_ofi: pd.Series,
    market_ofi: pd.Series,
) -> pd.Series:
    lagged_target = target_ofi.groupby(target_ofi.index.normalize()).shift(1)
    lagged_market = market_ofi.groupby(market_ofi.index.normalize()).shift(1)
    valid = lagged_target.notna() & lagged_market.notna()
    target_input = lagged_target.where(valid)
    market_input = lagged_market.where(valid)
    beta = pd.Series(np.nan, index=target_ofi.index, dtype=float)

    for _, day_market in market_input.groupby(
        market_input.index.normalize(), sort=False
    ):
        day_target = target_input.loc[day_market.index]
        covariance = day_target.rolling(
            ROLLING_WINDOW, min_periods=MIN_PERIODS
        ).cov(day_market)
        variance = day_market.rolling(
            ROLLING_WINDOW, min_periods=MIN_PERIODS
        ).var()
        beta.loc[day_market.index] = covariance.div(variance.where(variance.gt(0)))
    return beta.replace([np.inf, -np.inf], np.nan)


def _calculate_single_ofi_panel(
    ofi_panel: pd.DataFrame,
    amount_panel: pd.DataFrame,
    suffix: str,
) -> dict[str, pd.DataFrame]:
    lagged_amount = amount_panel.groupby(
        amount_panel.index.normalize(), sort=False
    ).shift(1)
    weights = lagged_amount.where(lagged_amount.gt(0))
    valid_weight = weights.notna() & ofi_panel.notna()
    effective_weights = weights.where(valid_weight)
    weighted_ofi = ofi_panel * effective_weights
    total_weight = effective_weights.sum(axis=1, min_count=1)
    total_weighted_ofi = weighted_ofi.sum(axis=1, min_count=1)

    dispersion_count = valid_weight.sum(axis=1)
    pool_mean = total_weighted_ofi.div(total_weight.where(total_weight.gt(0)))
    pool_variance = (
        effective_weights.mul(ofi_panel.sub(pool_mean, axis=0).pow(2))
        .sum(axis=1, min_count=1)
        .div(total_weight.where(total_weight.gt(0)))
    )
    dispersion = pool_variance.where(pool_variance.ge(0)).pow(0.5)
    dispersion = dispersion.where(dispersion_count.ge(2))

    results: dict[str, pd.DataFrame] = {}
    for fund_code in ofi_panel.columns:
        own_ofi = ofi_panel[fund_code]
        own_weight = effective_weights[fund_code]
        own_weighted_ofi = weighted_ofi[fund_code]
        peer_weight = total_weight.sub(own_weight, fill_value=np.nan)
        peer_weighted_ofi = total_weighted_ofi.sub(
            own_weighted_ofi, fill_value=np.nan
        )
        peer_count = dispersion_count.sub(valid_weight[fund_code].astype(int))
        market_ofi = peer_weighted_ofi.div(peer_weight.where(peer_weight.gt(0)))
        market_ofi = market_ofi.where(peer_count.ge(1))
        beta = _daily_rolling_beta(own_ofi, market_ofi)
        lead_market = market_ofi.groupby(market_ofi.index.normalize()).shift(1)
        own_valid = own_ofi.notna()
        results[fund_code] = pd.DataFrame(
            {
                f"idiosyncratic_ofi_{suffix}": (
                    own_ofi - beta * market_ofi
                ).where(own_valid),
                f"market_ofi_beta_{suffix}": beta.where(own_valid),
                f"lead_market_ofi_{suffix}": lead_market.where(own_valid),
                f"sector_ofi_dispersion_{suffix}": dispersion.where(own_valid),
            },
            index=ofi_panel.index,
        )
    return results


def calculate_group_ofi_factors(
    member_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if not member_frames:
        return {}

    amount_panel = pd.concat(
        {code: frame["amount"] for code, frame in member_frames.items()},
        axis=1,
    ).sort_index()
    results = {
        code: pd.DataFrame(np.nan, index=frame.index, columns=FACTOR_COLUMNS)
        for code, frame in member_frames.items()
    }
    for ofi_column in OFI_INPUT_COLUMNS:
        suffix = ofi_column.removeprefix("normalized_ofi_")
        if ofi_column == "normalized_mlofi_l5_60s":
            suffix = "mlofi_l5_60s"
        ofi_panel = pd.concat(
            {code: frame[ofi_column] for code, frame in member_frames.items()},
            axis=1,
        ).reindex(amount_panel.index)
        factor_results = _calculate_single_ofi_panel(
            ofi_panel, amount_panel, suffix
        )
        for fund_code, factor_frame in factor_results.items():
            results[fund_code].loc[:, factor_frame.columns] = factor_frame.reindex(
                results[fund_code].index
            )
    return results


def merge_factor_columns(
    existing: pd.DataFrame,
    factor_frame: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    result = existing.copy()
    timestamps = _timestamps_from_frame(result)
    if timestamps.has_duplicates:
        raise ValueError("CrossMarket output has duplicate minute timestamps")
    aligned = factor_frame.reindex(timestamps)
    incoming = timestamps.isin(factor_frame.index)

    for column in FACTOR_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        if overwrite:
            result.loc[incoming, column] = aligned.loc[incoming, column].to_numpy()
        else:
            missing = result[column].isna() & incoming
            result.loc[missing, column] = aligned.loc[missing, column].to_numpy()
    return result


def process_reference_group(
    reference_code: str,
    member_codes: tuple[str, ...],
    target_codes: tuple[str, ...],
    orderbook_files: dict[str, Path],
    crossmarket_files: dict[str, Path],
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    writable_targets = []
    for fund_code in target_codes:
        if fund_code in crossmarket_files:
            writable_targets.append(fund_code)
        else:
            results.append(
                {
                    "status": "missing_crossmarket_output",
                    "etf_code": fund_code,
                    "reference_code": reference_code,
                }
            )
    if not writable_targets:
        return results

    available_codes = [code for code in member_codes if code in orderbook_files]
    if len(available_codes) < 2:
        for fund_code in writable_targets:
            results.append(
                {
                    "status": "insufficient_peer_coverage",
                    "etf_code": fund_code,
                    "reference_code": reference_code,
                    "available_peer_files": len(available_codes),
                }
            )
        return results

    member_frames = {
        fund_code: filter_date_range(
            normalize_orderbook_frame(
                pd.read_parquet(
                    orderbook_files[fund_code],
                    columns=list(ORDERBOOK_READ_COLUMNS),
                )
            ),
            date_from,
            date_to,
        )
        for fund_code in available_codes
    }
    member_frames = {
        fund_code: frame
        for fund_code, frame in member_frames.items()
        if not frame.empty
    }
    if len(member_frames) < 2:
        for fund_code in writable_targets:
            results.append(
                {
                    "status": "insufficient_peer_coverage",
                    "etf_code": fund_code,
                    "reference_code": reference_code,
                    "available_peer_files": len(member_frames),
                }
            )
        return results

    factor_frames = calculate_group_ofi_factors(member_frames)
    for fund_code in writable_targets:
        output_path = crossmarket_files.get(fund_code)
        factor_frame = factor_frames.get(fund_code)
        if factor_frame is None:
            results.append(
                {
                    "status": "missing_orderbook_input",
                    "etf_code": fund_code,
                    "reference_code": reference_code,
                }
            )
            continue
        output = merge_factor_columns(
            pd.read_parquet(output_path), factor_frame, overwrite
        )
        output.to_parquet(output_path)
        results.append(
            {
                "status": "written",
                "etf_code": fund_code,
                "reference_code": reference_code,
                "output_path": output_path,
                "rows": len(output),
                "factor_non_null": int(
                    output.loc[:, FACTOR_COLUMNS].notna().sum().sum()
                ),
            }
        )
    return results


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
    if not args.enable_orderbook_factors:
        LOGGER.warning(
            "Orderbook-dependent CrossMarket OFI generation is disabled; "
            "rerun with --enable-orderbook-factors after coverage is available"
        )
        return 0

    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")

    mappings = load_mapping_records(args.mapping)
    orderbook_files = discover_symbol_files(args.orderbook_root, "orderbook")
    crossmarket_files = discover_symbol_files(
        args.crossmarket_root, "CrossMarket", allow_missing=True
    )
    requested = _requested_fund_codes(args)
    selected_codes = requested or sorted(mappings)
    missing_mapping = [code for code in selected_codes if code not in mappings]
    if missing_mapping:
        raise FileNotFoundError(
            "ETF codes missing from mapping: " + ", ".join(missing_mapping)
        )

    groups: dict[str, list[str]] = defaultdict(list)
    for fund_code, mapping in mappings.items():
        groups[mapping.reference_index_code].append(fund_code)
    selected_by_reference: dict[str, list[str]] = defaultdict(list)
    for fund_code in selected_codes:
        selected_by_reference[mappings[fund_code].reference_index_code].append(
            fund_code
        )

    jobs = [
        (
            reference_code,
            tuple(sorted(groups[reference_code])),
            tuple(sorted(target_codes)),
            orderbook_files,
            crossmarket_files,
            date_from,
            date_to,
            args.overwrite,
        )
        for reference_code, target_codes in selected_by_reference.items()
    ]
    LOGGER.info("Prepared %s same-index ETF groups", len(jobs))
    worker_count = max(1, int(args.workers))
    all_results: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    if worker_count == 1:
        for job in jobs:
            try:
                all_results.extend(process_reference_group(*job))
            except Exception as exc:  # noqa: BLE001
                failures.append((job[0], str(exc)))
                LOGGER.exception("Failed to process reference %s", job[0])
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(process_reference_group, *job): job[0]
                for job in jobs
            }
            for future in as_completed(future_map):
                reference_code = future_map[future]
                try:
                    all_results.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append((reference_code, str(exc)))
                    LOGGER.exception("Failed to process reference %s", reference_code)

    counts = pd.Series(
        [result["status"] for result in all_results], dtype="object"
    ).value_counts()
    LOGGER.info("Result counts: %s", counts.to_dict())
    if failures:
        LOGGER.error("Failed %s reference groups", len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
