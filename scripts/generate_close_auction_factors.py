"""Generate next-trading-day factors from the 14:57--15:00 closing auction.

The transaction-to-order fields are explicitly named ``*_proxy`` because the
source reports do not disclose their exact numerator and denominator.  They
measure the closing-match notional against orders submitted in that window,
linked by exchange order identifier. Full-day orders establish linkage; delayed
matches before 15:05 are included, while post-close fixed-price trades are not.
Shanghai ETFs before 2026-07-06 have no closing auction and are not applicable.
"""

from __future__ import annotations

import argparse
import logging
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if __package__:
    from scripts.auction_tick_cache import AuctionTickCache
    from scripts.generate_auction_factors import (
        ASSET_TYPES,
        DEFAULT_AUCTION_CACHE_ROOT,
        DEFAULT_ETF_MINUTE_ROOT,
        DEFAULT_STOCK_MINUTE_ROOT,
        DEFAULT_TICK_ROOT,
        build_asset_universe,
        discover_trade_date_dirs,
        group_symbol_paths,
        load_requested_codes,
        normalize_trade_date_arg,
    )
else:
    from auction_tick_cache import AuctionTickCache
    from generate_auction_factors import (
        ASSET_TYPES,
        DEFAULT_AUCTION_CACHE_ROOT,
        DEFAULT_ETF_MINUTE_ROOT,
        DEFAULT_STOCK_MINUTE_ROOT,
        DEFAULT_TICK_ROOT,
        build_asset_universe,
        discover_trade_date_dirs,
        group_symbol_paths,
        load_requested_codes,
        normalize_trade_date_arg,
    )


LOGGER = logging.getLogger("generate_close_auction_factors")

DEFAULT_STOCK_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\stock-factors\stock_close_auction_factors"
)
DEFAULT_ETF_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-factors\etf_close_auction_factors"
)

KEY_COLUMNS = ["trade_date", "available_time", "ts_code"]
DIAGNOSTIC_COLUMNS = [
    "close_auction_order_count",
    "close_auction_cancel_count",
    "close_auction_match_count",
    "close_auction_event_reconstruction_ok",
]
REFERENCE_COLUMNS = [
    "source_available_time",
    "close_auction_submitted_buy_notional",
    "close_auction_submitted_sell_notional",
    "close_auction_cancel_buy_notional",
    "close_auction_cancel_sell_notional",
    "close_auction_matched_notional",
    "close_auction_attributed_buy_match_notional",
    "close_auction_attributed_sell_match_notional",
]
FACTOR_COLUMNS = [
    "close_auction_order_imbalance",
    "close_auction_net_order_imbalance",
    "close_auction_cancel_notional_ratio",
    "close_auction_cancel_imbalance",
    "close_auction_buy_match_to_submitted_notional_proxy",
    "close_auction_sell_match_to_submitted_notional_proxy",
    "close_auction_match_to_submitted_notional_proxy",
]
SESSION_COLUMNS = ["auction_applicable", "auction_status"]
OUTPUT_COLUMNS = KEY_COLUMNS + SESSION_COLUMNS + DIAGNOSTIC_COLUMNS + REFERENCE_COLUMNS + FACTOR_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tick-root", type=Path, default=DEFAULT_TICK_ROOT)
    parser.add_argument("--asset-type", choices=ASSET_TYPES, default="both")
    parser.add_argument(
        "--stock-minute-root", type=Path, default=DEFAULT_STOCK_MINUTE_ROOT
    )
    parser.add_argument("--etf-minute-root", type=Path, default=DEFAULT_ETF_MINUTE_ROOT)
    parser.add_argument(
        "--stock-output-root", type=Path, default=DEFAULT_STOCK_OUTPUT_ROOT
    )
    parser.add_argument("--etf-output-root", type=Path, default=DEFAULT_ETF_OUTPUT_ROOT)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--auction-cache-root", type=Path, default=DEFAULT_AUCTION_CACHE_ROOT)
    parser.add_argument("--refresh-auction-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--calendar-file", type=Path, default=Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet"),
                        help="Parquet with trade_date from a complete market trading calendar; never inferred from tick folders.")
    parser.add_argument(
        "--refresh-existing-factors",
        action="store_true",
        help="Recompute and replace requested dates after factor formula changes.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _signed_imbalance(buy: float, sell: float) -> float:
    return _safe_ratio(buy - sell, buy + sell)


def _date_in_requested_range(
    date_text: str, date_from: str | None, date_to: str | None
) -> bool:
    return (date_from is None or date_text >= date_from) and (
        date_to is None or date_text <= date_to
    )


def _existing_trade_dates(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    existing = pd.read_parquet(output_path, columns=["trade_date"])
    return set(pd.to_datetime(existing["trade_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d"))


def _output_uses_current_schema(output_path: Path) -> bool:
    if not output_path.exists():
        return True
    try:
        columns = set(pq.ParquetFile(output_path).schema_arrow.names)
    except (OSError, ValueError):
        return False
    return set(OUTPUT_COLUMNS).issubset(columns)


def _valid_orders(orders: pd.DataFrame, exchange: str) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    work = orders.copy()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce")
    work["side"] = work["side"].astype(str).str.upper()
    work["order_type"] = work["order_type"].astype(str).str.upper()
    valid = work[
        work["order_id"].notna()
        & work["side"].isin(["B", "S"])
        & work["price"].gt(0)
        & work["quantity"].gt(0)
    ].copy()
    if exchange == "SH":
        adds = valid.loc[valid["order_type"].eq("A")].copy()
        cancels = valid.loc[valid["order_type"].eq("D")].copy()
    else:
        adds = valid.copy()
        cancels = valid.iloc[0:0].copy()
    adds["notional"] = adds["price"] * adds["quantity"]
    unique_ids = not bool(adds["order_id"].duplicated().any())
    return adds, cancels, unique_ids


def _sz_cancellations(transactions: pd.DataFrame) -> pd.DataFrame:
    cancelled = transactions.loc[transactions["trade_code"].eq("C")].copy()
    ask_present = cancelled["ask_order_id"].gt(0)
    bid_present = cancelled["bid_order_id"].gt(0)
    cancelled = cancelled.loc[ask_present ^ bid_present].copy()
    cancelled["order_id"] = cancelled["ask_order_id"].where(
        cancelled["ask_order_id"].gt(0), cancelled["bid_order_id"]
    )
    cancelled["side"] = np.where(cancelled["ask_order_id"].gt(0), "S", "B")
    return cancelled


def _cancel_notional_by_side(
    adds: pd.DataFrame, cancellations: pd.DataFrame
) -> tuple[dict[str, float], int, bool]:
    if cancellations.empty:
        return {"B": 0.0, "S": 0.0}, 0, True
    lookup = adds[["order_id", "side", "price", "quantity"]].rename(
        columns={"side": "add_side", "price": "add_price", "quantity": "add_quantity"}
    )
    cancelled = cancellations.merge(lookup, on="order_id", how="left", validate="many_to_one")
    matched = cancelled["add_price"].notna() & cancelled["add_quantity"].gt(0)
    all_matched = bool(matched.all())
    cancelled = cancelled.loc[matched].copy()
    if cancelled.empty:
        return {"B": 0.0, "S": 0.0}, 0, False
    cancelled["quantity"] = pd.to_numeric(cancelled["quantity"], errors="coerce")
    quantities_ok = bool((cancelled["quantity"].gt(0) & np.isfinite(cancelled["quantity"])).all())
    cancelled = cancelled.loc[cancelled["quantity"].gt(0)].copy()
    cancelled["notional"] = cancelled["add_price"] * cancelled["quantity"]
    totals = {
        side: float(cancelled.loc[cancelled["add_side"].eq(side), "notional"].sum())
        for side in ("B", "S")
    }
    cumulative = cancelled.groupby("order_id")["quantity"].sum()
    capacities = adds.set_index("order_id")["quantity"].reindex(cumulative.index)
    side_ok = "side" not in cancelled or bool(cancelled["side"].eq(cancelled["add_side"]).all())
    valid = all_matched and quantities_ok and side_ok and bool(cumulative.le(capacities).all())
    return totals, len(cancelled), valid


def _matched_notional_by_submitted_side(
    adds: pd.DataFrame, matches: pd.DataFrame
) -> dict[str, float]:
    lookup = adds[["order_id", "side"]]
    totals: dict[str, float] = {}
    for side, id_column in (("B", "bid_order_id"), ("S", "ask_order_id")):
        submitted = lookup.loc[lookup["side"].eq(side), ["order_id"]]
        linked = matches.merge(
            submitted,
            left_on=id_column,
            right_on="order_id",
            how="inner",
            validate="many_to_one",
        )
        totals[side] = float(linked["notional"].sum())
    return totals


def calculate_daily_close_auction_factors(
    orders: pd.DataFrame,
    transactions: pd.DataFrame,
    ts_code: str,
    next_trade_date: str | None,
    *,
    asset_type: str = "stock",
) -> dict[str, object]:
    """Use full-day order history, but retain the 14:57 new-order cohort."""
    if asset_type not in ("stock", "etf"):
        raise ValueError("asset_type must be stock or etf")
    if orders.empty:
        raise ValueError(f"Empty close-auction order frame for {ts_code}")
    trade_day = pd.Timestamp(orders["trade_time"].iloc[0]).normalize()
    close_time = trade_day + pd.Timedelta(hours=15)
    window_start = close_time - pd.Timedelta(minutes=3)
    if next_trade_date is not None and pd.Timestamp(next_trade_date).normalize() <= trade_day:
        raise ValueError("next_trade_date must be strictly after the source date")
    row: dict[str, object] = {column: np.nan for column in OUTPUT_COLUMNS}
    row.update(
        {
            "trade_date": trade_day.strftime("%Y-%m-%d"),
            "available_time": (
                pd.Timestamp(next_trade_date) + pd.Timedelta(hours=9, minutes=15)
                if next_trade_date is not None
                else pd.NaT
            ),
            "ts_code": ts_code,
            "source_available_time": close_time,
            "close_auction_event_reconstruction_ok": False,
            "auction_applicable": True,
            "auction_status": "invalid_event_coverage",
        }
    )
    exchange = ts_code.rsplit(".", 1)[-1].upper()
    # SSE fund closing auction took effect on 2026-07-06; earlier closes
    # are continuous trading, not missing auction observations.
    if asset_type == "etf" and exchange == "SH" and trade_day < pd.Timestamp("2026-07-06"):
        row.update(auction_applicable=False, auction_status="not_applicable")
        return row
    history = orders.loc[orders["trade_time"].ge(trade_day) & orders["trade_time"].lt(close_time)]
    all_adds, order_cancellations, unique_ids = _valid_orders(history, exchange)
    adds = all_adds.loc[all_adds["trade_time"].ge(window_start)].copy()
    if not unique_ids:
        raise ValueError(f"Duplicate close-auction order identifiers for {ts_code}")
    if exchange == "SZ":
        cancel_events = transactions.loc[transactions["trade_time"].ge(window_start) & transactions["trade_time"].lt(close_time)]
        cancellations = _sz_cancellations(cancel_events)
        cancel_shape_ok = len(cancellations) == int(cancel_events["trade_code"].eq("C").sum())
    else:
        cancellations = order_cancellations.loc[order_cancellations["trade_time"].ge(window_start)]
        cancel_shape_ok = True
    _, _, cancellation_ok = _cancel_notional_by_side(all_adds, cancellations)
    cohort_cancellations = cancellations.loc[cancellations["order_id"].isin(adds["order_id"])]
    cancel_notional, cancel_count, _ = _cancel_notional_by_side(adds, cohort_cancellations)

    work = transactions.copy()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce")
    matches = work.loc[
        work["trade_time"].ge(close_time)
        & work["trade_time"].lt(close_time + pd.Timedelta(minutes=5))
        & ~work["trade_code"].eq("C")
        & work["price"].gt(0)
        & work["quantity"].gt(0)
        & work["ask_order_id"].gt(0)
        & work["bid_order_id"].gt(0)
    ].copy()
    matches["notional"] = matches["price"] * matches["quantity"]
    submitted = {
        side: float(adds.loc[adds["side"].eq(side), "notional"].sum())
        for side in ("B", "S")
    }
    attributed_matches = _matched_notional_by_submitted_side(adds, matches)
    linkage_ok = all(
        matches[id_column].isin(all_adds.loc[all_adds["side"].eq(side), "order_id"]).all()
        for side, id_column in (("B", "bid_order_id"), ("S", "ask_order_id"))
    )
    valid = bool(unique_ids and cancellation_ok and cancel_shape_ok and linkage_ok
                 and not matches.empty and matches["price"].nunique() == 1)
    if not matches.empty:
        row["source_available_time"] = matches["trade_time"].max()
    row["auction_status"] = "ok" if valid else "invalid_event_coverage"

    row.update(
        {
            "close_auction_order_count": int(len(adds)),
            "close_auction_cancel_count": int(cancel_count),
            "close_auction_match_count": int(len(matches)),
            "close_auction_event_reconstruction_ok": valid,
            "close_auction_submitted_buy_notional": submitted["B"],
            "close_auction_submitted_sell_notional": submitted["S"],
            "close_auction_cancel_buy_notional": cancel_notional["B"],
            "close_auction_cancel_sell_notional": cancel_notional["S"],
            "close_auction_matched_notional": float(matches["notional"].sum()),
            "close_auction_attributed_buy_match_notional": attributed_matches["B"],
            "close_auction_attributed_sell_match_notional": attributed_matches["S"],
        }
    )
    submitted_total = submitted["B"] + submitted["S"]
    cancelled_total = cancel_notional["B"] + cancel_notional["S"]
    attributed_total = attributed_matches["B"] + attributed_matches["S"]
    row["close_auction_order_imbalance"] = _signed_imbalance(
        submitted["B"], submitted["S"]
    )
    row["close_auction_net_order_imbalance"] = _signed_imbalance(
        max(submitted["B"] - cancel_notional["B"], 0.0),
        max(submitted["S"] - cancel_notional["S"], 0.0),
    )
    row["close_auction_cancel_notional_ratio"] = _safe_ratio(
        cancelled_total, submitted_total
    )
    row["close_auction_cancel_imbalance"] = _signed_imbalance(
        cancel_notional["S"], cancel_notional["B"]
    )
    row["close_auction_buy_match_to_submitted_notional_proxy"] = _safe_ratio(
        attributed_matches["B"], submitted["B"]
    )
    row["close_auction_sell_match_to_submitted_notional_proxy"] = _safe_ratio(
        attributed_matches["S"], submitted["S"]
    )
    row["close_auction_match_to_submitted_notional_proxy"] = _safe_ratio(
        attributed_total, submitted_total
    )
    if not row["close_auction_event_reconstruction_ok"]:
        # Unknown event coverage must not masquerade as a valid zero signal.
        for column in FACTOR_COLUMNS + REFERENCE_COLUMNS[1:] + DIAGNOSTIC_COLUMNS[:3]:
            row[column] = np.nan
    return row


def merge_symbol_output(
    output_path: Path,
    requested: pd.DataFrame,
    overwrite: bool,
    replace_existing_dates: set[str] | None = None,
) -> pd.DataFrame:
    if output_path.exists():
        existing = pd.read_parquet(output_path).reindex(columns=OUTPUT_COLUMNS)
    else:
        existing = pd.DataFrame(columns=OUTPUT_COLUMNS)
    requested_dates = set(requested["trade_date"])
    replace_dates = requested_dates if overwrite else ((replace_existing_dates or set()) & requested_dates)
    if replace_dates:
        existing = existing.loc[~existing["trade_date"].astype(str).isin(replace_dates)]
        additions = requested
    else:
        additions = requested.loc[~requested["trade_date"].astype(str).isin(set(existing["trade_date"].astype(str)))]
    combined = pd.concat([existing, additions], ignore_index=True)
    if combined.empty:
        return combined.reindex(columns=OUTPUT_COLUMNS)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("available_time", "source_available_time"):
        combined[column] = pd.to_datetime(combined[column], errors="coerce")
    return combined.sort_values("trade_date", kind="mergesort").drop_duplicates(
        "trade_date", keep="last"
    ).reset_index(drop=True)[OUTPUT_COLUMNS]


def process_symbol_series(
    asset_type: str,
    ts_code: str,
    symbol_paths: list[Path],
    output_root: Path,
    date_from: str | None,
    date_to: str | None,
    overwrite: bool,
    next_trade_dates: dict[str, str],
    auction_cache_root: Path | None,
    refresh_auction_cache: bool,
    refresh_existing_factors: bool = False,
) -> tuple[str, Path, int]:
    output_path = output_root / f"{ts_code}.parquet"
    existing_dates = _existing_trade_dates(output_path)
    if output_path.exists():
        old = pd.read_parquet(output_path)
        if "available_time" in old:
            for date, available in zip(old["trade_date"], old["available_time"]):
                key = pd.Timestamp(date).strftime("%Y%m%d")
                expected = next_trade_dates.get(key)
                if expected is not None and (pd.isna(available) or pd.Timestamp(available).normalize() != pd.Timestamp(expected)):
                    existing_dates.discard(pd.Timestamp(date).strftime("%Y-%m-%d"))
    output_uses_current_schema = _output_uses_current_schema(output_path)
    requested_paths = [
        path
        for path in sorted(symbol_paths, key=lambda item: item.parent.name)
        if _date_in_requested_range(path.parent.name, date_from, date_to)
    ]
    replace_existing_dates: set[str] = set()
    if overwrite or refresh_existing_factors or not output_uses_current_schema:
        missing_paths = requested_paths
        replace_existing_dates = {
            pd.Timestamp(path.parent.name).strftime("%Y-%m-%d")
            for path in requested_paths
        }
        if not output_uses_current_schema and not (overwrite or refresh_existing_factors):
            LOGGER.info("%s backfilling current output schema for %s dates", ts_code, len(missing_paths))
    else:
        missing_paths = [
            path
            for path in requested_paths
            if pd.Timestamp(path.parent.name).strftime("%Y-%m-%d") not in existing_dates
        ]
        replace_existing_dates = {pd.Timestamp(path.parent.name).strftime("%Y-%m-%d") for path in missing_paths}
    if not missing_paths:
        LOGGER.info("%s skipped: requested=%s existing=%s missing=0", ts_code, len(requested_paths), len(existing_dates))
        return asset_type, output_path, 0
    cache = AuctionTickCache(auction_cache_root, refresh=refresh_auction_cache, validate_source=True)
    records: list[dict[str, object]] = []
    failed_dates = []
    for path in missing_paths:
        try:
            orders = cache.load_close_order_history(path)
            transactions = cache.load_close_transactions(path)
            records.append(
                calculate_daily_close_auction_factors(
                    orders,
                    transactions,
                    ts_code,
                    next_trade_dates.get(path.parent.name),
                    asset_type=asset_type,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failed_dates.append({"date": path.parent.name, "error": str(exc)})
            LOGGER.warning("%s skipped %s: %s", ts_code, path.parent.name, exc)
    requested = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    combined = merge_symbol_output(
        output_path,
        requested,
        overwrite,
        replace_existing_dates=replace_existing_dates,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.parquet")
    try:
        combined.to_parquet(temp, index=False)
        os.replace(temp, output_path)
    finally:
        temp.unlink(missing_ok=True)
    report_path = output_path.with_suffix(".failures.json")
    report_path.write_text(json.dumps(failed_dates, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("%s cache: hits=%s rebuilds=%s", ts_code, cache.stats.hits, cache.stats.rebuilds)
    if failed_dates:
        raise ValueError(f"{ts_code}: {len(failed_dates)} dates failed; successful dates saved; see {report_path}")
    return asset_type, output_path, len(requested)


def main() -> int:
    args = parse_args()
    configure_logging()
    date_from = normalize_trade_date_arg(args.date_from)
    date_to = normalize_trade_date_arg(args.date_to)
    if date_from and date_to and date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    assets = build_asset_universe(
        args.asset_type,
        args.stock_minute_root,
        args.etf_minute_root,
        load_requested_codes(args.symbols, args.symbols_file),
    )
    if args.limit is not None:
        assets = assets[: args.limit]
    date_dirs = discover_trade_date_dirs(args.tick_root, None)
    calendar = pd.read_parquet(args.calendar_file, columns=["trade_date"])
    dates = pd.DatetimeIndex(pd.to_datetime(calendar["trade_date"].astype(str), errors="raise")).normalize().unique().sort_values()
    next_trade_dates = {left.strftime("%Y%m%d"): right.strftime("%Y%m%d") for left, right in zip(dates[:-1], dates[1:])}
    requested_date_dirs = [
        path
        for path in date_dirs
        if _date_in_requested_range(path.name, date_from, date_to)
    ]
    grouped = group_symbol_paths(requested_date_dirs, {code for _, code, _ in assets})
    tasks = [
        (kind, symbol, grouped.get(code, []))
        for kind, code, symbol in assets
        if grouped.get(code)
    ]
    output_roots = {"stock": args.stock_output_root, "etf": args.etf_output_root}
    LOGGER.info("Processing %s close-auction symbols", len(tasks))
    failures: list[tuple[str, str]] = []
    written = 0
    worker_count = max(1, args.workers)
    task_kwargs = (
        date_from,
        date_to,
        args.overwrite,
        next_trade_dates,
        args.auction_cache_root,
        args.refresh_auction_cache,
        args.refresh_existing_factors,
    )
    if worker_count == 1:
        for kind, symbol, paths in tasks:
            try:
                _, output_path, row_count = process_symbol_series(kind, symbol, paths, output_roots[kind], *task_kwargs)
                written += int(row_count > 0)
                LOGGER.info("Wrote %s requested rows to %s", row_count, output_path)
            except Exception as exc:  # noqa: BLE001
                failures.append((symbol, str(exc)))
                LOGGER.exception("Failed to process %s", symbol)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_symbol_series, kind, symbol, paths, output_roots[kind], *task_kwargs): symbol
                for kind, symbol, paths in tasks
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    _, output_path, row_count = future.result()
                    written += int(row_count > 0)
                    LOGGER.info("Wrote %s requested rows to %s", row_count, output_path)
                except Exception as exc:  # noqa: BLE001
                    failures.append((symbol, str(exc)))
                    LOGGER.exception("Failed to process %s", symbol)
    LOGGER.info("Completed: %s symbol files written, %s failures", written, len(failures))
    if failures:
        for symbol, error in failures[:20]:
            LOGGER.error("%s: %s", symbol, error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
