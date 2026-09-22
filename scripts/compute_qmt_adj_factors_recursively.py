#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute and fill adj_factor for QMT 1d and 1min data using xtdata.get_divid_factors and recursive calculation."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_QMT_1D_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1d_qmt")
DEFAULT_QMT_1MIN_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_qmt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("qmt_recursive_adj")


def parse_date(value: str | None) -> pd.Timestamp | None:
    """Parse date string into a normalized pd.Timestamp."""
    if value is None:
        return None
    clean = re.sub(r"\D", "", str(value).strip())
    if not clean:
        return None
    return pd.to_datetime(clean, errors="coerce").normalize()


def fetch_all_qmt_divid_factors(symbols: list[str]) -> dict[str, dict[pd.Timestamp, float]]:
    """Fetch dividend / split factors from QMT's xtdata for given symbols."""
    try:
        from xtquant import xtdata
        xtdata.enable_hello = False
    except ImportError as exc:
        LOGGER.error("Failed to import xtquant: %s", exc)
        raise

    LOGGER.info("Fetching get_divid_factors from QMT for %d symbols...", len(symbols))
    started = time.monotonic()
    div_dict: dict[str, dict[pd.Timestamp, float]] = {}
    total_records = 0

    for idx, sym in enumerate(symbols, start=1):
        try:
            df = xtdata.get_divid_factors(sym)
            if df is not None and not df.empty and "dr" in df.columns:
                df.index = pd.to_datetime(df.index).normalize()
                # dr is the ex-dividend adjustment factor multiplier
                div_dict[sym] = df["dr"].to_dict()
                total_records += len(df)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Error getting divid factors for %s: %s", sym, exc)

        if idx % 200 == 0 or idx == len(symbols):
            elapsed = time.monotonic() - started
            LOGGER.info("Fetched %d/%d symbols (%.1f%%), %d with dividends, elapsed: %.1fs",
                        idx, len(symbols), idx / len(symbols) * 100, len(div_dict), elapsed)

    LOGGER.info("Done fetching divid factors: %d symbols have %d dividend/split events.", len(div_dict), total_records)
    return div_dict


def process_symbol_recursive(
    symbol: str,
    path_1d: str | None,
    path_1min: str | None,
    div_map: dict[pd.Timestamp, float] | None,
    date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
    ffill_subsequent: bool = True,
    dry_run: bool = False,
) -> tuple[str, bool, str, int, int]:
    """Recursively compute adj_factor and update both 1d and 1min parquet files.

    Returns (symbol, success, message, rows_1d_updated, rows_1min_updated).
    """
    p1d = Path(path_1d) if path_1d else None
    p1m = Path(path_1min) if path_1min else None

    if p1d is None or not p1d.exists():
        return (symbol, False, f"1d file not found: {p1d}", 0, 0)

    try:
        df_1d = pd.read_parquet(p1d)
        if df_1d.empty:
            return (symbol, True, "Empty 1d file", 0, 0)

        trade_dates_1d = pd.to_datetime(df_1d.index).normalize()
        unique_dates_1d = pd.DatetimeIndex(trade_dates_1d.unique()).sort_values()

        div_map = div_map or {}

        # Determine recursion starting point and dates to update
        if date_from is not None:
            prior_dates = unique_dates_1d[unique_dates_1d < date_from]
            if len(prior_dates) > 0:
                base_date = prior_dates[-1]
                base_adj = float(df_1d.loc[base_date, "adj_factor"]) if "adj_factor" in df_1d.columns else 1.0
            else:
                base_adj = 1.0

            # Target date range
            mask_dates = unique_dates_1d >= date_from
            if date_to is not None and not ffill_subsequent:
                mask_dates &= (unique_dates_1d <= date_to)

            calc_dates = unique_dates_1d[mask_dates]
            if len(calc_dates) == 0:
                return (symbol, True, "No 1d dates in range", 0, 0)

            # Recursive step
            curr_adj = base_adj
            computed_adj_by_date: dict[pd.Timestamp, float] = {}
            for d in calc_dates:
                # If date_to is specified and d > date_to, ffill without applying new events if requested
                if d in div_map and (date_to is None or d <= date_to):
                    curr_adj = float(curr_adj * div_map[d])
                computed_adj_by_date[d] = curr_adj

            # Mask for 1d dataframe
            mask_1d = trade_dates_1d >= date_from
            if date_to is not None and not ffill_subsequent:
                mask_1d &= (trade_dates_1d <= date_to)
        else:
            # Full history recursion from t0
            t0 = unique_dates_1d[0]
            base_adj = float(df_1d.loc[t0, "adj_factor"]) if "adj_factor" in df_1d.columns else 1.0
            curr_adj = base_adj
            computed_adj_by_date = {t0: curr_adj}
            for d in unique_dates_1d[1:]:
                if d in div_map:
                    curr_adj = float(curr_adj * div_map[d])
                computed_adj_by_date[d] = curr_adj

            mask_1d = np.ones(len(df_1d), dtype=bool)

        # Update 1d
        if "adj_factor" not in df_1d.columns:
            df_1d["adj_factor"] = 1.0

        updated_1d_vals = trade_dates_1d.map(computed_adj_by_date).to_numpy(dtype=float)
        df_1d.loc[mask_1d, "adj_factor"] = updated_1d_vals[mask_1d]

        rows_1d_updated = int(mask_1d.sum())
        if not dry_run:
            tmp_1d = p1d.with_name(f".{p1d.name}.{os.getpid()}.tmp")
            df_1d.to_parquet(tmp_1d)
            os.replace(tmp_1d, p1d)

        # Update 1min if provided
        rows_1min_updated = 0
        if p1m and p1m.exists():
            df_1m = pd.read_parquet(p1m)
            if not df_1m.empty:
                if isinstance(df_1m.index, pd.MultiIndex):
                    trade_dates_1m = pd.to_datetime(df_1m.index.get_level_values(0)).normalize()
                else:
                    trade_dates_1m = pd.to_datetime(df_1m.index).normalize()

                mask_1m = np.ones(len(df_1m), dtype=bool)
                if date_from is not None:
                    mask_1m &= (trade_dates_1m >= date_from)
                if date_to is not None and not ffill_subsequent:
                    mask_1m &= (trade_dates_1m <= date_to)

                if mask_1m.any():
                    if "adj_factor" not in df_1m.columns:
                        df_1m["adj_factor"] = 1.0
                    updated_1m_vals = trade_dates_1m.map(computed_adj_by_date).to_numpy(dtype=float)
                    df_1m.loc[mask_1m, "adj_factor"] = updated_1m_vals[mask_1m]
                    rows_1min_updated = int(mask_1m.sum())

                    if not dry_run:
                        tmp_1m = p1m.with_name(f".{p1m.name}.{os.getpid()}.tmp")
                        df_1m.to_parquet(tmp_1m)
                        os.replace(tmp_1m, p1m)

        return (symbol, True, "OK", rows_1d_updated, rows_1min_updated)
    except Exception as exc:  # noqa: BLE001
        return (symbol, False, str(exc), 0, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recursively compute adj_factor using QMT get_divid_factors.")
    parser.add_argument("--qmt-1d-root", type=Path, default=DEFAULT_QMT_1D_ROOT, help="Path to etf_1d_qmt")
    parser.add_argument("--qmt-1min-root", type=Path, default=DEFAULT_QMT_1MIN_ROOT, help="Path to etf_1min_qmt")
    parser.add_argument("--date-from", type=str, default=None, help="Start date (e.g. 2026-09-02)")
    parser.add_argument("--date-to", type=str, default=None, help="End date (e.g. 2026-09-11)")
    parser.add_argument("--ffill-subsequent", action="store_true", default=True, help="Forward-fill after date-to")
    parser.add_argument("--no-ffill-subsequent", dest="ffill_subsequent", action="store_false", help="Do not forward-fill after date-to")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    parser.add_argument("--symbols-file", type=Path, default=None, help="File of symbols")
    parser.add_argument("--workers", type=int, default=8, help="Worker count (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without writing files")

    args = parser.parse_args()

    date_from = parse_date(args.date_from)
    date_to = parse_date(args.date_to)

    # Discover symbols from 1d directory
    target_files_1d = sorted(args.qmt_1d_root.glob("*.parquet"))
    all_symbols = [f.name.replace(".parquet", "").upper() for f in target_files_1d]

    if args.symbols:
        requested = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        target_symbols = [s for s in all_symbols if s in requested]
    elif args.symbols_file and args.symbols_file.exists():
        requested = {line.strip().upper() for line in args.symbols_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        target_symbols = [s for s in all_symbols if s in requested]
    else:
        target_symbols = all_symbols

    LOGGER.info("Target symbols to process: %d (from %s)", len(target_symbols), args.qmt_1d_root)
    if not target_symbols:
        LOGGER.warning("No symbols to process.")
        return 0

    # 1. Fetch get_divid_factors from QMT in the main process
    div_data = fetch_all_qmt_divid_factors(target_symbols)

    # 2. Build tasks
    tasks = []
    for sym in target_symbols:
        path_1d = args.qmt_1d_root / f"{sym}.parquet"
        path_1min = args.qmt_1min_root / f"{sym}.parquet"
        div_map = div_data.get(sym)
        tasks.append((
            sym,
            str(path_1d),
            str(path_1min) if path_1min.exists() else None,
            div_map,
            date_from,
            date_to,
            args.ffill_subsequent,
            args.dry_run,
        ))

    # 3. Execute tasks
    started = time.monotonic()
    total = len(tasks)
    succeeded = 0
    failed = 0
    total_1d_rows = 0
    total_1min_rows = 0
    last_log_time = started

    LOGGER.info("Starting recursive calculation with %d workers (dry_run=%s)...", args.workers, args.dry_run)

    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            res = process_symbol_recursive(*task)
            if res[1]:
                succeeded += 1
                total_1d_rows += res[3]
                total_1min_rows += res[4]
            else:
                failed += 1
                LOGGER.warning("Failed on %s: %s", res[0], res[2])
            now = time.monotonic()
            if now - last_log_time >= 10.0 or idx == total:
                elapsed = now - started
                LOGGER.info("Progress: %d/%d (%.1f%%), %d failed, elapsed: %.1fs",
                            idx, total, idx / total * 100, failed, elapsed)
                last_log_time = now
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_symbol_recursive, *task): task[0] for task in tasks}
            for idx, future in enumerate(as_completed(futures), start=1):
                sym = futures[future]
                try:
                    res = future.result()
                    if res[1]:
                        succeeded += 1
                        total_1d_rows += res[3]
                        total_1min_rows += res[4]
                    else:
                        failed += 1
                        LOGGER.warning("Failed on %s: %s", sym, res[2])
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    LOGGER.exception("Exception on %s: %s", sym, exc)

                now = time.monotonic()
                if now - last_log_time >= 10.0 or idx == total:
                    elapsed = now - started
                    LOGGER.info("Progress: %d/%d (%.1f%%), %d failed, elapsed: %.1fs",
                                idx, total, idx / total * 100, failed, elapsed)
                    last_log_time = now

    total_time = time.monotonic() - started
    LOGGER.info("Completed in %.2fs: %d/%d succeeded, %d failed. Updated rows: 1d=%d, 1min=%d",
                total_time, succeeded, total, failed, total_1d_rows, total_1min_rows)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
