#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synchronize adj_factor from etf_daily.parquet into QMT datasets (1d, 1min, tick)."""

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
import pyarrow.parquet as pq

DEFAULT_DAILY_PATH = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_QMT_1D_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1d_qmt")
DEFAULT_QMT_1MIN_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_qmt")
DEFAULT_QMT_TICK_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_tick_qmt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("sync_adj_factors")


def parse_date(value: str | None) -> pd.Timestamp | None:
    """Parse date string into a normalized pd.Timestamp."""
    if value is None:
        return None
    clean = re.sub(r"\D", "", str(value).strip())
    if not clean:
        return None
    return pd.to_datetime(clean, errors="coerce").normalize()


def load_daily_adj_factors(daily_path: Path) -> dict[str, pd.Series]:
    """Read etf_daily.parquet and return mapping of ts_code -> pd.Series(adj_factor, index=trade_date)."""
    LOGGER.info("Loading daily adjustment factors from %s", daily_path)
    df = pd.read_parquet(daily_path, columns=["adj_factor"]).reset_index()
    df["ts_code"] = df["ts_code"].astype(str).str.strip().str.upper()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date", "adj_factor"]).sort_values(["ts_code", "trade_date"])
    df = df.drop_duplicates(["ts_code", "trade_date"], keep="last")

    adj_dict: dict[str, pd.Series] = {}
    for ts_code, group in df.groupby("ts_code"):
        series = group.set_index("trade_date")["adj_factor"].astype(float)
        adj_dict[ts_code] = series

    LOGGER.info("Loaded adj_factors for %s symbols from etf_daily", len(adj_dict))
    return adj_dict


def process_single_file(
    target_type: str,
    file_path: str,
    symbol: str,
    daily_series: pd.Series | None,
    date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
    ffill_subsequent: bool = False,
    dry_run: bool = False,
) -> tuple[str, str, int, bool, str]:
    """Process a single parquet file to update or add adj_factor.

    Returns (target_type, symbol, row_count, success, message).
    """
    path = Path(file_path)
    if not path.exists():
        return (target_type, symbol, 0, False, f"File does not exist: {path}")

    try:
        df = pd.read_parquet(path)
        row_count = len(df)
        if row_count == 0:
            return (target_type, symbol, 0, True, "Empty dataframe, skipped")

        # Extract dates from index
        if isinstance(df.index, pd.MultiIndex):
            if "trade_date" in df.index.names:
                trade_dates = pd.to_datetime(df.index.get_level_values("trade_date")).normalize()
            else:
                trade_dates = pd.to_datetime(df.index.get_level_values(0)).normalize()
        else:
            trade_dates = pd.to_datetime(df.index).normalize()

        # Date range filter mask
        mask = np.ones(row_count, dtype=bool)
        if date_from is not None:
            mask &= (trade_dates >= date_from)
        if date_to is not None and not ffill_subsequent:
            mask &= (trade_dates <= date_to)

        if not mask.any():
            return (target_type, symbol, 0, True, "No rows in requested date range, skipped")

        unique_dates = pd.DatetimeIndex(trade_dates.unique())

        if daily_series is not None and not daily_series.empty:
            full_series = daily_series.reindex(daily_series.index.union(unique_dates)).ffill().bfill()
            adj_map = full_series.to_dict()
            new_adj = trade_dates.map(adj_map).to_numpy(dtype=float)
            # If any NaN remain, fallback to 1.0
            if np.isnan(new_adj).any():
                new_adj = np.nan_to_num(new_adj, nan=1.0)
        else:
            # Symbol not in etf_daily at all, keep existing or default to 1.0
            if "adj_factor" in df.columns:
                new_adj = df["adj_factor"].to_numpy(dtype=float)
            else:
                new_adj = np.ones(row_count, dtype=float)

        if not dry_run:
            if date_from is not None or date_to is not None:
                if "adj_factor" not in df.columns:
                    df["adj_factor"] = 1.0
                df.loc[mask, "adj_factor"] = new_adj[mask]
            else:
                df["adj_factor"] = new_adj

            tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            df.to_parquet(tmp_path)
            os.replace(tmp_path, path)

        return (target_type, symbol, int(mask.sum()), True, "OK")
    except Exception as exc:  # noqa: BLE001
        return (target_type, symbol, 0, False, str(exc))


def sync_dataset(
    target_type: str,
    root_dir: Path,
    daily_factors: dict[str, pd.Series],
    symbols: list[str] | None = None,
    date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
    ffill_subsequent: bool = False,
    workers: int = 8,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Synchronize one dataset directory. Returns (total, succeeded, failed)."""
    date_msg = []
    if date_from is not None:
        date_msg.append(f"from {date_from.strftime('%Y-%m-%d')}")
    if date_to is not None:
        date_msg.append(f"to {date_to.strftime('%Y-%m-%d')}")
    date_desc = f" ({' '.join(date_msg)})" if date_msg else ""

    LOGGER.info("Starting synchronization for [%s] at %s%s", target_type, root_dir, date_desc)
    if not root_dir.exists():
        LOGGER.error("Directory does not exist: %s", root_dir)
        return (0, 0, 0)

    target_files = sorted(root_dir.glob("*.parquet"))
    if symbols:
        target_set = {s.upper() for s in symbols}
        target_files = [f for f in target_files if f.name.replace(".parquet", "").upper() in target_set]

    total = len(target_files)
    LOGGER.info("[%s] Found %s files to process", target_type, total)
    if total == 0:
        return (0, 0, 0)

    tasks = []
    for f in target_files:
        sym = f.name.replace(".parquet", "").upper()
        daily_series = daily_factors.get(sym)
        tasks.append((target_type, str(f), sym, daily_series, date_from, date_to, ffill_subsequent, dry_run))

    succeeded = 0
    failed = 0
    started = time.monotonic()
    last_log_time = started

    if workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            res = process_single_file(*task)
            if res[3]:
                succeeded += 1
            else:
                failed += 1
                LOGGER.warning("[%s] Failed on %s: %s", target_type, res[1], res[4])
            now = time.monotonic()
            if now - last_log_time >= 10.0 or idx == total:
                elapsed = now - started
                LOGGER.info("[%s] Progress: %s/%s (%.1f%%), %d failed, elapsed: %.1fs",
                            target_type, idx, total, idx / total * 100, failed, elapsed)
                last_log_time = now
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_file, *task): task[2] for task in tasks}
            for idx, future in enumerate(as_completed(futures), start=1):
                sym = futures[future]
                try:
                    res = future.result()
                    if res[3]:
                        succeeded += 1
                    else:
                        failed += 1
                        LOGGER.warning("[%s] Failed on %s: %s", target_type, sym, res[4])
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    LOGGER.exception("[%s] Unexpected exception on %s: %s", target_type, sym, exc)

                now = time.monotonic()
                if now - last_log_time >= 10.0 or idx == total:
                    elapsed = now - started
                    LOGGER.info("[%s] Progress: %s/%s (%.1f%%), %d failed, elapsed: %.1fs",
                                target_type, idx, total, idx / total * 100, failed, elapsed)
                    last_log_time = now

    total_time = time.monotonic() - started
    LOGGER.info("[%s] Complete: %d/%d succeeded, %d failed in %.2fs", target_type, succeeded, total, failed, total_time)
    return (total, succeeded, failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize adj_factor from etf_daily to QMT data.")
    parser.add_argument("--daily-path", type=Path, default=DEFAULT_DAILY_PATH, help="Path to etf_daily.parquet")
    parser.add_argument("--qmt-1d-root", type=Path, default=DEFAULT_QMT_1D_ROOT, help="Path to etf_1d_qmt directory")
    parser.add_argument("--qmt-1min-root", type=Path, default=DEFAULT_QMT_1MIN_ROOT, help="Path to etf_1min_qmt directory")
    parser.add_argument("--qmt-tick-root", type=Path, default=DEFAULT_QMT_TICK_ROOT, help="Path to etf_tick_qmt directory")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["1d", "1min", "tick", "all"],
        default=["all"],
        help="Target datasets to process (default: all)",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Start date inclusive (e.g. 2026-09-02 or 20260902)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="End date inclusive (e.g. 2026-09-11 or 20260911)",
    )
    parser.add_argument(
        "--ffill-subsequent",
        action="store_true",
        help="Also forward fill to dates after date-to",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to process (optional)",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Path to universe text file of symbols (optional)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbols_file and args.symbols_file.exists():
        symbols = [line.strip().upper() for line in args.symbols_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    date_from = parse_date(args.date_from)
    date_to = parse_date(args.date_to)

    daily_factors = load_daily_adj_factors(args.daily_path)

    target_map = {
        "1d": (args.qmt_1d_root,),
        "1min": (args.qmt_1min_root,),
        "tick": (args.qmt_tick_root,),
    }
    selected_targets = list(target_map.keys()) if "all" in args.targets else args.targets

    overall_failed = 0
    start_all = time.monotonic()
    for target in selected_targets:
        root_dir = target_map[target][0]
        _, _, failed = sync_dataset(
            target,
            root_dir,
            daily_factors,
            symbols=symbols,
            date_from=date_from,
            date_to=date_to,
            ffill_subsequent=args.ffill_subsequent,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        overall_failed += failed

    total_duration = time.monotonic() - start_all
    LOGGER.info("All tasks completed in %.2fs with %d total failures.", total_duration, overall_failed)
    return 1 if overall_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
