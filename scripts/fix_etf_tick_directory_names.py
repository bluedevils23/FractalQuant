"""Batch rename misnamed ETF directories in E:\逐笔数据 based on etf_1min benchmark."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_TICK_ROOT = Path(r"E:\逐笔数据")
DEFAULT_ETF_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_AUCTION_CACHE_ROOT = Path(r"D:\workspace\stockdata\auction_tick_cache")

LOGGER = logging.getLogger("fix_etf_tick_directory_names")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix misnamed ETF directories in tick data root."
    )
    parser.add_argument("--tick-root", type=Path, default=DEFAULT_TICK_ROOT)
    parser.add_argument("--etf-minute-root", type=Path, default=DEFAULT_ETF_MINUTE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_AUCTION_CACHE_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Preview renames without touching files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of renames for testing.")
    parser.add_argument("--clean-cache", action="store_true", help="Remove obsolete .SZ folders in auction tick cache.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_sh_etf_code_map(etf_minute_root: Path) -> dict[str, str]:
    """Map 6-digit code -> canonical symbol for Shanghai ETFs (.SH) only."""
    sh_map: dict[str, str] = {}
    pattern = re.compile(r"^(\d{6})\.SH$", re.IGNORECASE)
    for path in etf_minute_root.glob("*.parquet"):
        match = pattern.match(path.stem.upper())
        if match:
            code = match.group(1)
            sh_map[code] = f"{code}.SH"
    return sh_map


def main() -> int:
    args = parse_args()
    configure_logging()

    if not args.tick_root.exists():
        LOGGER.error("Tick root directory does not exist: %s", args.tick_root)
        return 1
    if not args.etf_minute_root.exists():
        LOGGER.error("ETF minute root directory does not exist: %s", args.etf_minute_root)
        return 1

    sh_map = build_sh_etf_code_map(args.etf_minute_root)
    LOGGER.info("Loaded %d Shanghai (.SH) ETFs from %s", len(sh_map), args.etf_minute_root)

    # Discover trade date directories
    LOGGER.info("Scanning trade date directories in %s...", args.tick_root)
    date_pattern = re.compile(r"^\d{8}$")
    date_dirs = sorted([
        path
        for path in args.tick_root.glob("*/*/*")
        if path.is_dir() and date_pattern.fullmatch(path.name)
    ])
    LOGGER.info("Found %d trade date directories.", len(date_dirs))

    # Scan and execute renames
    renamed_count = 0
    conflict_count = 0
    error_count = 0
    start_time = time.time()

    for d_idx, date_dir in enumerate(date_dirs):
        if (d_idx + 1) % 50 == 0 or d_idx == len(date_dirs) - 1:
            elapsed = time.time() - start_time
            LOGGER.info(
                "Progress: %d/%d dates processed (%d directories renamed, %.1fs elapsed)",
                d_idx + 1,
                len(date_dirs),
                renamed_count,
                elapsed,
            )

        existing_names = {p.name.upper(): p for p in date_dir.iterdir() if p.is_dir()}

        for code, expected_name in sh_map.items():
            wrong_name = f"{code}.SZ"
            if wrong_name in existing_names:
                src_path = existing_names[wrong_name]
                dst_path = date_dir / expected_name

                # Safety check: if target already exists, DO NOT OVERWRITE
                if dst_path.exists():
                    LOGGER.warning(
                        "Conflict on %s: both %s and %s exist! Skipping.",
                        date_dir.name,
                        src_path.name,
                        dst_path.name,
                    )
                    conflict_count += 1
                    continue

                if args.dry_run:
                    LOGGER.info("[DRY-RUN] Would rename: %s -> %s", src_path, dst_path)
                    renamed_count += 1
                else:
                    try:
                        os.replace(src_path, dst_path)
                        renamed_count += 1
                    except OSError as exc:
                        LOGGER.error("Failed to rename %s to %s: %s", src_path, dst_path, exc)
                        error_count += 1

                if args.limit is not None and renamed_count >= args.limit:
                    LOGGER.info("Reached limit of %d renames. Stopping.", args.limit)
                    break

        if args.limit is not None and renamed_count >= args.limit:
            break

    elapsed = time.time() - start_time
    mode_str = "DRY-RUN " if args.dry_run else ""
    LOGGER.info(
        "Finished: %srenamed=%d, conflicts=%d, errors=%d, total time=%.2fs",
        mode_str,
        renamed_count,
        conflict_count,
        error_count,
        elapsed,
    )

    # Optional: Clean up obsolete .SZ directories in auction tick cache
    if args.clean_cache and not args.dry_run and args.cache_root.exists():
        LOGGER.info("Cleaning obsolete .SZ directories in auction tick cache: %s", args.cache_root)
        cache_removed = 0
        for code in sh_map:
            for sz_cache_dir in args.cache_root.glob(f"*/*/*/{code}.SZ"):
                if sz_cache_dir.is_dir():
                    try:
                        import shutil
                        shutil.rmtree(sz_cache_dir)
                        cache_removed += 1
                    except OSError as exc:
                        LOGGER.warning("Could not remove cache dir %s: %s", sz_cache_dir, exc)
        LOGGER.info("Cleaned %d obsolete .SZ cache directories.", cache_removed)

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
