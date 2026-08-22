"""Fill only the ETF CaoMuJieBing column from daily tick-trade data.

The existing 38 FZ columns are reused. This avoids rerunning the expensive
minute-frequency factor pipeline when the independent tick source becomes
available.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_fz_daily_factors import (  # noqa: E402
    CAO_MU_REQUIRED_SOURCE_FIELDS,
    CAO_MU_SOURCE_COLUMNS,
    MANIFEST_NAME,
    assess_cao_mu_source_fields,
    calculate_retail_trade_ratio,
    load_csi_all_share_returns,
    normalize_cao_mu_source_frame,
    normalize_daily_frame,
)


DEFAULT_SYMBOLS = Path(
    r"D:\workspace\stockdata\etf-data\logs\non_day_turnover_stock_index_universe_available_20260809.txt"
)
DEFAULT_FACTOR_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_daily_fz_factors"
)
DEFAULT_DAILY_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_TICK_ROOT = Path(r"E:\逐笔数据")
DEFAULT_INDEX_ROOT = Path(
    r"D:\workspace\stockdata\指数数据\index_daily\000985.CSI.parquet"
)
DEFAULT_SOURCE = Path(
    r"D:\workspace\stockdata\etf-data\etf_cao_mu_jie_bing_source.parquet"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update only ETF CaoMuJieBing in existing FZ daily parquet files."
    )
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTOR_ROOT)
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    parser.add_argument("--tick-root", type=Path, default=DEFAULT_TICK_ROOT)
    parser.add_argument("--index-daily-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--date-from", type=pd.Timestamp, default=pd.Timestamp("2025-01-01"))
    parser.add_argument("--date-to", type=pd.Timestamp, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--skip-source-build",
        action="store_true",
        help="Reuse an existing source parquet and only rewrite CaoMuJieBing.",
    )
    return parser.parse_args(argv)


def read_symbols(path: Path) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            symbol = raw.split("#", 1)[0].strip()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol.removesuffix(".parquet"))
    return symbols


def resolve_warmup_start(
    daily_base: pd.DataFrame, date_from: pd.Timestamp, window_size: int = 20
) -> pd.Timestamp:
    dates = pd.DatetimeIndex(daily_base["trade_date"].drop_duplicates()).sort_values()
    index = dates.searchsorted(pd.Timestamp(date_from).normalize(), side="left")
    return dates[max(0, index - window_size)]


def _tick_ratio_task(
    task: tuple[str, pd.Timestamp, Path],
) -> tuple[str, pd.Timestamp, float | None]:
    code, trade_date, tick_path = task
    return code, trade_date, calculate_retail_trade_ratio(tick_path)


def collect_tick_ratios(
    tasks: list[tuple[str, pd.Timestamp, Path]], workers: int
) -> tuple[pd.DataFrame, int]:
    """Read existing tick files with bounded concurrency."""
    if not tasks:
        return pd.DataFrame(columns=["ts_code", "trade_date", "retail_trade_ratio"]), 0

    rows: list[dict[str, object]] = []
    invalid_tick_files = 0
    done_count = 0
    worker_count = max(1, workers)

    def consume(result: tuple[str, pd.Timestamp, float | None]) -> None:
        nonlocal invalid_tick_files, done_count
        done_count += 1
        code, trade_date, ratio = result
        if ratio is None:
            invalid_tick_files += 1
        else:
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "retail_trade_ratio": ratio,
                }
            )
        if done_count % 10000 == 0:
            print(
                f"Read {done_count}/{len(tasks)} tick files",
                flush=True,
            )

    if worker_count == 1:
        for task in tasks:
            consume(_tick_ratio_task(task))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pending = {}
            task_iter = iter(tasks)
            for _ in range(worker_count):
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                pending[executor.submit(_tick_ratio_task, task)] = None

            while pending:
                finished_futures, _ = wait(
                    pending, return_when=FIRST_COMPLETED
                )
                for future in finished_futures:
                    del pending[future]
                    consume(future.result())
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        continue
                    pending[executor.submit(_tick_ratio_task, task)] = None

    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=["ts_code", "trade_date", "retail_trade_ratio"])
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    return result, invalid_tick_files


def build_etf_cao_mu_source(
    daily_base: pd.DataFrame,
    tick_root: Path,
    index_daily_root: Path,
    source_path: Path,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp | None,
    workers: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    requested = daily_base[["ts_code", "trade_date"]].drop_duplicates().copy()
    requested = requested.loc[requested["trade_date"] >= date_from.normalize()]
    if date_to is not None:
        requested = requested.loc[requested["trade_date"] <= date_to.normalize()]
    requested = requested.sort_values(["ts_code", "trade_date"])

    if source_path.exists():
        existing = normalize_cao_mu_source_frame(pd.read_parquet(source_path))
    else:
        existing = pd.DataFrame(columns=CAO_MU_SOURCE_COLUMNS)

    existing_covered = (
        existing["retail_trade_ratio"].notna()
        & existing["csi_all_share_return"].notna()
    )
    covered = set(
        zip(
            existing.loc[existing_covered, "ts_code"],
            existing.loc[existing_covered, "trade_date"],
        )
    )
    missing = requested.loc[
        [key not in covered for key in zip(requested["ts_code"], requested["trade_date"])]
    ]

    tasks: list[tuple[str, pd.Timestamp, Path]] = []
    missing_tick_files = 0
    # E:\逐笔数据 is relatively slow for many individual exists() calls.
    # Index each trading-day directory once, then match the requested ETF
    # codes against that in-memory set.
    for trade_date, day_rows in missing.groupby("trade_date", sort=True):
        date_text = pd.Timestamp(trade_date).strftime("%Y%m%d")
        day_dir = (
            tick_root
            / date_text[:4]
            / date_text[:6]
            / date_text
        )
        try:
            available_codes = {
                entry.name
                for entry in os.scandir(day_dir)
                if entry.is_dir()
            }
        except OSError:
            missing_tick_files += len(day_rows)
            continue
        for code in day_rows["ts_code"]:
            if code not in available_codes:
                missing_tick_files += 1
                continue
            tasks.append(
                (
                    code,
                    pd.Timestamp(trade_date),
                    day_dir / code / "逐笔成交.csv",
                )
            )

    retail_df, invalid_tick_files = collect_tick_ratios(tasks, workers)
    benchmark = load_csi_all_share_returns(index_daily_root)
    new_source = missing.merge(retail_df, on=["ts_code", "trade_date"], how="left")
    new_source = new_source.merge(benchmark, on="trade_date", how="left")
    combined = pd.concat([existing, new_source], ignore_index=True)
    combined = normalize_cao_mu_source_frame(
        combined.drop_duplicates(["ts_code", "trade_date"], keep="last")
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(source_path, index=False)

    checked = combined.loc[
        combined["ts_code"].isin(requested["ts_code"])
        & combined["trade_date"].between(
            date_from.normalize(),
            date_to.normalize() if date_to is not None else pd.Timestamp.max,
        )
    ].copy()
    availability = assess_cao_mu_source_fields(checked)
    availability.update(
        {
            "requested_rows": int(len(requested)),
            "tick_files_read": len(tasks),
            "missing_tick_files": missing_tick_files,
            "invalid_tick_files": invalid_tick_files,
        }
    )
    return combined, availability


def update_factor_file(
    factor_path: Path,
    daily_base: pd.DataFrame,
    source: pd.DataFrame,
    date_from: pd.Timestamp,
    date_to: pd.Timestamp | None,
) -> tuple[str, int, int]:
    factor = pd.read_parquet(factor_path)
    factor.index = pd.to_datetime(factor.index).normalize()
    factor.index.name = "factor_date"
    code = factor_path.stem

    daily = daily_base.loc[
        daily_base["ts_code"].eq(code), ["trade_date", "close"]
    ].rename(columns={"trade_date": "factor_date"})
    src = source.loc[
        source["ts_code"].eq(code),
        ["trade_date", *CAO_MU_REQUIRED_SOURCE_FIELDS],
    ].rename(columns={"trade_date": "factor_date"})
    work = factor.reset_index().merge(daily, on="factor_date", how="left")
    work = work.merge(src, on="factor_date", how="left").sort_values("factor_date")

    stock_return = pd.to_numeric(work["close"], errors="coerce").pct_change(fill_method=None)
    market_return = pd.to_numeric(work["csi_all_share_return"], errors="coerce")
    fear = (
        (stock_return - market_return).abs()
        / (stock_return.abs() + market_return.abs() + 0.1)
    )
    decay = fear - (fear.shift(1) + fear.shift(2)) / 2.0
    positive_decay = decay.where(decay > 0)
    score = (
        pd.to_numeric(work["retail_trade_ratio"], errors="coerce")
        * pd.to_numeric(work["RiBoDongLv"], errors="coerce")
        * positive_decay
        * stock_return
    )
    updated = (
        score.rolling(20, min_periods=5).mean()
        + score.rolling(20, min_periods=5).std(ddof=1)
    ) / 2.0
    # A missing current-day source must not be silently carried forward by the
    # rolling window.  Keep the exposure null until both tick-derived inputs
    # and the current-day base inputs are present.
    current_inputs = (
        work[list(CAO_MU_REQUIRED_SOURCE_FIELDS)]
        .notna()
        .all(axis=1)
        & work["close"].notna()
        & work["RiBoDongLv"].notna()
    )
    updated = updated.where(current_inputs)

    output_mask = work["factor_date"].ge(date_from.normalize())
    if date_to is not None:
        output_mask &= work["factor_date"].le(date_to.normalize())
    factor.loc[
        work.loc[output_mask, "factor_date"], "CaoMuJieBing"
    ] = updated.loc[output_mask].to_numpy()

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=factor_path.parent, suffix=".parquet", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        factor.to_parquet(temporary_path)
        os.replace(temporary_path, factor_path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise
    return code, len(factor), int(factor["CaoMuJieBing"].notna().sum())


def update_manifest(
    factor_root: Path, availability: dict[str, object], updated: int, source_path: Path
) -> None:
    manifest_path = factor_root / MANIFEST_NAME
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.setdefault("factor_availability", {})["CaoMuJieBing"] = availability
    payload["cao_mu_update"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_symbols": updated,
        "source_path": str(source_path),
        "note": (
            "Only CaoMuJieBing was recomputed from ETF tick trades; "
            "other factor columns were preserved."
        ),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.tick_root.exists():
        raise FileNotFoundError(f"Tick root does not exist: {args.tick_root}")
    symbols = read_symbols(args.symbols_file)
    factor_files = [args.factor_root / f"{symbol}.parquet" for symbol in symbols]
    factor_files = [path for path in factor_files if path.exists()]
    if not factor_files:
        raise FileNotFoundError("No existing ETF FZ parquet files matched the symbols file.")

    factor_symbols = {path.stem for path in factor_files}
    daily = normalize_daily_frame(pd.read_parquet(args.daily_root))
    daily = daily.loc[daily["ts_code"].isin(factor_symbols)].copy()
    compute_from = resolve_warmup_start(daily, args.date_from)
    source_daily = daily.loc[daily["trade_date"] >= compute_from].copy()
    if args.date_to is not None:
        source_daily = source_daily.loc[
            source_daily["trade_date"] <= args.date_to.normalize()
        ]

    if args.skip_source_build:
        source = normalize_cao_mu_source_frame(pd.read_parquet(args.source_path))
        checked = source.loc[
            source["ts_code"].isin(factor_symbols)
            & source["trade_date"].ge(compute_from)
        ]
        if args.date_to is not None:
            checked = checked.loc[checked["trade_date"] <= args.date_to.normalize()]
        availability = assess_cao_mu_source_fields(checked)
    else:
        _, availability = build_etf_cao_mu_source(
            source_daily,
            args.tick_root,
            args.index_daily_root,
            args.source_path,
            compute_from,
            args.date_to,
            args.workers,
        )
    source = normalize_cao_mu_source_frame(pd.read_parquet(args.source_path))

    updated = 0
    available_values = 0
    for factor_path in factor_files:
        _, _, available = update_factor_file(
            factor_path, source_daily, source, args.date_from, args.date_to
        )
        updated += 1
        available_values += available

    update_manifest(args.factor_root, availability, updated, args.source_path)
    print(
        f"Updated {updated} ETF files; non-null CaoMuJieBing values: "
        f"{available_values}; source status: {availability['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
