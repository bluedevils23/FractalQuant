from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from auction_tick_cache import AuctionTickCache


UNIVERSE = Path(
    r"D:\workspace\stockdata\etf-factors\etf_universe\non_day_turnover_stock_index_universe.txt"
)
OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-factors\etf_auction_factors")
TICK_ROOT = Path(r"E:\逐笔数据")
REPORT = OUTPUT_ROOT / "quote_0925_0930_recovery_audit_progress.json"


def source_dir(symbol: str, trade_date: str) -> Path:
    compact = trade_date.replace("-", "")
    return TICK_ROOT / compact[:4] / compact[:6] / compact / symbol


def inspect_symbol(symbol: str) -> tuple[int, int, int, list[dict[str, object]]]:
    output = pd.read_parquet(
        OUTPUT_ROOT / f"{symbol}.parquet", columns=["trade_date", "auction_match_source"]
    )
    missing_dates = output.loc[
        output["auction_match_source"].eq("none"), "trade_date"
    ].astype(str)
    cache = AuctionTickCache(None)
    recoverable = 0
    missing_csv = 0
    samples: list[dict[str, object]] = []
    for trade_date in missing_dates:
        directory = source_dir(symbol, trade_date)
        if not (directory / "行情.csv").exists():
            missing_csv += 1
            continue
        quotes = cache.load_quote(directory)
        start = pd.Timestamp(f"{trade_date} 09:25:00")
        end = pd.Timestamp(f"{trade_date} 09:30:00")
        matches = quotes.loc[
            quotes["trade_time"].ge(start)
            & quotes["trade_time"].lt(end)
            & quotes["open_price"].notna()
            & quotes["trade_volume"].gt(0)
            & quotes["trade_amount"].gt(0)
        ]
        if matches.empty:
            continue
        recoverable += 1
        if len(samples) < 3:
            row = matches.iloc[0]
            samples.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "trade_time": str(row["trade_time"]),
                    "open_price": float(row["open_price"]),
                    "trade_volume": float(row["trade_volume"]),
                    "trade_amount": float(row["trade_amount"]),
                }
            )
    return len(missing_dates), recoverable, missing_csv, samples


symbols = [
    line.strip()
    for line in UNIVERSE.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
totals = {"none_rows": 0, "recoverable_quote_rows": 0, "missing_quote_csv_rows": 0}
samples: list[dict[str, object]] = []


def write_progress(processed: int, complete: bool) -> None:
    payload = {
        "processed_symbols": processed,
        "universe_symbols": len(symbols),
        **totals,
        "recovery_rate_among_none": (
            totals["recoverable_quote_rows"] / totals["none_rows"]
            if totals["none_rows"]
            else 0.0
        ),
        "sample_recoveries": samples[:20],
        "complete": complete,
    }
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(REPORT)


write_progress(0, False)
with ThreadPoolExecutor(max_workers=4) as executor:
    for processed, result in enumerate(executor.map(inspect_symbol, symbols), 1):
        none_rows, recoverable, missing_csv, entries = result
        totals["none_rows"] += none_rows
        totals["recoverable_quote_rows"] += recoverable
        totals["missing_quote_csv_rows"] += missing_csv
        samples.extend(entries)
        if processed % 50 == 0:
            write_progress(processed, False)
write_progress(len(symbols), True)
