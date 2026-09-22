from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd


UNIVERSE = Path(
    r"D:\workspace\stockdata\etf-factors\etf_universe\non_day_turnover_stock_index_universe.txt"
)
TICK_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_tick_qmt")
REPORT = Path(r"D:\workspace\stockdata\etf-factors\etf_auction_factors\qmt_tick_0925_volume_audit_progress.json")


def inspect(symbol: str) -> tuple[bool, int, int, int, int, int, int]:
    path = TICK_ROOT / f"{symbol}.parquet"
    if not path.exists():
        return False, 0, 0, 0, 0, 0, 0
    frame = pd.read_parquet(path, columns=["last_price", "volume", "amount"])
    frame = frame.reset_index()
    times = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.loc[
        times.dt.strftime("%H:%M:%S").between("09:25:00", "09:25:59")
    ].copy()
    if frame.empty:
        return True, 0, 0, 0, 0, 0, 0
    for column in ("last_price", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    valid = (
        frame["last_price"].gt(0)
        & frame["volume"].gt(0)
        & frame["amount"].gt(0)
    )
    final = frame.sort_values("trade_time", kind="mergesort").drop_duplicates(
        "trade_date", keep="last"
    )
    final_valid = (
        final["last_price"].gt(0)
        & final["volume"].gt(0)
        & final["amount"].gt(0)
    )
    return (
        True,
        len(frame),
        int(frame["volume"].eq(0).sum()),
        int(valid.sum()),
        len(final),
        int(final["volume"].eq(0).sum()),
        int(final_valid.sum()),
    )


symbols = [
    line.strip()
    for line in UNIVERSE.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
totals = [0] * 6
files = 0


def write_progress(processed: int) -> None:
    payload = {
        "processed_symbols": processed,
        "universe_symbols": len(symbols),
        "tick_files": files,
        "missing_files": processed - files,
        "all_0925_ticks": {
            "rows": totals[0],
            "volume_eq_0": totals[1],
            "valid_price_volume_amount": totals[2],
        },
        "last_tick_per_symbol_date": {
            "rows": totals[3],
            "volume_eq_0": totals[4],
            "valid_price_volume_amount": totals[5],
        },
        "complete": processed == len(symbols),
    }
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(REPORT)


write_progress(0)
with ThreadPoolExecutor(max_workers=4) as executor:
    for processed, (exists, *values) in enumerate(executor.map(inspect, symbols), 1):
        files += int(exists)
        totals = [left + right for left, right in zip(totals, values)]
        if processed % 50 == 0:
            write_progress(processed)

write_progress(len(symbols))

print(f"UNIVERSE={len(symbols)} TICK_FILES={files} MISSING_FILES={len(symbols) - files}")
print(
    "ALL_0925_TICKS "
    f"rows={totals[0]} volume_eq_0={totals[1]} "
    f"zero_rate={totals[1] / totals[0]:.8%} "
    f"valid_price_volume_amount={totals[2]} "
    f"valid_rate={totals[2] / totals[0]:.8%}"
)
print(
    "LAST_TICK_PER_SYMBOL_DATE "
    f"rows={totals[3]} volume_eq_0={totals[4]} "
    f"zero_rate={totals[4] / totals[3]:.8%} "
    f"valid_price_volume_amount={totals[5]} "
    f"valid_rate={totals[5] / totals[3]:.8%}"
)
