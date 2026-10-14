from __future__ import annotations

"""Generate previous-trading-day ETF daily states for minute factor jobs."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_etf_crossmarket_factors import (
    DEFAULT_ETF_ROOT,
    discover_etf_files,
    normalize_fund_code,
    normalize_minute_frame,
)


DEFAULT_INPUT = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_OUTPUT = Path(
    r"D:\workspace\stockdata\etf-data\etf_daily_states.parquet"
)
STATE_COLUMNS = ("prev_nav", "prev_total_size", "prev_total_share")


def generate_daily_states(
    input_path: Path,
    output_path: Path,
    *,
    symbols: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    etf_root: Path = DEFAULT_ETF_ROOT,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        input_path,
        columns=["ts_code", "trade_date", "nav", "total_size", "total_share"],
    )
    if "ts_code" not in frame.columns or "trade_date" not in frame.columns:
        frame = frame.reset_index()
    frame["ts_code"] = frame["ts_code"].map(normalize_fund_code)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["ts_code", "trade_date"])
    if symbols:
        requested = {normalize_fund_code(value) for value in symbols}
        frame = frame[frame["ts_code"].isin(requested)]
    for column in ("nav", "total_size", "total_share"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["ts_code", "trade_date"], kind="mergesort")
    frame = frame.drop_duplicates(["ts_code", "trade_date"], keep="last")
    minute_files = discover_etf_files(etf_root)
    records = []
    for code, history in frame.groupby("ts_code", sort=False):
        if code not in minute_files:
            continue
        minutes = normalize_minute_frame(pd.read_parquet(minute_files[code]))
        calendar = minutes.index.normalize().unique().sort_values()
        previous = pd.Series(calendar, index=calendar).shift(1)
        state = history.set_index("trade_date").reindex(pd.DatetimeIndex(previous))
        state = state[["nav", "total_size", "total_share"]].set_axis(calendar)
        state.columns = STATE_COLUMNS
        state["ts_code"] = code
        state["trade_date"] = calendar
        state["source_trade_date"] = previous.to_numpy()
        records.append(state.reset_index(drop=True))
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["ts_code", "trade_date", *STATE_COLUMNS, "source_trade_date"]
    )
    if date_from:
        result = result[result["trade_date"] >= pd.Timestamp(date_from).normalize()]
    if date_to:
        result = result[result["trade_date"] <= pd.Timestamp(date_to).normalize()]
    result = result.sort_values(["ts_code", "trade_date"], kind="mergesort")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--etf-root", type=Path, default=DEFAULT_ETF_ROOT)
    args = parser.parse_args()
    result = generate_daily_states(
        args.input,
        args.output,
        symbols=args.symbols,
        date_from=args.date_from,
        date_to=args.date_to,
        etf_root=args.etf_root,
    )
    print(f"wrote {len(result)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
