from __future__ import annotations

"""Generate previous-trading-day ETF daily states for minute factor jobs."""

import argparse
from pathlib import Path

import pandas as pd


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
) -> pd.DataFrame:
    frame = pd.read_parquet(
        input_path,
        columns=["ts_code", "trade_date", "nav", "total_size", "total_share"],
    )
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["ts_code", "trade_date"])
    if symbols:
        requested = {str(value).strip().upper() for value in symbols}
        frame = frame[frame["ts_code"].isin(requested)]
    for column in ("nav", "total_size", "total_share"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["ts_code", "trade_date"], kind="mergesort")
    frame = frame.drop_duplicates(["ts_code", "trade_date"], keep="last")
    grouped = frame.groupby("ts_code", sort=False)
    for source, target in zip(("nav", "total_size", "total_share"), STATE_COLUMNS):
        frame[target] = grouped[source].shift(1)
    result = frame[["ts_code", "trade_date", *STATE_COLUMNS]]
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
    args = parser.parse_args()
    result = generate_daily_states(
        args.input,
        args.output,
        symbols=args.symbols,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    print(f"wrote {len(result)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
