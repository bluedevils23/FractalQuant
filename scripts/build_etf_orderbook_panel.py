from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("build_etf_orderbook_panel")

DEFAULT_ORDERBOOK_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_orderbook_factors")
DEFAULT_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_orderbook_factors_hk_202602_panel")
DEFAULT_UNIVERSE_FILE = Path(__file__).resolve().parent / "hk_etf_universe.txt"

BASE_COLUMNS = ["ts_code", "open", "high", "low", "close", "volume", "amount", "adj_factor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-ETF research panels from daily ETF orderbook factor parquet files."
    )
    parser.add_argument("--orderbook-root", type=Path, default=DEFAULT_ORDERBOOK_ROOT)
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE_FILE)
    parser.add_argument("--date-from", type=str, required=True)
    parser.add_argument("--date-to", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def normalize_trade_date(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Invalid trade date: {value}")
    return digits


def load_universe_codes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file does not exist: {path}")
    codes: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        code = line.split("#", 1)[0].strip()
        if code:
            codes.append(code)
    if not codes:
        raise ValueError(f"No ETF codes found in {path}")
    return codes


def discover_trade_date_dirs(root: Path, date_from: str, date_to: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Orderbook root does not exist: {root}")
    date_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and date_from <= path.name <= date_to
    ]
    return date_dirs


def load_orderbook_panel(symbol: str, date_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date_dir in date_dirs:
        path = date_dir / f"{symbol}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"])
            df = df.set_index("trade_time")
        else:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "trade_time"
        df = df.sort_index()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=0)
    panel = panel[~panel.index.duplicated(keep="last")]
    panel = panel.sort_index()
    return panel


def load_minute_panel(symbol: str, minute_root: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    path = minute_root / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Minute parquet does not exist: {path}")
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name in {"trade_date", "trade_time"}:
        df = df.reset_index()
    if "trade_time" not in df.columns:
        raise ValueError(f"Missing trade_time in minute parquet: {path}")
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df[(df["trade_time"] >= start_ts) & (df["trade_time"] < end_ts)].copy()
    if df.empty:
        return df
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "ts_code" not in df.columns:
        df["ts_code"] = symbol
    else:
        df["ts_code"] = df["ts_code"].astype(str)
    df = df.sort_values("trade_time").drop_duplicates("trade_time", keep="last")
    df = df.set_index("trade_time")
    missing = [column for column in BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required minute columns for {symbol}: {missing}")
    return df[BASE_COLUMNS]


def build_symbol_panel(
    symbol: str,
    orderbook_root: Path,
    minute_root: Path,
    date_dirs: list[Path],
    date_from: str,
    date_to: str,
) -> tuple[str, pd.DataFrame | None, str | None]:
    orderbook_panel = load_orderbook_panel(symbol, date_dirs)
    if orderbook_panel.empty:
        return symbol, None, "missing_orderbook"

    start_ts = pd.Timestamp(date_from)
    end_ts = pd.Timestamp(date_to) + pd.Timedelta(days=1)
    minute_panel = load_minute_panel(symbol, minute_root, start_ts, end_ts)
    if minute_panel.empty:
        return symbol, None, "missing_minute_rows"

    merged = minute_panel.join(orderbook_panel.drop(columns=["ts_code"], errors="ignore"), how="inner")
    if merged.empty:
        return symbol, None, "no_time_overlap"

    merged["trade_date"] = merged["trade_date"].astype(str) if "trade_date" in merged.columns else merged.index.strftime("%Y%m%d")
    merged["ts_code"] = symbol
    return symbol, merged, None


def main() -> int:
    args = parse_args()
    configure_logging()

    date_from = normalize_trade_date(args.date_from)
    date_to = normalize_trade_date(args.date_to)
    if date_from > date_to:
        raise ValueError("--date-from cannot be later than --date-to")

    universe = load_universe_codes(args.universe_file)
    date_dirs = discover_trade_date_dirs(args.orderbook_root, date_from, date_to)
    if not date_dirs:
        raise FileNotFoundError("No orderbook trade-date directories matched the requested range.")

    LOGGER.info("Loaded %s ETF codes from universe", len(universe))
    LOGGER.info("Using %s orderbook date directories", len(date_dirs))

    args.output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped: list[tuple[str, str]] = []

    for symbol in universe:
        output_path = args.output_root / f"{symbol}.parquet"
        if output_path.exists() and not args.overwrite:
            LOGGER.info("Skipping existing output: %s", output_path)
            written += 1
            continue

        symbol_code, panel, reason = build_symbol_panel(
            symbol=symbol,
            orderbook_root=args.orderbook_root,
            minute_root=args.minute_root,
            date_dirs=date_dirs,
            date_from=date_from,
            date_to=date_to,
        )
        if panel is None:
            skipped.append((symbol_code, reason or "unknown"))
            LOGGER.warning("Skipping %s: %s", symbol_code, reason)
            continue

        panel.to_parquet(output_path)
        written += 1
        LOGGER.info("Wrote %s rows to %s", len(panel), output_path)

    LOGGER.info("Completed panel build: written=%s skipped=%s", written, len(skipped))
    if skipped:
        LOGGER.info(
            "Skipped symbols: %s",
            ", ".join(f"{symbol}({reason})" for symbol, reason in skipped[:20]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
