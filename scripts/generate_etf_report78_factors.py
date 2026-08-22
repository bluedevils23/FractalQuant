"""Generate causal daily ETF proxies for Kysec Quant Comment 78."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.etf_report78 import REPORT78_FACTOR_COLUMNS, build_report78_daily_panel  # noqa: E402

LOGGER = logging.getLogger("generate_etf_report78_factors")
DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_report78_factors")


def normalize_minute_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the ETF parquet layouts used by the local data directory."""

    frame = raw_df.copy()
    if isinstance(frame.index, pd.MultiIndex) and "trade_time" in frame.index.names:
        frame.index = pd.to_datetime(frame.index.get_level_values("trade_time"))
    elif "trade_time" in frame.columns:
        frame["trade_time"] = pd.to_datetime(frame["trade_time"])
        frame = frame.set_index("trade_time")
    elif "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index("datetime")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Cannot locate trade_time/datetime index or column")
    frame.index.name = "trade_time"
    frame = frame.rename(columns={"vol": "volume"}).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def process_file(input_path: Path, output_root: Path, overwrite: bool) -> tuple[str, Path]:
    output_path = output_root / input_path.name
    if output_path.exists() and not overwrite:
        return "skipped", output_path
    source = normalize_minute_frame(pd.read_parquet(input_path))
    panel = build_report78_daily_panel(source)
    panel = panel.reset_index()
    panel["available_date"] = panel["trade_date"].shift(-1)
    panel["available_time"] = panel["available_date"] + pd.Timedelta(hours=9, minutes=30)
    panel["ts_code"] = input_path.stem
    panel["source_level"] = "etf_minute_proxy"
    panel["amount_required"] = bool("amount" in source.columns)
    panel = panel[["ts_code", "trade_date", "available_date", "available_time", *REPORT78_FACTOR_COLUMNS, "source_level", "amount_required"]]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    output_root.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    return "written", output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    files = [args.input_root / f"{symbol}.parquet" for symbol in args.symbols] if args.symbols else sorted(args.input_root.glob("*.parquet"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No ETF parquet files found under {args.input_root}")
    for input_path in files:
        status, output_path = process_file(input_path, args.output_root, args.overwrite)
        LOGGER.info("%s %s", status, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

