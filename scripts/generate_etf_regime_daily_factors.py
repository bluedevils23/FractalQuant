"""Generate shared-market, causal daily ETF regime factors."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor.regime import (  # noqa: E402
    DAILY_REGIME_FEATURE_COLUMNS,
    DAILY_REGIME_OUTPUT_COLUMNS,
    build_daily_market_feature_panel,
    calculate_causal_daily_market_regime_features,
)

LOGGER = logging.getLogger("generate_etf_regime_daily_factors")

DEFAULT_ETF_DAILY_PATH = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_INDEX_DAILY_ROOT = Path(r"D:\workspace\stockdata\指数数据\index_daily")
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\etf-data\etf_regime_daily_factors"
)
DEFAULT_REFERENCE_CODES = (
    "000985.CSI",
    "000300.SH",
    "000905.SH",
    "000852.SH",
    "000012.SH",
    "000016.SH",
    "399006.SZ",
)
OUTPUT_COLUMNS = (
    "trade_date",
    "available_time",
    "source_trade_date",
    "ts_code",
) + DAILY_REGIME_OUTPUT_COLUMNS


def _normalize_codes(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip().upper() for value in values if str(value).strip()}


def _read_daily_frame(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Daily file does not exist: {path}")
    frame = pd.read_parquet(path)
    if isinstance(frame.index, pd.MultiIndex) or set(required).difference(frame.columns):
        frame = frame.reset_index()
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Daily file is missing columns {missing}: {path}")
    return frame


def load_reference_close_panel(
    index_daily_root: Path,
    reference_codes: tuple[str, ...] = DEFAULT_REFERENCE_CODES,
) -> pd.DataFrame:
    """Load and align the configured reference-index closes."""
    missing = [
        code
        for code in reference_codes
        if not (index_daily_root / f"{code}.parquet").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing configured reference index files under {index_daily_root}: {missing}"
        )
    series: list[pd.Series] = []
    for code in reference_codes:
        frame = pd.read_parquet(index_daily_root / f"{code}.parquet")
        if "trade_date" in frame.columns:
            dates = pd.to_datetime(frame.pop("trade_date"), errors="coerce").dt.normalize()
            frame = frame.set_index(dates)
        elif not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, errors="coerce").normalize()
        if "close" not in frame.columns:
            raise ValueError(f"Reference file is missing close: {code}")
        values = pd.to_numeric(frame["close"], errors="coerce")
        values.index = pd.DatetimeIndex(frame.index).normalize()
        values = values[~values.index.isna()].groupby(level=0).last()
        series.append(values.rename(code))
    panel = pd.concat(series, axis=1, sort=True).sort_index()
    if panel.empty:
        raise ValueError("Reference index panel is empty")
    return panel


def _filter_etf_codes(frame: pd.DataFrame, requested: set[str] | None) -> pd.DataFrame:
    if requested is None:
        return frame
    prefixes = {code.split(".", 1)[0] for code in requested}
    code_series = frame["ts_code"].astype(str).str.upper()
    return frame.loc[code_series.isin(requested) | code_series.str.split(".").str[0].isin(prefixes)]


def build_etf_regime_daily_frame(
    etf_daily: pd.DataFrame,
    regime_by_source_date: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
    requested_codes: set[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """Map source-date regime values to each ETF's next available trade date."""
    frame = etf_daily.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
    frame = frame.dropna(subset=["trade_date"])
    frame = _filter_etf_codes(frame, requested_codes)
    if date_from is not None:
        frame = frame.loc[frame["trade_date"] >= pd.Timestamp(date_from).normalize()]
    if date_to is not None:
        frame = frame.loc[frame["trade_date"] <= pd.Timestamp(date_to).normalize()]
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    market_dates = pd.DatetimeIndex(market_dates).sort_values().unique()
    target_dates = pd.DatetimeIndex(frame["trade_date"].unique()).sort_values()
    positions = market_dates.searchsorted(target_dates, side="left") - 1
    source_dates = pd.Series(
        [market_dates[position] if position >= 0 else pd.NaT for position in positions],
        index=target_dates,
    )
    selected = frame[["trade_date", "ts_code"]].drop_duplicates().copy()
    selected["source_trade_date"] = selected["trade_date"].map(source_dates)
    selected = selected.dropna(subset=["source_trade_date"])
    selected = selected.merge(
        regime_by_source_date.reset_index(names="source_trade_date"),
        on="source_trade_date",
        how="left",
    )
    selected["available_time"] = selected["trade_date"] + pd.Timedelta(hours=9, minutes=15)
    selected = selected.reindex(columns=OUTPUT_COLUMNS)
    return selected.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _merge_output(path: Path, requested: pd.DataFrame, overwrite: bool) -> None:
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    existing = existing.reindex(columns=OUTPUT_COLUMNS)
    dates = set(requested["trade_date"].astype(str))
    if overwrite:
        existing = existing.loc[~existing["trade_date"].astype(str).isin(dates)]
        additions = requested
    else:
        additions = requested.loc[
            ~requested["trade_date"].astype(str).isin(set(existing["trade_date"].astype(str)))
        ]
    combined = pd.concat([existing, additions], ignore_index=True)
    combined = combined.drop_duplicates(["trade_date", "ts_code"], keep="last")
    combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce").dt.normalize()
    combined["source_trade_date"] = pd.to_datetime(
        combined["source_trade_date"], errors="coerce"
    ).dt.normalize()
    combined = combined.sort_values("trade_date", kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)


def write_outputs(frame: pd.DataFrame, output_root: Path, overwrite: bool) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for code, group in frame.groupby("ts_code", sort=True):
        path = output_root / f"{code}.parquet"
        _merge_output(path, group, overwrite)
        written.append(str(path))
    return written


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etf-daily", type=Path, default=DEFAULT_ETF_DAILY_PATH)
    parser.add_argument("--index-daily-root", type=Path, default=DEFAULT_INDEX_DAILY_ROOT)
    parser.add_argument("--reference-codes", nargs="+", default=list(DEFAULT_REFERENCE_CODES))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--training-days", type=int, default=756)
    parser.add_argument("--min-training-days", type=int, default=252)
    parser.add_argument("--refit-days", type=int, default=21)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reference_codes = tuple(str(code).strip().upper() for code in args.reference_codes)
    close_panel = load_reference_close_panel(args.index_daily_root, reference_codes)
    market_features = build_daily_market_feature_panel(close_panel)
    regime = calculate_causal_daily_market_regime_features(
        market_features,
        training_days=args.training_days,
        min_training_days=args.min_training_days,
        refit_days=args.refit_days,
    )
    etf = _read_daily_frame(args.etf_daily, ("trade_date", "ts_code"))
    requested = _normalize_codes(args.symbols)
    if args.symbols_file is not None:
        file_codes = {
            line.strip().upper()
            for line in args.symbols_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        requested = (requested or set()) | file_codes
    output = build_etf_regime_daily_frame(
        etf,
        regime,
        close_panel.index,
        requested_codes=requested or None,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    files = write_outputs(output, args.output_root, args.overwrite) if not output.empty else []
    null_counts = {
        column: int(output[column].isna().sum()) for column in DAILY_REGIME_OUTPUT_COLUMNS
    }
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_codes": list(reference_codes),
        "training_days": args.training_days,
        "min_training_days": args.min_training_days,
        "refit_days": args.refit_days,
        "feature_columns": list(DAILY_REGIME_FEATURE_COLUMNS),
        "output_columns": list(OUTPUT_COLUMNS),
        "source_date_from": str(close_panel.index.min().date()),
        "source_date_to": str(close_panel.index.max().date()),
        "target_date_from": str(output["trade_date"].min().date()) if not output.empty else None,
        "target_date_to": str(output["trade_date"].max().date()) if not output.empty else None,
        "etf_count": int(output["ts_code"].nunique()) if not output.empty else 0,
        "output_rows": int(len(output)),
        "output_files": files,
        "model_fit_failure_dates": regime.attrs.get("model_fit_failure_dates", []),
        "insufficient_history_dates": regime.attrs.get("insufficient_history_dates", []),
    }
    state_counts = (
        output["regime_state"].value_counts(dropna=False).to_dict()
        if not output.empty
        else {}
    )
    finite_state_count = sum(
        int(value) for key, value in state_counts.items() if pd.notna(key)
    )
    report = {
        "output_rows": int(len(output)),
        "null_counts": null_counts,
        "state_counts": state_counts,
        "state_occupancy": {
            str(key): (int(value) / finite_state_count if finite_state_count else 0.0)
            for key, value in state_counts.items()
            if pd.notna(key)
        },
        "schema": list(output.columns) if not output.empty else list(OUTPUT_COLUMNS),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "_regime_manifest.json", manifest)
    _write_json(args.output_root / "_regime_report.json", report)
    LOGGER.info("Generated %d regime rows for %d ETFs", len(output), manifest["etf_count"])


if __name__ == "__main__":
    main()
