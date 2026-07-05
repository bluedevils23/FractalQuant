from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.noise_area import NoiseArea  # noqa: E402


LOGGER = logging.getLogger("generate_etf_noise_bound_ratio_factors")

STOCKDATA_ROOT = Path(r"D:\workspace\stockdata")
ETF_META_PATH = STOCKDATA_ROOT / "etf-data" / "etf_basic_data.parquet"
DEFAULT_INPUT_ROOT = STOCKDATA_ROOT / "etf-data" / "etf_1min"
INDEX_MINUTE_DIR = STOCKDATA_ROOT / "index-data" / "index_1min"
DEFAULT_OUTPUT_ROOT = STOCKDATA_ROOT / "etf-data" / "etf_noise_bound_ratio_factors"
WINDOW = 14
CURRENT_INDEX_TO_ETF = {
    "000300.SH": "510300.SH",
    "000016.SH": "510050.SH",
    "000905.SH": "510500.SH",
    "000852.SH": "159845.SZ",
}
CANDIDATE_ETFS_PATHS = [
    Path(__file__).resolve().parent
    / "validation_outputs_core_broad_sector_merged"
    / "candidate_etfs.csv",
    Path(
        r"D:\workspace\stock-playbook\QuantsPlaybook\C-择时类\另类ETF交易策略：日内动量\validation_outputs_core_broad_sector_merged\candidate_etfs.csv"
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ETF noise bound ratio factors from local parquet files."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional ETF symbols. If provided, they override any symbols file.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional ETF symbol list file.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def normalize_symbol_id(value: str) -> str:
    symbol = str(value).strip()
    if symbol.lower().endswith(".parquet"):
        symbol = symbol[:-8]
    return symbol


def read_symbol_list_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file does not exist: {path}")
    symbols: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                symbols.append(normalize_symbol_id(line))
    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def load_requested_symbols(symbols: list[str] | None, symbols_file: Path | None) -> list[str] | None:
    if symbols:
        requested = [normalize_symbol_id(symbol) for symbol in symbols]
    else:
        if symbols_file is None:
            return None
        requested = read_symbol_list_file(symbols_file)

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in requested:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def resolve_input_root(input_root: Path) -> Path:
    candidate = input_root / "etf_1min"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return input_root


def resolve_candidate_etfs_path() -> Path | None:
    for path in CANDIDATE_ETFS_PATHS:
        if path.exists():
            return path
    return None


def discover_input_files(input_root: Path, symbols: list[str] | None) -> list[Path]:
    resolved_root = resolve_input_root(input_root)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {resolved_root}")
    files = sorted(resolved_root.glob("*.parquet"))
    if symbols:
        wanted = [normalize_symbol_id(symbol) for symbol in symbols]
        file_map = {normalize_symbol_id(path.name): path for path in files}
        missing = [symbol for symbol in wanted if symbol not in file_map]
        if missing:
            raise FileNotFoundError("Missing input parquet files: " + ", ".join(missing[:10]))
        files = [file_map[symbol] for symbol in wanted]
    if not files:
        raise FileNotFoundError(f"No minute parquet files found in: {resolved_root}")
    return files


def discover_representative_etfs(etf_minute_dir: Path, index_minute_dir: Path) -> pd.DataFrame:
    etf_meta = pd.read_parquet(ETF_META_PATH).copy()
    etf_meta = etf_meta.loc[etf_meta["ts_code"].notna() & etf_meta["index_code"].notna()].copy()
    etf_meta["ts_code"] = etf_meta["ts_code"].astype(str)
    etf_meta["index_code"] = etf_meta["index_code"].astype(str)
    etf_meta["etf_name"] = etf_meta["extname"].fillna(etf_meta["csname"]).astype(str)
    etf_meta["index_name"] = etf_meta["index_name"].fillna("").astype(str)
    etf_meta["list_date"] = pd.to_datetime(etf_meta["list_date"], format="%Y%m%d", errors="coerce")
    etf_meta["etf_file_exists"] = etf_meta["ts_code"].map(lambda code: (etf_minute_dir / f"{code}.parquet").exists())
    etf_meta["index_file_exists"] = etf_meta["index_code"].map(lambda code: (index_minute_dir / f"{code}.parquet").exists())
    etf_meta = etf_meta.loc[etf_meta["etf_file_exists"] & etf_meta["index_file_exists"]].copy()

    candidate_path = resolve_candidate_etfs_path()
    if candidate_path is not None:
        curated = pd.read_csv(candidate_path).rename(columns={"code": "ts_code"})
        curated["priority_curated"] = 1
        etf_meta = etf_meta.merge(
            curated[["ts_code", "index_code", "priority_curated"]],
            on=["ts_code", "index_code"],
            how="left",
        )
        etf_meta["priority_curated"] = etf_meta["priority_curated"].fillna(0).astype(int)
    else:
        etf_meta["priority_curated"] = 0

    etf_meta["priority_current"] = etf_meta.apply(
        lambda row: int(CURRENT_INDEX_TO_ETF.get(row["index_code"]) == row["ts_code"]),
        axis=1,
    )

    selected = (
        etf_meta.sort_values(
            ["index_code", "priority_current", "priority_curated", "list_date", "ts_code"],
            ascending=[True, False, False, True, True],
        )
        .groupby("index_code", as_index=False)
        .first()
    )
    return selected[["ts_code", "etf_name", "index_code", "index_name", "list_date"]].rename(columns={"ts_code": "code"})


def normalize_minute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index().rename(columns={"ts_code": "code", "vol": "volume"})
    out["trade_time"] = pd.to_datetime(out["trade_time"])
    out["code"] = out["code"].astype(str)
    return out


def build_factor_frame_for_mapping(
    mapping: pd.Series, etf_minute_dir: Path, index_minute_dir: Path
) -> pd.DataFrame:
    index_path = index_minute_dir / f"{mapping['index_code']}.parquet"
    etf_path = etf_minute_dir / f"{mapping['code']}.parquet"

    index_price = normalize_minute_frame(pd.read_parquet(index_path))
    index_price = index_price[["trade_time", "code", "open", "high", "low", "close", "volume", "amount"]].sort_values(["trade_time", "code"])
    noise = NoiseArea(index_price)
    upperbound = noise.calculate_bound(window=WINDOW, method="U")
    lowerbound = noise.calculate_bound(window=WINDOW, method="L")
    index_close = noise.close

    factor_frame = pd.concat(
        [
            upperbound.stack().to_frame(name="upperbound"),
            lowerbound.stack().to_frame(name="lowerbound"),
            index_close.stack().to_frame(name="index_close"),
        ],
        axis=1,
    ).reset_index()
    factor_frame = factor_frame.rename(columns={"level_0": "trade_time", "code": "index_code"})
    factor_frame["code"] = mapping["code"]
    factor_frame["etf_name"] = mapping["etf_name"]
    factor_frame["index_name"] = mapping["index_name"]
    factor_frame["upper_ratio"] = factor_frame["upperbound"] / factor_frame["index_close"]
    factor_frame["lower_ratio"] = factor_frame["lowerbound"] / factor_frame["index_close"]
    factor_frame["upper_strength"] = (1.0 - factor_frame["upper_ratio"]).clip(lower=0.0)
    factor_frame["lower_strength"] = (factor_frame["lower_ratio"] - 1.0).clip(lower=0.0)
    factor_frame["net_strength"] = factor_frame["upper_strength"] - factor_frame["lower_strength"]
    factor_frame["dominant_signal"] = "neutral"
    factor_frame.loc[factor_frame["upper_strength"] > factor_frame["lower_strength"], "dominant_signal"] = "long"
    factor_frame.loc[factor_frame["lower_strength"] > factor_frame["upper_strength"], "dominant_signal"] = "short"

    etf_price = normalize_minute_frame(pd.read_parquet(etf_path))
    etf_price = etf_price[["trade_time", "code"]].drop_duplicates(subset=["trade_time", "code"])
    factor_frame = factor_frame.merge(
        etf_price,
        on=["trade_time", "code"],
        how="inner",
        validate="one_to_one",
    )
    return factor_frame[[
        "trade_time",
        "code",
        "etf_name",
        "index_code",
        "index_name",
        "upperbound",
        "lowerbound",
        "index_close",
        "upper_ratio",
        "lower_ratio",
        "upper_strength",
        "lower_strength",
        "net_strength",
        "dominant_signal",
    ]]


def process_mapping(
    mapping: pd.Series,
    output_root: Path,
    etf_minute_dir: Path,
    index_minute_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    output_path = output_root / f"{mapping['code']}.parquet"
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "code": mapping["code"], "output_path": output_path}
    factor_frame = build_factor_frame_for_mapping(mapping, etf_minute_dir, index_minute_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    factor_frame.to_parquet(output_path)
    return {"status": "written", "code": mapping["code"], "output_path": output_path, "rows": len(factor_frame)}


def main() -> int:
    args = parse_args()
    configure_logging()

    etf_minute_dir = resolve_input_root(args.input_root)
    requested_symbols = load_requested_symbols(args.symbols, args.symbols_file)
    if args.symbols:
        LOGGER.info("Using explicit --symbols filter with %s ETFs", len(requested_symbols))
    elif args.symbols_file is not None:
        LOGGER.info("Using ETF universe file: %s", args.symbols_file)
    else:
        LOGGER.info("No ETF universe file provided; processing all representative ETFs")
    if requested_symbols is None:
        selected = discover_representative_etfs(etf_minute_dir, INDEX_MINUTE_DIR)
    else:
        selected = discover_representative_etfs(etf_minute_dir, INDEX_MINUTE_DIR)
        selected = selected.loc[selected["code"].isin(requested_symbols)].copy()
    if args.limit is not None:
        selected = selected.head(args.limit)
    if selected.empty:
        LOGGER.warning("No ETF mappings matched the requested inputs.")
        return 0

    if args.workers <= 1:
        for _, row in selected.iterrows():
            result = process_mapping(
                row,
                args.output_root,
                etf_minute_dir,
                INDEX_MINUTE_DIR,
                args.overwrite,
            )
            LOGGER.info("%s %s", result["status"], result["code"])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    process_mapping,
                    row,
                    args.output_root,
                    etf_minute_dir,
                    INDEX_MINUTE_DIR,
                    args.overwrite,
                ): row["code"]
                for _, row in selected.iterrows()
            }
            for future in as_completed(future_map):
                result = future.result()
                LOGGER.info("%s %s", result["status"], result["code"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
