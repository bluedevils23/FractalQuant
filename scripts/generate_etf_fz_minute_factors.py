from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"
REPLICATION_ROOT = PROJECT_ROOT.parent / "Replication-of-Minute-Frequency-Factor-refer-FZ"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(REPLICATION_ROOT))

import MinuteFrequentFactorCalculateMethodsFZ as fz_methods  # noqa: E402


LOGGER = logging.getLogger("generate_etf_fz_minute_factors")

DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
DEFAULT_DAILY_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min_fz_factors")
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
BASE_OUTPUT_COLUMNS = (
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adj_factor",
)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    function: Callable[..., pl.DataFrame | None]
    needs_daily_pv: bool = False


RAW_FACTOR_SPECS = (
    FactorSpec("YaoYanBoDongLv", fz_methods.cal_YaoYanBoDongLv),
    FactorSpec("YaoYanShouYiLv", fz_methods.cal_YaoYanShouYiLv),
    FactorSpec("QiangShiBanChaoXi", fz_methods.cal_QiangShiBanChaoXi),
    FactorSpec("RuoShiBanChaoXi", fz_methods.cal_RuoShiBanChaoXi),
    FactorSpec("MoHuGuanLianDu", fz_methods.cal_MoHuGuanLianDu),
    FactorSpec("MoHuJinEBi", fz_methods.cal_MoHuJinEBi),
    FactorSpec("MoHuJiaCha", fz_methods.cal_MoHuJiaCha),
    FactorSpec("PanDeng", fz_methods.cal_PanDeng),
    FactorSpec("TiaoYueDu", fz_methods.cal_TiaoYueDu),
    FactorSpec("RiBoDongLv", fz_methods.cal_RiBoDongLv),
    FactorSpec("GuYanChuQun", fz_methods.cal_GuYanChuQun),
    FactorSpec("GaoDiECha", fz_methods.cal_GaoDiECha, needs_daily_pv=True),
    FactorSpec("ZhaoMoChenWu", fz_methods.cal_ZhaoMoChenWu),
    FactorSpec("WuBiGuMu", fz_methods.cal_WuBiGuMu),
    FactorSpec("YeMianShuangLu_t_intercept", fz_methods.cal_YeMianShuangLu_t_intercept),
    FactorSpec("GenSuiXiShu", fz_methods.cal_GenSuiXiShu),
    FactorSpec("ChengJiaoLiangBoYi_ShouYiLv", fz_methods.cal_ChengJiaoLiangBoYi_ShouYiLv),
    FactorSpec(
        "ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi",
        fz_methods.cal_ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi,
    ),
    FactorSpec("ZhenFuBoYi", fz_methods.cal_ZhenFuBoYi),
    FactorSpec("ChengJiaoLiangXieTong", fz_methods.cal_ChengJiaoLiangXieTong),
    FactorSpec("XieTongJiaCha", fz_methods.cal_XieTongJiaCha),
)

COMPOSED_FACTOR_SPECS = (
    FactorSpec("ShiDuMaoXian", fz_methods.cal_ShiDuMaoXian),
    FactorSpec("ChaoXi", fz_methods.cal_ChaoXi),
    FactorSpec("YunKaiWuSan", fz_methods.cal_YunKaiWuSan),
    FactorSpec("YongPanGaoFeng", fz_methods.cal_YongPanGaoFeng),
    FactorSpec("FeiEPuHuo", fz_methods.cal_FeiEPuHuo),
    FactorSpec("CaoMuJieBing", fz_methods.cal_CaoMuJieBing),
    FactorSpec("SuiBoZhuLiu", fz_methods.cal_SuiBoZhuLiu),
    FactorSpec("ShuiZhongXingZhou", fz_methods.cal_ShuiZhongXingZhou),
    FactorSpec("YeMianShuangLu", fz_methods.cal_YeMianShuangLu),
    FactorSpec("HuaYinLinJian", fz_methods.cal_HuaYinLinJian),
    FactorSpec("DaiZhuErJiu", fz_methods.cal_DaiZhuErJiu),
    FactorSpec("DuoKongBoYi", fz_methods.cal_DuoKongBoYi),
    FactorSpec("XieTongXiaoYing", fz_methods.cal_XieTongXiaoYing),
)

ALL_FACTOR_NAMES = tuple(
    spec.name
    for spec in (
        FactorSpec("YaoYanBoDongLv", fz_methods.cal_YaoYanBoDongLv),
        FactorSpec("YaoYanShouYiLv", fz_methods.cal_YaoYanShouYiLv),
        FactorSpec("ShiDuMaoXian", fz_methods.cal_ShiDuMaoXian),
        FactorSpec("QiangShiBanChaoXi", fz_methods.cal_QiangShiBanChaoXi),
        FactorSpec("RuoShiBanChaoXi", fz_methods.cal_RuoShiBanChaoXi),
        FactorSpec("ChaoXi", fz_methods.cal_ChaoXi),
        FactorSpec("MoHuGuanLianDu", fz_methods.cal_MoHuGuanLianDu),
        FactorSpec("MoHuJinEBi", fz_methods.cal_MoHuJinEBi),
        FactorSpec("MoHuJiaCha", fz_methods.cal_MoHuJiaCha),
        FactorSpec("YunKaiWuSan", fz_methods.cal_YunKaiWuSan),
        FactorSpec("PanDeng", fz_methods.cal_PanDeng),
        FactorSpec("YongPanGaoFeng", fz_methods.cal_YongPanGaoFeng),
        FactorSpec("TiaoYueDu", fz_methods.cal_TiaoYueDu),
        FactorSpec("FeiEPuHuo", fz_methods.cal_FeiEPuHuo),
        FactorSpec("RiBoDongLv", fz_methods.cal_RiBoDongLv),
        FactorSpec("CaoMuJieBing", fz_methods.cal_CaoMuJieBing),
        FactorSpec("GuYanChuQun", fz_methods.cal_GuYanChuQun),
        FactorSpec("GaoDiECha", fz_methods.cal_GaoDiECha),
        FactorSpec("SuiBoZhuLiu", fz_methods.cal_SuiBoZhuLiu),
        FactorSpec("ShuiZhongXingZhou", fz_methods.cal_ShuiZhongXingZhou),
        FactorSpec("ZhaoMoChenWu", fz_methods.cal_ZhaoMoChenWu),
        FactorSpec("WuBiGuMu", fz_methods.cal_WuBiGuMu),
        FactorSpec("YeMianShuangLu_t_intercept", fz_methods.cal_YeMianShuangLu_t_intercept),
        FactorSpec("YeMianShuangLu", fz_methods.cal_YeMianShuangLu),
        FactorSpec("HuaYinLinJian", fz_methods.cal_HuaYinLinJian),
        FactorSpec("GenSuiXiShu", fz_methods.cal_GenSuiXiShu),
        FactorSpec("DaiZhuErJiu", fz_methods.cal_DaiZhuErJiu),
        FactorSpec("ChengJiaoLiangBoYi_ShouYiLv", fz_methods.cal_ChengJiaoLiangBoYi_ShouYiLv),
        FactorSpec(
            "ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi",
            fz_methods.cal_ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi,
        ),
        FactorSpec("ZhenFuBoYi", fz_methods.cal_ZhenFuBoYi),
        FactorSpec("DuoKongBoYi", fz_methods.cal_DuoKongBoYi),
        FactorSpec("ChengJiaoLiangXieTong", fz_methods.cal_ChengJiaoLiangXieTong),
        FactorSpec("XieTongJiaCha", fz_methods.cal_XieTongJiaCha),
        FactorSpec("XieTongXiaoYing", fz_methods.cal_XieTongXiaoYing),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ETF FangZheng minute factors from local parquet files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing ETF minute parquet files.",
    )
    parser.add_argument(
        "--daily-root",
        type=Path,
        default=DEFAULT_DAILY_ROOT,
        help="ETF daily parquet file used by GaoDiECha-related factors.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where factor parquet files will be written.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional ETF symbols such as 159001.SZ 510300.SH.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional ETF symbol list file with one symbol per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N parquet files after filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel workers to use for file-based stages.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


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


def load_requested_symbols(
    symbols: list[str] | None, symbols_file: Path | None
) -> list[str] | None:
    requested: list[str] = []
    if symbols_file is not None:
        requested.extend(read_symbol_list_file(symbols_file))
    if symbols:
        requested.extend(normalize_symbol_id(symbol) for symbol in symbols)
    if not requested:
        return None

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


def discover_input_files(input_root: Path, symbols: list[str] | None) -> list[Path]:
    resolved_root = resolve_input_root(input_root)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {resolved_root}")

    if symbols:
        files = [resolved_root / f"{symbol}.parquet" for symbol in symbols]
    else:
        files = sorted(resolved_root.glob("*.parquet"))

    missing_files = [path for path in files if not path.exists()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(f"Missing input parquet files: {missing_text}")

    return files


def normalize_minute_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    if isinstance(df.index, pd.MultiIndex) and "trade_time" in df.index.names:
        trade_time = pd.to_datetime(df.index.get_level_values("trade_time"))
        df.index = trade_time
        df.index.name = "trade_time"
    elif "trade_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["trade_time"])
        df = df.set_index("trade_time")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df.index.name = "trade_time"
    else:
        raise ValueError("Cannot locate trade_time/datetime index or column.")

    df = df.rename(columns={"vol": "volume"})

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "ts_code" not in df.columns:
        raise ValueError("Missing ts_code column in source minute parquet.")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    numeric_columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "amount", "adj_factor")
        if column in df.columns
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ts_code"] = df["ts_code"].astype(str)
    return df


def normalize_daily_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    if isinstance(df.index, pd.MultiIndex) and {"trade_date", "ts_code"} <= set(df.index.names):
        df = df.reset_index()
    elif not {"trade_date", "ts_code"} <= set(df.columns):
        raise ValueError("Cannot locate trade_date/ts_code in ETF daily parquet.")

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df["ts_code"] = df["ts_code"].astype(str)
    df = df.rename(columns={"vol": "volume"})

    numeric_columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "amount", "total_size", "total_share")
        if column in df.columns
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "total_size" in df.columns:
        cmc = df["total_size"].copy()
    else:
        cmc = pd.Series(index=df.index, dtype="float64")
    if "total_share" in df.columns and "close" in df.columns:
        cmc = cmc.fillna(df["total_share"] * df["close"])
    df["cmc"] = cmc
    return df


def convert_time_to_int(series: pd.Series) -> pd.Series:
    trade_time = pd.to_datetime(series)
    return trade_time.dt.hour * 10000000 + trade_time.dt.minute * 100000


def build_trade_day_slice(input_path: Path, stage_root: Path) -> tuple[str, int]:
    raw_df = pd.read_parquet(input_path)
    minute_df = normalize_minute_frame(raw_df).reset_index()
    minute_df["trade_date"] = pd.to_datetime(minute_df["trade_time"]).dt.normalize()
    export_columns = [
        column
        for column in ("trade_time", "trade_date", "ts_code", "open", "high", "low", "close", "volume", "amount")
        if column in minute_df.columns
    ]
    exported_days = 0
    for trade_date, day_frame in minute_df[export_columns].groupby("trade_date", sort=True):
        day_dir = stage_root / pd.Timestamp(trade_date).strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        day_frame.to_parquet(day_dir / input_path.name, index=False)
        exported_days += 1
    return (input_path.name, exported_days)


def load_day_minute_panel(day_dir: Path) -> pl.DataFrame:
    day_frames = [pd.read_parquet(path) for path in sorted(day_dir.glob("*.parquet"))]
    if not day_frames:
        raise ValueError(f"No staged day files found in {day_dir}")

    panel = pd.concat(day_frames, ignore_index=True)
    panel["trade_time"] = pd.to_datetime(panel["trade_time"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    panel["ts_code"] = panel["ts_code"].astype(str)
    panel = panel.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    factor_input = pd.DataFrame(
        {
            "code": panel["ts_code"],
            "date": panel["trade_date"].dt.date,
            "time": convert_time_to_int(panel["trade_time"]).astype("int64"),
            "open": pd.to_numeric(panel["open"], errors="coerce"),
            "high": pd.to_numeric(panel["high"], errors="coerce"),
            "low": pd.to_numeric(panel["low"], errors="coerce"),
            "close": pd.to_numeric(panel["close"], errors="coerce"),
            "volume": pd.to_numeric(panel["volume"], errors="coerce"),
            "amount": pd.to_numeric(panel["amount"], errors="coerce"),
        }
    )
    return pl.from_pandas(factor_input, include_index=False)


def load_daily_inputs(
    daily_root: Path, symbols: set[str]
) -> tuple[pd.DataFrame, pl.DataFrame]:
    if not daily_root.exists():
        raise FileNotFoundError(f"Daily parquet does not exist: {daily_root}")

    daily_df = normalize_daily_frame(pd.read_parquet(daily_root))
    daily_df = daily_df.loc[daily_df["ts_code"].isin(symbols)].copy()
    if daily_df.empty:
        raise ValueError("No ETF daily rows matched the requested symbols.")

    daily_df = daily_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    missing_cmc = int(daily_df["cmc"].isna().sum())
    if missing_cmc:
        LOGGER.warning(
            "Daily cmc missing for %s code-date rows; GaoDiECha-related factors may be null there.",
            missing_cmc,
        )

    daily_base = daily_df[
        [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "cmc",
        ]
    ].copy()

    daily_pv = daily_df[["ts_code", "trade_date", "open", "close", "cmc"]].copy()
    daily_pv = daily_pv.rename(
        columns={
            "ts_code": "Stkcd",
            "trade_date": "Trddt",
            "open": "Opnprc",
            "close": "Clsprc",
            "cmc": "Dsmvosd",
        }
    )
    daily_pv["Trddt"] = pd.to_datetime(daily_pv["Trddt"]).dt.strftime("%Y-%m-%d")
    daily_pv_pl = pl.from_pandas(daily_pv, include_index=False)
    return daily_base, daily_pv_pl


def build_base_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(["code", "date"]).unique().sort(["code", "date"])


def normalize_factor_output(
    factor_name: str,
    factor_df: pl.DataFrame | None,
    base_keys: pl.DataFrame,
) -> pl.DataFrame:
    if factor_df is None or factor_df.is_empty():
        return base_keys.with_columns(pl.lit(None).alias(factor_name))

    value_columns = [
        column for column in factor_df.columns if column not in {"code", "date"}
    ]
    if len(value_columns) != 1:
        raise ValueError(
            f"Factor {factor_name} returned {len(value_columns)} value columns: {value_columns}"
        )

    value_column = value_columns[0]
    if value_column != factor_name:
        factor_df = factor_df.rename({value_column: factor_name})

    return base_keys.join(
        factor_df.select(["code", "date", factor_name]),
        on=["code", "date"],
        how="left",
    )


def calculate_raw_daily_panel(
    stage_root: Path,
    daily_pv: pl.DataFrame,
) -> pl.DataFrame:
    daily_frames: list[pl.DataFrame] = []
    date_dirs = sorted(path for path in stage_root.iterdir() if path.is_dir())
    for date_dir in date_dirs:
        minute_panel = load_day_minute_panel(date_dir)
        base_keys = build_base_keys(minute_panel)
        day_exposure = base_keys
        for spec in RAW_FACTOR_SPECS:
            if spec.needs_daily_pv:
                factor_df = spec.function(minute_panel, daily_pv)
            else:
                factor_df = spec.function(minute_panel)
            day_exposure = day_exposure.join(
                normalize_factor_output(spec.name, factor_df, base_keys),
                on=["code", "date"],
                how="left",
            )
        daily_frames.append(day_exposure)

    if not daily_frames:
        raise ValueError("No daily minute slices were staged for FZ computation.")
    return pl.concat(daily_frames, how="vertical").sort(["code", "date"])


def enrich_with_daily_base(
    raw_panel: pl.DataFrame,
    daily_base: pd.DataFrame,
) -> pl.DataFrame:
    daily_base_pl = pl.from_pandas(
        daily_base.rename(columns={"ts_code": "code", "trade_date": "date"}).assign(
            date=lambda frame: pd.to_datetime(frame["date"]).dt.date
        ),
        include_index=False,
    )
    panel = raw_panel.join(
        daily_base_pl.select(["code", "date", "open", "high", "low", "close", "volume", "amount", "cmc"]),
        on=["code", "date"],
        how="left",
    ).sort(["code", "date"])

    missing_close = panel.select(pl.col("close").is_null().sum()).item()
    if missing_close:
        LOGGER.warning(
            "Daily OHLC rows missing for %s code-date rows; some composed FZ factors may be null.",
            int(missing_close),
        )
    return panel


def calculate_composed_panel(panel: pl.DataFrame) -> pl.DataFrame:
    keys = build_base_keys(panel)
    output = panel
    for spec in COMPOSED_FACTOR_SPECS:
        factor_df = spec.function(output)
        normalized = normalize_factor_output(spec.name, factor_df, keys)
        output = output.join(normalized, on=["code", "date"], how="left")
    return output.sort(["code", "date"])


def build_final_daily_factor_frame(
    raw_panel: pl.DataFrame,
    composed_panel: pl.DataFrame,
) -> pd.DataFrame:
    factor_columns = [name for name in ALL_FACTOR_NAMES if name in composed_panel.columns]
    output = composed_panel.select(["code", "date", *factor_columns]).to_pandas()
    output["date"] = pd.to_datetime(output["date"])
    return output.sort_values(["code", "date"]).reset_index(drop=True)


def merge_factors_for_symbol(
    input_path: Path,
    output_root: Path,
    overwrite: bool,
    factor_frame: pd.DataFrame,
) -> tuple[str, Path, int | None, int | None]:
    output_path = output_root / input_path.name
    if output_path.exists() and not overwrite:
        return ("skipped", output_path, None, None)

    raw_df = pd.read_parquet(input_path)
    minute_df = normalize_minute_frame(raw_df)
    minute_df["trade_date"] = pd.to_datetime(minute_df.index).normalize()
    symbol = normalize_symbol_id(input_path.name)
    symbol_factors = factor_frame.loc[factor_frame["code"] == symbol].copy()
    symbol_factors = symbol_factors.rename(columns={"code": "ts_code", "date": "trade_date"})

    result = minute_df.reset_index().merge(
        symbol_factors,
        on=["ts_code", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    result = result.set_index("trade_time")
    result.index.name = "trade_time"
    result = result.drop(columns=["trade_date"])

    ordered_columns = [
        column for column in BASE_OUTPUT_COLUMNS if column in result.columns
    ]
    ordered_columns.extend(
        column for column in ALL_FACTOR_NAMES if column in result.columns
    )
    remaining_columns = [
        column for column in result.columns if column not in ordered_columns
    ]
    result = result[ordered_columns + remaining_columns]

    output_root.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    return ("written", output_path, len(result), len(result.columns))


def main() -> int:
    args = parse_args()
    configure_logging()

    requested_symbols = load_requested_symbols(args.symbols, args.symbols_file)
    files = discover_input_files(args.input_root, requested_symbols)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        LOGGER.warning("No parquet files matched the requested inputs.")
        return 0

    symbols = {normalize_symbol_id(path.name) for path in files}
    daily_base, daily_pv = load_daily_inputs(args.daily_root, symbols)

    worker_count = max(1, args.workers)
    LOGGER.info("Processing %s ETF minute parquet files for FZ factors", len(files))

    with tempfile.TemporaryDirectory(prefix="etf_fz_stage_") as stage_dir_name:
        stage_root = Path(stage_dir_name)
        if worker_count == 1:
            for input_path in files:
                _, exported_days = build_trade_day_slice(input_path, stage_root)
                LOGGER.info("Staged %s trade days for %s", exported_days, input_path.name)
        else:
            LOGGER.info("Using %s workers for staging", worker_count)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(build_trade_day_slice, input_path, stage_root): input_path
                    for input_path in files
                }
                for future in as_completed(future_map):
                    input_path = future_map[future]
                    file_name, exported_days = future.result()
                    LOGGER.info("Staged %s trade days for %s", exported_days, file_name)

        raw_panel = calculate_raw_daily_panel(stage_root, daily_pv)
        panel_with_daily = enrich_with_daily_base(raw_panel, daily_base)
        composed_panel = calculate_composed_panel(panel_with_daily)
        final_factor_frame = build_final_daily_factor_frame(raw_panel, composed_panel)

    failures: list[tuple[Path, str]] = []
    if worker_count == 1:
        for input_path in files:
            try:
                status, output_path, row_count, column_count = merge_factors_for_symbol(
                    input_path,
                    args.output_root,
                    args.overwrite,
                    final_factor_frame,
                )
                if status == "skipped":
                    LOGGER.info("Skipping existing output: %s", output_path)
                else:
                    LOGGER.info(
                        "Wrote %s rows and %s columns to %s",
                        row_count,
                        column_count,
                        output_path,
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append((input_path, str(exc)))
                LOGGER.exception("Failed to process %s", input_path)
    else:
        LOGGER.info("Using %s workers for output merge", worker_count)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    merge_factors_for_symbol,
                    input_path,
                    args.output_root,
                    args.overwrite,
                    final_factor_frame,
                ): input_path
                for input_path in files
            }
            for future in as_completed(future_map):
                input_path = future_map[future]
                try:
                    status, output_path, row_count, column_count = future.result()
                    if status == "skipped":
                        LOGGER.info("Skipping existing output: %s", output_path)
                    else:
                        LOGGER.info(
                            "Wrote %s rows and %s columns to %s",
                            row_count,
                            column_count,
                            output_path,
                        )
                except Exception as exc:  # noqa: BLE001
                    failures.append((input_path, str(exc)))
                    LOGGER.exception("Failed to process %s", input_path)

    if failures:
        LOGGER.error("Completed with %s failures", len(failures))
        for failed_path, reason in failures[:10]:
            LOGGER.error("  %s -> %s", failed_path, reason)
        return 1

    LOGGER.info("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
