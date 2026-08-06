from __future__ import annotations

import argparse
from bisect import bisect_right
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FractalQuant.factor import fz_methods  # noqa: E402


LOGGER = logging.getLogger("generate_fz_daily_factors")

ETF_DEFAULT_INPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
ETF_DEFAULT_DAILY_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
ETF_DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_daily_fz_factors")
STOCK_DEFAULT_INPUT_ROOT = Path(
    r"D:\workspace\stockdata\stock-data\行情数据\stock_1min"
)
STOCK_DEFAULT_DAILY_ROOT = Path(
    r"D:\workspace\stockdata\stock-data\行情数据\stock_daily.parquet"
)
STOCK_DEFAULT_OUTPUT_ROOT = Path(
    r"D:\workspace\stockdata\stock-factors\stock_fz_daily_factors"
)
ASSET_DEFAULTS = {
    "etf": (
        ETF_DEFAULT_INPUT_ROOT,
        ETF_DEFAULT_DAILY_ROOT,
        ETF_DEFAULT_OUTPUT_ROOT,
    ),
    "stock": (
        STOCK_DEFAULT_INPUT_ROOT,
        STOCK_DEFAULT_DAILY_ROOT,
        STOCK_DEFAULT_OUTPUT_ROOT,
    ),
}
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
MORNING_MINUTES = pd.date_range("09:30", "11:30", freq="min").time
AFTERNOON_MINUTES = pd.date_range("13:01", "15:00", freq="min").time
EXPECTED_MINUTE_TIMES = frozenset((*MORNING_MINUTES, *AFTERNOON_MINUTES))
EXPECTED_MINUTE_ROWS = len(EXPECTED_MINUTE_TIMES)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    function: Callable[..., pl.DataFrame | pl.LazyFrame | None]
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


def parse_args(
    argv: list[str] | None = None,
    default_asset_type: str = "etf",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stock or ETF FangZheng daily factor exposures from local "
            "1-minute parquet files."
        )
    )
    parser.add_argument(
        "--asset-type",
        choices=tuple(ASSET_DEFAULTS),
        default=default_asset_type,
        help="Asset defaults to use when input, daily, or output paths are omitted.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Directory containing one minute parquet file per symbol. Defaults "
            "to the stock_1min or etf_1min directory selected by --asset-type."
        ),
    )
    parser.add_argument(
        "--daily-root",
        type=Path,
        default=None,
        help=(
            "Daily parquet used by GaoDiECha-related factors. Defaults to the "
            "stock or ETF daily parquet selected by --asset-type."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory where one daily factor parquet file per symbol will be "
            "written. Defaults to stock_fz_daily_factors or "
            "etf_daily_fz_factors."
        ),
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbols such as 000001.SZ or 510300.SH.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional symbol list file with one symbol per line.",
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
    parser.add_argument(
        "--date-from",
        type=pd.Timestamp,
        default=None,
        help="First factor_date to write, inclusive (for example 2022-01-01).",
    )
    parser.add_argument(
        "--date-to",
        type=pd.Timestamp,
        default=None,
        help="Last factor_date to write, inclusive (defaults to the latest input date).",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=None,
        help="Optional directory for temporary day-by-symbol minute partitions.",
    )
    args = parser.parse_args(argv)
    default_input, default_daily, default_output = ASSET_DEFAULTS[args.asset_type]
    args.input_root = args.input_root or default_input
    args.daily_root = args.daily_root or default_daily
    args.output_root = args.output_root or default_output
    return args


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
    for child_name in ("etf_1min", "stock_1min"):
        candidate = input_root / child_name
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
        raise ValueError("Cannot locate trade_date/ts_code in daily parquet.")

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


def validate_fz_trade_day(day_frame: pd.DataFrame) -> str | None:
    trade_times = pd.to_datetime(day_frame["trade_time"])
    if trade_times.duplicated().any():
        return "duplicate trade_time"
    if len(trade_times) != EXPECTED_MINUTE_ROWS:
        return f"expected {EXPECTED_MINUTE_ROWS} rows, found {len(trade_times)}"
    minute_times = frozenset(trade_times.dt.time)
    if minute_times != EXPECTED_MINUTE_TIMES:
        missing = len(EXPECTED_MINUTE_TIMES - minute_times)
        unexpected = len(minute_times - EXPECTED_MINUTE_TIMES)
        return f"invalid minute grid (missing={missing}, unexpected={unexpected})"
    return None


def build_trade_day_slice(
    input_path: Path,
    stage_root: Path,
    compute_date_from: pd.Timestamp | None = None,
    date_to: pd.Timestamp | None = None,
) -> tuple[str, int, list[tuple[str, str]]]:
    raw_df = pd.read_parquet(input_path)
    minute_df = normalize_minute_frame(raw_df).reset_index()
    minute_df["trade_date"] = pd.to_datetime(minute_df["trade_time"]).dt.normalize()
    if compute_date_from is not None:
        minute_df = minute_df.loc[
            minute_df["trade_date"] >= pd.Timestamp(compute_date_from).normalize()
        ]
    if date_to is not None:
        minute_df = minute_df.loc[
            minute_df["trade_date"] <= pd.Timestamp(date_to).normalize()
        ]
    export_columns = [
        column
        for column in ("trade_time", "trade_date", "ts_code", "open", "high", "low", "close", "volume", "amount")
        if column in minute_df.columns
    ]
    exported_days = 0
    skipped_days: list[tuple[str, str]] = []
    for trade_date, day_frame in minute_df[export_columns].groupby("trade_date", sort=True):
        reason = validate_fz_trade_day(day_frame)
        date_text = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        if reason is not None:
            skipped_days.append((date_text, reason))
            continue
        day_dir = stage_root / pd.Timestamp(trade_date).strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        day_frame.to_parquet(day_dir / input_path.name, index=False)
        exported_days += 1
    return (input_path.name, exported_days, skipped_days)


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
        raise ValueError("No daily rows matched the requested symbols.")

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
    factor_df: pl.DataFrame | pl.LazyFrame | None,
    base_keys: pl.DataFrame,
) -> pl.DataFrame:
    if factor_df is None:
        return base_keys.with_columns(pl.lit(None).alias(factor_name))

    if isinstance(factor_df, pl.LazyFrame):
        factor_df = factor_df.collect()
    if not isinstance(factor_df, pl.DataFrame):
        raise TypeError(
            f"Factor {factor_name} returned {type(factor_df).__name__}, expected Polars DataFrame."
        )
    if factor_df.is_empty():
        return base_keys.with_columns(pl.lit(None).alias(factor_name))

    missing_keys = {"code", "date"} - set(factor_df.columns)
    if missing_keys:
        raise ValueError(f"Factor {factor_name} is missing key columns: {sorted(missing_keys)}")
    duplicate_keys = factor_df.group_by(["code", "date"]).len().filter(pl.col("len") > 1)
    if duplicate_keys.height:
        raise ValueError(f"Factor {factor_name} returned duplicate code/date keys.")

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


def partition_daily_pv_by_date(
    daily_pv: pl.DataFrame,
) -> tuple[list[str], dict[str, pl.DataFrame]]:
    partitions = daily_pv.partition_by("Trddt", as_dict=True)
    by_date = {str(key[0]): frame for key, frame in partitions.items()}
    return sorted(by_date), by_date


def select_daily_pv_window(
    daily_dates: list[str],
    daily_pv_by_date: dict[str, pl.DataFrame],
    factor_date: pd.Timestamp,
    window_size: int = 20,
) -> pl.DataFrame:
    date_text = pd.Timestamp(factor_date).strftime("%Y-%m-%d")
    end = bisect_right(daily_dates, date_text)
    selected_dates = daily_dates[max(0, end - window_size):end]
    if not selected_dates:
        return pl.DataFrame(schema=daily_pv_by_date[daily_dates[0]].schema)
    return pl.concat([daily_pv_by_date[date] for date in selected_dates], how="vertical")


def calculate_raw_daily_exposure(
    date_dir: Path, daily_pv_window: pl.DataFrame
) -> pl.DataFrame:
    minute_panel = load_day_minute_panel(date_dir)
    base_keys = build_base_keys(minute_panel)
    day_exposure = base_keys
    for spec in RAW_FACTOR_SPECS:
        factor_df = (
            spec.function(minute_panel, daily_pv_window)
            if spec.needs_daily_pv
            else spec.function(minute_panel)
        )
        day_exposure = day_exposure.join(
            normalize_factor_output(spec.name, factor_df, base_keys),
            on=["code", "date"],
            how="left",
        )
    return day_exposure


def calculate_raw_daily_panel(
    stage_root: Path,
    daily_pv: pl.DataFrame,
    workers: int,
) -> pl.DataFrame:
    date_dirs = sorted(path for path in stage_root.iterdir() if path.is_dir())
    daily_dates, daily_pv_by_date = partition_daily_pv_by_date(daily_pv)
    date_inputs = [
        (
            date_dir,
            select_daily_pv_window(
                daily_dates, daily_pv_by_date, pd.Timestamp(date_dir.name)
            ),
        )
        for date_dir in date_dirs
    ]
    if workers == 1:
        daily_frames = [
            calculate_raw_daily_exposure(date_dir, daily_pv_window)
            for date_dir, daily_pv_window in date_inputs
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            daily_frames = list(
                executor.map(
                    calculate_raw_daily_exposure,
                    (date_dir for date_dir, _ in date_inputs),
                    (daily_pv_window for _, daily_pv_window in date_inputs),
                )
            )

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
    composed_panel: pl.DataFrame,
) -> pd.DataFrame:
    missing_factor_columns = [
        name for name in ALL_FACTOR_NAMES if name not in composed_panel.columns
    ]
    if missing_factor_columns:
        composed_panel = composed_panel.with_columns(
            pl.lit(None).alias(name) for name in missing_factor_columns
        )
    output = composed_panel.select(["code", "date", *ALL_FACTOR_NAMES]).to_pandas()
    output["date"] = pd.to_datetime(output["date"])
    return output.sort_values(["code", "date"]).reset_index(drop=True)


def resolve_compute_date_from(
    daily_base: pd.DataFrame,
    output_date_from: pd.Timestamp | None,
    warmup_days: int = 20,
) -> pd.Timestamp | None:
    if output_date_from is None:
        return None

    trading_dates = pd.DatetimeIndex(daily_base["trade_date"].drop_duplicates()).sort_values()
    start = pd.Timestamp(output_date_from).normalize()
    first_output_index = trading_dates.searchsorted(start, side="left")
    if first_output_index == len(trading_dates):
        return start
    return trading_dates[max(0, first_output_index - warmup_days)]


def filter_factor_date_range(
    factor_frame: pd.DataFrame,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
) -> pd.DataFrame:
    result = factor_frame
    if date_from is not None:
        result = result.loc[result["date"] >= pd.Timestamp(date_from).normalize()]
    if date_to is not None:
        result = result.loc[result["date"] <= pd.Timestamp(date_to).normalize()]
    return result.reset_index(drop=True)


def write_daily_factors_for_symbol(
    input_path: Path,
    output_root: Path,
    overwrite: bool,
    factor_frame: pd.DataFrame,
) -> tuple[str, Path, int | None, int | None]:
    output_path = output_root / input_path.name
    if output_path.exists() and not overwrite:
        return ("skipped", output_path, None, None)

    symbol = normalize_symbol_id(input_path.name)
    symbol_factors = factor_frame.loc[factor_frame["code"] == symbol].copy()
    result = symbol_factors.rename(
        columns={"code": "ts_code", "date": "factor_date"}
    ).sort_values("factor_date")
    result = result.set_index("factor_date")
    result.index.name = "factor_date"
    result = result[["ts_code", *ALL_FACTOR_NAMES]]

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_root, suffix=".parquet", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        result.to_parquet(temporary_path)
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise
    return ("written", output_path, len(result), len(result.columns))


def main(
    argv: list[str] | None = None,
    default_asset_type: str = "etf",
) -> int:
    args = parse_args(argv, default_asset_type=default_asset_type)
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
    compute_date_from = resolve_compute_date_from(daily_base, args.date_from)
    if args.date_from is not None and args.date_to is not None and args.date_from > args.date_to:
        raise ValueError("--date-from must be on or before --date-to")

    worker_count = max(1, args.workers)
    LOGGER.info(
        "Processing %s %s minute parquet files for FZ daily factors",
        len(files),
        args.asset_type,
    )
    if args.date_from is not None or args.date_to is not None:
        LOGGER.info(
            "Writing factor dates from %s to %s (compute starts at %s for 20-day warmup)",
            args.date_from.date() if args.date_from is not None else "first available",
            args.date_to.date() if args.date_to is not None else "latest available",
            compute_date_from.date() if compute_date_from is not None else "first available",
        )
    if len(files) < 10:
        LOGGER.warning(
            "Only %s symbols selected; cross-sectional FZ outputs are suitable for smoke testing only.",
            len(files),
        )

    if args.stage_root is not None:
        args.stage_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{args.asset_type}_fz_stage_", dir=args.stage_root
    ) as stage_dir_name:
        stage_root = Path(stage_dir_name)
        skipped_days: list[tuple[str, str, str]] = []
        if worker_count == 1:
            for input_path in files:
                file_name, exported_days, invalid_days = build_trade_day_slice(
                    input_path, stage_root, compute_date_from, args.date_to
                )
                skipped_days.extend(
                    (file_name, trade_date, reason)
                    for trade_date, reason in invalid_days
                )
                LOGGER.info("Staged %s trade days for %s", exported_days, input_path.name)
        else:
            LOGGER.info("Using %s workers for staging", worker_count)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        build_trade_day_slice,
                        input_path,
                        stage_root,
                        compute_date_from,
                        args.date_to,
                    ): input_path
                    for input_path in files
                }
                for future in as_completed(future_map):
                    input_path = future_map[future]
                    file_name, exported_days, invalid_days = future.result()
                    skipped_days.extend(
                        (file_name, trade_date, reason)
                        for trade_date, reason in invalid_days
                    )
                    LOGGER.info("Staged %s trade days for %s", exported_days, file_name)

        if skipped_days:
            LOGGER.warning(
                "Skipped %s incomplete FZ code/date inputs; first entries: %s",
                len(skipped_days),
                skipped_days[:10],
            )

        raw_panel = calculate_raw_daily_panel(stage_root, daily_pv, worker_count)
        panel_with_daily = enrich_with_daily_base(raw_panel, daily_base)
        composed_panel = calculate_composed_panel(panel_with_daily)
        final_factor_frame = build_final_daily_factor_frame(composed_panel)
        final_factor_frame = filter_factor_date_range(
            final_factor_frame, args.date_from, args.date_to
        )

    failures: list[tuple[Path, str]] = []
    if worker_count == 1:
        for input_path in files:
            try:
                status, output_path, row_count, column_count = write_daily_factors_for_symbol(
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
                    write_daily_factors_for_symbol,
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
