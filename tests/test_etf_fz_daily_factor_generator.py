from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from scripts import generate_etf_fz_daily_factors as daily_entry
from scripts import generate_etf_fz_minute_factors as legacy_entry
from scripts import generate_fz_daily_factors as fz

from FractalQuant.factor.fz_factor import MinFreqFactor


def _complete_day_frame(trade_date: str = "2026-01-05") -> pd.DataFrame:
    morning = pd.date_range(f"{trade_date} 09:30", f"{trade_date} 11:30", freq="min")
    afternoon = pd.date_range(f"{trade_date} 13:01", f"{trade_date} 15:00", freq="min")
    return pd.DataFrame({"trade_time": morning.append(afternoon)})


def _base_keys() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": ["159001.SZ", "510300.SH"],
            "date": [date(2026, 1, 5), date(2026, 1, 5)],
        }
    )


def test_241_minute_grid_matches_orderbook_convention() -> None:
    frame = _complete_day_frame()

    assert len(frame) == 241
    assert fz.validate_fz_trade_day(frame) is None
    assert frame["trade_time"].dt.strftime("%H:%M").iloc[[0, 120, 121, -1]].tolist() == [
        "09:30",
        "11:30",
        "13:01",
        "15:00",
    ]


@pytest.mark.parametrize(
    ("frame", "reason_fragment"),
    [
        (_complete_day_frame().iloc[1:].copy(), "expected 241 rows"),
        (_complete_day_frame().iloc[:-1].copy(), "expected 241 rows"),
        (
            pd.concat([_complete_day_frame(), _complete_day_frame().iloc[[0]]], ignore_index=True),
            "duplicate trade_time",
        ),
        (
            pd.concat(
                [
                    _complete_day_frame().iloc[:-1],
                    pd.DataFrame({"trade_time": [pd.Timestamp("2026-01-05 13:00")] }),
                ],
                ignore_index=True,
            ),
            "invalid minute grid",
        ),
    ],
)
def test_invalid_fz_minute_grids_are_rejected(
    frame: pd.DataFrame, reason_fragment: str
) -> None:
    reason = fz.validate_fz_trade_day(frame)

    assert reason is not None
    assert reason_fragment in reason


def test_normalize_factor_output_collects_lazy_frame() -> None:
    base_keys = _base_keys()
    lazy_output = base_keys.with_columns(pl.Series("raw_value", [1.0, 2.0])).lazy()

    result = fz.normalize_factor_output("example_factor", lazy_output, base_keys)

    assert result.columns == ["code", "date", "example_factor"]
    assert result["example_factor"].to_list() == [1.0, 2.0]


def test_normalize_factor_output_handles_empty_and_rejects_bad_keys() -> None:
    base_keys = _base_keys()
    empty = pl.DataFrame(
        {
            "code": pl.Series([], dtype=pl.String),
            "date": pl.Series([], dtype=pl.Date),
            "value": pl.Series([], dtype=pl.Float64),
        }
    )
    empty_result = fz.normalize_factor_output("empty_factor", empty, base_keys)
    assert empty_result["empty_factor"].null_count() == len(base_keys)

    duplicate = pl.DataFrame(
        {
            "code": ["159001.SZ", "159001.SZ"],
            "date": [date(2026, 1, 5), date(2026, 1, 5)],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate code/date"):
        fz.normalize_factor_output("duplicate_factor", duplicate, base_keys)

    bad_columns = base_keys.with_columns(pl.lit(1.0).alias("left"), pl.lit(2.0).alias("right"))
    with pytest.raises(ValueError, match="returned 2 value columns"):
        fz.normalize_factor_output("bad_factor", bad_columns, base_keys)


def test_daily_writer_uses_factor_date_index_and_all_registered_factors(tmp_path: Path) -> None:
    factor_frame = pd.DataFrame(
        {
            "code": ["159001.SZ", "159001.SZ"],
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            **{name: [float(index), float(index + 1)] for index, name in enumerate(fz.ALL_FACTOR_NAMES)},
        }
    )
    status, output_path, row_count, column_count = fz.write_daily_factors_for_symbol(
        Path("159001.SZ.parquet"), tmp_path, False, factor_frame
    )
    written = pd.read_parquet(output_path)

    assert status == "written"
    assert row_count == 2
    assert column_count == 35
    assert written.index.name == "factor_date"
    assert written.columns.tolist() == ["ts_code", *fz.ALL_FACTOR_NAMES]
    assert written.index.tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    assert fz.write_daily_factors_for_symbol(
        Path("159001.SZ.parquet"), tmp_path, False, factor_frame
    )[0] == "skipped"


def test_compatibility_entrypoints_reuse_daily_engine() -> None:
    assert daily_entry.normalize_factor_output is fz.normalize_factor_output
    assert legacy_entry.normalize_factor_output is fz.normalize_factor_output


def test_asset_defaults_cover_stock_and_etf() -> None:
    stock_args = fz.parse_args(["--asset-type", "stock"])
    etf_args = fz.parse_args(["--asset-type", "etf"])

    assert stock_args.input_root == fz.STOCK_DEFAULT_INPUT_ROOT
    assert stock_args.daily_root == fz.STOCK_DEFAULT_DAILY_ROOT
    assert stock_args.output_root == fz.STOCK_DEFAULT_OUTPUT_ROOT
    assert etf_args.input_root == fz.ETF_DEFAULT_INPUT_ROOT
    assert etf_args.daily_root == fz.ETF_DEFAULT_DAILY_ROOT
    assert etf_args.output_root == fz.ETF_DEFAULT_OUTPUT_ROOT


def test_replication_api_collects_and_validates_daily_results() -> None:
    base = pl.DataFrame(
        {
            "code": ["000001.SZ", "000002.SZ"],
            "date": [date(2026, 1, 5), date(2026, 1, 5)],
            "factor": [1.0, 2.0],
        }
    )
    result = MinFreqFactor._normalize_daily_result(base.lazy())

    assert MinFreqFactor._normalize_daily_result(None) is None
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["code", "date", "factor"]

    empty = base.head(0)
    assert MinFreqFactor._normalize_daily_result(empty).is_empty()

    duplicate = pl.concat([base.head(1), base.head(1)])
    with pytest.raises(ValueError, match="重复的code/date"):
        MinFreqFactor._normalize_daily_result(duplicate)

    missing_keys = pl.DataFrame({"factor": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="缺少键列"):
        MinFreqFactor._normalize_daily_result(missing_keys)


def test_composed_factor_keeps_daily_keys_after_20_day_warmup() -> None:
    dates = pd.bdate_range("2026-01-01", periods=25).date
    codes = [f"{index:06d}.SZ" for index in range(10)]
    rows = [
        {
            "code": code,
            "date": factor_date,
            "YaoYanBoDongLv": float(code_index + day_index),
            "YaoYanShouYiLv": float(code_index * 2 - day_index),
        }
        for code_index, code in enumerate(codes)
        for day_index, factor_date in enumerate(dates)
    ]

    result = fz.fz_methods.cal_ShiDuMaoXian(pl.DataFrame(rows))

    assert result.height == 250
    assert result.select(["code", "date"]).n_unique() == 250
    assert result.filter(pl.col("ShiDuMaoXian").is_not_null()).height == 60


def test_date_range_keeps_20_trading_day_warmup_and_filters_output() -> None:
    daily_base = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2021-11-01", periods=46),
        }
    )
    factor_frame = pd.DataFrame(
        {
            "code": ["000001.SZ"] * 4,
            "date": pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03", "2022-01-04"]),
        }
    )

    assert fz.resolve_compute_date_from(daily_base, pd.Timestamp("2022-01-03")) == pd.Timestamp(
        "2021-12-06"
    )
    filtered = fz.filter_factor_date_range(
        factor_frame, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-03")
    )
    assert filtered["date"].tolist() == [pd.Timestamp("2022-01-03")]


def test_daily_pv_window_is_limited_to_20_dates_ending_on_factor_date() -> None:
    dates = pd.bdate_range("2021-11-01", periods=25).strftime("%Y-%m-%d").tolist()
    daily_pv = pl.DataFrame(
        {
            "Stkcd": ["000001.SZ"] * len(dates),
            "Trddt": dates,
            "Opnprc": [10.0] * len(dates),
            "Clsprc": [11.0] * len(dates),
            "Dsmvosd": [100.0] * len(dates),
        }
    )
    daily_dates, by_date = fz.partition_daily_pv_by_date(daily_pv)

    window = fz.select_daily_pv_window(
        daily_dates, by_date, pd.Timestamp(dates[-1])
    )

    assert window.height == 20
    assert window["Trddt"].to_list() == dates[-20:]
