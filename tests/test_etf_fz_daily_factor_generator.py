from __future__ import annotations

from datetime import date
import json
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
    assert column_count == 39
    assert written.index.name == "factor_date"
    assert written.columns.tolist() == ["ts_code", *fz.ALL_FACTOR_NAMES]
    assert written.index.tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    assert fz.write_daily_factors_for_symbol(
        Path("159001.SZ.parquet"), tmp_path, False, factor_frame
    )[0] == "skipped"


def test_compatibility_entrypoints_reuse_daily_engine() -> None:
    assert daily_entry.normalize_factor_output is fz.normalize_factor_output
    assert legacy_entry.normalize_factor_output is fz.normalize_factor_output


def test_asset_defaults_cover_stock_and_etf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    default_cao_mu_source = tmp_path / "stock_cao_mu_jie_bing_source.parquet"
    default_cao_mu_source.touch()
    monkeypatch.setattr(fz, "STOCK_DEFAULT_CAO_MU_SOURCE", default_cao_mu_source)
    stock_args = fz.parse_args(["--asset-type", "stock"])
    etf_args = fz.parse_args(["--asset-type", "etf"])
    strict_args = fz.parse_args(["--strict-source-fields"])
    build_args = fz.parse_args(["--asset-type", "stock", "--build-cao-mu-source"])

    assert stock_args.input_root == fz.STOCK_DEFAULT_INPUT_ROOT
    assert stock_args.daily_root == fz.STOCK_DEFAULT_DAILY_ROOT
    assert stock_args.output_root == fz.STOCK_DEFAULT_OUTPUT_ROOT
    assert etf_args.input_root == fz.ETF_DEFAULT_INPUT_ROOT
    assert etf_args.daily_root == fz.ETF_DEFAULT_DAILY_ROOT
    assert etf_args.output_root == fz.ETF_DEFAULT_OUTPUT_ROOT
    assert strict_args.strict_source_fields is True
    assert stock_args.cao_mu_source == default_cao_mu_source
    assert etf_args.cao_mu_source is None
    assert build_args.cao_mu_source == default_cao_mu_source


def test_daily_normalization_preserves_cao_mu_source_fields() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-01-05")],
            "ts_code": ["000001.SZ"],
            "open": [10.0],
            "close": [10.1],
            "total_share": [100.0],
            "retail_trade_ratio": ["0.25"],
            "csi_all_share_return": ["0.01"],
        }
    )

    result = fz.normalize_daily_frame(source)

    assert result.loc[0, "retail_trade_ratio"] == 0.25
    assert result.loc[0, "csi_all_share_return"] == 0.01


def test_cao_mu_source_availability_and_strict_policy() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    missing = pd.DataFrame({"trade_date": dates})
    partial = pd.DataFrame(
        {
            "trade_date": dates,
            "retail_trade_ratio": [0.2, None],
            "csi_all_share_return": [0.01, 0.02],
        }
    )
    complete = partial.fillna({"retail_trade_ratio": 0.3})

    unavailable = fz.assess_cao_mu_source_fields(missing)
    partial_result = fz.assess_cao_mu_source_fields(partial)
    available = fz.assess_cao_mu_source_fields(complete)

    assert unavailable["status"] == "unavailable"
    assert unavailable["missing_fields"] == list(fz.CAO_MU_REQUIRED_SOURCE_FIELDS)
    assert partial_result["status"] == "partial"
    assert partial_result["null_rows"]["retail_trade_ratio"] == 1
    assert available["status"] == "available"
    fz.enforce_source_field_policy(unavailable, strict=False)
    with pytest.raises(ValueError, match="not fully available"):
        fz.enforce_source_field_policy(partial_result, strict=True)


def test_calculate_retail_trade_ratio_uses_only_matched_buy_sell_ticks(
    tmp_path: Path,
) -> None:
    tick_path = tmp_path / "逐笔成交.csv"
    pd.DataFrame(
        {
            "成交代码": ["0", "0", "0", "0", "C"],
            "BS标志": ["B", "S", "B", "S", "B"],
            "成交价格": [100_000, 100_000, 100_000, 100_000, 100_000],
            "成交数量": [100, 200, 5_000, 6_000, 100_000],
        }
    ).to_csv(tick_path, index=False, encoding="gb18030")

    result = fz.calculate_retail_trade_ratio(tick_path)

    assert result == pytest.approx((1_000 + 2_000) / (2 * 113_000))


def test_build_and_merge_cao_mu_source_from_ticks_and_index_daily(
    tmp_path: Path,
) -> None:
    tick_root = tmp_path / "ticks"
    trade_date = pd.Timestamp("2026-01-05")
    tick_path = fz.tick_trade_path(tick_root, "000001.SZ", trade_date)
    tick_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "成交代码": ["0", "0", "0"],
            "BS标志": ["B", "S", "B"],
            "成交价格": [100_000, 100_000, 100_000],
            "成交数量": [100, 200, 5_000],
        }
    ).to_csv(tick_path, index=False, encoding="gb18030")

    index_path = tmp_path / "000985.CSI.parquet"
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "close": [100.0, 101.0, 99.0],
        }
    ).set_index("trade_date").to_parquet(index_path)
    daily_base = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "close": [10.0, 10.1],
        }
    )
    source_path = tmp_path / "cao_mu_source.parquet"

    source = fz.build_cao_mu_source(
        daily_base, tick_root, index_path, source_path
    )
    merged = fz.merge_cao_mu_source(daily_base, source_path)
    rebuilt = fz.build_cao_mu_source(
        daily_base, tick_root, index_path, source_path
    )

    first_ratio = (1_000 + 2_000) / (2 * 53_000)
    assert source_path.exists()
    assert source["retail_trade_ratio"].iloc[0] == pytest.approx(first_ratio)
    assert pd.isna(source["retail_trade_ratio"].iloc[1])
    assert source["csi_all_share_return"].tolist() == pytest.approx(
        [0.01, (99 / 101) - 1]
    )
    assert merged["retail_trade_ratio"].iloc[0] == pytest.approx(first_ratio)
    assert merged["csi_all_share_return"].iloc[1] == pytest.approx((99 / 101) - 1)
    assert rebuilt.equals(source)


def test_merge_cao_mu_source_preserves_explicit_daily_source_values(tmp_path: Path) -> None:
    source_path = tmp_path / "cao_mu_source.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [pd.Timestamp("2026-01-05")],
            "retail_trade_ratio": [0.2],
            "csi_all_share_return": [0.01],
        }
    ).to_parquet(source_path, index=False)
    daily_base = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [pd.Timestamp("2026-01-05")],
            "retail_trade_ratio": [0.3],
        }
    )

    merged = fz.merge_cao_mu_source(daily_base, source_path)

    assert merged["retail_trade_ratio"].iloc[0] == 0.3
    assert merged["csi_all_share_return"].iloc[0] == 0.01


def test_generation_manifest_records_schema_sources_and_target_scope(
    tmp_path: Path,
) -> None:
    source_availability = {
        "status": "unavailable",
        "required_fields": list(fz.CAO_MU_REQUIRED_SOURCE_FIELDS),
        "missing_fields": list(fz.CAO_MU_REQUIRED_SOURCE_FIELDS),
        "null_rows": {},
        "checked_rows": 10,
    }

    manifest_path = fz.write_generation_manifest(
        tmp_path,
        asset_type="stock",
        symbols_file=Path("csi300.txt"),
        symbol_count=10,
        source_availability=source_availability,
        skipped_days=[
            ("000001.SZ.parquet", "2026-01-05", "expected 241 rows, found 240"),
            ("000002.SZ.parquet", "2026-01-05", "expected 241 rows, found 240"),
        ],
        written_count=10,
        skipped_existing_count=0,
        date_from=pd.Timestamp("2026-01-01"),
        date_to=pd.Timestamp("2026-02-28"),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == fz.MANIFEST_NAME
    assert manifest["output_schema"]["column_count"] == 39
    assert manifest["factor_availability"]["CaoMuJieBing"]["status"] == "unavailable"
    assert manifest["calculation_universe"]["mode"] == "target_symbols"
    assert manifest["input_validation"]["invalid_code_dates"] == 2
    assert manifest["outputs"]["written_symbols"] == 10


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


def test_candidate_factor_columns_append_to_the_existing_schema() -> None:
    existing = (
        "YaoYanBoDongLv",
        "YaoYanShouYiLv",
        "ShiDuMaoXian",
        "QiangShiBanChaoXi",
        "RuoShiBanChaoXi",
        "ChaoXi",
        "MoHuGuanLianDu",
        "MoHuJinEBi",
        "MoHuJiaCha",
        "YunKaiWuSan",
        "PanDeng",
        "YongPanGaoFeng",
        "TiaoYueDu",
        "FeiEPuHuo",
        "RiBoDongLv",
        "CaoMuJieBing",
        "GuYanChuQun",
        "GaoDiECha",
        "SuiBoZhuLiu",
        "ShuiZhongXingZhou",
        "ZhaoMoChenWu",
        "WuBiGuMu",
        "YeMianShuangLu_t_intercept",
        "YeMianShuangLu",
        "HuaYinLinJian",
        "GenSuiXiShu",
        "DaiZhuErJiu",
        "ChengJiaoLiangBoYi_ShouYiLv",
        "ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi",
        "ZhenFuBoYi",
        "DuoKongBoYi",
        "ChengJiaoLiangXieTong",
        "XieTongJiaCha",
        "XieTongXiaoYing",
    )

    assert fz.ALL_FACTOR_NAMES[: len(existing)] == existing
    assert fz.ALL_FACTOR_NAMES[len(existing) :] == (
        "ChongJian",
        "ZaiHouChongJian",
        "YueYaoYanBoDongLv",
        "YueYaoYanShouYiLv",
    )


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
