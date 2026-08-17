from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from FractalQuant.factor import fz_methods
from scripts import generate_fz_daily_factors as generator


def _complete_minute_panel(factor_date: date, code: str = "A") -> pl.DataFrame:
    rows = []
    for minute_index, minute_time in enumerate(sorted(generator.EXPECTED_MINUTE_TIMES)):
        time = minute_time.hour * 10_000_000 + minute_time.minute * 100_000
        close = 100.0 + minute_index * 0.03 + (minute_index % 7) * 0.01
        rows.append(
            {
                "code": code,
                "date": factor_date,
                "time": time,
                "open": close - 0.01,
                "high": close + 0.05 + (minute_index % 3) * 0.01,
                "low": close - 0.04 - (minute_index % 2) * 0.01,
                "close": close,
                "volume": 100.0 + minute_index,
                "amount": 1_000.0 + minute_index,
            }
        )
    return pl.DataFrame(rows)


def _tide_panel(
    surge_volume: float,
    ebb_volume: float,
    peak_position: int = 121,
) -> pl.DataFrame:
    panel = _complete_minute_panel(date(2024, 1, 2)).to_pandas()
    panel["close"] = 100.0
    panel["volume"] = 100.0
    panel.loc[panel["time"].isin([93000000, 150000000]), "volume"] = 1e12

    retained = panel.loc[~panel["time"].isin([93000000, 150000000])].copy()
    retained["tide_position"] = np.arange(1, len(retained) + 1)
    for center, value in (
        (5, surge_volume),
        (peak_position, 1_000.0),
        (233, ebb_volume),
    ):
        positions = range(center - 4, center + 5)
        selected_times = retained.loc[
            retained["tide_position"].isin(positions), "time"
        ]
        panel.loc[panel["time"].isin(selected_times), "volume"] = value

    close_by_position = {5: 90.0, peak_position: 110.0, 233: 105.0}
    for position, close in close_by_position.items():
        selected_time = retained.loc[
            retained["tide_position"].eq(position), "time"
        ].iloc[0]
        panel.loc[panel["time"].eq(selected_time), "close"] = close
    return pl.from_pandas(panel, include_index=False)


def test_tide_uses_report_boundaries_and_excludes_open_close_minutes() -> None:
    panel = _tide_panel(surge_volume=1.0, ebb_volume=2.0)

    result = fz_methods._calculate_tidal_half_factors(panel)

    expected_rise = (110.0 / 90.0 - 1) / (121 - 5)
    expected_ebb = (105.0 / 110.0 - 1) / (233 - 121)
    np.testing.assert_allclose(result["QiangShiBanChaoXi"], [expected_rise])
    np.testing.assert_allclose(result["RuoShiBanChaoXi"], [expected_ebb])


def test_tide_assigns_ebb_to_strong_half_when_ebb_volume_is_lower() -> None:
    panel = _tide_panel(surge_volume=2.0, ebb_volume=1.0)

    result = fz_methods._calculate_tidal_half_factors(panel)

    expected_rise = (110.0 / 90.0 - 1) / (121 - 5)
    expected_ebb = (105.0 / 110.0 - 1) / (233 - 121)
    np.testing.assert_allclose(result["QiangShiBanChaoXi"], [expected_ebb])
    np.testing.assert_allclose(result["RuoShiBanChaoXi"], [expected_rise])


def test_tide_uses_earliest_minute_when_peak_volumes_tie() -> None:
    panel = _tide_panel(surge_volume=1.0, ebb_volume=2.0).to_pandas()
    retained = panel.loc[~panel["time"].isin([93000000, 150000000])].copy()
    retained["tide_position"] = np.arange(1, len(retained) + 1)
    tied_peak_times = retained.loc[
        retained["tide_position"].isin(range(76, 85)), "time"
    ]
    panel.loc[panel["time"].isin(tied_peak_times), "volume"] = 1_000.0
    earliest_peak_time = retained.loc[
        retained["tide_position"].eq(80), "time"
    ].iloc[0]
    panel.loc[panel["time"].eq(earliest_peak_time), "close"] = 120.0

    result = fz_methods._calculate_tidal_half_factors(
        pl.from_pandas(panel, include_index=False)
    )

    expected_rise = (120.0 / 90.0 - 1) / (80 - 5)
    np.testing.assert_allclose(result["QiangShiBanChaoXi"], [expected_rise])


def test_tide_returns_null_when_peak_has_no_valid_preceding_segment() -> None:
    panel = _tide_panel(surge_volume=1_000.0, ebb_volume=2.0, peak_position=5)

    result = fz_methods._calculate_tidal_half_factors(panel)

    assert result["QiangShiBanChaoXi"].null_count() == 1
    assert result["RuoShiBanChaoXi"].null_count() == 1


def test_tide_returns_null_for_invalid_daily_price() -> None:
    panel = _tide_panel(surge_volume=1.0, ebb_volume=2.0).with_columns(
        pl.when(pl.col("time") == 100000000)
        .then(0.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    result = fz_methods._calculate_tidal_half_factors(panel)

    assert result["QiangShiBanChaoXi"].null_count() == 1
    assert result["RuoShiBanChaoXi"].null_count() == 1


def test_chao_xi_does_not_compress_missing_trading_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=35)
    strong = [float(index) for index in range(len(dates))]
    weak = [float(index) / 2 for index in range(len(dates))]
    strong[9] = None
    panel = pl.DataFrame(
        {
            "code": ["A"] * len(dates),
            "date": [value.date() for value in dates],
            "QiangShiBanChaoXi": strong,
            "RuoShiBanChaoXi": weak,
        }
    )

    result = fz_methods.cal_ChaoXi(panel)
    valid_dates = result.filter(pl.col("ChaoXi").is_not_null())["date"].to_list()

    assert valid_dates[0] == dates[29].date()


def test_better_volatility_uses_the_report_233_minute_window() -> None:
    panel = _complete_minute_panel(date(2024, 1, 2))

    result = fz_methods._better_volatility_panel(panel).collect()

    assert result.height == 233
    assert result["trade_minute"].min() == 5
    assert result["trade_minute"].max() == 237
    assert result["time"].to_list()[0] == 93400000
    assert result["time"].to_list()[-1] == 145600000


def test_chong_jian_matches_the_daily_covariance_of_report_inputs() -> None:
    panel = _complete_minute_panel(date(2024, 1, 2))
    prepared = fz_methods._better_volatility_panel(panel).collect().drop_nulls(
        ["ret_to_vol", "better_volatility"]
    )
    expected = np.cov(
        prepared["ret_to_vol"].to_numpy(),
        prepared["better_volatility"].to_numpy(),
        ddof=1,
    )[0, 1]

    result = fz_methods.cal_ChongJian(panel)

    np.testing.assert_allclose(result["ChongJian"].to_numpy(), [expected])


def test_pan_deng_uses_the_same_prepared_data_as_chong_jian() -> None:
    panel = _complete_minute_panel(date(2024, 1, 2))
    prepared = fz_methods._better_volatility_panel(panel).collect().drop_nulls(
        ["ret_to_vol", "better_volatility"]
    )
    high_volatility = prepared.filter(
        pl.col("better_volatility")
        >= pl.col("better_volatility").mean() + pl.col("better_volatility").std()
    )
    expected = np.cov(
        high_volatility["ret_to_vol"].to_numpy(),
        high_volatility["better_volatility"].to_numpy(),
        ddof=1,
    )[0, 1]

    result = fz_methods.cal_PanDeng(panel)

    np.testing.assert_allclose(result["PanDeng"].to_numpy(), [expected])


def test_candidate_composed_factors_start_after_20_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=25)
    panel = pl.DataFrame(
        {
            "code": ["A"] * len(dates),
            "date": [value.date() for value in dates],
            "ChongJian": [float(index) for index in range(len(dates))],
            "YaoYanBoDongLv": [float(index) / 10 for index in range(len(dates))],
            "YaoYanShouYiLv": [float(index) / 5 for index in range(len(dates))],
        }
    )

    reconstruction = fz_methods.cal_ZaiHouChongJian(panel)
    volatility = fz_methods.cal_YueYaoYanBoDongLv(panel)
    returns = fz_methods.cal_YueYaoYanShouYiLv(panel)

    for result, column in (
        (reconstruction, "ZaiHouChongJian"),
        (volatility, "YueYaoYanBoDongLv"),
        (returns, "YueYaoYanShouYiLv"),
    ):
        assert result.filter(pl.col(column).is_not_null()).height == 6


def test_off_diagonal_correlation_excludes_self() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    result = fz_methods._mean_abs_off_diagonal_corr(corr)

    np.testing.assert_allclose(result.to_numpy(), [0.3, 0.4, 0.5])


def test_yao_yan_return_uses_close_to_close_return() -> None:
    rows = []
    for code, scale in (("A", 1.0), ("B", 1.5)):
        for offset in range(12):
            rows.append(
                {
                    "code": code,
                    "date": date(2024, 1, 2),
                    "time": 93600000 + offset * 100000,
                    "open": 100.0 + offset,
                    "close": (100.0 + offset * scale),
                    "volume": 100.0 + (1000.0 if offset == 6 else offset),
                }
            )
    minute = pl.DataFrame(rows)
    changed_open = minute.with_columns((pl.col("open") * 3).alias("open"))

    original = fz_methods.cal_YaoYanShouYiLv(minute).sort("code")
    changed = fz_methods.cal_YaoYanShouYiLv(changed_open).sort("code")

    np.testing.assert_allclose(
        original["YaoYanShouYiLv"].to_numpy(),
        changed["YaoYanShouYiLv"].to_numpy(),
        equal_nan=True,
    )
    assert original["YaoYanShouYiLv"].min() >= 0


def test_tiao_yue_du_is_the_intraday_residual_mean() -> None:
    close = np.array([100.0, 101.0, 103.0])
    minute = pl.DataFrame(
        {
            "code": ["A"] * len(close),
            "date": [date(2024, 1, 2)] * len(close),
            "time": [93500000, 93600000, 93700000],
            "close": close,
        }
    )
    simple_return = close[1:] / close[:-1] - 1
    log_return = np.log(close[1:] / close[:-1])
    expected = np.mean(2 * (simple_return - log_return) - log_return**2)

    result = fz_methods.cal_TiaoYueDu(minute)

    np.testing.assert_allclose(result["TiaoYueDu"].to_numpy(), [expected])


def test_sui_bo_zhu_liu_starts_after_exactly_20_dates() -> None:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=20)
    for index, factor_date in enumerate(dates):
        for code, multiplier in (("A", 1.0), ("B", 2.0), ("C", -1.0)):
            rows.append(
                {
                    "code": code,
                    "date": factor_date.date(),
                    "GaoDiECha": multiplier * index + (index % 3),
                }
            )

    result = fz_methods.cal_SuiBoZhuLiu(pl.DataFrame(rows))

    assert result is not None
    assert result["date"].unique().to_list() == [dates[-1].date()]


def test_gao_di_e_cha_uses_open_and_excludes_equal_threshold() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    daily = pl.DataFrame(
        {
            "Stkcd": ["A"] * len(dates),
            "Trddt": dates.strftime("%Y-%m-%d").tolist(),
            "Opnprc": [100.0] * len(dates),
            "Clsprc": [101.0] * len(dates),
            "Dsmvosd": [100.0] * len(dates),
        }
    )
    minute = pl.DataFrame(
        {
            "code": ["A", "A"],
            "date": [dates[-1].date()] * 2,
            "time": [93600000, 93700000],
            "open": [100.0, 100.0],
            "close": [101.0, 102.0],
            "amount": [1.0, 1.0],
        }
    )

    result = fz_methods.cal_GaoDiECha(minute, daily)

    # The first bar equals the 1% threshold and is neither high nor low;
    # the second bar is high relative to the opening price.
    assert result["GaoDiECha"].to_list() == [1e-05]


def test_cheng_jiao_liang_xie_tong_is_stable_for_unordered_input() -> None:
    rows = []
    for code, base in (("A", 100.0), ("B", 100.5), ("C", 101.0)):
        for offset in range(8):
            close = base + offset * (1.0 if code != "B" else -0.5)
            rows.append(
                {
                    "code": code,
                    "date": date(2024, 1, 2),
                    "time": 93600000 + offset * 100000,
                    "open": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 100.0 + offset * (2 if code == "A" else 3),
                }
            )
    minute = pl.DataFrame(rows)
    shuffled = minute.sample(fraction=1.0, shuffle=True, seed=7)

    original = fz_methods.cal_ChengJiaoLiangXieTong(minute).sort("code")
    changed = fz_methods.cal_ChengJiaoLiangXieTong(shuffled).sort("code")

    np.testing.assert_allclose(
        original["ChengJiaoLiangXieTong"].to_numpy(),
        changed["ChengJiaoLiangXieTong"].to_numpy(),
        equal_nan=True,
    )


def test_cheng_jiao_liang_xie_tong_includes_the_current_stock_in_group_volume() -> None:
    rows = []
    for code, base_volume in (("A", 100.0), ("B", 200.0)):
        for offset in range(8):
            price = 100.0 + offset
            rows.append(
                {
                    "code": code,
                    "date": date(2024, 1, 2),
                    "time": 93600000 + offset * 100000,
                    "open": price,
                    "high": price + 0.1,
                    "low": price - 0.1,
                    "close": price,
                    "volume": base_volume + offset,
                }
            )

    result = fz_methods.cal_ChengJiaoLiangXieTong(pl.DataFrame(rows))

    # Both stocks stay in the same neutral state, so the group volume is the
    # total market share and has zero variance; it must not become -own_share.
    assert result["ChengJiaoLiangXieTong"].null_count() == 2


def test_hua_yin_lin_jian_does_not_turn_missing_correlation_into_zero() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    panel = pl.DataFrame(
        {
            "code": ["A"] * len(dates),
            "date": [value.date() for value in dates],
            "ZhaoMoChenWu": [1.0] * len(dates),
            "WuBiGuMu": [1.0] * len(dates),
            "YeMianShuangLu": [None] * len(dates),
        }
    )

    result = fz_methods.cal_HuaYinLinJian(panel)

    assert result["HuaYinLinJian"].null_count() == len(dates)


def test_all_raw_factors_accept_a_complete_multi_symbol_day() -> None:
    factor_date = date(2024, 2, 1)
    codes = ["A", "B", "C"]
    times = sorted(generator.EXPECTED_MINUTE_TIMES)
    minute_rows = []
    for code_index, code in enumerate(codes):
        for minute_index, minute_time in enumerate(times):
            timestamp = minute_time.hour * 10000000 + minute_time.minute * 100000
            close = 100.0 + code_index + minute_index * 0.01
            minute_rows.append(
                {
                    "code": code,
                    "date": factor_date,
                    "time": timestamp,
                    "open": close - 0.02,
                    "high": close + 0.03,
                    "low": close - 0.03,
                    "close": close,
                    "volume": 100.0 + minute_index + code_index,
                    "amount": 1000.0 + minute_index * 2 + code_index,
                }
            )
    minute = pl.DataFrame(minute_rows)
    daily_dates = pd.bdate_range(end=pd.Timestamp(factor_date), periods=20)
    daily_rows = [
        {
            "Stkcd": code,
            "Trddt": value.strftime("%Y-%m-%d"),
            "Opnprc": 99.0 + code_index,
            "Clsprc": 100.0 + code_index,
            "Dsmvosd": 1_000.0,
        }
        for code_index, code in enumerate(codes)
        for value in daily_dates
    ]
    daily_pv = pl.DataFrame(daily_rows)
    base_keys = generator.build_base_keys(minute)

    for spec in generator.RAW_FACTOR_SPECS:
        raw = (
            spec.function(minute, daily_pv)
            if spec.needs_daily_pv
            else spec.function(minute)
        )
        normalized = generator.normalize_factor_output(spec.name, raw, base_keys)
        assert normalized.height == len(codes)
        assert spec.name in normalized.columns


def test_cao_mu_jie_bing_is_null_without_required_source_fields() -> None:
    minute = pl.DataFrame(
        {
            "code": ["A", "A"],
            "date": [date(2024, 1, 2)] * 2,
            "time": [93500000, 93600000],
            "close": [10.0, 10.1],
            "RiBoDongLv": [0.02, 0.02],
            "cmc": [100.0, 100.0],
        }
    )

    result = fz_methods.cal_CaoMuJieBing(minute)

    assert result.columns == ["code", "date", "CaoMuJieBing"]
    assert result["CaoMuJieBing"].null_count() == 1


def test_cao_mu_jie_bing_matches_report_formula_and_five_observation_minimum() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    daily_returns = np.linspace(0.01, 0.12, len(dates) - 1)
    closes = [100.0]
    for daily_return in daily_returns:
        closes.append(closes[-1] * (1 + daily_return))
    panel = pl.DataFrame(
        {
            "code": ["A"] * len(dates),
            "date": [value.date() for value in dates],
            "close": closes,
            "RiBoDongLv": [0.02] * len(dates),
            "retail_trade_ratio": [0.4] * len(dates),
            "csi_all_share_return": [0.0] * len(dates),
        }
    )

    result = fz_methods.cal_CaoMuJieBing(panel).to_pandas()
    expected = pd.DataFrame({"close": closes})
    expected["daily_return"] = expected["close"].pct_change(fill_method=None)
    expected["fear"] = expected["daily_return"].abs() / (
        expected["daily_return"].abs() + 0.1
    )
    expected["fear_decay"] = expected["fear"] - (
        expected["fear"].shift(1) + expected["fear"].shift(2)
    ) / 2
    expected.loc[expected["fear_decay"] <= 0, "fear_decay"] = np.nan
    expected["daily_score"] = (
        0.4 * 0.02 * expected["fear_decay"] * expected["daily_return"]
    )
    expected["factor"] = (
        expected["daily_score"].rolling(20, min_periods=5).mean()
        + expected["daily_score"].rolling(20, min_periods=5).std(ddof=1)
    ) / 2

    np.testing.assert_allclose(
        result["CaoMuJieBing"], expected["factor"], equal_nan=True
    )
    assert result["CaoMuJieBing"].first_valid_index() == 7


def test_xie_tong_jia_cha_uses_daily_previous_close() -> None:
    rows = []
    for code in ("A", "B"):
        for offset in range(10):
            close = 100.0 + offset if code == "A" else 100.0 + offset * 0.5
            rows.append(
                {
                    "code": code,
                    "date": date(2024, 1, 2),
                    "time": 93500000 + offset * 100000,
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": 100.0 + offset,
                }
            )
    minute = pl.DataFrame(rows)
    daily = pl.DataFrame(
        {
            "Stkcd": ["A", "B", "A", "B"],
            "Trddt": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "Clsprc": [100.0, 100.0, 109.0, 104.5],
        }
    )

    result = fz_methods.cal_XieTongJiaCha(minute, daily)

    result = result.sort("code")
    assert result.height == 2
    assert result.schema["XieTongJiaCha"] == pl.Float64
    np.testing.assert_allclose(
        result["XieTongJiaCha"].to_numpy(),
        [0.045, -0.045],
    )
