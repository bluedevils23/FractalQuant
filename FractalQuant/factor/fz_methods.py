import pandas as pd
import polars as pl
import statsmodels.api as sm
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import torch

"""
    =========================
        方正金工分钟频因子
    =========================
"""


# 常用函数

def r_mean_20_ts_pl(col_name: str) -> pl.Expr:
    """时序滚动20日均值"""
    return (
        pl.col(col_name)
        .rolling_mean(20, min_samples=20)
        .over('code')
        .alias(f'{col_name}_mean')
    )


def r_std_20_ts_pl(col_name: str) -> pl.Expr:
    """时序滚动20日标准差"""
    return (
        pl.col(col_name)
        .rolling_std(20, min_samples=20, ddof=0)
        .over('code')
        .alias(f'{col_name}_std')
    )


def standardize_cs_pl(col_name: str) -> pl.Expr:
    """截面标准化"""
    return (
        pl.when(
            pl.col(col_name).std(ddof=0) != 0
        ).then(
            (
                    pl.col(col_name) - pl.col(col_name).mean()
            ) / pl.col(col_name).std(ddof=0)
        ).otherwise(
            pl.col(col_name) - pl.col(col_name).mean()
        ).over('date')
        .alias(col_name)
    )


def _mean_abs_off_diagonal_corr(corr: pd.DataFrame) -> pd.Series:
    """Return each column's mean absolute correlation with the other columns."""
    if corr.empty:
        return pd.Series(dtype="float64")
    values = corr.abs().to_numpy(copy=True, dtype="float64")
    np.fill_diagonal(values, np.nan)
    valid_counts = np.sum(~np.isnan(values), axis=1)
    means = np.divide(
        np.nansum(values, axis=1),
        valid_counts,
        out=np.full(values.shape[0], np.nan, dtype="float64"),
        where=valid_counts != 0,
    )
    return pd.Series(means, index=corr.index)


def _trade_minute_expr() -> pl.Expr:
    """Return the one-based minute number on the validated 241-minute grid."""
    clock_minute = (
        pl.col('time') // 10_000_000 * 60
        + (pl.col('time') % 10_000_000 // 100_000)
    )
    return (
        pl.when(clock_minute < 720)
        .then(clock_minute - 569)
        .otherwise(clock_minute - 659)
        .cast(pl.Int64)
    )


def _report_minute_mask() -> pl.Expr:
    """The 233 observations used after the five-minute warm-up."""
    return (pl.col('trade_minute') >= 5) & (pl.col('trade_minute') <= 237)


def _better_volatility_panel(df: pl.DataFrame) -> pl.LazyFrame:
    """Build the report's OHLC-based volatility and return/volatility ratio."""
    return (
        df.lazy()
        .sort(['code', 'date', 'time'])
        .with_columns(
            _trade_minute_expr().alias('trade_minute'),
            pl.concat_arr(
                pl.col(col).shift(i).over(['code', 'date'])
                for i in range(0, 5)
                for col in ['open', 'high', 'low', 'close']
            ).alias('info_arr'),
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('return'),
        )
        .with_columns(
            (
                pl.col('info_arr').arr.std(ddof=0)
                / pl.col('info_arr').arr.mean()
            ).pow(2).alias('better_volatility')
        )
        .with_columns(
            pl.when(
                pl.col('better_volatility').is_not_null()
                & (pl.col('better_volatility') != 0)
            )
            .then(pl.col('return') / pl.col('better_volatility'))
            .otherwise(None)
            .alias('ret_to_vol')
        )
        .filter(_report_minute_mask())
    )


# 适度冒险：耀眼波动率和耀眼收益率的20日均值与20日标准差4个因子等权合成

def cal_YaoYanBoDongLv(df: pl.DataFrame):
    """
    适度冒险因子的中间因子：耀眼波动率
    :param df:
    :return:
    """
    return (
        df.lazy().sort(['code', 'date', 'time']).with_columns(
            _trade_minute_expr().alias('trade_minute'),
            pl.col('volume')
            .diff()
            .over(['code', 'date'])
            .alias('vol_diff'),
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            (
                pl.col('pct_change')
                .shift(-i)
                .over(['code', 'date'])
                .alias(f'pct_change_shift_-{i}')
            ) for i in range(0, 5)
        ).filter(_report_minute_mask()).with_columns(
            pl.when(
                pl.col('vol_diff') > (
                        pl.col('vol_diff').mean()
                        + pl.col('vol_diff').std(ddof=0)
                ).over(['code', 'date'])
            )
            .then(
                pl.concat_arr(
                    pl.col(f'pct_change_shift_-{i}') for i in range(0, 5)
                ).arr.std(ddof=0)
            )
            .otherwise(None)
            .alias('YaoYanBoDongLv')
        ).group_by(['code', 'date']).agg(
            pl.col('YaoYanBoDongLv')
            .mean()
        ).with_columns(
            (
                    pl.col('YaoYanBoDongLv')
                    - pl.col('YaoYanBoDongLv').mean()
            )
            .abs()
            .over('date')
            .alias('YaoYanBoDongLv')
        ).collect()
    )


def cal_YaoYanShouYiLv(df: pl.DataFrame):
    """
    适度冒险因子的中间因子：耀眼收益率
    :param df:
    :return:
    """
    return (
        df.lazy().sort(['code', 'date', 'time']).with_columns(
            _trade_minute_expr().alias('trade_minute'),
            pl.col('volume')
            .diff()
            .over(['code', 'date'])
            .alias('vol_diff'),
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).filter(_report_minute_mask()).with_columns(
            pl.when(
                pl.col('vol_diff') > (
                        pl.col('vol_diff').mean()
                        + pl.col('vol_diff').std(ddof=0)
                ).over(['code', 'date'])
            )
            .then(1)
            .otherwise(0)
            .alias('raise_moment')
        ).with_columns(
            pl.when(pl.col('raise_moment') == 1)
            .then(
                pl.col('pct_change')
            )
            .otherwise(None)
            .alias('YaoYanShouYiLv')
        ).group_by(['code', 'date']).agg(
            pl.col('YaoYanShouYiLv')
            .mean()
        ).with_columns(
            (
                    pl.col('YaoYanShouYiLv')
                    - pl.col('YaoYanShouYiLv').mean()
            ).abs().over('date')
            .alias('YaoYanShouYiLv')
        ).collect()
    )


def cal_ShiDuMaoXian(df: pl.DataFrame):
    """
    适度冒险计算方法
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .with_columns(
            (
                    r_mean_20_ts_pl('YaoYanBoDongLv') + r_std_20_ts_pl('YaoYanBoDongLv')
            ).alias('YaoYanBoDongLv'),
            (
                    r_mean_20_ts_pl('YaoYanShouYiLv') + r_std_20_ts_pl('YaoYanShouYiLv')
            ).alias('YaoYanShouYiLv')
        ).with_columns(
            standardize_cs_pl('YaoYanBoDongLv'),
            standardize_cs_pl('YaoYanShouYiLv')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('YaoYanBoDongLv') + pl.col('YaoYanShouYiLv'))
            .alias('ShiDuMaoXian')
        ).collect()
    )


# 潮汐因子：强势半潮汐的20日均值和弱势半潮汐的20日标准差等权合成

def _calculate_tidal_half_factors(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate the report-defined strong and weak half-tide daily factors."""
    schema = {
        'code': pl.String,
        'date': pl.Date,
        'QiangShiBanChaoXi': pl.Float64,
        'RuoShiBanChaoXi': pl.Float64,
    }
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    minute = df.select(
        ['code', 'date', 'time', 'close', 'volume']
    ).to_pandas().sort_values(['code', 'date', 'time'], kind='mergesort')
    results: list[dict[str, object]] = []

    for (code, factor_date), day in minute.groupby(
        ['code', 'date'], sort=False, observed=True
    ):
        day = day.loc[~day['time'].isin([93000000, 150000000])].copy()
        strong_rate = np.nan
        weak_rate = np.nan

        numeric = day[['close', 'volume']].to_numpy(dtype='float64', copy=False)
        valid_day = (
            len(day) == 239
            and np.isfinite(numeric).all()
            and (day['close'] > 0).all()
            and (day['volume'] >= 0).all()
        )
        if valid_day:
            day['tide_minute'] = np.arange(1, len(day) + 1)
            day['neighborhood_volume'] = day['volume'].rolling(
                window=9, min_periods=9, center=True
            ).sum()
            peak_candidates = day.dropna(subset=['neighborhood_volume']).sort_values(
                ['neighborhood_volume', 'tide_minute'],
                ascending=[False, True],
                kind='mergesort',
            )
            if not peak_candidates.empty:
                peak = peak_candidates.iloc[0]
                pre_peak = day.loc[
                    day['tide_minute'].between(5, int(peak['tide_minute']) - 1)
                ].sort_values(
                    ['neighborhood_volume', 'tide_minute'],
                    ascending=[True, True],
                    kind='mergesort',
                )
                post_peak = day.loc[
                    (day['tide_minute'] > peak['tide_minute'])
                    & (day['tide_minute'] <= 233)
                ].sort_values(
                    ['neighborhood_volume', 'tide_minute'],
                    ascending=[True, True],
                    kind='mergesort',
                )

                if not pre_peak.empty and not post_peak.empty:
                    surge = pre_peak.iloc[0]
                    ebb = post_peak.iloc[0]
                    rise_minutes = peak['tide_minute'] - surge['tide_minute']
                    ebb_minutes = ebb['tide_minute'] - peak['tide_minute']
                    if rise_minutes > 0 and ebb_minutes > 0:
                        rise_rate = (
                            peak['close'] / surge['close'] - 1
                        ) / rise_minutes
                        ebb_rate = (
                            ebb['close'] / peak['close'] - 1
                        ) / ebb_minutes
                        if surge['neighborhood_volume'] < ebb['neighborhood_volume']:
                            strong_rate, weak_rate = rise_rate, ebb_rate
                        else:
                            strong_rate, weak_rate = ebb_rate, rise_rate

        results.append(
            {
                'code': str(code),
                'date': factor_date,
                'QiangShiBanChaoXi': strong_rate,
                'RuoShiBanChaoXi': weak_rate,
            }
        )

    return pl.from_pandas(pd.DataFrame(results), schema_overrides=schema)


def cal_QiangShiBanChaoXi(df: pl.DataFrame):
    """潮汐因子的中间因子：强势半潮汐。"""
    return _calculate_tidal_half_factors(df).select(
        ['code', 'date', 'QiangShiBanChaoXi']
    )


def cal_ChaoXi(df: pl.DataFrame):
    """
    潮汐因子的计算方式
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .with_columns(
            r_mean_20_ts_pl('QiangShiBanChaoXi'),
            r_std_20_ts_pl('RuoShiBanChaoXi')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                    pl.col('QiangShiBanChaoXi_mean')
                    + pl.col('RuoShiBanChaoXi_std')
            ).alias('ChaoXi')
        ).collect()
        .sort(by=['code', 'date'])
    )


# 云开雾散：模糊关联度、模糊金额比和修正模糊价差的20日均值等权合成
### 修正模糊价差的算法为将模糊价差负的部分先除以过去10日负的部分的标准差，再按截面进行调整，除以负的部分的和再乘以总的和。

def cal_MoHuGuanLianDu(df: pl.DataFrame):
    """
    云开雾散的中间因子：模糊关联度
    :param df:
    :return:
    """
    return (
        df.lazy().sort(['code', 'date', 'time']).filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            pl.col('pct_change')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('volatility')
        ).with_columns(
            pl.col('volatility')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('ambiguity')
        ).filter(
            pl.col('time') >= 94000000
        ).group_by(['code', 'date']).agg(
            pl.corr(
                pl.col('ambiguity'),
                pl.col('amount')
            )
            .alias('MoHuGuanLianDu')
        ).collect()
    )


def cal_MoHuJinEBi(df: pl.DataFrame):
    """
    云开雾散的中间因子：模糊金额比
    :param df:
    :return:
    """
    return (
        df.lazy().sort(['code', 'date', 'time']).filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            pl.col('pct_change')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('volatility')
        ).with_columns(
            pl.col('volatility')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('ambiguity')
        ).filter(
            pl.col('time') >= 94000000
        ).group_by(['code', 'date']).agg(
            (
                    pl.col('amount')
                    .filter(
                        pl.col('ambiguity') > pl.col('ambiguity').mean()
                    )
                    .mean() / pl.col('amount').mean()
            ).alias('MoHuJinEBi')
        ).collect()
    )


def cal_MoHuJiaCha(df: pl.DataFrame):
    """
    云开雾散的中间因子：模糊价差
    :param df:
    :return:
    """
    return (
        df.lazy().sort(['code', 'date', 'time']).filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            pl.col('pct_change')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('volatility')
        ).with_columns(
            pl.col('volatility')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('ambiguity')
        ).filter(
            pl.col('time') >= 94000000
        ).group_by(['code', 'date']).agg(
            (
                (
                    pl.col('amount').filter(
                        pl.col('ambiguity') > pl.col('ambiguity').mean()
                    ).mean() / pl.col('amount').mean()
                ) - (
                    pl.col('volume').filter(
                        pl.col('ambiguity') > pl.col('ambiguity').mean()
                    ).mean() / pl.col('volume').mean()
                )
            ).alias('MoHuJiaCha')
        ).collect()
    )


def cal_YunKaiWuSan(df: pl.DataFrame):
    """
    云开雾散计算方法
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .filter(
            (~pl.col('MoHuJiaCha').is_null()) & (~pl.col('MoHuJiaCha').is_nan())
        )
        .with_columns(
            pl.col('MoHuJiaCha')
            .rolling_std(10, min_samples=10, ddof=0)
            .over('code')
            .alias('MoHuJiaCha_std')
        ).with_columns(
            pl.when(
                (pl.col('MoHuJiaCha') < 0)
                & (pl.col('MoHuJiaCha_std') != 0)
            ).then(
                pl.col('MoHuJiaCha') / pl.col('MoHuJiaCha_std')
            ).otherwise(pl.col('MoHuJiaCha'))
            .alias('XiuZhengMoHuJiaCha_raw'),
            pl.col('MoHuJiaCha')
            .filter(pl.col('MoHuJiaCha') < 0)
            .sum()
            .over('date')
            .alias('s1')
        ).with_columns(
            pl.col('XiuZhengMoHuJiaCha_raw')
            .filter(pl.col('XiuZhengMoHuJiaCha_raw') < 0)
            .sum()
            .over('date')
            .alias('s2')
        ).with_columns(
            pl.when(
                (pl.col('XiuZhengMoHuJiaCha_raw') < 0)
                & (pl.col('s2') != 0)
            ).then(
                pl.col('XiuZhengMoHuJiaCha_raw')
                * pl.col('s1')
                / pl.col('s2')
            ).otherwise(pl.col('XiuZhengMoHuJiaCha_raw'))
            .alias('XiuZhengMoHuJiaCha')
        ).drop_nans()
        .with_columns(
            r_mean_20_ts_pl('MoHuGuanLianDu'),
            r_mean_20_ts_pl('MoHuJinEBi'),
            r_mean_20_ts_pl('XiuZhengMoHuJiaCha')
        ).with_columns(
            standardize_cs_pl('MoHuGuanLianDu_mean'),
            standardize_cs_pl('MoHuJinEBi_mean'),
            standardize_cs_pl('XiuZhengMoHuJiaCha_mean')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('MoHuGuanLianDu_mean')
                + pl.col('MoHuJinEBi_mean')
                + pl.col('XiuZhengMoHuJiaCha_mean')
            ).alias('YunKaiWuSan')
        ).collect()
    )


# 灾后重建与勇攀高峰：共享更优波动率，分别使用全日和高波动时段协方差

def cal_ChongJian(df: pl.DataFrame):
    """灾后重建的日频中间因子：全天收益波动比与更优波动率的协方差。"""
    return (
        _better_volatility_panel(df)
        .group_by(['code', 'date'])
        .agg(
            pl.cov(pl.col('ret_to_vol'), pl.col('better_volatility'))
            .alias('ChongJian')
        ).collect()
    )


def cal_ZaiHouChongJian(df: pl.DataFrame):
    """灾后重建：日协方差的20日均值与标准差之和。"""
    return (
        df.lazy().sort(['code', 'date'])
        .with_columns(
            r_mean_20_ts_pl('ChongJian'),
            r_std_20_ts_pl('ChongJian'),
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('ChongJian_mean') + pl.col('ChongJian_std'))
            .alias('ZaiHouChongJian'),
        ).collect()
    )


def cal_PanDeng(df: pl.DataFrame):
    """勇攀高峰的中间因子：高波动时段的收益波动比协方差。"""
    return (
        _better_volatility_panel(df)
        .filter(
            pl.col('better_volatility') >= (
                pl.col('better_volatility').mean()
                + pl.col('better_volatility').std()
            ).over(['code', 'date'])
        ).group_by(['code', 'date']).agg(
            pl.cov(pl.col('ret_to_vol'), pl.col('better_volatility'))
            .alias('PanDeng')
        ).collect()
    )


def cal_YongPanGaoFeng(df: pl.DataFrame):
    """
    勇攀高峰计算方法
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .drop_nans()
        .with_columns(
            r_mean_20_ts_pl('PanDeng'),
            r_std_20_ts_pl('PanDeng')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('PanDeng_mean') + pl.col('PanDeng_std'))
            .alias('YongPanGaoFeng')
        ).collect()
    )


# 飞蛾扑火：跳跃度的20日均值与标准差、修正振幅1和修正振幅2的20日均值四者等权合成

def cal_TiaoYueDu(df: pl.DataFrame):
    """
    飞蛾扑火的中间因子：跳跃度
    :param df:
    :return:
    """
    return (
        df.lazy().sort(['code', 'date', 'time']).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change'),
            pl.col('close')
            .log()
            .diff()
            .over(['code', 'date'])
            .alias('log_change')
        ).filter(
            (pl.col('time') >= 93500000)
            & (pl.col('time') <= 145300000)
        ).with_columns(
            (
                    (pl.col('pct_change') - pl.col('log_change')) * 2
                    - pl.col('log_change').pow(2)
            )
            .alias('taylor_residual')
        ).group_by(['code', 'date']).agg(
            pl.col('taylor_residual')
            .mean()
            .alias('TiaoYueDu')
        ).collect()
    )


def cal_FeiEPuHuo(df: pl.DataFrame):
    """
    飞蛾扑火计算方法。需要使用到量价数据：high/low/close
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .with_columns(
            ((pl.col('high') - pl.col('low')) / pl.col('close').shift(1))
            .over('code')
            .alias('range'),
            (pl.col('high') / pl.col('low').shift(1) - 1)
            .over('code')
            .alias('single_return'),
            (pl.col('high') / pl.col('low').shift(1))
            .over('code')
            .log()
            .alias('log_return')
        ).with_columns(
            pl.when(
                pl.col('TiaoYueDu') >= pl.col('TiaoYueDu').mean().over('date')
            )
            .then(pl.col('range'))
            .otherwise(-pl.col('range'))
            .alias('FanZhuanZhenFu'),
            (
                    (pl.col('single_return') - pl.col('log_return')) * 2
                    - pl.col('log_return').pow(2)
            )
            .alias('taylor_residual')
        ).with_columns(
            pl.when(
                pl.col('taylor_residual')
                >= pl.col('taylor_residual').mean().over('date')
            )
            .then(pl.col('range'))
            .otherwise(-pl.col('range'))
            .alias('FanZhuanZhenFu2')
        ).drop_nans()
        .with_columns(
            (r_mean_20_ts_pl('TiaoYueDu') + r_std_20_ts_pl('TiaoYueDu'))
            .alias('TiaoYueDu'),
            r_mean_20_ts_pl('FanZhuanZhenFu'),
            r_mean_20_ts_pl('FanZhuanZhenFu2')
        ).with_columns(
            standardize_cs_pl('TiaoYueDu'),
            standardize_cs_pl('FanZhuanZhenFu_mean'),
            standardize_cs_pl('FanZhuanZhenFu2_mean')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                    pl.col('TiaoYueDu')
                    + pl.col('FanZhuanZhenFu_mean')
                    + pl.col('FanZhuanZhenFu2_mean')
            ).alias('FeiEPuHuo')
        ).collect()
    )


# 草木皆兵

def cal_RiBoDongLv(df: pl.DataFrame):
    """草木皆兵的中间因子：日波动率"""
    return (
        df.lazy().sort(['code', 'date', 'time']).select(
            pl.col('code'),
            pl.col('date'),
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('min_return')
        ).drop_nans()
        .group_by(['code', 'date']).agg(
            pl.col('min_return').std()
            .alias('RiBoDongLv')
        ).collect()
    )


def cal_CaoMuJieBing(df: pl.DataFrame):
    """草木皆兵因子计算方法"""
    keys = df.select('code', 'date').unique().sort(['code', 'date'])
    required_columns = {'retail_trade_ratio', 'csi_all_share_return'}
    if not required_columns.issubset(df.columns):
        return keys.with_columns(
            pl.lit(None, dtype=pl.Float64).alias('CaoMuJieBing')
        )
    return (
        df.lazy().sort(['code', 'date']).with_columns(
            pl.col('close')
            .pct_change()
            .over('code')
            .alias('daily_return')
        ).with_columns(
            (
                (
                    pl.col('daily_return') - pl.col('csi_all_share_return')
                ) / (
                    pl.col('daily_return').abs()
                    + pl.col('csi_all_share_return').abs()
                    + 0.1
                )
            ).abs().alias('JingKongDu')
        ).with_columns(
            (
                pl.col('JingKongDu')
                - (
                    pl.col('JingKongDu').shift(1)
                    + pl.col('JingKongDu').shift(2)
                ) / 2
            )
            .over('code')
            .alias('JingKongDu_ShuaiJian')
        ).with_columns(
            pl.when(pl.col('JingKongDu_ShuaiJian') > 0)
            .then(pl.col('JingKongDu_ShuaiJian'))
            .otherwise(None)
            .alias('JingKongDu_ShuaiJian')
        ).with_columns(
            (
                pl.col('retail_trade_ratio')
                * pl.col('RiBoDongLv')
                * pl.col('JingKongDu_ShuaiJian')
                * pl.col('daily_return')
            ).alias('CaoMuJieBing_daily')
        ).with_columns(
            pl.col('CaoMuJieBing_daily')
            .rolling_mean(20, min_samples=5)
            .over('code')
            .alias('CaoMuJieBing_mean'),
            pl.col('CaoMuJieBing_daily')
            .rolling_std(20, min_samples=5, ddof=1)
            .over('code')
            .alias('CaoMuJieBing_std'),
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                (
                    pl.col('CaoMuJieBing_mean')
                    + pl.col('CaoMuJieBing_std')
                ) / 2
            ).alias('CaoMuJieBing')
        ).collect()
    )


# 水中行舟：孤雁出群的20日均值与标准差和随波逐流三者等权合成

def cal_GuYanChuQun(df: pl.DataFrame):
    """
    水中行舟的中间因子：孤雁出群
    :param df:
    :return:
    """
    pivot_df = (
        df.sort(['code', 'date', 'time']).filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            pl.col('pct_change')
            .std()
            .over(['date', 'time'])
            .alias('differentiation')
        ).filter(
            pl.col('differentiation') < pl.col('differentiation').mean().over('date')
        ).select(
            'code', 'time', 'amount', 'date'
        ).pivot(on='time', index='code', values='amount')
    )
    if pivot_df.is_empty():
        return None
    pivot_pd = pivot_df.to_pandas().set_index('code')
    mean_corr = _mean_abs_off_diagonal_corr(pivot_pd.T.corr())
    return pl.DataFrame(
        {
            'code': mean_corr.index.astype(str),
            'date': [df['date'].first()] * len(mean_corr),
            'GuYanChuQun': mean_corr.to_numpy(),
        }
    )


def cal_YueYaoYanBoDongLv(df: pl.DataFrame):
    """月耀眼波动率：适度日耀眼波动率的20日均值与标准差之和。"""
    return (
        df.lazy().sort(['code', 'date'])
        .with_columns(
            r_mean_20_ts_pl('YaoYanBoDongLv'),
            r_std_20_ts_pl('YaoYanBoDongLv'),
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('YaoYanBoDongLv_mean')
                + pl.col('YaoYanBoDongLv_std')
            ).alias('YueYaoYanBoDongLv'),
        ).collect()
    )


def cal_YueYaoYanShouYiLv(df: pl.DataFrame):
    """月耀眼收益率：适度日耀眼收益率的20日均值与标准差之和。"""
    return (
        df.lazy().sort(['code', 'date'])
        .with_columns(
            r_mean_20_ts_pl('YaoYanShouYiLv'),
            r_std_20_ts_pl('YaoYanShouYiLv'),
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('YaoYanShouYiLv_mean')
                + pl.col('YaoYanShouYiLv_std')
            ).alias('YueYaoYanShouYiLv'),
        ).collect()
    )


def cal_GaoDiECha(df: pl.DataFrame, daily_pv: pl.DataFrame):
    """
    水中行舟的中间因子：随波逐流的中间因子：高低额差
    :param df:
    :param daily_pv:
    :return:
    """
    daily_context = (
        daily_pv.lazy()
        .sort(['Stkcd', 'Trddt'])
        .with_columns(
            pl.col('Stkcd').cast(pl.String).alias('code'),
            pl.col('Trddt').str.to_date(format='%Y-%m-%d').alias('date'),
            pl.col('Dsmvosd').alias('mv'),
            (pl.col('Clsprc') / pl.col('Opnprc') - 1)
            .rolling_mean(20, min_samples=20)
            .over('Stkcd')
            .alias('reasonable_return'),
        )
        .filter(pl.col('mv') != 0)
        .select(['code', 'date', 'mv', 'reasonable_return'])
    )
    return (
        df.lazy()
        .sort(['code', 'date', 'time'])
        .filter(pl.col('close') != 0)
        .join(daily_context, on=['code', 'date'], how='left')
        .with_columns(
            (pl.col('close') / pl.col('open').first().over(['code', 'date']) - 1)
            .alias('intraday_return')
        )
        .with_columns(
            pl.when(pl.col('intraday_return') > pl.col('reasonable_return'))
            .then(pl.col('amount'))
            .otherwise(0)
            .alias('high_amount'),
            pl.when(pl.col('intraday_return') < pl.col('reasonable_return'))
            .then(pl.col('amount'))
            .otherwise(0)
            .alias('low_amount'),
        )
        .group_by(['code', 'date'])
        .agg(
            pl.col('high_amount').sum(),
            pl.col('low_amount').sum(),
            pl.col('mv').last(),
        )
        .select(
            pl.col('code'),
            pl.col('date'),
            ((pl.col('high_amount') - pl.col('low_amount')) / (pl.col('mv') * 1000))
            .alias('GaoDiECha'),
        )
        .collect()
    )


def cal_SuiBoZhuLiu(df: pl.DataFrame):
    """
    水中行舟的中间因子：随波逐流
    :param df:
    :return:
    """
    pivot_data = (
        df.select(
            'date', 'code', 'GaoDiECha'
        ).sort(by=['date', 'code'])
        .pivot(index='date', on='code', values='GaoDiECha')
        .to_pandas()
    )

    def _single_corr_cal(df_rolling: pd.DataFrame, i):
        window = df_rolling.iloc[i - 19:i + 1].set_index('date').dropna(axis=1)
        mean_corr = _mean_abs_off_diagonal_corr(
            window.corr(method='spearman')
        )
        return mean_corr.rename(pivot_data.loc[i, 'date']).to_frame().T

    valid_results = []
    if pivot_data.shape[0] >= 20:
        results = Parallel(n_jobs=-1)(
            delayed(_single_corr_cal)(
                pivot_data,
                i
            )
            for i in tqdm(range(19, pivot_data.shape[0]), desc='Processing')
        )
        valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        return None
    else:
        return (
            pl.from_pandas(
                pd.concat(valid_results).unstack(level=1).reset_index()
            ).select(
                pl.col('level_0')
                .alias('code'),
                pl.col('level_1')
                .cast(pl.Date)
                .alias('date'),
                pl.col('0')
                .alias('SuiBoZhuLiu')
            ).drop_nulls()
            .sort(by=['code', 'date'])
        )


def cal_ShuiZhongXingZhou(df: pl.DataFrame):
    """
    水中行舟计算方法
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .drop_nans()
        .with_columns(
            (r_mean_20_ts_pl('GuYanChuQun') + r_std_20_ts_pl('GuYanChuQun'))
            .alias('GuYanChuQun')
        ).with_columns(
            standardize_cs_pl('GuYanChuQun'),
            standardize_cs_pl('SuiBoZhuLiu')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('GuYanChuQun') + pl.col('SuiBoZhuLiu'))
            .alias('ShuiZhongXingZhou')
        ).collect()
    )


# 花隐林间：朝没晨雾和午避古木的20日均值与该股票与当期截面所有股票过去20天的夜眠霜路的相关系数的绝对值等权合成

def _HuaYinLinJian_preprocess(df: pl.DataFrame) -> pl.DataFrame:
    """花隐林间中间因子统一的预处理方式"""
    return (
        df.sort(['code', 'date', 'time']).filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change'),
            pl.col('volume')
            .diff()
            .over(['code', 'date'])
            .alias('vol_diff')
        ).with_columns(
            pl.col('vol_diff')
            .shift(i)
            .over(['code', 'date'])
            .alias(f'vol_diff_shift_{i}') for i in range(0, 6)
        ).filter(
            (pl.col('time') > 93700000) & (pl.col('time') <= 145300000)
        ).drop_nulls()
    )


def cal_ZhaoMoChenWu(df: pl.DataFrame):
    """
    花隐林间的中间因子：朝没晨雾
    :param df:
    :return:
    """
    return pl.from_pandas(
        _HuaYinLinJian_preprocess(df)
        .to_pandas()
        .groupby(['code', 'date']).apply(
            lambda g: sm.OLS(
                g['pct_change'], sm.add_constant(
                    g[[f'vol_diff_shift_{i}' for i in range(0, 6)]]
                )
            ).fit().tvalues,
            include_groups=False
        ).reset_index(level=['code', 'date'])
    ).select(
        pl.col('code'),
        pl.col('date')
        .cast(pl.Date),
        pl.concat_arr(
            pl.col(f'vol_diff_shift_{i}') for i in range(1, 6)
        ).arr.std()
        .alias('ZhaoMoChenWu')
    )


def cal_WuBiGuMu(df: pl.DataFrame):
    """
    花隐林间的中间因子：午避古木
    :param df:
    :return:
    """
    return pl.from_pandas(
        _HuaYinLinJian_preprocess(df)
        .to_pandas()
        .groupby(['code', 'date']).apply(
            lambda g: sm.OLS(
                g['pct_change'], sm.add_constant(
                    g[[f'vol_diff_shift_{i}' for i in range(0, 6)]]
                )
            ).fit(),
            include_groups=False
        ).apply(
            lambda m: pd.Series(
                [m.tvalues.iloc[0], m.fvalue],
                index=['t_intercept', 'f']
            )
        ).reset_index(level=['code', 'date'])
    ).select(
        pl.col('code'),
        pl.col('date')
        .cast(pl.Date),
        pl.when(pl.col('f') < pl.col('f').mean())
        .then(pl.col('t_intercept').abs() * -1)
        .otherwise(pl.col('t_intercept').abs())
        .alias('WuBiGuMu')
    )


def cal_YeMianShuangLu_t_intercept(df: pl.DataFrame):
    """
    花隐林间的中间因子的中间因子：夜眠霜路的中间因子：t_intercept
    :param df:
    :return:
    """
    return pl.from_pandas(
        _HuaYinLinJian_preprocess(df)
        .to_pandas()
        .groupby(['code', 'date']).apply(
            lambda g: sm.OLS(
                g['pct_change'], sm.add_constant(
                    g[[f'vol_diff_shift_{i}' for i in range(0, 6)]]
                )
            ).fit().tvalues.iloc[0],
            include_groups=False
        ).reset_index(level=['code', 'date'])
    ).select(
        pl.col('code'),
        pl.col('date')
        .cast(pl.Date),
        pl.col('0')
        .alias('YeMianShuangLu_t_intercept')
    )


def cal_YeMianShuangLu(df: pl.DataFrame):
    """
    花隐林间的中间因子：夜眠霜路
    :param df:
    :return:
    """
    pivot_data = (
        df.select(
            'date', 'code', 'YeMianShuangLu_t_intercept'
        ).sort(by=['date', 'code'])
        .pivot(index='date', on='code', values='YeMianShuangLu_t_intercept')
        .to_pandas()
    )
    def _single_corr_cal(df_rolling: pd.DataFrame, i):
        window = df_rolling.iloc[i - 19:i + 1].set_index('date')
        mean_corr = _mean_abs_off_diagonal_corr(window.corr())
        return mean_corr.rename(pivot_data.loc[i, 'date']).to_frame().T

    valid_results = []
    if pivot_data.shape[0] >= 20:
        results = Parallel(n_jobs=-1)(
            delayed(_single_corr_cal)(
                pivot_data,
                i
            )
            for i in tqdm(range(19, pivot_data.shape[0]), desc='Processing')
        )
        valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        return None
    else:
        return (
            pl.from_pandas(
                pd.concat(valid_results).unstack(level=1).reset_index()
            ).select(
                pl.col('level_0')
                .alias('code'),
                pl.col('level_1')
                .cast(pl.Date)
                .alias('date'),
                pl.col('0')
                .alias('YeMianShuangLu')
            )
        )


def cal_HuaYinLinJian(df: pl.DataFrame):
    """
    花隐林间计算方法。
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .drop_nans()
        .select(
            pl.col('code'),
            pl.col('date'),
            (
                (
                    pl.col('ZhaoMoChenWu')
                    .rolling_mean(20, min_samples=20)
                ) + (
                    pl.col('WuBiGuMu')
                    .rolling_mean(20, min_samples=20)
                ) + (
                    pl.col('YeMianShuangLu')
                )
            ).over('code')
            .alias('HuaYinLinJian')
        ).collect()
    )


# 待著而救：由跟随系数的20日均值与标准差等权合成

def cal_DaiZhuErJiu(df: pl.DataFrame):
    """待著而救因子计算方法"""
    return (
        df.lazy().sort(['code', 'date']).drop_nans()
        .with_columns(
            r_mean_20_ts_pl('GenSuiXiShu'),
            r_std_20_ts_pl('GenSuiXiShu')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('GenSuiXiShu_mean') + pl.col('GenSuiXiShu_std'))
            .alias('DaiZhuErJiu')
        ).collect()
    )


# 多空博弈：由均值距离化后的成交量博弈-收益率、成交量博弈-日内相对位置和振幅博弈的20日均值与标准差等权合成

def cal_ChengJiaoLiangBoYi_ShouYiLv(df: pl.DataFrame):
    """多空博弈的中间因子：成交量博弈-收益率"""
    return (
        df.lazy().sort(['code', 'date', 'time']).with_columns(
            (pl.col('close') / pl.col('close').shift(5) - 1)
            .over(['code', 'date'])
            .alias('return_past_5_min')
        ).filter(
            (pl.col('time') > 93500000) & (pl.col('time') < 145700000)
        ).with_columns(
            pl.col('volume')
            .sort_by('return_past_5_min', descending=False)
            .over(['code', 'date'])
            .alias('volume_ascend'),
            pl.col('volume')
            .sort_by('return_past_5_min', descending=True)
            .over(['code', 'date'])
            .alias('volume_descend')
        ).group_by(['code', 'date']).agg(
            (pl.col('volume_ascend').cum_sum() - pl.col('volume_descend').cum_sum())
            .sum()
            .alias('ChengJiaoLiangBoYi_ShouYiLv')
        ).collect()
    )


def cal_ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi(df: pl.DataFrame):
    """多空博弈的中间因子：成交量博弈-日内相对位置"""
    return (
        df.lazy().sort(['code', 'date', 'time']).with_columns(
            pl.col('low')
            .cum_min()
            .shift(1)
            .over(['code', 'date'])
            .alias('former_low'),
            pl.col('high')
            .cum_max()
            .shift(1)
            .over(['code', 'date'])
            .alias('former_high')
        ).with_columns(
            (
                (
                    (
                        pl.col('close') / pl.col('former_high')
                    ) + (
                        pl.col('close') / pl.col('former_low')
                    )
                ) / 2 - 1
            ).alias('position')
        ).filter(
            (pl.col('time') > 93500000) & (pl.col('time') < 145700000)
        ).with_columns(
            pl.col('volume')
            .sort_by('position', descending=False)
            .over(['code', 'date'])
            .alias('volume_ascend'),
            pl.col('volume')
            .sort_by('position', descending=True)
            .over(['code', 'date'])
            .alias('volume_descend')
        ).group_by(['code', 'date']).agg(
            (pl.col('volume_ascend').cum_sum() - pl.col('volume_descend').cum_sum())
            .sum()
            .alias('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
        ).collect()
    )


def cal_ZhenFuBoYi(df: pl.DataFrame):
    """多空博弈的中间因子：振幅博弈"""
    return (
        df.lazy().sort(['code', 'date', 'time']).with_columns(
            (pl.col('close') / pl.col('close').shift(5) - 1)
            .over(['code', 'date'])
            .alias('return_past_5_min'),
            ((pl.col('high') - pl.col('low')) / pl.col('close'))
            .alias('range')
        ).filter(
            (pl.col('time') > 93500000) & (pl.col('time') < 145700000)
        ).with_columns(
            pl.col('range')
            .sort_by('return_past_5_min', descending=False)
            .over(['code', 'date'])
            .alias('range_ascend'),
            pl.col('range')
            .sort_by('return_past_5_min', descending=True)
            .over(['code', 'date'])
            .alias('range_descend')
        ).group_by(['code', 'date']).agg(
            (pl.col('range_ascend').cum_sum() - pl.col('range_descend').cum_sum())
            .sum()
            .alias('ZhenFuBoYi')
        ).collect()
    )


def cal_DuoKongBoYi(df: pl.DataFrame):
    """多空博弈计算方法"""
    return (
        df.lazy().drop_nans()
        .with_columns(
            standardize_cs_pl('ChengJiaoLiangBoYi_ShouYiLv'),
            standardize_cs_pl('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
        ).with_columns(
            (
                pl.col('ChengJiaoLiangBoYi_ShouYiLv')
                - pl.col('ChengJiaoLiangBoYi_ShouYiLv').mean()
            ).abs()
            .over('date')
            .alias('ChengJiaoLiangBoYi_ShouYiLv'),
            (
                pl.col('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
                - pl.col('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi').mean()
            ).abs()
            .over('date')
            .alias('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi'),
            (
                    pl.col('ZhenFuBoYi')
                    - pl.col('ZhenFuBoYi').mean()
            ).abs()
            .over('date')
            .alias('ZhenFuBoYi')
        ).with_columns(
            (
                r_mean_20_ts_pl('ChengJiaoLiangBoYi_ShouYiLv')
                + r_std_20_ts_pl('ChengJiaoLiangBoYi_ShouYiLv')
            ).alias('ChengJiaoLiangBoYi_ShouYiLv'),
            (
                r_mean_20_ts_pl('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
                + r_std_20_ts_pl('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
            ).alias('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi'),
            (
                r_mean_20_ts_pl('ZhenFuBoYi')
                + r_std_20_ts_pl('ZhenFuBoYi')
            ).alias('ZhenFuBoYi')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('ChengJiaoLiangBoYi_ShouYiLv')
                + pl.col('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
                + pl.col('ZhenFuBoYi') * 2
            ).alias('DuoKongBoYi')
        ).collect()
    )


# 协同效应：成交量协同和协同价差的20日均值与标准差

def cal_ChengJiaoLiangXieTong(df: pl.DataFrame):
    """协同效应的中间因子：成交量协同"""
    return (
        df.lazy().sort(['code', 'date', 'time']).with_columns(
            pl.concat_arr(
                pl.col(col).shift(i).over(['code', 'date'])
                for i in range(0, 5)
                for col in ('high', 'low', 'open', 'close')
            ).alias('past_20_variable')
        ).filter(
            pl.col('time') > 93500000
        ).with_columns(
            pl.col('past_20_variable')
            .arr.mean()
            .alias('past_mean'),
            pl.col('past_20_variable')
            .arr.std()
            .alias('past_std')
        ).with_columns(
            (pl.col('past_mean') + pl.col('past_std'))
            .alias('upper_track'),
            (pl.col('past_mean') - pl.col('past_std'))
            .alias('lower_track')
        ).with_columns(
            pl.when(
                pl.col('close') > pl.col('upper_track')
            ).then(
                1
            ).otherwise(
                pl.when(
                    pl.col('close') < pl.col('lower_track')
                )
                .then(-1)
                .otherwise(0)
            ).alias('co_state'),
            pl.when(
                pl.col('volume').sum().over(['date', 'time']) != 0
            )
            .then(
                pl.col('volume')
                / pl.col('volume').sum().over(['date', 'time'])
            )
            .otherwise(None)
            .alias('volume_d')
        ).with_columns(
            pl.col('volume_d')
            .sum()
            .over(['date', 'time', 'co_state'])
            .alias('co_volume_d')
        ).group_by(['code', 'date']).agg(
            pl.corr(
                pl.col('volume_d'),
                pl.col('co_volume_d')
            ).fill_nan(None)
            .alias('ChengJiaoLiangXieTong')
        ).collect()
    )


def cal_XieTongJiaCha(
    df: pl.DataFrame,
    daily_pv: pl.DataFrame | None = None,
):
    """协同价差：按三类分钟符号选择最多30个同向邻居。"""
    minute = df.to_pandas().copy()
    if minute.empty:
        return pl.DataFrame(
            schema={
                'code': pl.String,
                'date': pl.Date,
                'XieTongJiaCha': pl.Float64,
            }
        )
    minute['date'] = pd.to_datetime(minute['date']).dt.date
    minute = minute.sort_values(['code', 'date', 'time'])
    grouped = minute.groupby(['code', 'date'], sort=False)
    minute['minute_return'] = grouped['close'].pct_change(fill_method=None)
    minute['past_return_mean'] = grouped['minute_return'].transform(
        lambda values: values.shift(1).rolling(5, min_periods=5).mean()
    )
    minute['past_close_mean'] = grouped['close'].transform(
        lambda values: values.shift(1).rolling(5, min_periods=5).mean()
    )
    minute['past_volume_mean'] = grouped['volume'].transform(
        lambda values: values.shift(1).rolling(5, min_periods=5).mean()
    )

    previous_close: dict[tuple[str, object], float] = {}
    if daily_pv is not None and not daily_pv.is_empty():
        daily = daily_pv.to_pandas().copy()
        daily['date'] = pd.to_datetime(daily['Trddt']).dt.date
        daily = daily.sort_values(['Stkcd', 'date'])
        daily['previous_close'] = daily.groupby('Stkcd')['Clsprc'].shift(1)
        previous_close = {
            (str(row.Stkcd), row.date): row.previous_close
            for row in daily.itertuples()
            if pd.notna(row.previous_close)
        }
    minute['previous_close'] = [
        previous_close.get((str(code), date), np.nan)
        for code, date in zip(minute['code'], minute['date'])
    ]

    # A zero return falls back to the preceding five-minute price, then the
    # previous daily close, as specified by the report.
    direction = np.sign(minute['minute_return'])
    fallback_price = minute['close'] - minute['past_close_mean']
    fallback_price = fallback_price.where(
        fallback_price != 0,
        minute['close'] - minute['previous_close'],
    )
    direction = direction.where(direction != 0, np.sign(fallback_price))
    relative_direction = np.sign(minute['minute_return'] - minute['past_return_mean'])
    relative_direction = relative_direction.where(relative_direction != 0, direction)
    volume_direction = np.sign(minute['volume'] - minute['past_volume_mean'])
    minute['signal_1'] = direction
    minute['signal_2'] = relative_direction
    minute['signal_3'] = volume_direction
    minute = minute.loc[
        (minute['time'] >= 93500000) & (minute['time'] < 145700000)
    ]
    if minute.empty:
        return pl.DataFrame(
            schema={
                'code': pl.String,
                'date': pl.Date,
                'XieTongJiaCha': pl.Float64,
            }
        )

    results = []
    for date, day in minute.groupby('date', sort=True):
        codes = sorted(day['code'].astype(str).unique())
        signals = [
            day.pivot(index='code', columns='time', values=column)
            .reindex(index=codes)
            .fillna(0)
            .to_numpy()
            for column in ('signal_1', 'signal_2', 'signal_3')
        ]
        similarity = np.zeros((len(codes), len(codes)), dtype='float64')
        for signal in signals:
            similarity += (signal[:, None, :] == signal[None, :, :]).sum(axis=2)
        np.fill_diagonal(similarity, -np.inf)
        returns = day.groupby('code')['close'].agg(lambda values: values.iloc[-1]).reindex(codes)
        returns = returns / pd.Series(
            [previous_close.get((code, date), np.nan) for code in codes], index=codes
        ) - 1
        for row_index, code in enumerate(codes):
            neighbor_count = min(30, max(0, len(codes) - 1))
            if neighbor_count == 0 or pd.isna(returns.loc[code]):
                spread = np.nan
            else:
                neighbors = np.argsort(-similarity[row_index])[:neighbor_count]
                neighbor_returns = returns.iloc[neighbors].dropna()
                spread = (
                    returns.loc[code] - neighbor_returns.mean()
                    if len(neighbor_returns)
                    else np.nan
                )
            results.append({'code': code, 'date': date, 'XieTongJiaCha': spread})
    return pl.from_pandas(pd.DataFrame(results), include_index=False)


def cal_XieTongXiaoYing(df: pl.DataFrame):
    """协同效应计算方法"""
    return (
        df.lazy().drop_nans()
        .with_columns(
            (
                r_mean_20_ts_pl('ChengJiaoLiangXieTong')
                + r_std_20_ts_pl('ChengJiaoLiangXieTong')
            ).alias('ChengJiaoLiangXieTong'),
            (
                r_mean_20_ts_pl('XieTongJiaCha')
                + r_std_20_ts_pl('XieTongJiaCha')
            ).alias('XieTongJiaCha')
        ).with_columns(
            standardize_cs_pl('ChengJiaoLiangXieTong'),
            standardize_cs_pl('XieTongJiaCha')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('ChengJiaoLiangXieTong')
                + pl.col('XieTongJiaCha')
            ).alias('XieTongXiaoYing')
        ).collect()
    )


def cal_RuoShiBanChaoXi(df: pl.DataFrame):
    """潮汐因子的中间因子：弱势半潮汐。"""
    return _calculate_tidal_half_factors(df).select(
        ['code', 'date', 'RuoShiBanChaoXi']
    )


def cal_GenSuiXiShu(df: pl.DataFrame):
    return (
        df.lazy().sort(['code', 'date', 'time']).filter(
            pl.col('time') >= 94500000
        ).with_columns(
            pl.when(pl.col('volume') >= pl.col('volume').top_k(10).min())
            .then(pl.col('time'))
            .otherwise(None)
            .over(['code', 'date'])
            .alias('high_volume_moment')
        ).with_columns(
            (pl.col('high_volume_moment') / pl.col('high_volume_moment'))
            .alias('advantage_moment'),
            pl.col('high_volume_moment')
            .forward_fill()
            .diff()
            .over(['code', 'date'])
            .alias('moment_diff')
        ).with_columns(
            pl.when(pl.col('moment_diff') < 500000)
            .then(None)
            .otherwise(pl.col('advantage_moment'))
            .fill_null(0)
            .alias('advantage_moment')
        ).with_columns(
            (
                pl.col('advantage_moment').shift(1)
                + pl.col('advantage_moment').shift(2)
                + pl.col('advantage_moment').shift(3)
                + pl.col('advantage_moment').shift(4)
                + pl.col('advantage_moment').shift(5)
            )
            .over(['code', 'date'])
            .fill_null(0)
            .alias('follow_moment_raw')
        ).with_columns(
            pl.when(pl.col('follow_moment_raw') > 0)
            .then(1)
            .otherwise(0)
            .alias('follow_moment')
        ).group_by(['code', 'date']).agg(
            pl.when(
                ((pl.col('advantage_moment') * pl.col('volume')).sum()) != 0
            ).then(
                ((pl.col('follow_moment') * pl.col('volume')).sum())
                / ((pl.col('advantage_moment') * pl.col('volume')).sum())
            ).otherwise(None)
            .alias('GenSuiXiShu')
        ).collect()
    )
