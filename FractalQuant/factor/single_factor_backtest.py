"""
单因子回测模块 - 基于 Alphalens 方法论

整合了以下特性：
1. 标准化的因子预处理（去极值、标准化）
2. 因子中性化（行业、市值等）
3. IC/RankIC 分析
4. 分组回测（10分组）
5. 多空组合分析
6. 性能指标计算
7. 可视化报告生成

参考：
- TEJ Alphalens 教程
- 方正证券多因子系列
- alphalens-reloaded
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FactorBacktestConfig:
    """因子回测配置"""
    # 分组参数
    n_quantiles: int = 10
    quantile_labels: bool = True

    # 中性化参数
    neutralize_industry: bool = False
    neutralize_market_cap: bool = False

    # 数据清洗参数
    winsorize_method: Literal['mad', 'percentile'] = 'mad'
    winsorize_std: float = 3.0
    winsorize_percentile: tuple[float, float] = (0.025, 0.975)
    standardize: bool = True

    # 前向收益期
    forward_periods: list[int] = None  # [1, 5, 10, 20] 对应1日、5日、10日、20日

    # 回测参数
    commission: float = 0.0003  # 单边手续费
    slippage: float = 0.0001    # 滑点

    def __post_init__(self):
        if self.forward_periods is None:
            self.forward_periods = [1, 5, 10, 20]


class FactorPreprocessor:
    """因子预处理器"""

    @staticmethod
    def mad_winsorize(series: pd.Series, n_std: float = 3.0) -> pd.Series:
        """
        MAD去极值

        Args:
            series: 原始序列
            n_std: 标准差倍数

        Returns:
            处理后的序列
        """
        median = series.median()
        mad = (series - median).abs().median()
        std = 1.4826 * mad  # MAD to std conversion

        lower = median - n_std * std
        upper = median + n_std * std

        # 线性压缩而非截断
        below_mask = series < lower
        above_mask = series > upper

        result = series.copy()
        if below_mask.any():
            result[below_mask] = np.linspace(
                median - 3.5 * std,
                median - 3.0 * std,
                num=below_mask.sum()
            )
        if above_mask.any():
            result[above_mask] = np.linspace(
                median + 3.0 * std,
                median + 3.5 * std,
                num=above_mask.sum()
            )

        return result

    @staticmethod
    def percentile_winsorize(
        series: pd.Series,
        percentile: tuple[float, float] = (0.025, 0.975)
    ) -> pd.Series:
        """
        百分位去极值

        Args:
            series: 原始序列
            percentile: 上下分位点

        Returns:
            处理后的序列
        """
        lower = series.quantile(percentile[0])
        upper = series.quantile(percentile[1])
        return series.clip(lower, upper)

    @staticmethod
    def standardize(series: pd.Series) -> pd.Series:
        """
        标准化（Z-score）

        Args:
            series: 原始序列

        Returns:
            标准化后的序列
        """
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0:
            return pd.Series(0, index=series.index)
        return (series - mean) / std

    def process_factor(
        self,
        factor_series: pd.Series,
        winsorize_method: str = 'mad',
        winsorize_std: float = 3.0,
        winsorize_percentile: tuple[float, float] = (0.025, 0.975),
        standardize: bool = True
    ) -> pd.Series:
        """
        处理因子值

        Args:
            factor_series: 原始因子序列
            winsorize_method: 去极值方法 ('mad' or 'percentile')
            winsorize_std: MAD标准差倍数
            winsorize_percentile: 百分位截断点
            standardize: 是否标准化

        Returns:
            处理后的因子序列
        """
        result = factor_series.copy()

        # 去极值
        if winsorize_method == 'mad':
            result = self.mad_winsorize(result, n_std=winsorize_std)
        elif winsorize_method == 'percentile':
            result = self.percentile_winsorize(result, percentile=winsorize_percentile)

        # 标准化
        if standardize:
            result = self.standardize(result)

        return result


class FactorNeutralizer:
    """因子中性化"""

    @staticmethod
    def neutralize(
        factor_df: pd.DataFrame,
        exposure_df: pd.DataFrame,
        factor_col: str = 'factor'
    ) -> pd.Series:
        """
        因子中性化（回归残差）

        Args:
            factor_df: 因子数据，包含因子值和其他列
            exposure_df: 暴露度数据（如行业哑变量、市值等）
            factor_col: 因子列名

        Returns:
            中性化后的因子（残差）
        """
        y = factor_df[factor_col].astype(float)
        X = exposure_df.astype(float)

        # OLS回归
        model = sm.OLS(y, X, hasconst=False, missing='drop').fit()
        residuals = model.resid

        return residuals


class ICAnalyzer:
    """IC分析器"""

    @staticmethod
    def calculate_ic(
        factor_values: pd.Series,
        forward_returns: pd.Series
    ) -> dict[str, float]:
        """
        计算IC指标

        Args:
            factor_values: 因子值
            forward_returns: 前向收益

        Returns:
            IC指标字典
        """
        # 去除缺失值
        valid_mask = factor_values.notna() & forward_returns.notna()
        factor_clean = factor_values[valid_mask]
        returns_clean = forward_returns[valid_mask]

        if len(factor_clean) < 2:
            return {
                'IC': np.nan,
                'RankIC': np.nan,
                'IC_pvalue': np.nan,
                'RankIC_pvalue': np.nan
            }

        # Pearson IC
        ic, ic_pvalue = stats.pearsonr(factor_clean, returns_clean)

        # Spearman RankIC
        rank_ic, rank_ic_pvalue = stats.spearmanr(factor_clean, returns_clean)

        return {
            'IC': ic,
            'RankIC': rank_ic,
            'IC_pvalue': ic_pvalue,
            'RankIC_pvalue': rank_ic_pvalue
        }

    @staticmethod
    def calculate_time_series_ic(
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        date_col: str = 'trade_date',
        factor_col: str = 'factor',
        return_col: str = 'forward_return'
    ) -> pd.DataFrame:
        """
        计算时间序列IC

        Args:
            factor_df: 因子数据
            returns_df: 收益数据
            date_col: 日期列名
            factor_col: 因子列名
            return_col: 收益列名

        Returns:
            IC时间序列
        """
        # Match each factor observation to the same asset's forward return.
        # Joining on date alone creates a per-day Cartesian product for
        # cross-sectional panels and can exhaust memory.
        merge_keys = [date_col]
        if 'stock_code' in factor_df.columns and 'stock_code' in returns_df.columns:
            merge_keys.append('stock_code')

        # 合并数据
        merged = pd.merge(
            factor_df[[*merge_keys, factor_col]],
            returns_df[[*merge_keys, return_col]],
            on=merge_keys,
            how='inner'
        )

        ic_results = []

        for date in tqdm(merged[date_col].unique(), desc="Calculating IC"):
            date_data = merged[merged[date_col] == date]
            ic_dict = ICAnalyzer.calculate_ic(
                date_data[factor_col],
                date_data[return_col]
            )
            ic_results.append({
                date_col: date,
                **ic_dict
            })

        return pd.DataFrame(ic_results)


class QuantileBacktest:
    """分组回测"""

    def __init__(self, config: FactorBacktestConfig):
        self.config = config

    def assign_quantiles(
        self,
        factor_series: pd.Series,
        n_quantiles: int = 10
    ) -> pd.Series:
        """
        分配分位数组

        Args:
            factor_series: 因子值
            n_quantiles: 分组数

        Returns:
            分组标签
        """
        # 从小到大分组，label 1-10
        return pd.qcut(
            -factor_series.rank(method='first', ascending=False),
            n_quantiles,
            labels=list(range(1, n_quantiles + 1)),
            duplicates='drop'
        )

    def calculate_quantile_returns(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        date_col: str = 'trade_date',
        stock_col: str = 'stock_code',
        factor_col: str = 'factor',
        return_col: str = 'forward_return'
    ) -> pd.DataFrame:
        """
        计算分组收益

        Args:
            factor_df: 因子数据 (date, stock, factor)
            returns_df: 收益数据 (date, stock, forward_return)
            date_col: 日期列
            stock_col: 股票列
            factor_col: 因子列
            return_col: 收益列

        Returns:
            分组收益时间序列
        """
        # 合并数据
        merged = pd.merge(
            factor_df,
            returns_df,
            on=[date_col, stock_col],
            how='inner'
        )

        merged = merged.sort_values(by=date_col)
        all_dates = merged[date_col].unique()

        # 分组
        merged['quantile'] = merged.groupby(date_col)[factor_col].transform(
            lambda x: self.assign_quantiles(x, self.config.n_quantiles)
        )

        # The original implementation repeatedly filtered the full panel for
        # every date and quantile (quadratic in the number of dates). Map each
        # signal date to its next date once, then perform one keyed join.
        next_date_map = pd.Series(all_dates[1:], index=all_dates[:-1])
        merged['_next_date'] = merged[date_col].map(next_date_map)
        next_returns = returns_df[[date_col, stock_col, return_col]].rename(
            columns={date_col: '_next_date'}
        )
        joined = merged[['_next_date', stock_col, 'quantile']].merge(
            next_returns,
            on=['_next_date', stock_col],
            how='inner'
        )
        quantile_returns = joined.groupby(
            ['_next_date', 'quantile'], observed=True
        )[return_col].mean().unstack('quantile')
        quantile_returns.index.name = date_col
        quantile_returns.columns = [f'Q{int(q)}' for q in quantile_returns.columns]
        return quantile_returns

    def calculate_long_short_returns(
        self,
        quantile_returns: pd.DataFrame
    ) -> pd.Series:
        """
        计算多空组合收益

        Args:
            quantile_returns: 分组收益 DataFrame

        Returns:
            多空收益序列
        """
        return quantile_returns['Q1'] - quantile_returns[f'Q{self.config.n_quantiles}']

    def calculate_long_average_returns(
        self,
        quantile_returns: pd.DataFrame
    ) -> pd.Series:
        """
        计算多头-平均组合收益

        Args:
            quantile_returns: 分组收益 DataFrame

        Returns:
            多头-平均收益序列
        """
        # 剩余组平均
        other_cols = [f'Q{i}' for i in range(2, self.config.n_quantiles + 1)]
        average_returns = quantile_returns[other_cols].mean(axis=1)
        return quantile_returns['Q1'] - average_returns


class PerformanceMetrics:
    """性能指标计算"""

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """最大回撤"""
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return np.nan
        return returns.mean() / returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def calculate_metrics(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> dict[str, float]:
        """
        计算全部性能指标

        Args:
            returns: 收益序列
            periods_per_year: 年化周期数

        Returns:
            指标字典
        """
        cumulative = (1 + returns.fillna(0)).cumprod()

        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1 if len(returns) > 0 else 0

        volatility = returns.std() * np.sqrt(periods_per_year)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns, periods_per_year)
        max_dd = PerformanceMetrics.calculate_max_drawdown(cumulative)

        win_rate = (returns > 0).mean()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'n_periods': len(returns)
        }


class SingleFactorBacktest:
    """单因子回测主类"""

    def __init__(self, config: FactorBacktestConfig = None):
        self.config = config or FactorBacktestConfig()
        self.preprocessor = FactorPreprocessor()
        self.neutralizer = FactorNeutralizer()
        self.ic_analyzer = ICAnalyzer()
        self.quantile_backtest = QuantileBacktest(self.config)

        self.results: dict = {}

    def prepare_factor_data(
        self,
        factor_df: pd.DataFrame,
        date_col: str = 'trade_date',
        stock_col: str = 'stock_code',
        factor_col: str = 'factor_value'
    ) -> pd.DataFrame:
        """
        准备因子数据（预处理）

        Args:
            factor_df: 原始因子数据
            date_col: 日期列
            stock_col: 股票列
            factor_col: 因子列

        Returns:
            处理后的因子数据
        """
        result = factor_df.copy()

        # 按日期分组处理
        result['factor_processed'] = result.groupby(date_col)[factor_col].transform(
            lambda x: self.preprocessor.process_factor(
                x,
                winsorize_method=self.config.winsorize_method,
                winsorize_std=self.config.winsorize_std,
                winsorize_percentile=self.config.winsorize_percentile,
                standardize=self.config.standardize
            )
        )

        return result

    def run_backtest(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        date_col: str = 'trade_date',
        stock_col: str = 'stock_code',
        factor_col: str = 'factor_value',
        price_col: str = 'close',
        industry_df: Optional[pd.DataFrame] = None,
        market_cap_df: Optional[pd.DataFrame] = None
    ) -> dict:
        """
        运行完整回测

        Args:
            factor_df: 因子数据
            price_df: 价格数据
            date_col: 日期列
            stock_col: 股票列
            factor_col: 因子列
            price_col: 价格列
            industry_df: 行业数据（用于中性化）
            market_cap_df: 市值数据（用于中性化）

        Returns:
            回测结果字典
        """
        logger.info("开始单因子回测...")

        # 1. 数据预处理
        logger.info("预处理因子数据...")
        factor_processed = self.prepare_factor_data(
            factor_df, date_col, stock_col, factor_col
        )

        # 2. 因子中性化（如果需要）
        if self.config.neutralize_industry or self.config.neutralize_market_cap:
            logger.info("因子中性化...")
            # TODO: 实现中性化逻辑
            pass

        # 3. 计算前向收益
        logger.info("计算前向收益...")
        returns_dfs = {}
        for period in self.config.forward_periods:
            returns_df = self._calculate_forward_returns(
                price_df, period, date_col, stock_col, price_col
            )
            returns_dfs[f'{period}d'] = returns_df

        # 4. IC分析
        logger.info("计算IC...")
        ic_results = {}
        for period_name, returns_df in returns_dfs.items():
            ic_ts = self.ic_analyzer.calculate_time_series_ic(
                factor_processed.rename(columns={'factor_processed': 'factor'}),
                returns_df,
                date_col=date_col,
                factor_col='factor',
                return_col='forward_return'
            )
            ic_results[period_name] = ic_ts

        # 5. 分组回测
        logger.info("分组回测...")
        quantile_results = {}
        for period_name, returns_df in returns_dfs.items():
            q_returns = self.quantile_backtest.calculate_quantile_returns(
                factor_processed.rename(columns={'factor_processed': 'factor'}),
                returns_df,
                date_col=date_col,
                stock_col=stock_col,
                factor_col='factor',
                return_col='forward_return'
            )
            quantile_results[period_name] = q_returns

        # 6. 多空组合
        logger.info("计算多空组合...")
        long_short_results = {}
        long_average_results = {}
        for period_name, q_returns in quantile_results.items():
            long_short_results[period_name] = self.quantile_backtest.calculate_long_short_returns(q_returns)
            long_average_results[period_name] = self.quantile_backtest.calculate_long_average_returns(q_returns)

        # 7. 性能指标
        logger.info("计算性能指标...")
        performance = {}
        for period_name, ls_returns in long_short_results.items():
            performance[period_name] = {
                'long_short': PerformanceMetrics.calculate_metrics(ls_returns),
                'long_average': PerformanceMetrics.calculate_metrics(long_average_results[period_name]),
                'q1': PerformanceMetrics.calculate_metrics(quantile_results[period_name]['Q1'])
            }

        self.results = {
            'ic_results': ic_results,
            'quantile_returns': quantile_results,
            'long_short_returns': long_short_results,
            'long_average_returns': long_average_results,
            'performance': performance
        }

        logger.info("回测完成!")
        return self.results

    def _calculate_forward_returns(
        self,
        price_df: pd.DataFrame,
        period: int,
        date_col: str,
        stock_col: str,
        price_col: str
    ) -> pd.DataFrame:
        """计算前向收益"""
        price_sorted = price_df.sort_values([stock_col, date_col])
        price_sorted['forward_price'] = price_sorted.groupby(stock_col)[price_col].shift(-period)
        price_sorted['forward_return'] = price_sorted['forward_price'] / price_sorted[price_col] - 1

        return price_sorted[[date_col, stock_col, 'forward_return']].dropna()

    def generate_report(
        self,
        output_dir: Path,
        factor_name: str = 'factor'
    ):
        """
        生成回测报告

        Args:
            output_dir: 输出目录
            factor_name: 因子名称
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"生成报告到 {output_dir}...")

        # 1. IC报告
        self._plot_ic_analysis(output_dir, factor_name)

        # 2. 分组收益图
        self._plot_quantile_returns(output_dir, factor_name)

        # 3. 多空组合图
        self._plot_long_short(output_dir, factor_name)

        # 4. 性能指标表
        self._save_performance_table(output_dir, factor_name)

        logger.info("报告生成完成!")

    def _plot_ic_analysis(self, output_dir: Path, factor_name: str):
        """绘制IC分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, (period_name, ic_df) in enumerate(self.results['ic_results'].items()):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            # IC时间序列
            ax.plot(ic_df.index, ic_df['IC'], label='IC', alpha=0.7)
            ax.plot(ic_df.index, ic_df['RankIC'], label='RankIC', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_title(f'{factor_name} - {period_name} IC Time Series')
            ax.set_xlabel('Date')
            ax.set_ylabel('IC Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{factor_name}_ic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_quantile_returns(self, output_dir: Path, factor_name: str):
        """绘制分组收益图"""
        n_periods = len(self.results['quantile_returns'])
        fig, axes = plt.subplots(n_periods, 1, figsize=(14, 6 * n_periods))

        if n_periods == 1:
            axes = [axes]

        for idx, (period_name, q_returns) in enumerate(self.results['quantile_returns'].items()):
            ax = axes[idx]
            cumulative = (1 + q_returns.fillna(0)).cumprod()

            for col in cumulative.columns:
                ax.plot(cumulative.index, cumulative[col], label=col)

            ax.set_title(f'{factor_name} - {period_name} Quantile Cumulative Returns')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend(ncol=5)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{factor_name}_quantile_returns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_long_short(self, output_dir: Path, factor_name: str):
        """绘制多空组合图"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # 多空组合
        ax = axes[0]
        for period_name, ls_returns in self.results['long_short_returns'].items():
            cumulative = (1 + ls_returns.fillna(0)).cumprod()
            ax.plot(cumulative.index, cumulative, label=f'Long-Short {period_name}')

        ax.set_title(f'{factor_name} - Long-Short Portfolio')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 多头-平均
        ax = axes[1]
        for period_name, la_returns in self.results['long_average_returns'].items():
            cumulative = (1 + la_returns.fillna(0)).cumprod()
            ax.plot(cumulative.index, cumulative, label=f'Long-Average {period_name}')

        ax.set_title(f'{factor_name} - Long-Average Portfolio')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{factor_name}_long_short.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _save_performance_table(self, output_dir: Path, factor_name: str):
        """保存性能指标表"""
        rows = []

        for period_name, perf_dict in self.results['performance'].items():
            for strategy, metrics in perf_dict.items():
                row = {
                    'Period': period_name,
                    'Strategy': strategy,
                    **metrics
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        # 格式化百分比
        pct_cols = ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'win_rate']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f'{x:.2%}' if pd.notna(x) else '')

        # 格式化数值
        if 'sharpe_ratio' in df.columns:
            df['sharpe_ratio'] = df['sharpe_ratio'].apply(lambda x: f'{x:.3f}' if pd.notna(x) else '')

        df.to_csv(output_dir / f'{factor_name}_performance.csv', index=False)
        logger.info(f"Performance table saved to {output_dir / f'{factor_name}_performance.csv'}")


# ===== 便捷函数 =====

def quick_backtest(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    output_dir: Path | str,
    factor_name: str = 'factor',
    config: Optional[FactorBacktestConfig] = None,
    **kwargs
) -> dict:
    """
    快速回测接口

    Args:
        factor_df: 因子数据，必须包含 ['trade_date', 'stock_code', 'factor_value']
        price_df: 价格数据，必须包含 ['trade_date', 'stock_code', 'close']
        output_dir: 输出目录
        factor_name: 因子名称
        config: 回测配置
        **kwargs: 传递给 run_backtest 的其他参数

    Returns:
        回测结果字典
    """
    backtest = SingleFactorBacktest(config)
    results = backtest.run_backtest(factor_df, price_df, **kwargs)
    backtest.generate_report(Path(output_dir), factor_name)
    return results


if __name__ == '__main__':
    # 示例用法
    logging.basicConfig(level=logging.INFO)

    # 生成示例数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    stocks = [f'stock_{i:03d}' for i in range(100)]

    # 因子数据
    factor_data = []
    for date in dates:
        for stock in stocks:
            factor_data.append({
                'trade_date': date,
                'stock_code': stock,
                'factor_value': np.random.randn()
            })
    factor_df = pd.DataFrame(factor_data)

    # 价格数据
    price_data = []
    for date in dates:
        for stock in stocks:
            price_data.append({
                'trade_date': date,
                'stock_code': stock,
                'close': 100 * (1 + np.random.randn() * 0.02)
            })
    price_df = pd.DataFrame(price_data)

    # 运行回测
    config = FactorBacktestConfig(
        n_quantiles=10,
        forward_periods=[1, 5, 10, 20]
    )

    results = quick_backtest(
        factor_df=factor_df,
        price_df=price_df,
        output_dir='./backtest_output',
        factor_name='example_factor',
        config=config
    )

    print("回测完成！")
