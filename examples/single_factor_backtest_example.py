"""
单因子回测示例

演示如何使用 single_factor_backtest 模块进行因子分析
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from FractalQuant.factor.single_factor_backtest import (
    FactorBacktestConfig,
    SingleFactorBacktest,
    quick_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_backtest():
    """示例1：基础回测"""
    logger.info("=" * 80)
    logger.info("示例1：基础单因子回测")
    logger.info("=" * 80)

    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')  # 交易日
    stocks = [f'{i:06d}.SH' for i in range(600000, 600100)]  # 100只股票

    # 因子数据 - 模拟一个有效的动量因子
    factor_data = []
    for date in dates:
        for stock in stocks:
            # 添加一些序列相关性，使因子更真实
            factor_value = np.random.randn() + 0.1 * np.sin(len(factor_data) / 100)
            factor_data.append({
                'trade_date': date,
                'stock_code': stock,
                'factor_value': factor_value
            })
    factor_df = pd.DataFrame(factor_data)

    # 价格数据 - 让价格与因子有一定相关性
    price_data = []
    base_prices = {stock: 100.0 for stock in stocks}

    for i, date in enumerate(dates):
        for stock in stocks:
            # 获取该日期该股票的因子值
            factor_val = factor_df[
                (factor_df['trade_date'] == date) &
                (factor_df['stock_code'] == stock)
            ]['factor_value'].iloc[0]

            # 价格变动部分受因子影响
            factor_effect = factor_val * 0.005  # 因子效应
            random_noise = np.random.randn() * 0.015  # 随机噪声
            return_today = factor_effect + random_noise

            base_prices[stock] *= (1 + return_today)

            price_data.append({
                'trade_date': date,
                'stock_code': stock,
                'close': base_prices[stock]
            })

    price_df = pd.DataFrame(price_data)

    logger.info(f"因子数据形状: {factor_df.shape}")
    logger.info(f"价格数据形状: {price_df.shape}")
    logger.info(f"日期范围: {dates.min()} 到 {dates.max()}")
    logger.info(f"股票数量: {len(stocks)}")

    # 配置回测参数
    config = FactorBacktestConfig(
        n_quantiles=10,
        forward_periods=[1, 5, 10, 20],  # 1日、5日、10日、20日收益
        winsorize_method='mad',
        winsorize_std=3.0,
        standardize=True,
        commission=0.0003,
        slippage=0.0001
    )

    # 运行回测
    output_dir = Path('./backtest_output/example_1_basic')
    results = quick_backtest(
        factor_df=factor_df,
        price_df=price_df,
        output_dir=output_dir,
        factor_name='momentum_factor',
        config=config
    )

    # 打印关键指标
    logger.info("\n" + "=" * 80)
    logger.info("回测结果摘要")
    logger.info("=" * 80)

    for period_name, perf_dict in results['performance'].items():
        logger.info(f"\n{period_name} 收益期:")
        for strategy, metrics in perf_dict.items():
            logger.info(f"  {strategy}:")
            logger.info(f"    年化收益: {metrics['annualized_return']:.2%}")
            logger.info(f"    夏普比率: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"    最大回撤: {metrics['max_drawdown']:.2%}")
            logger.info(f"    胜率: {metrics['win_rate']:.2%}")

    # IC分析
    logger.info("\nIC统计:")
    for period_name, ic_df in results['ic_results'].items():
        ic_mean = ic_df['IC'].mean()
        ic_std = ic_df['IC'].std()
        rank_ic_mean = ic_df['RankIC'].mean()
        ir = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else np.nan

        logger.info(f"  {period_name}:")
        logger.info(f"    IC均值: {ic_mean:.4f}")
        logger.info(f"    RankIC均值: {rank_ic_mean:.4f}")
        logger.info(f"    IR (信息比率): {ir:.3f}")

    logger.info(f"\n详细报告已保存到: {output_dir}")


def example_2_value_factors():
    """示例2：价值因子回测（模拟教程中的价值因子）"""
    logger.info("\n" + "=" * 80)
    logger.info("示例2：价值因子回测（PB、PE等）")
    logger.info("=" * 80)

    np.random.seed(123)
    dates = pd.date_range('2018-01-01', '2024-03-31', freq='M')  # 月度数据
    stocks = [f'{i:06d}.SH' for i in range(600000, 600300)]  # 300只股票

    # 模拟价值因子数据
    factor_data = []
    for date in dates:
        for stock in stocks:
            # PB倒数
            pb = np.random.uniform(0.5, 5.0)
            pb_inv_rank = 1 / pb

            # ROE
            roe = np.random.uniform(-0.1, 0.3)
            roe_rank = roe

            # BPROE因子 = PB倒数排名 + ROE排名（简化版）
            factor_value = pb_inv_rank * 0.5 + roe_rank * 0.5

            factor_data.append({
                'trade_date': date,
                'stock_code': stock,
                'factor_value': factor_value,
                'pb': pb,
                'roe': roe
            })

    factor_df = pd.DataFrame(factor_data)

    # 价格数据 - 价值因子通常有反转效应
    price_data = []
    base_prices = {stock: 50.0 for stock in stocks}

    for date in dates:
        for stock in stocks:
            factor_val = factor_df[
                (factor_df['trade_date'] == date) &
                (factor_df['stock_code'] == stock)
            ]['factor_value'].iloc[0]

            # 价值因子效应（反向）+ 噪声
            factor_effect = factor_val * 0.008
            random_noise = np.random.randn() * 0.04
            return_today = factor_effect + random_noise

            base_prices[stock] *= (1 + return_today)

            price_data.append({
                'trade_date': date,
                'stock_code': stock,
                'close': base_prices[stock]
            })

    price_df = pd.DataFrame(price_data)

    # 配置（月度回测）
    config = FactorBacktestConfig(
        n_quantiles=10,
        forward_periods=[1],  # 月度只看1期前向收益
        winsorize_method='mad',
        standardize=True
    )

    # 运行回测
    output_dir = Path('./backtest_output/example_2_value')
    results = quick_backtest(
        factor_df=factor_df[['trade_date', 'stock_code', 'factor_value']],
        price_df=price_df,
        output_dir=output_dir,
        factor_name='BPROE_value_factor',
        config=config
    )

    logger.info("\n价值因子回测完成！")
    logger.info(f"报告保存到: {output_dir}")


def example_3_compare_multiple_factors():
    """示例3：多个因子对比分析"""
    logger.info("\n" + "=" * 80)
    logger.info("示例3：多因子对比分析")
    logger.info("=" * 80)

    np.random.seed(456)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
    stocks = [f'{i:06d}.SZ' for i in range(300000, 300200)]

    # 生成多个因子
    factors = {
        'momentum_20d': lambda: np.random.randn() + 0.05,  # 动量因子
        'reversal_5d': lambda: -np.random.randn() * 0.8,   # 反转因子
        'volatility': lambda: -abs(np.random.randn()),     # 低波动因子
    }

    # 价格数据
    price_data = []
    base_prices = {stock: 100.0 for stock in stocks}

    for date in dates:
        for stock in stocks:
            return_today = np.random.randn() * 0.015
            base_prices[stock] *= (1 + return_today)
            price_data.append({
                'trade_date': date,
                'stock_code': stock,
                'close': base_prices[stock]
            })

    price_df = pd.DataFrame(price_data)

    # 对每个因子运行回测
    config = FactorBacktestConfig(
        n_quantiles=10,
        forward_periods=[1, 5, 10],
        standardize=True
    )

    all_results = {}

    for factor_name, factor_func in factors.items():
        logger.info(f"\n回测因子: {factor_name}")

        # 生成因子数据
        factor_data = []
        for date in dates:
            for stock in stocks:
                factor_data.append({
                    'trade_date': date,
                    'stock_code': stock,
                    'factor_value': factor_func()
                })
        factor_df = pd.DataFrame(factor_data)

        # 运行回测
        output_dir = Path(f'./backtest_output/example_3_compare/{factor_name}')
        results = quick_backtest(
            factor_df=factor_df,
            price_df=price_df,
            output_dir=output_dir,
            factor_name=factor_name,
            config=config
        )

        all_results[factor_name] = results

    # 对比结果
    logger.info("\n" + "=" * 80)
    logger.info("因子对比结果（5日收益）")
    logger.info("=" * 80)

    comparison_data = []
    for factor_name, results in all_results.items():
        if '5d' in results['performance']:
            metrics = results['performance']['5d']['long_short']
            ic_mean = results['ic_results']['5d']['IC'].mean()
            rank_ic_mean = results['ic_results']['5d']['RankIC'].mean()

            comparison_data.append({
                '因子': factor_name,
                'IC均值': f"{ic_mean:.4f}",
                'RankIC均值': f"{rank_ic_mean:.4f}",
                '年化收益': f"{metrics['annualized_return']:.2%}",
                '夏普比率': f"{metrics['sharpe_ratio']:.3f}",
                '最大回撤': f"{metrics['max_drawdown']:.2%}",
                '胜率': f"{metrics['win_rate']:.2%}"
            })

    comparison_df = pd.DataFrame(comparison_data)
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    # 保存对比结果
    output_dir = Path('./backtest_output/example_3_compare')
    comparison_df.to_csv(output_dir / 'factor_comparison.csv', index=False)
    logger.info(f"\n对比结果保存到: {output_dir / 'factor_comparison.csv'}")


def example_4_real_world_workflow():
    """示例4：真实场景工作流"""
    logger.info("\n" + "=" * 80)
    logger.info("示例4：真实场景工作流示例")
    logger.info("=" * 80)

    # 这个示例展示如何从真实数据文件读取并进行回测
    # 假设你有以下数据文件：
    # - factor_data.csv: 包含 trade_date, stock_code, factor_value
    # - price_data.csv: 包含 trade_date, stock_code, close

    logger.info("工作流步骤:")
    logger.info("1. 从文件/数据库读取因子数据")
    logger.info("2. 从文件/数据库读取价格数据")
    logger.info("3. 配置回测参数")
    logger.info("4. 运行回测")
    logger.info("5. 分析结果并生成报告")

    # 示例代码（注释掉，因为没有真实数据）
    """
    # 读取数据
    factor_df = pd.read_csv('path/to/factor_data.csv')
    factor_df['trade_date'] = pd.to_datetime(factor_df['trade_date'])

    price_df = pd.read_csv('path/to/price_data.csv')
    price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])

    # 数据清洗
    factor_df = factor_df.dropna()
    price_df = price_df.dropna()

    # 确保数据对齐
    common_dates = set(factor_df['trade_date']) & set(price_df['trade_date'])
    common_stocks = set(factor_df['stock_code']) & set(price_df['stock_code'])

    factor_df = factor_df[
        factor_df['trade_date'].isin(common_dates) &
        factor_df['stock_code'].isin(common_stocks)
    ]
    price_df = price_df[
        price_df['trade_date'].isin(common_dates) &
        price_df['stock_code'].isin(common_stocks)
    ]

    # 配置回测
    config = FactorBacktestConfig(
        n_quantiles=10,
        forward_periods=[1, 5, 10, 20],
        winsorize_method='mad',
        winsorize_std=3.0,
        standardize=True,
        neutralize_industry=False,  # 如果有行业数据可以设为True
        neutralize_market_cap=False,  # 如果有市值数据可以设为True
        commission=0.0003,
        slippage=0.0001
    )

    # 运行回测
    results = quick_backtest(
        factor_df=factor_df,
        price_df=price_df,
        output_dir='./backtest_output/real_world',
        factor_name='my_alpha_factor',
        config=config
    )

    # 分析结果
    print("回测完成！查看以下文件：")
    print("- my_alpha_factor_ic_analysis.png")
    print("- my_alpha_factor_quantile_returns.png")
    print("- my_alpha_factor_long_short.png")
    print("- my_alpha_factor_performance.csv")
    """

    logger.info("\n使用真实数据时，请参考上述注释代码")


def main():
    """运行所有示例"""
    logger.info("单因子回测示例集")
    logger.info("基于 Alphalens 方法论的完整回测流程")

    try:
        # 示例1：基础回测
        example_1_basic_backtest()

        # 示例2：价值因子
        example_2_value_factors()

        # 示例3：多因子对比
        example_3_compare_multiple_factors()

        # 示例4：真实场景
        example_4_real_world_workflow()

        logger.info("\n" + "=" * 80)
        logger.info("所有示例运行完成！")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"运行示例时出错: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
