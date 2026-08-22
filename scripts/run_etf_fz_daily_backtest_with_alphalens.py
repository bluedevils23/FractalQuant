"""
使用新的 Alphalens 方法对 ETF FZ 日度因子进行回测

基于 single_factor_backtest 模块，对所有方正因子进行标准化的回测分析
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT / "FractalQuant"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from factor.single_factor_backtest import (
    FactorBacktestConfig,
    SingleFactorBacktest,
)

LOGGER = logging.getLogger("run_etf_fz_daily_backtest_with_alphalens")

# 默认路径
DEFAULT_FACTOR_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_daily_fz_factors")
DEFAULT_DAILY_PRICES = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
DEFAULT_OUTPUT_ROOT = Path(r"D:\workspace\stock-alphalens-reloaded\analysis_outputs\etf_fz_daily_alphalens_backtest")

# 方正因子列表（从 generate_fz_daily_factors.py 提取）
FZ_FACTOR_NAMES = (
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
    "ChongJian",
    "ZaiHouChongJian",
    "GuYanChuQun",
    "GaoDiECha",
    "ZhaoMoChenWu",
    "WuBiGuMu",
    "SuiBoZhuLiu",
    "ShuiZhongXingZhou",
    "YeMianShuangLu",
    "YeMianShuangLu_t_intercept",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Alphalens-style backtest on ETF FZ daily factors"
    )
    parser.add_argument(
        "--factor-root",
        type=Path,
        default=DEFAULT_FACTOR_ROOT,
        help="Root directory containing factor parquet files"
    )
    parser.add_argument(
        "--daily-prices",
        type=Path,
        default=DEFAULT_DAILY_PRICES,
        help="Daily price parquet file"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for backtest results"
    )
    parser.add_argument(
        "--factors",
        nargs="*",
        default=None,
        help="Specific factors to backtest (default: all)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date for backtest"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-12-31",
        help="End date for backtest"
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=10,
        help="Number of quantiles for grouping"
    )
    parser.add_argument(
        "--forward-periods",
        nargs="*",
        type=int,
        default=[1, 5, 10, 20],
        help="Forward return periods (in days)"
    )
    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_factor_data(
    factor_root: Path,
    factor_name: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    加载因子数据

    Args:
        factor_root: 因子数据根目录
        factor_name: 因子名称
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        因子数据 DataFrame
    """
    factor_files = list(factor_root.glob("*.parquet"))

    if not factor_files:
        raise FileNotFoundError(f"No factor files found in {factor_root}")

    LOGGER.info(f"Loading factor {factor_name} from {len(factor_files)} files...")

    factor_dfs = []
    for factor_file in tqdm(factor_files, desc=f"Loading {factor_name}"):
        df = pd.read_parquet(factor_file)

        # FZ factor files store the date in the ``factor_date`` index and
        # the ETF code in a ``ts_code`` column.
        if "trade_date" not in df.columns:
            if "factor_date" in df.index.names:
                df = df.reset_index().rename(columns={"factor_date": "trade_date"})
            elif "trade_date" in df.index.names:
                df = df.reset_index()

        if factor_name not in df.columns:
            continue

        # 提取需要的列
        factor_df = df[['trade_date', factor_name]].copy()
        factor_df['stock_code'] = (
            df['ts_code'].astype(str).to_numpy()
            if 'ts_code' in df.columns
            else factor_file.stem
        )  # ETF代码
        factor_df = factor_df.rename(columns={factor_name: 'factor_value'})

        factor_dfs.append(factor_df)

    if not factor_dfs:
        raise ValueError(f"Factor {factor_name} not found in any files")

    # 合并所有ETF的数据
    result = pd.concat(factor_dfs, ignore_index=True)
    result['trade_date'] = pd.to_datetime(result['trade_date'])

    # 过滤日期范围
    result = result[
        (result['trade_date'] >= start_date) &
        (result['trade_date'] <= end_date)
    ]

    LOGGER.info(f"Loaded {len(result)} rows for factor {factor_name}")

    return result


def load_price_data(
    daily_prices: Path,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    加载价格数据

    Args:
        daily_prices: 日度价格文件路径
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        价格数据 DataFrame
    """
    LOGGER.info(f"Loading daily prices from {daily_prices}...")

    if not daily_prices.exists():
        raise FileNotFoundError(f"Daily prices file not found: {daily_prices}")

    df = pd.read_parquet(daily_prices)
    if 'trade_date' not in df.columns and 'trade_date' in df.index.names:
        df = df.reset_index()
    if 'ts_code' not in df.columns and 'ts_code' in df.index.names:
        df = df.reset_index()
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # 过滤日期范围
    df = df[
        (df['trade_date'] >= start_date) &
        (df['trade_date'] <= end_date)
    ]

    # 重命名列以匹配回测模块
    if 'ts_code' in df.columns:
        df = df.rename(columns={'ts_code': 'stock_code'})

    result = df[['trade_date', 'stock_code', 'close']].copy()

    LOGGER.info(f"Loaded {len(result)} price records")

    return result


def run_single_factor_backtest(
    factor_name: str,
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    output_dir: Path,
    config: FactorBacktestConfig
) -> dict:
    """
    对单个因子运行回测

    Args:
        factor_name: 因子名称
        factor_df: 因子数据
        price_df: 价格数据
        output_dir: 输出目录
        config: 回测配置

    Returns:
        回测结果字典
    """
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Backtesting factor: {factor_name}")
    LOGGER.info(f"{'='*80}")

    try:
        # 创建回测实例
        backtest = SingleFactorBacktest(config)

        # 运行回测
        results = backtest.run_backtest(
            factor_df=factor_df,
            price_df=price_df,
            date_col='trade_date',
            stock_col='stock_code',
            factor_col='factor_value',
            price_col='close'
        )

        # 生成报告
        factor_output_dir = output_dir / factor_name
        backtest.generate_report(
            output_dir=factor_output_dir,
            factor_name=factor_name
        )

        LOGGER.info(f"Completed backtest for {factor_name}")
        LOGGER.info(f"Reports saved to {factor_output_dir}")

        return results

    except Exception as e:
        LOGGER.error(f"Error backtesting {factor_name}: {e}", exc_info=True)
        return None


def summarize_results(
    all_results: dict[str, dict],
    output_root: Path
):
    """
    汇总所有因子的回测结果

    Args:
        all_results: 所有因子的回测结果
        output_root: 输出根目录
    """
    LOGGER.info("\n" + "="*80)
    LOGGER.info("Generating summary report...")
    LOGGER.info("="*80)

    summary_rows = []

    for factor_name, results in all_results.items():
        if results is None:
            continue

        try:
            # 提取关键指标（以5日收益为代表）
            if '5d' not in results['performance']:
                continue

            perf = results['performance']['5d']
            ic_df = results['ic_results']['5d']

            ic_mean = ic_df['IC'].mean()
            ic_std = ic_df['IC'].std()
            rank_ic_mean = ic_df['RankIC'].mean()
            ir = ic_mean / ic_std * (252 ** 0.5) if ic_std > 0 else 0

            # 多空组合指标
            ls_metrics = perf['long_short']

            # 多头-平均指标
            la_metrics = perf['long_average']

            # Q1组指标
            q1_metrics = perf['q1']

            summary_rows.append({
                '因子': factor_name,
                'IC均值': f"{ic_mean:.4f}",
                'IC标准差': f"{ic_std:.4f}",
                'RankIC均值': f"{rank_ic_mean:.4f}",
                'IR': f"{ir:.3f}",
                '多空年化收益': f"{ls_metrics['annualized_return']:.2%}",
                '多空夏普': f"{ls_metrics['sharpe_ratio']:.3f}",
                '多空最大回撤': f"{ls_metrics['max_drawdown']:.2%}",
                '多空胜率': f"{ls_metrics['win_rate']:.2%}",
                '多头年化收益': f"{q1_metrics['annualized_return']:.2%}",
                '多头-平均年化': f"{la_metrics['annualized_return']:.2%}",
            })

        except Exception as e:
            LOGGER.error(f"Error summarizing {factor_name}: {e}")
            continue

    if not summary_rows:
        LOGGER.warning("No valid results to summarize")
        return

    summary_df = pd.DataFrame(summary_rows)

    # 按IC均值排序
    summary_df['IC_abs'] = summary_df['IC均值'].str.rstrip('%').astype(float).abs()
    summary_df = summary_df.sort_values('IC_abs', ascending=False)
    summary_df = summary_df.drop('IC_abs', axis=1)

    # 保存汇总表
    summary_file = output_root / "factor_summary_alphalens.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')

    LOGGER.info(f"\nSummary saved to {summary_file}")
    LOGGER.info(f"\nTop 10 factors by |IC|:")
    print(summary_df.head(10).to_string(index=False))


def main() -> int:
    args = parse_args()
    configure_logging()

    LOGGER.info("="*80)
    LOGGER.info("ETF FZ Daily Factor Backtest - Alphalens Style")
    LOGGER.info("="*80)

    # 检查输入路径
    if not args.factor_root.exists():
        LOGGER.error(f"Factor root not found: {args.factor_root}")
        return 1

    if not args.daily_prices.exists():
        LOGGER.error(f"Daily prices file not found: {args.daily_prices}")
        return 1

    # 创建输出目录
    args.output_root.mkdir(parents=True, exist_ok=True)

    # 确定要回测的因子
    factors_to_test = args.factors if args.factors else FZ_FACTOR_NAMES
    LOGGER.info(f"Will backtest {len(factors_to_test)} factors")

    # 配置回测参数
    config = FactorBacktestConfig(
        n_quantiles=args.n_quantiles,
        forward_periods=args.forward_periods,
        winsorize_method='mad',
        winsorize_std=3.0,
        standardize=True,
        neutralize_industry=False,
        neutralize_market_cap=False,
        commission=0.0003,
        slippage=0.0001
    )

    LOGGER.info(f"Backtest config: n_quantiles={config.n_quantiles}, "
                f"forward_periods={config.forward_periods}")

    # 加载价格数据（一次性加载）
    price_df = load_price_data(
        args.daily_prices,
        args.start_date,
        args.end_date
    )

    # 对每个因子运行回测
    all_results = {}

    for factor_name in factors_to_test:
        try:
            # 加载因子数据
            factor_df = load_factor_data(
                args.factor_root,
                factor_name,
                args.start_date,
                args.end_date
            )

            # 运行回测
            results = run_single_factor_backtest(
                factor_name,
                factor_df,
                price_df,
                args.output_root,
                config
            )

            if results is not None:
                all_results[factor_name] = results

        except Exception as e:
            LOGGER.error(f"Failed to backtest {factor_name}: {e}", exc_info=True)
            continue

    # 生成汇总报告
    if all_results:
        summarize_results(all_results, args.output_root)
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Backtest completed! Results saved to {args.output_root}")
        LOGGER.info(f"{'='*80}")
    else:
        LOGGER.warning("No factors were successfully backtested")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
