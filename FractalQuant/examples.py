"""
策略使用示例
"""
import pandas as pd
import numpy as np
from datetime import datetime

from config.config import config
from strategy.strategy import HighFrequencyTradingStrategy, ArbitrageStrategy, MeanReversionStrategy, MomentumStrategy
from factor.combiner import FactorCombiner, MultiFactorSignal
from factor.price import ReturnsFactor, PriceMomentumFactor
from factor.volatility import HistoricalVolatilityFactor
from factor.trend import MACDFactor, RSIFactor
from factor.advanced import HurstExponentFactor, LyapunovExponentFactor
from factor.ml import MLForecastFactor, MLAnomalyDetectionFactor
from factor.microstructure import OrderFlowImbalanceFactor
from factor.crossmarket import CrossMarketCorrelationFactor
from backtest.engine import BacktestEngine
from factor.selector import FactorSelector, WeightOptimizer, FactorEnsemble

def generate_sample_data(n_bars: int = 1000) -> pd.DataFrame:
    """生成示例数据"""
    np.random.seed(42)
    
    timestamps = pd.date_range(start='2023-01-01', periods=n_bars, freq='1min')
    
    # 生成价格序列（带趋势和波动）
    returns = np.random.normal(0.0005, 0.01, n_bars)
    prices = 100 * (1 + pd.Series(returns).cumsum())
    
    # 生成成交量
    volumes = np.random.normal(1000, 200, n_bars)
    volumes = np.abs(volumes)
    
    # 生成订单簿数据
    bid_prices = prices - 0.01
    ask_prices = prices + 0.01
    bid_volumes = volumes * np.random.uniform(0.4, 0.6, n_bars)
    ask_volumes = volumes * np.random.uniform(0.4, 0.6, n_bars)
    
    df = pd.DataFrame({
        'open': prices * np.random.uniform(0.99, 1.01, n_bars),
        'high': prices * np.random.uniform(1.0, 1.02, n_bars),
        'low': prices * np.random.uniform(0.98, 1.0, n_bars),
        'close': prices,
        'volume': volumes,
        'bid_price': bid_prices,
        'ask_price': ask_prices,
        'bid_volume': bid_volumes,
        'ask_volume': ask_volumes,
    }, index=timestamps)
    
    return df

def example_basic_backtest():
    """示例1：基本回测"""
    print("=" * 80)
    print("示例1：基本回测")
    print("=" * 80)
    
    # 生成示例数据
    df = generate_sample_data(1000)
    
    # 创建策略
    strategy = HighFrequencyTradingStrategy()
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        leverage=10
    )
    
    # 定义信号生成函数
    def signal_generator(data, symbols):
        signals = {}
        for symbol in symbols:
            if symbol in data.columns:
                symbol_data = data[symbol]
                signal_data = strategy.generate_signal(symbol_data)
                signals[symbol] = signal_data['signal']
        return signals
    
    # 运行回测
    result = engine.run(df, signal_generator, ['BTC/USDT'])
    
    # 打印结果
    print(f"\n回测结果:")
    print(f"  总收益率: {result.total_return*100:.2f}%")
    print(f"  年化收益率: {result.annual_return*100:.2f}%")
    print(f"  最大回撤: {result.max_drawdown*100:.2f}%")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  索提诺比率: {result.sortino_ratio:.2f}")
    print(f"  卡玛比率: {result.calmar_ratio:.2f}")
    print(f"  信息比率: {result.information_ratio:.2f}")
    print(f"  胜率: {result.win_rate*100:.2f}%")
    print(f"  盈亏比: {result.profit_factor:.2f}")
    print(f"  总交易数: {result.total_trades}")
    
    # 详细性能分析
    performance = engine.analyze_performance()
    print(f"\n详细性能指标:")
    for key, value in list(performance.items())[:10]:
        print(f"  {key}: {value:.4f}")

def example_factor_combination():
    """示例2：因子组合"""
    print("\n" + "=" * 80)
    print("示例2：因子组合")
    print("=" * 80)
    
    # 生成示例数据
    df = generate_sample_data(500)
    
    # 创建因子组合器
    factors = [
        ReturnsFactor(window=5),
        PriceMomentumFactor(window=20),
        HistoricalVolatilityFactor(window=20),
        MACDFactor(),
        RSIFactor(),
        HurstExponentFactor(window=50),
        LyapunovExponentFactor(window=50),
        MLForecastFactor(model_type='linear'),
        MLAnomalyDetectionFactor(),
        OrderFlowImbalanceFactor(),
        CrossMarketCorrelationFactor(),
    ]
    
    # 使用不同方法组合因子
    methods = ['equal', 'rank', 'zscore', 'pca', 'optimal', 'adaptive']
    
    for method in methods:
        combiner = FactorCombiner(factors=factors, method=method)
        combined = combiner.combine_factors(df)
        
        print(f"\n{method.upper()} 方法:")
        print(f"  最新组合值: {combined.iloc[-1]:.4f}")
        print(f"  均值: {combined.mean():.4f}")
        print(f"  标准差: {combined.std():.4f}")
        print(f"  最大值: {combined.max():.4f}")
        print(f"  最小值: {combined.min():.4f}")

def example_multi_factor_signal():
    """示例3：多因子信号生成"""
    print("\n" + "=" * 80)
    print("示例3：多因子信号生成")
    print("=" * 80)
    
    # 生成示例数据
    df = generate_sample_data(300)
    
    # 创建多因子信号生成器
    multi_factor = MultiFactorSignal()
    
    # 生成信号
    signal_data = multi_factor.generate_signal(df)
    
    print(f"\n信号数据:")
    print(f"  信号: {signal_data['signal']}")
    print(f"  强度: {signal_data['strength']:.4f}")
    print(f"  得分: {signal_data['score']:.4f}")
    
    print(f"\n因子值:")
    for factor_name, factor_value in signal_data['factors'].items():
        print(f"  {factor_name}: {factor_value:.4f}")

def example_factor_selection():
    """示例4：因子选择"""
    print("\n" + "=" * 80)
    print("示例4：因子选择")
    print("=" * 80)
    
    # 生成示例数据
    df = generate_sample_data(500)
    
    # 创建多因子信号生成器
    multi_factor = MultiFactorSignal()
    
    # 计算所有因子
    factor_df = multi_factor.combiner.calculate_all(df)
    returns = df['close'].pct_change().shift(-1)
    
    # 因子选择
    selector = FactorSelector()
    
    # 不同方法选择因子
    methods = ['correlation', 'information', 'stability', 'diversification', 'all']
    
    for method in methods:
        selected = selector.select_factors(factor_df, returns, method=method, target_count=5)
        print(f"\n{method.upper()} 方法选中的因子 ({len(selected)} 个):")
        for factor in selected:
            print(f"  - {factor}")

def example_weight_optimization():
    """示例5：权重优化"""
    print("\n" + "=" * 80)
    print("示例5：权重优化")
    print("=" * 80)
    
    # 生成示例数据
    df = generate_sample_data(500)
    
    # 创建多因子信号生成器
    multi_factor = MultiFactorSignal()
    
    # 计算所有因子
    factor_df = multi_factor.combiner.calculate_all(df)
    returns = df['close'].pct_change()
    
    # 权重优化
    methods = ['equal', 'risk_parity', 'mean_var', 'adaptive']
    
    for method in methods:
        optimizer = WeightOptimizer(method=method)
        weights = optimizer.optimize(factor_df, returns)
        
        print(f"\n{method.upper()} 方法权重:")
        for factor, weight in list(weights.items())[:5]:
            print(f"  {factor}: {weight:.4f}")
    
    # 集成权重
    ensemble = FactorEnsemble()
    ensembled_weights = ensemble.ensemble(factor_df, returns)
    
    print(f"\n集成权重:")
    for factor, weight in list(ensembled_weights.items())[:5]:
        print(f"  {factor}: {weight:.4f}")

def example_strategy_types():
    """示例6：不同策略类型"""
    print("\n" + "=" * 80)
    print("示例6：不同策略类型")
    print("=" * 80)
    
    # 生成示例数据
    df = generate_sample_data(300)
    
    # 创建不同策略
    strategies = {
        '均值回归': MeanReversionStrategy(zscore_threshold=2.0, window=50),
        '动量': MomentumStrategy(momentum_window=20, volume_threshold=1.5),
        '套利': ArbitrageStrategy(spread_threshold=0.001),
    }
    
    for name, strategy in strategies.items():
        signal_data = strategy.generate_signal(df)
        
        print(f"\n{name} 策略:")
        print(f"  信号: {signal_data['signal']}")
        print(f"  强度: {signal_data['strength']:.4f}")
        if 'reason' in signal_data:
            print(f"  原因: {signal_data['reason']}")

def example_advanced_factors():
    """示例7：高级因子计算"""
    print("\n" + "=" * 80)
    print("示例7：高级因子计算")
    print("=" * 80)
    
    # 生成示例数据
    df = generate_sample_data(200)
    
    # 创建高级因子
    advanced_factors = [
        ('李雅普诺夫指数', LyapunovExponentFactor(window=50)),
        ('Hurst指数', HurstExponentFactor(window=50)),
        ('混沌指示器', MLAnomalyDetectionFactor()),
        ('订单流失衡', OrderFlowImbalanceFactor()),
        ('跨市场相关性', CrossMarketCorrelationFactor()),
    ]
    
    print("\n高级因子计算结果:")
    for name, factor in advanced_factors:
        try:
            result = factor.calculate(df)
            latest_value = result.iloc[-1]
            mean_value = result.mean()
            
            print(f"\n{name}:")
            print(f"  最新值: {latest_value:.4f}")
            print(f"  均值: {mean_value:.4f}")
            print(f"  标准差: {result.std():.4f}")
        except Exception as e:
            print(f"\n{name}: 计算失败 - {e}")

def example_performance_analysis():
    """示例8：性能分析"""
    print("\n" + "=" * 80)
    print("示例8：性能分析")
    print("=" * 80)
    
    # 生成示例数据
    df = generate_sample_data(1000)
    
    # 创建策略
    strategy = HighFrequencyTradingStrategy()
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        leverage=10
    )
    
    # 定义信号生成函数
    def signal_generator(data, symbols):
        signals = {}
        for symbol in symbols:
            if symbol in data.columns:
                symbol_data = data[symbol]
                signal_data = strategy.generate_signal(symbol_data)
                signals[symbol] = signal_data['signal']
        return signals
    
    # 运行回测
    result = engine.run(df, signal_generator, ['BTC/USDT'])
    
    # 详细性能分析
    performance = engine.analyze_performance()
    
    print("\n详细性能分析:")
    print(f"\n主要指标:")
    print(f"  总收益率: {performance['total_return']*100:.2f}%")
    print(f"  年化收益率: {performance['annual_return']*100:.2f}%")
    print(f"  最大回撤: {performance['max_drawdown']*100:.2f}%")
    print(f"  夏普比率: {performance['sharpe_ratio']:.2f}")
    print(f"  索提诺比率: {performance['sortino_ratio']:.2f}")
    print(f"  卡玛比率: {performance['calmar_ratio']:.2f}")
    print(f"  信息比率: {performance['information_ratio']:.2f}")
    
    print(f"\n交易统计:")
    print(f"  总交易数: {performance['total_trades']}")
    print(f"  平均盈利: {performance['avg_profit']:.4f}")
    print(f"  平均亏损: {performance['avg_loss']:.4f}")
    print(f"  盈亏比: {performance['profit_ratio']:.2f}")
    print(f"  胜率: {performance['win_rate']*100:.2f}%")
    
    print(f"\n连续盈亏:")
    print(f"  最大连续盈利: {performance['max_consecutive_wins']}")
    print(f"  最大连续亏损: {performance['max_consecutive_losses']}")
    
    print(f"\n收益分布:")
    print(f"  平均正收益: {performance['avg_positive_return']:.4f}")
    print(f"  平均负收益: {performance['avg_negative_return']:.4f}")
    print(f"  收益偏度: {performance['return_skew']:.4f}")
    
    print(f"\n波动率:")
    print(f"  年化波动率: {performance['volatility']*100:.2f}%")
    print(f"  日均收益: {performance['avg_daily_return']*100:.4f}%")
    print(f"  日波动率: {performance['daily_volatility']*100:.4f}%")

def main():
    """主函数"""
    print("高频交易策略示例")
    print("=" * 80)
    
    # 运行所有示例
    example_basic_backtest()
    example_factor_combination()
    example_multi_factor_signal()
    example_factor_selection()
    example_weight_optimization()
    example_strategy_types()
    example_advanced_factors()
    example_performance_analysis()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
