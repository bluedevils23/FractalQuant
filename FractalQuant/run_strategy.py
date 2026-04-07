"""
完整策略运行脚本
"""
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

from config.config import config
from data.fetcher import exchange_manager
from data.store import DataStore
from strategy.strategy import HighFrequencyTradingStrategy, MultiExchangeStrategy, ArbitrageStrategy, MeanReversionStrategy, MomentumStrategy
from factor.selector import FactorSelector, WeightOptimizer, FactorEnsemble, FactorBacktest
from factor.combiner import MultiFactorSignal, FactorCombiner
from backtest.engine import BacktestEngine
from risk.manager import RiskManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(symbols: List[str], timeframe: str = '1m', days: int = 30) -> pd.DataFrame:
    """加载数据"""
    logger.info(f"Loading data for {symbols}")
    
    # 使用数据存储
    store = DataStore()
    all_data = []
    
    for symbol in symbols:
        try:
            df = store.load_data(symbol, timeframe, days)
            if not df.empty:
                df['symbol'] = symbol
                all_data.append(df)
                logger.info(f"Loaded {len(df)} bars for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
    
    if not all_data:
        logger.warning("No data loaded, using empty DataFrame")
        return pd.DataFrame()
    
    combined_data = pd.concat(all_data, axis=0)
    return combined_data

def run_backtest(
    data: pd.DataFrame,
    strategy: HighFrequencyTradingStrategy,
    symbols: List[str],
    initial_capital: float = 100000
) -> Dict:
    """运行回测"""
    logger.info("Starting backtest...")
    
    def signal_generator(df, symbols):
        signals = {}
        for symbol in symbols:
            if symbol in df.columns:
                symbol_data = df[symbol]
                signal_data = strategy.generate_signal(symbol_data)
                signals[symbol] = signal_data['signal']
        return signals
    
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=config.backtest.commission,
        slippage=config.backtest.slippage,
        leverage=config.risk.leverage
    )
    
    result = engine.run(data, signal_generator, symbols)
    
    # 详细性能分析
    performance = engine.analyze_performance()
    
    return {
        'result': result,
        'performance': performance,
        'equity_curve': engine.get_equity_curve(),
        'trades': engine.get_trades()
    }

def analyze_factors(
    data: pd.DataFrame,
    strategy: HighFrequencyTradingStrategy
) -> Dict:
    """因子分析"""
    logger.info("Analyzing factors...")
    
    # 获取因子数据
    factor_df = strategy.multi_factor.combiner.calculate_all(data)
    returns = data['close'].pct_change().shift(-1)
    
    # 因子选择
    selector = FactorSelector()
    selected_factors = selector.select_factors(factor_df, returns, method='all', target_count=10)
    
    logger.info(f"Selected {len(selected_factors)} factors: {selected_factors}")
    
    # 因子分析
    factor_analysis = strategy.multi_factor.get_factor_analysis(data)
    
    # 因子回测
    backtest = FactorBacktest()
    factor_backtest_results = backtest.backtest_factor_pairs(factor_df, returns, top_n=5)
    
    return {
        'selected_factors': selected_factors,
        'factor_analysis': factor_analysis,
        'factor_backtest_results': factor_backtest_results,
    }

def optimize_weights(
    data: pd.DataFrame,
    strategy: HighFrequencyTradingStrategy
) -> Dict[str, float]:
    """优化因子权重"""
    logger.info("Optimizing weights...")
    
    # 获取因子数据
    factor_df = strategy.multi_factor.combiner.calculate_all(data)
    returns = data['close'].pct_change()
    
    # 使用多种方法优化
    methods = ['equal', 'risk_parity', 'mean_var', 'adaptive']
    all_weights = []
    
    for method in methods:
        optimizer = WeightOptimizer(method=method)
        weights = optimizer.optimize(factor_df, returns)
        all_weights.append(weights)
        logger.info(f"Method {method} weights: {weights}")
    
    # 集成权重
    ensemble = FactorEnsemble()
    ensembled_weights = ensemble.ensemble(factor_df, returns)
    
    logger.info(f"Ensembled weights: {ensembled_weights}")
    
    return ensembled_weights

def run_live_trading(
    strategy: HighFrequencyTradingStrategy,
    symbols: List[str],
    exchange_name: str = 'binance'
):
    """运行实盘交易"""
    logger.info(f"Starting live trading on {exchange_name}")
    
    # 初始化执行器
    strategy.initialize_executor(exchange_name)
    
    # 主循环
    while True:
        try:
            for symbol in symbols:
                # 获取市场数据
                df = strategy.load_historical_data(symbol)
                
                if df.empty:
                    continue
                    
                # 生成信号
                signal_data = strategy.generate_signal(df)
                
                if signal_data['signal'] == 0:
                    continue
                    
                # 检查风险
                current_price = df['close'].iloc[-1]
                capital = strategy._get_available_capital()
                
                if not strategy.check_risk(
                    symbol=symbol,
                    signal=signal_data['signal'],
                    quantity=1,
                    capital=capital,
                    current_price=current_price
                ):
                    continue
                    
                # 计算仓位大小
                position_size = strategy.calculate_position_size(
                    capital=capital,
                    signal_strength=signal_data['strength'],
                    volatility=df['close'].pct_change().std()
                )
                
                # 执行交易
                if strategy.executor:
                    logger.info(f"Executing signal for {symbol}: {signal_data['signal']}, strength: {signal_data['strength']:.2f}")
                    
                # 更新仓位
                strategy.update_position(
                    symbol=symbol,
                    quantity=position_size,
                    current_price=current_price,
                    side='long' if signal_data['signal'] > 0 else 'short'
                )
                
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
            
        # 等待下一次循环
        import time
        time.sleep(60)  # 1分钟

def print_performance_report(result: Dict, performance: Dict):
    """打印性能报告"""
    print("\n" + "="*80)
    print("回测性能报告")
    print("="*80)
    
    print("\n主要指标:")
    print(f"  总收益率: {result.total_return*100:.2f}%")
    print(f"  年化收益率: {result.annual_return*100:.2f}%")
    print(f"  最大回撤: {result.max_drawdown*100:.2f}%")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  索提诺比率: {result.sortino_ratio:.2f}")
    print(f"  卡玛比率: {result.calmar_ratio:.2f}")
    print(f"  信息比率: {result.information_ratio:.2f}")
    print(f"  胜率: {result.win_rate*100:.2f}%")
    print(f"  盈亏比: {result.profit_factor:.2f}")
    
    print("\n交易统计:")
    print(f"  总交易数: {result.total_trades}")
    print(f"  平均持仓时间: {result.avg_trade_duration:.2f} 分钟")
    print(f"  平均盈利: {result.avg_profit:.4f}")
    print(f"  平均亏损: {result.avg_loss:.4f}")
    
    print("\n详细性能指标:")
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*80)

def print_factor_analysis(factor_analysis: Dict):
    """打印因子分析报告"""
    print("\n" + "="*80)
    print("因子分析报告")
    print("="*80)
    
    print(f"\n选中的因子 ({len(factor_analysis['selected_factors'])} 个):")
    for factor in factor_analysis['selected_factors']:
        print(f"  - {factor}")
    
    print("\n因子统计:")
    for factor, stats in factor_analysis['factor_statistics'].items():
        print(f"\n  {factor}:")
        print(f"    均值: {stats['mean']:.4f}")
        print(f"    标准差: {stats['std']:.4f}")
        print(f"    最小值: {stats['min']:.4f}")
        print(f"    最大值: {stats['max']:.4f}")
        print(f"    偏度: {stats['skewness']:.4f}")
        print(f"    峰度: {stats['kurtosis']:.4f}")
        print(f"    与收益相关性: {stats['correlation_with_returns']:.4f}")
    
    print("\n" + "="*80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='High-Frequency Trading Strategy')
    parser.add_argument('--mode', type=str, default='backtest', 
                       choices=['backtest', 'live', 'factor_analysis'],
                       help='运行模式')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'],
                       help='交易对列表')
    parser.add_argument('--timeframe', type=str, default='1m',
                       help='时间框架')
    parser.add_argument('--days', type=int, default=30,
                       help='数据天数')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='交易所')
    parser.add_argument('--initial_capital', type=float, default=100000,
                       help='初始资金')
    parser.add_argument('--strategy_type', type=str, default='hf',
                       choices=['hf', 'arbitrage', 'mean_reversion', 'momentum'],
                       help='策略类型')
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_data(args.symbols, args.timeframe, args.days)
    
    if data.empty:
        logger.error("No data loaded. Exiting.")
        return
    
    # 根据策略类型选择策略
    if args.strategy_type == 'hf':
        strategy = HighFrequencyTradingStrategy()
    elif args.strategy_type == 'arbitrage':
        strategy = ArbitrageStrategy()
    elif args.strategy_type == 'mean_reversion':
        strategy = MeanReversionStrategy()
    elif args.strategy_type == 'momentum':
        strategy = MomentumStrategy()
    else:
        strategy = HighFrequencyTradingStrategy()
    
    if args.mode == 'backtest':
        # 运行回测
        result = run_backtest(
            data=data,
            strategy=strategy,
            symbols=args.symbols,
            initial_capital=args.initial_capital
        )
        
        # 打印性能报告
        print_performance_report(result['result'], result['performance'])
        
        # 因子分析
        factor_analysis = analyze_factors(data, strategy)
        print_factor_analysis(factor_analysis)
        
        # 权重优化
        optimized_weights = optimize_weights(data, strategy)
        print("\n优化后的权重:")
        for factor, weight in optimized_weights.items():
            print(f"  {factor}: {weight:.4f}")
            
    elif args.mode == 'factor_analysis':
        # 因子分析
        factor_analysis = analyze_factors(data, strategy)
        print_factor_analysis(factor_analysis)
        
        # 因子回测结果
        print("\n因子回测结果:")
        print(factor_analysis['factor_backtest_results'])
        
    elif args.mode == 'live':
        # 运行实盘交易
        run_live_trading(strategy, args.symbols, args.exchange)

if __name__ == '__main__':
    main()
