"""
主程序入口
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse
import asyncio
import sys

from config.config import config, logger
from strategy.strategy import (
    HighFrequencyTradingStrategy,
    MultiExchangeStrategy,
    ArbitrageStrategy,
    MeanReversionStrategy,
    MomentumStrategy
)
from backtest.engine import BacktestEngine
from data.store import DataStore

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_sample_data(symbols: list, days: int = 30) -> pd.DataFrame:
    """生成示例数据（实际应用中应该从交易所获取）"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_data = {}
    
    for symbol in symbols:
        # 生成随机价格数据
        np.random.seed(hash(symbol) % 2**32)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        n_points = len(dates)
        
        # 生成价格路径（随机游走）
        price = 10000  # 初始价格
        prices = [price]
        
        for _ in range(n_points - 1):
            change = np.random.normal(0, 0.002)  # 0.2%的波动
            price = price * (1 + change)
            prices.append(price)
        
        # 生成OHLC数据
        opens = prices[:-1]
        closes = prices[1:]
        highs = [max(o, c) * (1 + np.random.uniform(0, 0.001)) for o, c in zip(opens, closes)]
        lows = [min(o, c) * (1 - np.random.uniform(0, 0.001)) for o, c in zip(opens, closes)]
        volumes = [np.random.uniform(100, 1000) for _ in range(n_points - 1)]
        
        # 创建DataFrame
        symbol_data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates[:-1])
        
        all_data[symbol] = symbol_data
    
    # 创建MultiIndex DataFrame
    data = pd.concat(all_data, axis=1)
    data.columns = pd.MultiIndex.from_tuples(
        [(col, idx) for idx, cols in all_data.items() for col in cols.columns],
        names=['symbol', 'field']
    )
    
    return data

def run_backtest():
    """运行回测"""
    logger.info("Starting backtest...")
    
    # 生成示例数据
    symbols = config.get_all_pairs()
    data = generate_sample_data(symbols, days=30)
    
    # 创建策略
    strategy = HighFrequencyTradingStrategy()
    
    # 运行回测
    result = strategy.run_backtest(data)
    
    # 打印结果
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    r = result['result']
    print(f"Total Return: {r.total_return:.2%}")
    print(f"Annual Return: {r.annual_return:.2%}")
    print(f"Max Drawdown: {r.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {r.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {r.sortino_ratio:.2f}")
    print(f"Win Rate: {r.win_rate:.2%}")
    print(f"Profit Factor: {r.profit_factor:.2f}")
    print(f"Total Trades: {r.total_trades}")
    print(f"Avg Trade Duration: {r.avg_trade_duration:.2f} min")
    print(f"Avg Profit: {r.avg_profit:.2f}")
    print(f"Avg Loss: {r.avg_loss:.2f}")
    print("="*60)
    
    # 保存结果
    data_store = DataStore()
    data_store.save_bars('BTC/USDT', [], 'backtest')
    
    logger.info("Backtest completed")
    
    return result

def run_live_trading(exchange: str = 'binance'):
    """运行实盘交易（模拟）"""
    logger.info(f"Starting live trading on {exchange}...")
    
    # 创建策略
    strategy = HighFrequencyTradingStrategy()
    
    # 初始化执行器
    strategy.initialize_executor(exchange)
    
    # 获取交易对
    symbols = config.get_all_pairs()
    
    logger.info(f"Monitoring {len(symbols)} symbols: {symbols}")
    
    # 模拟主循环
    try:
        for i in range(10):  # 模拟10次循环
            logger.info(f"\n--- Loop {i+1} ---")
            
            for symbol in symbols:
                # 加载数据
                df = strategy.load_historical_data(symbol)
                
                if df.empty:
                    continue
                    
                # 生成信号
                signal_data = strategy.generate_signal(df)
                
                logger.info(f"{symbol}: Signal={signal_data['signal']}, "
                          f"Strength={signal_data['strength']:.2f}, "
                          f"Reason={signal_data['reason']}")
                
                # 检查是否需要交易
                if signal_data['signal'] != 0:
                    current_price = df['close'].iloc[-1]
                    capital = strategy._get_available_capital()
                    
                    # 计算仓位大小
                    position_size = strategy.calculate_position_size(
                        capital=capital,
                        signal_strength=signal_data['strength'],
                        volatility=df['close'].pct_change().std()
                    )
                    
                    logger.info(f"Executing {signal_data['signal'] > 0 and 'BUY' or 'SELL'} "
                              f"order for {symbol}: {position_size:.4f} units at {current_price:.2f}")
                    
                    # 更新仓位
                    strategy.update_position(
                        symbol=symbol,
                        quantity=position_size,
                        price=current_price,
                        side='long' if signal_data['signal'] > 0 else 'short'
                    )
                    
            # 等待下一次循环
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
        
    # 打印最终仓位
    if strategy.positions:
        logger.info("\nFinal Positions:")
        for symbol, position in strategy.positions.items():
            logger.info(f"  {symbol}: {position}")

def run_arbitrage():
    """运行套利策略"""
    logger.info("Starting arbitrage strategy...")
    
    # 创建多交易所策略
    strategy = MultiExchangeStrategy()
    
    # 添加交易所
    for exchange_name in config.exchanges.keys():
        strategy.add_exchange(exchange_name)
        
    # 获取交易对
    symbols = config.get_all_pairs()
    
    # 寻找套利机会
    opportunity = strategy.get_best_opportunity(symbols)
    
    if opportunity:
        logger.info(f"Found arbitrage opportunity: {opportunity}")
    else:
        logger.info("No arbitrage opportunities found")
        
    return opportunity

def run_mean_reversion():
    """运行均值回归策略"""
    logger.info("Starting mean reversion strategy...")
    
    # 生成示例数据
    symbols = config.get_all_pairs()
    data = generate_sample_data(symbols, days=30)
    
    # 创建策略
    strategy = MeanReversionStrategy(zscore_threshold=2.0, window=50)
    
    # 运行回测
    result = strategy.run_backtest(data)
    
    # 打印结果
    print("\nMean Reversion Strategy Results:")
    r = result['result']
    print(f"Total Return: {r.total_return:.2%}")
    print(f"Sharpe Ratio: {r.sharpe_ratio:.2f}")
    
    return result

def run_momentum():
    """运行动量策略"""
    logger.info("Starting momentum strategy...")
    
    # 生成示例数据
    symbols = config.get_all_pairs()
    data = generate_sample_data(symbols, days=30)
    
    # 创建策略
    strategy = MomentumStrategy(momentum_window=20, volume_threshold=1.5)
    
    # 运行回测
    result = strategy.run_backtest(data)
    
    # 打印结果
    print("\nMomentum Strategy Results:")
    r = result['result']
    print(f"Total Return: {r.total_return:.2%}")
    print(f"Sharpe Ratio: {r.sharpe_ratio:.2f}")
    
    return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加密货币高频交易策略')
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'live', 'arbitrage', 'mean_reversion', 'momentum'],
                       help='运行模式')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='交易所名称')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 运行
    if args.mode == 'backtest':
        run_backtest()
    elif args.mode == 'live':
        run_live_trading(args.exchange)
    elif args.mode == 'arbitrage':
        run_arbitrage()
    elif args.mode == 'mean_reversion':
        run_mean_reversion()
    elif args.mode == 'momentum':
        run_momentum()

if __name__ == '__main__':
    main()