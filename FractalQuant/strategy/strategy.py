"""
主策略模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from config.config import config
from data.models import BarData, MarketData
from data.fetcher import exchange_manager
from factor.combiner import MultiFactorSignal, FactorCombiner
from factor.price import ReturnsFactor, PriceMomentumFactor
from factor.volatility import HistoricalVolatilityFactor, ParkinsonVolatilityFactor
from factor.trend import MACDFactor, RSIFactor
from factor.orderbook import OrderBookImbalanceFactor
from factor.advanced import FutureReturnsFactor, CorrelationFactor, HurstExponentFactor, LyapunovExponentFactor
from factor.ml import MLForecastFactor, MLAnomalyDetectionFactor, ClusteringRegimeFactor
from factor.microstructure import OrderFlowImbalanceFactor, LiquidityRatioFactor
from factor.crossmarket import CrossMarketCorrelationFactor, ArbitrageOpportunityFactor
from signal.generator import SignalGenerator, SignalFilter, SignalOptimizer
from signal.optimizer import PortfolioOptimizer, PositionSizer
from risk.manager import RiskManager, StopLossManager, VolatilityRiskManager
from execution.executor import OrderExecutor, MarketOrderExecutor
from backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)

class HighFrequencyTradingStrategy:
    """高频交易策略"""
    
    def __init__(self):
        """初始化策略"""
        # 初始化组件
        self.signal_generator = SignalGenerator(
            threshold=0.0,
            min_confidence=0.6,
            cooldown_period=5
        )
        
        self.signal_filter = SignalFilter(
            min_volume=100,
            max_spread=0.002,
            min_trend_strength=0.5
        )
        
        self.signal_optimizer = SignalOptimizer(
            lookback_window=30,
            min_signals=5
        )
        
        self.portfolio_optimizer = PortfolioOptimizer(
            max_positions=10,
            min_sharpe_ratio=0.5,
            max_volatility=0.3
        )
        
        self.position_sizer = PositionSizer(
            max_position_size=0.1,
            risk_per_trade=0.02,
            leverage=config.risk.leverage
        )
        
        self.risk_manager = RiskManager(
            max_drawdown=config.risk.max_drawdown,
            max_position=config.risk.max_position_size,
            max_positions=config.risk.max_positions,
            stop_loss=config.risk.stop_loss_threshold,
            take_profit=config.risk.take_profit_threshold,
            max_daily_loss=0.10,
            max_volatility=0.3
        )
        
        self.stop_loss_manager = StopLossManager(
            fixed_stop_loss=config.risk.stop_loss_threshold,
            atr_multiplier=2.0,
            trailing_stop=0.03
        )
        
        self.volatility_manager = VolatilityRiskManager(
            max_volatility=0.3,
            min_volatility=0.01,
            position_reduction=0.5
        )
        
        # 初始化多因子信号生成器
        self.multi_factor = MultiFactorSignal(
            ml_factors=[
                MLForecastFactor(model_type='linear'),
                MLAnomalyDetectionFactor(),
                ClusteringRegimeFactor(),
            ],
            microstructure_factors=[
                OrderFlowImbalanceFactor(),
                LiquidityRatioFactor(),
            ],
            crossmarket_factors=[
                CrossMarketCorrelationFactor(),
                ArbitrageOpportunityFactor(),
            ]
        )
        
        # 市场数据
        self.market_data = MarketData()
        
        # 订单执行器
        self.executor = None
        
        # 当前仓位
        self.positions: Dict[str, Dict] = {}
        
        # 历史信号
        self.signal_history: Dict[str, List[int]] = {}
        
    def initialize_executor(self, exchange_name: str):
        """初始化执行器"""
        exchange_config = config.get_exchange_config(exchange_name)
        if exchange_config:
            self.executor = MarketOrderExecutor(
                commission=config.backtest.commission,
                slippage=config.backtest.slippage
            )
            logger.info(f"Initialized executor for {exchange_name}")
        else:
            logger.error(f"Exchange {exchange_name} not configured")
            
    def load_historical_data(self, symbol: str, timeframe: str = '1m', days: int = 30):
        """加载历史数据（简化版本）"""
        # 实际应用中应该从数据存储中加载
        # 这里简化为返回空数据
        logger.info(f"Loading historical data for {symbol}")
        return pd.DataFrame()
        
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子（包括高级因子）"""
        factor_values = {}
        
        # 价格因子
        price_factors = [
            ReturnsFactor(window=5),
            PriceMomentumFactor(window=20),
        ]
        
        for factor in price_factors:
            try:
                factor_values[factor.name] = factor.calculate(df)
            except Exception as e:
                logger.error(f"Error calculating {factor.name}: {e}")
                
        # 波动率因子
        volatility_factors = [
            HistoricalVolatilityFactor(window=20),
            ParkinsonVolatilityFactor(window=20),
        ]
        
        for factor in volatility_factors:
            try:
                factor_values[factor.name] = factor.calculate(df)
            except Exception as e:
                logger.error(f"Error calculating {factor.name}: {e}")
                
        # 趋势因子
        trend_factors = [
            MACDFactor(),
            RSIFactor(),
        ]
        
        for factor in trend_factors:
            try:
                factor_values[factor.name] = factor.calculate(df)
            except Exception as e:
                logger.error(f"Error calculating {factor.name}: {e}")
                
        # 高级统计因子
        advanced_factors = [
            HurstExponentFactor(window=50),
            LyapunovExponentFactor(window=50),
        ]
        
        for factor in advanced_factors:
            try:
                factor_values[factor.name] = factor.calculate(df)
            except Exception as e:
                logger.error(f"Error calculating {factor.name}: {e}")
                
        # 机器学习因子
        ml_factors = [
            MLForecastFactor(model_type='linear'),
            MLAnomalyDetectionFactor(),
            ClusteringRegimeFactor(),
        ]
        
        for factor in ml_factors:
            try:
                factor_values[factor.name] = factor.calculate(df)
            except Exception as e:
                logger.error(f"Error calculating {factor.name}: {e}")
                
        # 微观结构因子
        microstructure_factors = [
            OrderFlowImbalanceFactor(),
            LiquidityRatioFactor(),
        ]
        
        for factor in microstructure_factors:
            try:
                factor_values[factor.name] = factor.calculate(df)
            except Exception as e:
                logger.error(f"Error calculating {factor.name}: {e}")
                
        # 跨市场因子
        crossmarket_factors = [
            CrossMarketCorrelationFactor(),
            ArbitrageOpportunityFactor(),
        ]
        
        for factor in crossmarket_factors:
            try:
                factor_values[factor.name] = factor.calculate(df)
            except Exception as e:
                logger.error(f"Error calculating {factor.name}: {e}")
                
        return pd.DataFrame(factor_values)
        
    def generate_signal(self, df: pd.DataFrame, orderbook: Dict = None) -> Dict:
        """生成交易信号"""
        # 计算因子
        factors = self.calculate_factors(df)
        
        # 生成多因子信号
        signal_data = self.multi_factor.generate_signal(df, orderbook)
        
        # 应用信号过滤器
        if not self.signal_filter.filter_signal(signal_data['signal'], df, orderbook):
            signal_data['signal'] = 0
            signal_data['reason'] = 'filtered'
            
        # 优化信号
        symbol = df.index.get_level_values('symbol')[0] if 'symbol' in df.index.names else 'BTC/USDT'
        
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
            
        signal_data['signal'] = self.signal_optimizer.optimize_signal(
            signal_data['signal'],
            self.signal_history[symbol]
        )
        
        # 更新信号历史
        self.signal_history[symbol].append(signal_data['signal'])
        if len(self.signal_history[symbol]) > 30:
            self.signal_history[symbol] = self.signal_history[symbol][-30:]
            
        return signal_data
        
    def calculate_position_size(
        self,
        capital: float,
        signal_strength: float,
        volatility: float
    ) -> float:
        """计算仓位大小"""
        return self.position_sizer.calculate_position_size(
            capital=capital,
            signal_strength=signal_strength,
            volatility=volatility
        )
        
    def check_risk(
        self,
        symbol: str,
        signal: int,
        quantity: float,
        capital: float,
        current_price: float,
        entry_price: Optional[float] = None
    ) -> bool:
        """检查风险"""
        return self.risk_manager.check_risk(
            symbol=symbol,
            signal=signal,
            quantity=quantity,
            capital=capital,
            current_price=current_price,
            entry_price=entry_price
        )
        
    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ):
        """更新仓位"""
        self.risk_manager.update_position(symbol, quantity, price, side)
        
        if symbol not in self.positions:
            self.positions[symbol] = {}
            
        self.positions[symbol]['quantity'] = quantity
        self.positions[symbol]['price'] = price
        self.positions[symbol]['side'] = side
        
    def close_position(self, symbol: str, current_price: float) -> float:
        """平仓"""
        if symbol not in self.positions:
            return 0
            
        position = self.positions[symbol]
        pnl = 0
        
        if position['side'] == 'long':
            pnl = (current_price - position['price']) * position['quantity']
        else:
            pnl = (position['price'] - current_price) * position['quantity']
            
        del self.positions[symbol]
        return pnl
        
    def run_backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = None
    ) -> Dict:
        """运行回测"""
        if initial_capital is None:
            initial_capital = config.backtest.initial_capital
            
        def signal_generator(df, symbols):
            signals = {}
            for symbol in symbols:
                if symbol in df.columns:
                    symbol_data = df[symbol]
                    signal_data = self.generate_signal(symbol_data)
                    signals[symbol] = signal_data['signal']
            return signals
            
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=config.backtest.commission,
            slippage=config.backtest.slippage,
            leverage=config.risk.leverage
        )
        
        symbols = list(self.positions.keys()) if self.positions else config.get_all_pairs()
        
        result = engine.run(data, signal_generator, symbols)
        
        return {
            'result': result,
            'equity_curve': engine.get_equity_curve(),
            'trades': engine.get_trades()
        }
        
    async def run_live_trading(self, exchange_name: str = 'binance'):
        """运行实盘交易（模拟）"""
        logger.info(f"Starting live trading on {exchange_name}")
        
        # 初始化执行器
        self.initialize_executor(exchange_name)
        
        # 获取交易对
        symbols = config.get_all_pairs()
        
        # 主循环
        while True:
            try:
                for symbol in symbols:
                    # 获取市场数据
                    df = self.load_historical_data(symbol)
                    
                    if df.empty:
                        continue
                        
                    # 生成信号
                    signal_data = self.generate_signal(df)
                    
                    if signal_data['signal'] == 0:
                        continue
                        
                    # 检查风险
                    current_price = df['close'].iloc[-1]
                    capital = self._get_available_capital()
                    
                    if not self.check_risk(
                        symbol=symbol,
                        signal=signal_data['signal'],
                        quantity=1,
                        capital=capital,
                        current_price=current_price
                    ):
                        continue
                        
                    # 计算仓位大小
                    position_size = self.calculate_position_size(
                        capital=capital,
                        signal_strength=signal_data['strength'],
                        volatility=df['close'].pct_change().std()
                    )
                    
                    # 执行交易
                    if self.executor:
                        order = await self.executor.execute_signal(
                            symbol=symbol,
                            signal=signal_data['signal'],
                            quantity=position_size
                        )
                        
                        if order:
                            # 更新仓位
                            self.update_position(
                                symbol=symbol,
                                quantity=position_size,
                                price=current_price,
                                side='long' if signal_data['signal'] > 0 else 'short'
                            )
                            
                # 等待下一次循环
                await asyncio.sleep(60)  # 1分钟
                
            except Exception as e:
                logger.error(f"Error in live trading: {e}")
                await asyncio.sleep(60)
                
    def _get_available_capital(self) -> float:
        """获取可用资金（简化版本）"""
        return config.backtest.initial_capital * 0.9
        
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        metrics = self.risk_manager.get_risk_metrics()
        
        # 计算因子表现
        if self.positions:
            metrics['positions'] = self.positions
            
        return metrics

class MultiExchangeStrategy(HighFrequencyTradingStrategy):
    """多交易所策略"""
    
    def __init__(self):
        super().__init__()
        self.exchanges: Dict[str, HighFrequencyTradingStrategy] = {}
        
    def add_exchange(self, exchange_name: str):
        """添加交易所"""
        strategy = HighFrequencyTradingStrategy()
        strategy.initialize_executor(exchange_name)
        self.exchanges[exchange_name] = strategy
        
    def get_best_opportunity(self, symbols: List[str]) -> Optional[Dict]:
        """获取最佳机会"""
        opportunities = []
        
        for exchange_name, strategy in self.exchanges.items():
            for symbol in symbols:
                df = strategy.load_historical_data(symbol)
                
                if not df.empty:
                    signal_data = strategy.generate_signal(df)
                    
                    if signal_data['signal'] != 0:
                        opportunities.append({
                            'exchange': exchange_name,
                            'symbol': symbol,
                            'signal': signal_data['signal'],
                            'strength': signal_data['strength'],
                            'price': df['close'].iloc[-1]
                        })
                        
        if opportunities:
            opportunities.sort(key=lambda x: abs(x['strength']), reverse=True)
            return opportunities[0]
            
        return None

class ArbitrageStrategy(HighFrequencyTradingStrategy):
    """套利策略"""
    
    def __init__(self, spread_threshold: float = 0.001):
        super().__init__()
        self.spread_threshold = spread_threshold
        
    def find_arbitrage_opportunity(
        self,
        symbol: str,
        exchanges: List[str]
    ) -> Optional[Dict]:
        """
        寻找套利机会
        
        Args:
            symbol: 交易对
            exchanges: 交易所列表
            
        Returns:
            套利机会
        """
        prices = {}
        
        for exchange_name in exchanges:
            if exchange_name in self.exchanges:
                strategy = self.exchanges[exchange_name]
                df = strategy.load_historical_data(symbol)
                
                if not df.empty:
                    prices[exchange_name] = df['close'].iloc[-1]
                    
        if len(prices) < 2:
            return None
            
        # 计算价差
        exchange_list = list(prices.keys())
        for i in range(len(exchange_list)):
            for j in range(i + 1, len(exchange_list)):
                spread = abs(prices[exchange_list[i]] - prices[exchange_list[j]]) / \
                        ((prices[exchange_list[i]] + prices[exchange_list[j]]) / 2)
                        
                if spread > self.spread_threshold:
                    buy_exchange = exchange_list[i] if prices[exchange_list[i]] < prices[exchange_list[j]] else exchange_list[j]
                    sell_exchange = exchange_list[j] if prices[exchange_list[i]] < prices[exchange_list[j]] else exchange_list[i]
                    
                    return {
                        'buy_exchange': buy_exchange,
                        'sell_exchange': sell_exchange,
                        'symbol': symbol,
                        'spread': spread,
                        'buy_price': prices[buy_exchange],
                        'sell_price': prices[sell_exchange]
                    }
                    
        return None

class MeanReversionStrategy(HighFrequencyTradingStrategy):
    """均值回归策略"""
    
    def __init__(self, zscore_threshold: float = 2.0, window: int = 50):
        super().__init__()
        self.zscore_threshold = zscore_threshold
        self.window = window
        
    def generate_signal(self, df: pd.DataFrame, orderbook: Dict = None) -> Dict:
        """生成交易信号"""
        # 计算Z-score
        close = df['close']
        mean = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()
        zscore = (close - mean) / (std + 1e-8)
        
        # 生成信号
        if zscore.iloc[-1] > self.zscore_threshold:
            signal = -1  # 卖出
            strength = abs(zscore.iloc[-1]) / (self.zscore_threshold * 2)
        elif zscore.iloc[-1] < -self.zscore_threshold:
            signal = 1  # 买入
            strength = abs(zscore.iloc[-1]) / (self.zscore_threshold * 2)
        else:
            signal = 0
            strength = 0
            
        return {
            'signal': signal,
            'strength': strength,
            'zscore': zscore.iloc[-1],
            'reason': f"Z-score: {zscore.iloc[-1]:.2f}"
        }

class MomentumStrategy(HighFrequencyTradingStrategy):
    """动量策略"""
    
    def __init__(
        self,
        momentum_window: int = 20,
        volume_threshold: float = 1.5
    ):
        super().__init__()
        self.momentum_window = momentum_window
        self.volume_threshold = volume_threshold
        
    def generate_signal(self, df: pd.DataFrame, orderbook: Dict = None) -> Dict:
        """生成交易信号"""
        # 计算动量
        returns = df['close'].pct_change()
        momentum = returns.rolling(window=self.momentum_window).sum()
        
        # 计算成交量变化
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / (volume_ma + 1e-8)
        
        # 生成信号
        if momentum.iloc[-1] > 0 and volume_ratio.iloc[-1] > self.volume_threshold:
            signal = 1
            strength = min(momentum.iloc[-1] * volume_ratio.iloc[-1] / 10, 1.0)
        elif momentum.iloc[-1] < 0 and volume_ratio.iloc[-1] > self.volume_threshold:
            signal = -1
            strength = min(abs(momentum.iloc[-1]) * volume_ratio.iloc[-1] / 10, 1.0)
        else:
            signal = 0
            strength = 0
            
        return {
            'signal': signal,
            'strength': strength,
            'momentum': momentum.iloc[-1],
            'volume_ratio': volume_ratio.iloc[-1],
            'reason': f"Momentum: {momentum.iloc[-1]:.4f}, Volume Ratio: {volume_ratio.iloc[-1]:.2f}"
        }