"""
回测引擎
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)

class Position:
    """仓位类"""
    
    def __init__(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        entry_time: datetime
    ):
        """
        初始化仓位
        
        Args:
            symbol: 交易对
            side: 买卖方向
            quantity: 数量
            entry_price: 入场价格
            entry_time: 入场时间
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0
        self.pnl_ratio = 0
        self.fee = 0
        
    def close(self, exit_price: float, exit_time: datetime, fee: float = 0):
        """
        平仓
        
        Args:
            exit_price: 平仓价格
            exit_time: 平仓时间
            fee: 手续费
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.fee = fee
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_ratio = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_ratio = (self.entry_price - exit_price) / self.entry_price
            
        self.pnl -= fee
        
    def is_open(self) -> bool:
        """检查是否持仓中"""
        return self.exit_price is None
    
    def current_value(self, current_price: float) -> float:
        """计算当前价值"""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

class Trade:
    """交易类"""
    
    def __init__(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time: datetime,
        fee: float = 0
    ):
        """
        初始化交易
        
        Args:
            symbol: 交易对
            side: 买卖方向
            quantity: 数量
            price: 价格
            time: 时间
            fee: 手续费
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.time = time
        self.fee = fee
        self.profit = 0
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'time': self.time.isoformat(),
            'fee': self.fee,
            'profit': self.profit
        }

@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    avg_profit: float
    avg_loss: float
    pnl_series: pd.Series
    trades: List[Dict]
    equity_curve: pd.Series
    drawdown_series: pd.Series
    monthly_returns: pd.Series
    trade_details: List[Dict]

class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        leverage: int = 1
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费
            slippage: 滑点
            leverage: 杠杆
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
        self.max_equity = initial_capital
        self.max_drawdown = 0
        self.rolling_max_drawdown = 0
        
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
        symbols: List[str] = None
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            data: 数据
            signal_generator: 信号生成函数
            symbols: 交易对列表
            
        Returns:
            回测结果
        """
        if symbols is None:
            symbols = data.columns.get_level_values(0).unique().tolist()
            
        # 初始化
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.timestamps = [data.index[0]]
        self.max_equity = self.initial_capital
        self.max_drawdown = 0
        
        # 遍历数据
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_data = data.iloc[:i+1]
            
            # 生成信号
            signals = signal_generator(current_data, symbols)
            
            # 执行信号
            self._execute_signals(signals, data.iloc[i], current_time)
            
            # 更新权益曲线
            self._update_equity(data.iloc[i], current_time)
            
        # 计算结果
        return self._calculate_result(data)
    
    def _execute_signals(
        self,
        signals: Dict[str, int],
        current_bar: pd.Series,
        current_time: datetime
    ):
        """
        执行信号
        
        Args:
            signals: 信号字典
            current_bar: 当前K线
            current_time: 当前时间
        """
        for symbol, signal in signals.items():
            if symbol not in current_bar.index:
                continue
                
            current_price = current_bar[symbol]['close']
            
            if signal > 0:
                # 买入
                self._open_position(symbol, 'long', current_price, current_time)
            elif signal < 0:
                # 卖出
                self._open_position(symbol, 'short', current_price, current_time)
            else:
                # 平仓
                self._close_position(symbol, current_price, current_time)
    
    def _open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        time: datetime
    ):
        """
        开仓
        
        Args:
            symbol: 交易对
            side: 买卖方向
            price: 价格
            time: 时间
        """
        # 计算可用资金
        available_capital = self.capital * 0.9  # 留10%作为保证金
        
        # 计算仓位大小
        position_size = available_capital / price * self.leverage
        
        # 计算手续费
        fee = position_size * price * self.commission
        
        # 开仓
        position = Position(
            symbol=symbol,
            side=side,
            quantity=position_size,
            entry_price=price,
            entry_time=time
        )
        
        self.positions[symbol] = position
        self.capital -= fee
        
        # 记录交易
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=position_size,
            price=price,
            time=time,
            fee=fee
        )
        self.trades.append(trade)
        
    def _close_position(
        self,
        symbol: str,
        price: float,
        time: datetime
    ):
        """
        平仓
        
        Args:
            symbol: 交易对
            price: 价格
            time: 时间
        """
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # 计算手续费
        fee = position.quantity * price * self.commission
        
        # 平仓
        position.close(price, time, fee)
        
        # 更新资金
        self.capital += position.pnl
        
        # 记录交易
        if self.trades:
            self.trades[-1].profit = position.pnl
            
        # 删除仓位
        del self.positions[symbol]
        
    def _update_equity(self, current_bar: pd.Series, current_time: datetime):
        """
        更新权益曲线
        
        Args:
            current_bar: 当前K线
            current_time: 当前时间
        """
        # 计算持仓价值
        position_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_bar.index:
                current_price = current_bar[symbol]['close']
                position_value += position.current_value(current_price)
                
        # 计算权益
        equity = self.capital + position_value
        self.equity_curve.append(equity)
        self.timestamps.append(current_time)
        
        # 更新最大权益和回撤
        if equity > self.max_equity:
            self.max_equity = equity
            
        drawdown = (self.max_equity - equity) / self.max_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
    def _calculate_result(self, data: pd.DataFrame) -> BacktestResult:
        """计算回测结果"""
        # 计算收益序列
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        returns = equity_series.pct_change().fillna(0)
        
        # 计算回撤序列
        running_max = equity_series.cummax()
        drawdown_series = (equity_series - running_max) / running_max
        
        # 计算月度收益
        monthly_returns = equity_series.resample('D').last().pct_change()
        
        # 计算各项指标
        total_return = equity_series.iloc[-1] / self.initial_capital - 1
        
        # 年化收益
        days = (self.timestamps[-1] - self.timestamps[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
            
        # 夏普比率
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)
        else:
            sharpe_ratio = 0
            
        # 索提诺比率
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252 * 24 * 60)
        else:
            sortino_ratio = 0
            
        # 卡玛比率
        if self.max_drawdown > 0:
            calmar_ratio = annual_return / abs(self.max_drawdown)
        else:
            calmar_ratio = 0
            
        # 信息比率
        benchmark_returns = returns.rolling(20).mean().shift(1)
        active_returns = returns - benchmark_returns
        if active_returns.std() > 0:
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252 * 24 * 60)
        else:
            information_ratio = 0
            
        # 胜率和盈亏比
        winning_trades = [t for t in self.trades if t.profit > 0]
        losing_trades = [t for t in self.trades if t.profit < 0]
        
        if len(self.trades) > 0:
            win_rate = len(winning_trades) / len(self.trades)
        else:
            win_rate = 0
            
        if len(losing_trades) > 0:
            profit_factor = sum(t.profit for t in winning_trades) / abs(sum(t.profit for t in losing_trades))
        else:
            profit_factor = float('inf') if len(winning_trades) > 0 else 0
            
        # 平均盈亏
        if winning_trades:
            avg_profit = sum(t.profit for t in winning_trades) / len(winning_trades)
        else:
            avg_profit = 0
            
        if losing_trades:
            avg_loss = sum(t.profit for t in losing_trades) / len(losing_trades)
        else:
            avg_loss = 0
            
        # 平均持仓时间
        trade_durations = []
        for i in range(0, len(self.trades) - 1, 2):
            if i + 1 < len(self.trades):
                duration = (self.trades[i+1].time - self.trades[i].time).total_seconds() / 60
                trade_durations.append(duration)
                
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # 转换交易记录
        trades_dict = [t.to_dict() for t in self.trades]
        
        # 构建交易详情
        trade_details = []
        for i in range(0, len(self.trades) - 1, 2):
            if i + 1 < len(self.trades):
                entry_trade = self.trades[i]
                exit_trade = self.trades[i + 1]
                trade_detail = {
                    'entry_time': entry_trade.time,
                    'exit_time': exit_trade.time,
                    'entry_price': entry_trade.price,
                    'exit_price': exit_trade.price,
                    'profit': exit_trade.profit,
                    'duration': (exit_trade.time - entry_trade.time).total_seconds() / 60,
                    'symbol': entry_trade.symbol,
                    'side': entry_trade.side,
                }
                trade_details.append(trade_detail)
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_trade_duration=avg_trade_duration,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            pnl_series=returns,
            trades=trades_dict,
            equity_curve=equity_series,
            drawdown_series=drawdown_series,
            monthly_returns=monthly_returns,
            trade_details=trade_details
        )
    
    def get_equity_curve(self) -> pd.Series:
        """获取权益曲线"""
        return pd.Series(self.equity_curve, index=self.timestamps)
    
    def get_positions(self) -> Dict[str, Position]:
        """获取当前仓位"""
        return self.positions
    
    def get_trades(self) -> List[Trade]:
        """获取交易记录"""
        return self.trades
    
    def analyze_performance(self) -> Dict:
        """
        详细性能分析
        
        Returns:
            性能分析字典
        """
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        returns = equity_series.pct_change().fillna(0)
        
        # 计算统计指标
        stats = {
            'total_return': equity_series.iloc[-1] / self.initial_capital - 1,
            'annual_return': (1 + equity_series.iloc[-1] / self.initial_capital) ** (365 / ((self.timestamps[-1] - self.timestamps[0]).days + 1)) - 1,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0,
            'sortino_ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252 * 24 * 60) if len(returns[returns < 0]) > 0 and returns[returns < 0].std() > 0 else 0,
            'calmar_ratio': (1 + equity_series.iloc[-1] / self.initial_capital) ** (365 / ((self.timestamps[-1] - self.timestamps[0]).days + 1)) - 1 / abs(self.max_drawdown) if self.max_drawdown > 0 else 0,
            'win_rate': len([t for t in self.trades if t.profit > 0]) / len(self.trades) if self.trades else 0,
            'profit_factor': sum(t.profit for t in self.trades if t.profit > 0) / abs(sum(t.profit for t in self.trades if t.profit < 0)) if any(t.profit < 0 for t in self.trades) else float('inf') if any(t.profit > 0 for t in self.trades) else 0,
            'total_trades': len(self.trades),
            'avg_trade': sum(t.profit for t in self.trades) / len(self.trades) if self.trades else 0,
            'skewness': stats.skew(returns.dropna()),
            'kurtosis': stats.kurtosis(returns.dropna()),
            'volatility': returns.std() * np.sqrt(252 * 24 * 60),
        }
        
        # 计算最大连续盈利/亏损
        consecutive_wins = 0
        consecutive_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in self.trades:
            if trade.profit > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_wins = max(max_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_losses = max(max_losses, consecutive_losses)
        
        stats['max_consecutive_wins'] = max_wins
        stats['max_consecutive_losses'] = max_losses
        
        # 计算盈亏比分布
        profits = [t.profit for t in self.trades if t.profit > 0]
        losses = [t.profit for t in self.trades if t.profit < 0]
        
        stats['avg_profit'] = np.mean(profits) if profits else 0
        stats['avg_loss'] = np.mean(losses) if losses else 0
        stats['profit_ratio'] = stats['avg_profit'] / abs(stats['avg_loss']) if stats['avg_loss'] != 0 else 0
        
        # 计算不同时间段的表现
        equity_df = pd.DataFrame({'equity': self.equity_curve}, index=self.timestamps)
        equity_df['date'] = equity_df.index.date
        
        daily_returns = equity_df['equity'].pct_change()
        stats['avg_daily_return'] = daily_returns.mean()
        stats['daily_volatility'] = daily_returns.std()
        
        # 计算夏普比率分解
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        stats['avg_positive_return'] = positive_returns.mean() if len(positive_returns) > 0 else 0
        stats['avg_negative_return'] = negative_returns.mean() if len(negative_returns) > 0 else 0
        stats['return_skew'] = stats.skew(returns.dropna())
        
        return stats

class MultiAssetBacktestEngine:
    """多资产回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        leverage: int = 1
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        
        self.engines: Dict[str, BacktestEngine] = {}
        
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
        symbols: List[str]
    ) -> Dict[str, BacktestResult]:
        """
        运行多资产回测
        
        Args:
            data: 数据
            signal_generator: 信号生成函数
            symbols: 交易对列表
            
        Returns:
            回测结果字典
        """
        results = {}
        
        for symbol in symbols:
            if symbol in data.columns:
                # 提取单个资产的数据
                symbol_data = data[symbol]
                
                # 创建回测引擎
                engine = BacktestEngine(
                    initial_capital=self.initial_capital / len(symbols),
                    commission=self.commission,
                    slippage=self.slippage,
                    leverage=self.leverage
                )
                
                # 运行回测
                result = engine.run(symbol_data, signal_generator, [symbol])
                results[symbol] = result
                
        return results

class WalkForwardBacktest:
    """滚动窗口回测"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        leverage: int = 1,
        train_ratio: float = 0.7,
        rolling_window: int = 30
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.train_ratio = train_ratio
        self.rolling_window = rolling_window
        
    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
        symbols: List[str] = None
    ) -> List[BacktestResult]:
        """
        运行滚动窗口回测
        
        Args:
            data: 数据
            signal_generator: 信号生成函数
            symbols: 交易对列表
            
        Returns:
            回测结果列表
        """
        if symbols is None:
            symbols = data.columns.get_level_values(0).unique().tolist()
            
        results = []
        
        # 滚动窗口
        for i in range(0, len(data) - self.rolling_window, self.rolling_window):
            # 训练集
            train_data = data.iloc[i:i+int(len(data) * self.train_ratio)]
            
            # 测试集
            test_data = data.iloc[i+int(len(data) * self.train_ratio):i+self.rolling_window]
            
            if len(test_data) == 0:
                break
                
            # 在训练集上优化参数（简化版本）
            # 实际应用中可以在这里进行参数优化
            
            # 在测试集上回测
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage,
                leverage=self.leverage
            )
            
            result = engine.run(test_data, signal_generator, symbols)
            results.append(result)
            
        return results