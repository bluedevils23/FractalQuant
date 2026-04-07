"""
组合优化模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from scipy.optimize import minimize, LinearConstraint

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(
        self,
        max_positions: int = 10,
        min_sharpe_ratio: float = 0.5,
        max_volatility: float = 0.3
    ):
        """
        初始化组合优化器
        
        Args:
            max_positions: 最大持仓数量
            min_sharpe_ratio: 最小夏普比率
            max_volatility: 最大波动率
        """
        self.max_positions = max_positions
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_volatility = max_volatility
        
    def optimize_weights(
        self,
        returns: pd.DataFrame,
        signals: pd.DataFrame
    ) -> Dict[str, float]:
        """
        优化权重
        
        Args:
            returns: 收益率数据
            signals: 信号数据
            
        Returns:
            权重字典
        """
        if returns.empty or signals.empty:
            return {}
            
        # 计算预期收益和协方差
        expected_returns = returns.mean() * 252 * 24 * 60
        cov_matrix = returns.cov() * 252 * 24 * 60
        
        # 获取有效信号
        valid_signals = signals.iloc[-1]
        valid_assets = valid_signals[valid_signals.abs() > 0].index.tolist()
        
        if len(valid_assets) == 0:
            return {}
            
        # 限制持仓数量
        valid_assets = valid_assets[:self.max_positions]
        
        # 优化权重
        try:
            weights = self._mean_variance_optimization(
                expected_returns[valid_assets],
                cov_matrix.loc[valid_assets, valid_assets],
                valid_signals[valid_assets]
            )
            
            return weights
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # 返回等权权重
            return {asset: 1.0 / len(valid_assets) for asset in valid_assets}
    
    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        signals: pd.Series
    ) -> Dict[str, float]:
        """
        均值方差优化
        
        Args:
            expected_returns: 预期收益
            cov_matrix: 协方差矩阵
            signals: 信号
            
        Returns:
            权重字典
        """
        n_assets = len(expected_returns)
        
        def negative_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility
        
        # 初始权重（基于信号强度）
        initial_weights = np.abs(signals.values)
        initial_weights = initial_weights / initial_weights.sum()
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: self.max_volatility - 
             np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))}
        ]
        
        # 边界条件
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # 优化
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = dict(zip(expected_returns.index, result.x))
            return weights
        else:
            # 返回等权权重
            return {asset: 1.0 / n_assets for asset in expected_returns.index}

class PositionSizer:
    """仓位管理器"""
    
    def __init__(
        self,
        max_position_size: float = 0.1,
        risk_per_trade: float = 0.02,
        leverage: int = 10
    ):
        """
        初始化仓位管理器
        
        Args:
            max_position_size: 最大仓位比例
            risk_per_trade: 每笔交易风险
            leverage: 杠杆倍数
        """
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        
    def calculate_position_size(
        self,
        capital: float,
        signal_strength: float,
        volatility: float,
        stop_loss: float = 0.05
    ) -> float:
        """
        计算仓位大小
        
        Args:
            capital: 资本
            signal_strength: 信号强度
            volatility: 波动率
            stop_loss: 止损幅度
            
        Returns:
            仓位大小（比例）
        """
        # 根据信号强度调整仓位
        base_position = signal_strength * self.max_position_size
        
        # 根据波动率调整仓位（波动率越高，仓位越小）
        volatility_adjustment = 1.0 / (1.0 + volatility * 100)
        
        # 根据风险调整仓位
        risk_position = (capital * self.risk_per_trade) / (stop_loss * capital)
        risk_position = min(risk_position, self.max_position_size)
        
        # 综合计算仓位
        position = base_position * volatility_adjustment * risk_position
        position = min(position, self.max_position_size)
        
        # 应用杠杆
        position_with_leverage = position * self.leverage
        
        return min(position_with_leverage, self.max_position_size * self.leverage)

class RiskAdjustedSignal:
    """风险调整信号"""
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        max_position: float = 0.2,
        stop_loss: float = 0.05,
        take_profit: float = 0.10
    ):
        """
        初始化风险调整信号
        
        Args:
            max_drawdown: 最大回撤
            max_position: 最大仓位
            stop_loss: 止损
            take_profit: 止盈
        """
        self.max_drawdown = max_drawdown
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
    def adjust_signal(
        self,
        signal: int,
        drawdown: float,
        position: float,
        current_price: float,
        entry_price: float
    ) -> int:
        """
        调整信号
        
        Args:
            signal: 原始信号
            drawdown: 当前回撤
            position: 当前仓位
            current_price: 当前价格
            entry_price: 入场价格
            
        Returns:
            调整后的信号
        """
        # 检查最大回撤
        if drawdown > self.max_drawdown:
            return 0  # 清仓
            
        # 检查止损
        if signal > 0 and current_price < entry_price * (1 - self.stop_loss):
            return -1  # 止损卖出
        elif signal < 0 and current_price > entry_price * (1 + self.stop_loss):
            return 1  # 止损买入
            
        # 检查止盈
        if signal > 0 and current_price > entry_price * (1 + self.take_profit):
            return -1  # 止盈卖出
        elif signal < 0 and current_price < entry_price * (1 - self.take_profit):
            return 1  # 止盈买入
            
        # 检查仓位限制
        if abs(position) > self.max_position:
            return 0
            
        return signal

class CorrelationBasedDiversification:
    """基于相关性的分散化"""
    
    def __init__(
        self,
        max_correlation: float = 0.7
    ):
        """
        初始化相关性分散化
        
        Args:
            max_correlation: 最大相关性
        """
        self.max_correlation = max_correlation
        
    def select_diversified_assets(
        self,
        returns: pd.DataFrame,
        signals: pd.Series,
        n_assets: int
    ) -> List[str]:
        """
        选择分散化的资产
        
        Args:
            returns: 收益率数据
            signals: 信号
            n_assets: 资产数量
            
        Returns:
            资产列表
        """
        # 获取有信号的资产
        valid_assets = signals[signals.abs() > 0].index.tolist()
        
        if len(valid_assets) <= n_assets:
            return valid_assets
            
        # 计算相关性矩阵
        corr_matrix = returns[valid_assets].corr()
        
        # 选择低相关性的资产组合
        selected_assets = []
        
        for asset in valid_assets:
            if len(selected_assets) >= n_assets:
                break
                
            if len(selected_assets) == 0:
                selected_assets.append(asset)
            else:
                # 检查与已选资产的相关性
                max_corr = max([corr_matrix.loc[asset, a] for a in selected_assets])
                if max_corr < self.max_correlation:
                    selected_assets.append(asset)
                    
        return selected_assets