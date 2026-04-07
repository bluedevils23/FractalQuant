"""
风险管理模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """风险管理器"""
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        max_position: float = 0.2,
        max_positions: int = 10,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        max_daily_loss: float = 0.10,
        max_volatility: float = 0.3
    ):
        """
        初始化风险管理器
        
        Args:
            max_drawdown: 最大回撤
            max_position: 最大仓位
            max_positions: 最大持仓数量
            stop_loss: 止损
            take_profit: 止盈
            max_daily_loss: 最大日损失
            max_volatility: 最大波动率
        """
        self.max_drawdown = max_drawdown
        self.max_position = max_position
        self.max_positions = max_positions
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_daily_loss = max_daily_loss
        self.max_volatility = max_volatility
        
        self.positions: Dict[str, Dict] = {}
        self.daily_pnl: List[float] = []
        self.trades: List[Dict] = []
        
    def check_risk(
        self,
        symbol: str,
        signal: int,
        quantity: float,
        capital: float,
        current_price: float,
        entry_price: Optional[float] = None
    ) -> bool:
        """
        检查风险
        
        Args:
            symbol: 交易对
            signal: 信号
            quantity: 数量
            capital: 资本
            current_price: 当前价格
            entry_price: 入场价格
            
        Returns:
            是否允许交易
        """
        # 检查最大回撤
        if self._calculate_drawdown() > self.max_drawdown:
            logger.warning("Maximum drawdown exceeded")
            return False
            
        # 检查最大仓位
        if abs(quantity) > self.max_position * capital:
            logger.warning("Maximum position size exceeded")
            return False
            
        # 检查最大持仓数量
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            logger.warning("Maximum number of positions exceeded")
            return False
            
        # 检查止损
        if entry_price and signal > 0 and current_price < entry_price * (1 - self.stop_loss):
            logger.warning("Stop loss triggered")
            return False
            
        if entry_price and signal < 0 and current_price > entry_price * (1 + self.stop_loss):
            logger.warning("Stop loss triggered")
            return False
            
        # 检查日损失
        if self._calculate_daily_loss() > self.max_daily_loss:
            logger.warning("Maximum daily loss exceeded")
            return False
            
        return True
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ):
        """
        更新仓位
        
        Args:
            symbol: 交易对
            quantity: 数量
            price: 价格
            side: 买卖方向
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'entry_price': 0,
                'total_cost': 0
            }
            
        position = self.positions[symbol]
        
        if side == 'buy':
            # 买入
            total_cost = position['total_cost'] + quantity * price
            total_quantity = position['quantity'] + quantity
            position['entry_price'] = total_cost / total_quantity if total_quantity > 0 else 0
            position['total_cost'] = total_cost
            position['quantity'] = total_quantity
        else:
            # 卖出
            position['quantity'] -= quantity
            if position['quantity'] <= 0:
                del self.positions[symbol]
                
    def update_daily_pnl(self, pnl: float):
        """
        更新日收益
        
        Args:
            pnl: 收益
        """
        self.daily_pnl.append(pnl)
        
    def add_trade(self, trade: Dict):
        """
        添加交易
        
        Args:
            trade: 交易信息
        """
        self.trades.append(trade)
        
    def _calculate_drawdown(self) -> float:
        """计算当前回撤"""
        if not self.daily_pnl:
            return 0
            
        cumulative = np.cumsum(self.daily_pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / (running_max + 1e-8)
        
        return drawdown[-1] if len(drawdown) > 0 else 0
    
    def _calculate_daily_loss(self) -> float:
        """计算当日损失"""
        if not self.daily_pnl:
            return 0
            
        today = datetime.now().date()
        today_pnl = [pnl for pnl, timestamp in zip(self.daily_pnl, self.trades) 
                    if timestamp.get('timestamp', datetime.now()).date() == today]
        
        return abs(sum(pnl for pnl in today_pnl if pnl < 0))
    
    def get_risk_metrics(self) -> Dict:
        """获取风险指标"""
        if not self.daily_pnl:
            return {
                'drawdown': 0,
                'daily_loss': 0,
                'num_positions': len(self.positions),
                'positions': self.positions
            }
            
        return {
            'drawdown': self._calculate_drawdown(),
            'daily_loss': self._calculate_daily_loss(),
            'num_positions': len(self.positions),
            'positions': self.positions,
            'total_trades': len(self.trades)
        }

class PositionRiskManager:
    """仓位风险管理员"""
    
    def __init__(
        self,
        risk_per_trade: float = 0.02,
        max_risk_ratio: float = 3.0,
        volatility_adjustment: bool = True
    ):
        """
        初始化仓位风险管理员
        
        Args:
            risk_per_trade: 每笔交易风险
            max_risk_ratio: 最大风险比率
            volatility_adjustment: 是否根据波动率调整
        """
        self.risk_per_trade = risk_per_trade
        self.max_risk_ratio = max_risk_ratio
        self.volatility_adjustment = volatility_adjustment
        
    def calculate_position_size(
        self,
        capital: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        计算仓位大小
        
        Args:
            capital: 资本
            stop_loss: 止损幅度
            volatility: 波动率
            
        Returns:
            仓位大小
        """
        # 基础仓位
        position_size = (capital * self.risk_per_trade) / stop_loss
        
        # 根据波动率调整
        if self.volatility_adjustment and volatility:
            position_size = position_size / (1 + volatility * 100)
            
        # 限制最大风险比率
        max_position = capital * self.max_risk_ratio
        position_size = min(position_size, max_position)
        
        return position_size

class StopLossManager:
    """止损管理员"""
    
    def __init__(
        self,
        fixed_stop_loss: float = 0.05,
        atr_multiplier: float = 2.0,
        trailing_stop: float = 0.03
    ):
        """
        初始化止损管理员
        
        Args:
            fixed_stop_loss: 固定止损
            atr_multiplier: ATR倍数
            trailing_stop: 追踪止损
        """
        self.fixed_stop_loss = fixed_stop_loss
        self.atr_multiplier = atr_multiplier
        self.trailing_stop = trailing_stop
        
        self.entry_prices: Dict[str, float] = {}
        self.high_prices: Dict[str, float] = {}
        
    def set_entry_price(self, symbol: str, price: float):
        """设置入场价格"""
        self.entry_prices[symbol] = price
        self.high_prices[symbol] = price
        
    def update_high_price(self, symbol: str, price: float):
        """更新最高价格（用于追踪止损）"""
        if symbol in self.high_prices:
            self.high_prices[symbol] = max(self.high_prices[symbol], price)
            
    def check_stop_loss(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float] = None
    ) -> bool:
        """
        检查是否触发止损
        
        Args:
            symbol: 交易对
            current_price: 当前价格
            atr: ATR值
            
        Returns:
            是否触发止损
        """
        if symbol not in self.entry_prices:
            return False
            
        entry_price = self.entry_prices[symbol]
        
        # 固定止损
        if current_price < entry_price * (1 - self.fixed_stop_loss):
            return True
            
        # ATR止损
        if atr and current_price < entry_price - self.atr_multiplier * atr:
            return True
            
        # 追踪止损
        if symbol in self.high_prices:
            high_price = self.high_prices[symbol]
            if current_price < high_price * (1 - self.trailing_stop):
                return True
                
        return False
    
    def get_stop_price(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float] = None
    ) -> Optional[float]:
        """
        获取止损价格
        
        Args:
            symbol: 交易对
            current_price: 当前价格
            atr: ATR值
            
        Returns:
            止损价格
        """
        if symbol not in self.entry_prices:
            return None
            
        entry_price = self.entry_prices[symbol]
        
        # 使用追踪止损
        if symbol in self.high_prices:
            high_price = self.high_prices[symbol]
            return high_price * (1 - self.trailing_stop)
            
        # 使用固定止损
        return entry_price * (1 - self.fixed_stop_loss)

class VolatilityRiskManager:
    """波动率风险管理员"""
    
    def __init__(
        self,
        max_volatility: float = 0.3,
        min_volatility: float = 0.01,
        position_reduction: float = 0.5
    ):
        """
        初始化波动率风险管理员
        
        Args:
            max_volatility: 最大波动率
            min_volatility: 最小波动率
            position_reduction: 仓位减少比例
        """
        self.max_volatility = max_volatility
        self.min_volatility = min_volatility
        self.position_reduction = position_reduction
        
    def adjust_position_for_volatility(
        self,
        current_position: float,
        volatility: float
    ) -> float:
        """
        根据波动率调整仓位
        
        Args:
            current_position: 当前仓位
            volatility: 波动率
            
        Returns:
            调整后的仓位
        """
        if volatility > self.max_volatility:
            return current_position * self.position_reduction
        elif volatility < self.min_volatility:
            return current_position * 1.2
        return current_position
    
    def should_trade(self, volatility: float) -> bool:
        """
        判断是否应该交易
        
        Args:
            volatility: 波动率
            
        Returns:
            是否应该交易
        """
        return self.min_volatility <= volatility <= self.max_volatility

class CorrelationRiskManager:
    """相关性风险管理员"""
    
    def __init__(
        self,
        max_correlation: float = 0.7,
        max_concentration: float = 0.3
    ):
        """
        初始化相关性风险管理员
        
        Args:
            max_correlation: 最大相关性
            max_concentration: 最大集中度
        """
        self.max_correlation = max_correlation
        self.max_concentration = max_concentration
        
    def check_diversification(
        self,
        correlations: Dict[str, Dict[str, float]],
        positions: Dict[str, float]
    ) -> bool:
        """
        检查分散化
        
        Args:
            correlations: 相关性矩阵
            positions: 仓位
            
        Returns:
            是否满足分散化要求
        """
        total_position = sum(abs(p) for p in positions.values())
        
        for symbol1, pos1 in positions.items():
            if abs(pos1) / total_position > self.max_concentration:
                return False
                
            for symbol2, pos2 in positions.items():
                if symbol1 != symbol2:
                    corr = correlations.get(symbol1, {}).get(symbol2, 0)
                    if abs(corr) > self.max_correlation:
                        return False
                        
        return True