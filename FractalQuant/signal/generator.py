"""
信号生成模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

from factor.combiner import MultiFactorSignal, FactorCombiner
from factor.price import ReturnsFactor, PriceMomentumFactor
from factor.volatility import HistoricalVolatilityFactor
from factor.trend import MACDFactor, RSIFactor
from factor.orderbook import OrderBookImbalanceFactor

logger = logging.getLogger(__name__)

class SignalGenerator:
    """信号生成器"""
    
    def __init__(
        self, 
        threshold: float = 0.0,
        min_confidence: float = 0.6,
        cooldown_period: int = 5
    ):
        """
        初始化信号生成器
        
        Args:
            threshold: 信号阈值
            min_confidence: 最小置信度
            cooldown_period: 冷却期（分钟）
        """
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.cooldown_period = cooldown_period
        self.last_signal_time = None
        self.last_position = None
        
        # 初始化多因子信号生成器
        self.multi_factor = MultiFactorSignal()
        
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        orderbook: Dict = None,
        current_position: float = 0
    ) -> Dict:
        """
        生成交易信号
        
        Args:
            df: K线数据
            orderbook: 订单簿数据
            current_position: 当前仓位
            
        Returns:
            信号字典
        """
        # 生成多因子信号
        signal_data = self.multi_factor.generate_signal(df, orderbook)
        
        # 检查冷却期
        if self._in_cooldown():
            signal_data['signal'] = 0
            signal_data['reason'] = 'cooldown'
            return signal_data
        
        # 调整信号强度
        signal_strength = signal_data['strength']
        
        # 根据当前仓位调整信号
        if current_position > 0 and signal_data['signal'] == -1:
            signal_strength = max(signal_strength, 0.8)
        elif current_position < 0 and signal_data['signal'] == 1:
            signal_strength = max(signal_strength, 0.8)
        
        # 生成最终信号
        if signal_strength >= self.min_confidence:
            final_signal = signal_data['signal']
        else:
            final_signal = 0
            
        # 更新最后信号时间
        if final_signal != 0:
            self.last_signal_time = datetime.now()
            self.last_position = current_position
            
        signal_data['final_signal'] = final_signal
        signal_data['confidence'] = signal_strength
        signal_data['reason'] = self._generate_reason(signal_data)
        
        return signal_data
    
    def _in_cooldown(self) -> bool:
        """检查是否在冷却期内"""
        if self.last_signal_time is None:
            return False
        
        from datetime import timedelta
        cooldown_delta = timedelta(minutes=self.cooldown_period)
        return datetime.now() - self.last_signal_time < cooldown_delta
    
    def _generate_reason(self, signal_data: Dict) -> str:
        """生成信号原因"""
        factors = signal_data.get('factors', {})
        signal = signal_data.get('signal', 0)
        
        reasons = []
        
        # 检查各因子贡献
        for factor_name, value in factors.items():
            if abs(value) > 1:
                if value > 0:
                    reasons.append(f"{factor_name}多")
                else:
                    reasons.append(f"{factor_name}空")
        
        if signal == 1:
            return f"多头信号: {' '.join(reasons[:3])}" if reasons else "多头信号"
        elif signal == -1:
            return f"空头信号: {' '.join(reasons[:3])}" if reasons else "空头信号"
        else:
            return "观望"
    
    def adjust_signal_for_risk(
        self, 
        signal: int, 
        volatility: float,
        max_volatility: float = 0.05
    ) -> int:
        """
        根据风险调整信号
        
        Args:
            signal: 原始信号
            volatility: 波动率
            max_volatility: 最大允许波动率
            
        Returns:
            调整后的信号
        """
        if volatility > max_volatility:
            return 0  # 高波动时观望
        return signal

class SignalOptimizer:
    """信号优化器"""
    
    def __init__(
        self, 
        lookback_window: int = 30,
        min_signals: int = 5
    ):
        """
        初始化信号优化器
        
        Args:
            lookback_window: 回看窗口
            min_signals: 最小信号数量
        """
        self.lookback_window = lookback_window
        self.min_signals = min_signals
        self.signal_history = []
        
    def optimize_signal(
        self, 
        signal: int, 
        historical_signals: List[int]
    ) -> int:
        """
        优化信号
        
        Args:
            signal: 原始信号
            historical_signals: 历史信号
            
        Returns:
            优化后的信号
        """
        if len(historical_signals) < self.min_signals:
            return signal
        
        # 计算信号一致性
        recent_signals = historical_signals[-self.lookback_window:]
        positive_signals = sum(1 for s in recent_signals if s > 0)
        negative_signals = sum(1 for s in recent_signals if s < 0)
        total_signals = len(recent_signals)
        
        # 如果历史信号与当前信号一致，增强信号
        if (signal > 0 and positive_signals / total_signals > 0.7) or \
           (signal < 0 and negative_signals / total_signals > 0.7):
            return signal * 2  # 增强信号
            
        # 如果历史信号与当前信号不一致，减弱信号
        if (signal > 0 and negative_signals / total_signals > 0.6) or \
           (signal < 0 and positive_signals / total_signals > 0.6):
            return signal // 2  # 减弱信号
            
        return signal
    
    def add_signal(self, signal: int):
        """添加信号到历史"""
        self.signal_history.append(signal)
        if len(self.signal_history) > self.lookback_window * 2:
            self.signal_history = self.signal_history[-self.lookback_window * 2:]

class SignalFilter:
    """信号过滤器"""
    
    def __init__(
        self,
        min_volume: float = 100,
        max_spread: float = 0.002,
        min_trend_strength: float = 0.5
    ):
        """
        初始化信号过滤器
        
        Args:
            min_volume: 最小成交量
            max_spread: 最大价差
            min_trend_strength: 最小趋势强度
        """
        self.min_volume = min_volume
        self.max_spread = max_spread
        self.min_trend_strength = min_trend_strength
        
    def filter_signal(
        self, 
        signal: int, 
        df: pd.DataFrame,
        orderbook: Dict = None
    ) -> bool:
        """
        过滤信号
        
        Args:
            signal: 信号
            df: K线数据
            orderbook: 订单簿数据
            
        Returns:
            是否保留信号
        """
        if signal == 0:
            return True
            
        # 检查成交量
        if df['volume'].iloc[-1] < self.min_volume:
            return False
            
        # 检查价差
        if orderbook:
            spread = self._calculate_spread(orderbook)
            if spread > self.max_spread:
                return False
                
        # 检查趋势强度
        trend_strength = self._calculate_trend_strength(df)
        if abs(trend_strength) < self.min_trend_strength:
            return False
            
        return True
    
    def _calculate_spread(self, orderbook: Dict) -> float:
        """计算价差"""
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return 0
            
        bid_price = orderbook['bids'][0][0]
        ask_price = orderbook['asks'][0][0]
        
        return (ask_price - bid_price) / (bid_price + ask_price) * 2
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度"""
        if len(df) < 20:
            return 0
            
        returns = df['close'].pct_change()
        trend = returns.rolling(window=10).mean()
        volatility = returns.rolling(window=20).std()
        
        if volatility.iloc[-1] == 0:
            return 0
            
        return trend.iloc[-1] / volatility.iloc[-1]