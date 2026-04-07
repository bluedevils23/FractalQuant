"""
市场微观结构因子（订单流、流动性、订单簿分析等）
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats
from .base import BaseFactor

class OrderFlowImbalanceFactor(BaseFactor):
    """订单流失衡因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('order_flow_imbalance', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单流失衡（买方/卖方压力）"""
        close = df['close']
        volume = df['volume']
        
        def calc_imbalance(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            
            buy_pressure = np.sum(returns[returns > 0] * volume[1:][returns > 0])
            sell_pressure = np.sum(np.abs(returns[returns < 0] * volume[1:][returns < 0]))
            
            total_pressure = buy_pressure + sell_pressure
            
            if total_pressure > 0:
                imbalance = (buy_pressure - sell_pressure) / total_pressure
                return imbalance
            return 0
        
        imbalance = close.rolling(window=self.window).apply(calc_imbalance)
        return imbalance

class LiquidityRatioFactor(BaseFactor):
    """流动性比率因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('liquidity_ratio', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流动性比率（买卖价差和成交量）"""
        close = df['close']
        volume = df['volume']
        
        def calc_liquidity(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            
            volatility = np.std(returns)
            avg_volume = np.mean(volume[1:])
            
            if volatility > 0:
                liquidity = avg_volume / (volatility * 100)
                return liquidity
            return 0
        
        liquidity = close.rolling(window=self.window).apply(calc_liquidity)
        return liquidity

class VolumeWeightedPriceFactor(BaseFactor):
    """成交量加权价格因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('volume_weighted_price', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量加权平均价格"""
        close = df['close']
        volume = df['volume']
        
        def calc_vwap(x):
            if len(x) < 20:
                return 0
            
            prices = x
            vols = volume[-len(prices):]
            
            if len(vols) != len(prices):
                vols = vols[-len(prices):]
            
            vwap = np.sum(prices * vols) / (np.sum(vols) + 1e-8)
            return (vwap - prices[-1]) / (prices[-1] + 1e-8)
        
        vwap = close.rolling(window=self.window).apply(calc_vwap)
        return vwap

class OrderBookPressureFactor(BaseFactor):
    """订单簿压力因子"""
    
    def __init__(self, window: int = 50, levels: int = 5):
        super().__init__('orderbook_pressure', window)
        self.levels = levels
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单簿压力（基于买卖盘口）"""
        close = df['close']
        
        def calc_pressure(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            
            buy_count = np.sum(returns > 0)
            sell_count = np.sum(returns < 0)
            
            total = buy_count + sell_count
            
            if total > 0:
                pressure = (buy_count - sell_count) / total
                return pressure
            return 0
        
        pressure = close.rolling(window=self.window).apply(calc_pressure)
        return pressure

class TradeSizeDistributionFactor(BaseFactor):
    """交易规模分布因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('trade_size_distribution', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算交易规模分布特征"""
        close = df['close']
        volume = df['volume']
        
        def calc_distribution(x):
            if len(x) < 20:
                return 0
            
            prices = x
            vols = volume[-len(prices):]
            
            trade_sizes = np.abs(prices * vols)
            
            if len(trade_sizes) < 10:
                return 0
            
            try:
                skewness = stats.skew(trade_sizes)
                kurtosis = stats.kurtosis(trade_sizes)
                
                return abs(skewness) + kurtosis / 10
            except:
                return 0
        
        distribution = close.rolling(window=self.window).apply(calc_distribution)
        return distribution

class VolatilityAdjustedVolumeFactor(BaseFactor):
    """波动率调整成交量因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('volatility_adj_volume', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率调整后的成交量"""
        close = df['close']
        volume = df['volume']
        
        def calc_adj_volume(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            current_vol = np.std(returns)
            
            avg_vol = np.mean(np.std(returns.reshape(-1, 5), axis=1))
            
            current_volume = volume[-1]
            
            if avg_vol > 0:
                adj_volume = current_volume * (current_vol / avg_vol)
                return adj_volume / (np.mean(volume) + 1e-8)
            return 0
        
        adj_volume = close.rolling(window=self.window).apply(calc_adj_volume)
        return adj_volume

class PriceVelocityFactor(BaseFactor):
    """价格速度因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('price_velocity', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算价格变化速度"""
        close = df['close']
        
        def calc_velocity(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            
            velocity = np.mean(returns)
            volatility = np.std(returns)
            
            if volatility > 0:
                return velocity / volatility
            return 0
        
        velocity = close.rolling(window=self.window).apply(calc_velocity)
        return velocity

class MomentumAccelerationFactor(BaseFactor):
    """动量加速度因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('momentum_acceleration', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算动量加速度（价格变化的加速度）"""
        close = df['close']
        
        def calc_acceleration(x):
            if len(x) < 30:
                return 0
            
            returns = np.diff(x)
            
            momentum = np.diff(returns)
            
            if len(momentum) < 10:
                return 0
            
            acceleration = np.mean(momentum)
            momentum_vol = np.std(momentum)
            
            if momentum_vol > 0:
                return acceleration / momentum_vol
            return 0
        
        acceleration = close.rolling(window=self.window).apply(calc_acceleration)
        return acceleration

class VolumeSpikeFactor(BaseFactor):
    """成交量激增因子"""
    
    def __init__(self, window: int = 50, threshold: float = 2.0):
        super().__init__('volume_spike', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测成交量激增事件"""
        close = df['close']
        volume = df['volume']
        
        def calc_spike(x):
            if len(x) < 20:
                return 0
            
            current_volume = volume[-1]
            avg_volume = np.mean(volume[:-1])
            
            if avg_volume > 0:
                spike_ratio = current_volume / avg_volume
                return max(0, spike_ratio - self.threshold)
            return 0
        
        spike = close.rolling(window=self.window).apply(calc_spike)
        return spike

class LiquidityShockFactor(BaseFactor):
    """流动性冲击因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('liquidity_shock', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流动性冲击（价格变动与成交量的关系）"""
        close = df['close']
        volume = df['volume']
        
        def calc_shock(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            vols = volume[1:]
            
            price_impact = np.abs(returns) * vols
            
            if len(price_impact) < 10:
                return 0
            
            current_impact = price_impact[-1]
            avg_impact = np.mean(price_impact[:-1])
            
            if avg_impact > 0:
                return (current_impact - avg_impact) / (np.std(price_impact[:-1]) + 1e-8)
            return 0
        
        shock = close.rolling(window=self.window).apply(calc_shock)
        return shock

class OrderBookAsymmetryFactor(BaseFactor):
    """订单簿不对称性因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('orderbook_asymmetry', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单簿不对称性"""
        close = df['close']
        
        def calc_asymmetry(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            
            buy_returns = returns[returns > 0]
            sell_returns = returns[returns < 0]
            
            if len(buy_returns) == 0 or len(sell_returns) == 0:
                return 0
            
            buy_magnitude = np.mean(np.abs(buy_returns))
            sell_magnitude = np.mean(np.abs(sell_returns))
            
            if sell_magnitude > 0:
                asymmetry = buy_magnitude / sell_magnitude
                return asymmetry - 1
            return 0
        
        asymmetry = close.rolling(window=self.window).apply(calc_asymmetry)
        return asymmetry

class TradeDirectionPersistenceFactor(BaseFactor):
    """交易方向持续性因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('trade_direction_persistence', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算交易方向持续性（连续同向交易的概率）"""
        close = df['close']
        
        def calc_persistence(x):
            if len(x) < 30:
                return 0
            
            returns = np.diff(x)
            directions = np.sign(returns)
            
            consecutive_same = 0
            total_transitions = 0
            
            for i in range(1, len(directions)):
                if directions[i] == directions[i-1]:
                    consecutive_same += 1
                total_transitions += 1
            
            if total_transitions > 0:
                persistence = consecutive_same / total_transitions
                return persistence * 2 - 1
            return 0
        
        persistence = close.rolling(window=self.window).apply(calc_persistence)
        return persistence

class MarketImpactFactor(BaseFactor):
    """市场冲击因子"""
    
    def __init__(self, window: int = 50, alpha: float = 0.5):
        super().__init__('market_impact', window)
        self.alpha = alpha
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算市场冲击（订单流对价格的影响）"""
        close = df['close']
        volume = df['volume']
        
        def calc_impact(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            vols = volume[1:]
            
            order_flow = vols * np.sign(returns)
            
            if len(order_flow) < 10:
                return 0
            
            current_impact = np.sum(order_flow[-5:]) / (np.std(close[-10:]) * 100)
            
            return current_impact

        impact = close.rolling(window=self.window).apply(calc_impact)
        return impact

class LiquidityDepthFactor(BaseFactor):
    """流动性深度因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('liquidity_depth', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流动性深度（价格变动所需的成交量）"""
        close = df['close']
        volume = df['volume']
        
        def calc_depth(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            vols = volume[1:]
            
            price_changes = np.abs(returns)
            volumes = vols
            
            if len(price_changes) < 10:
                return 0
            
            try:
                slope, _ = stats.linregress(price_changes, volumes)[:2]
                return slope / (np.mean(volumes) + 1e-8)
            except:
                return 0
        
        depth = close.rolling(window=self.window).apply(calc_depth)
        return depth

class OrderFlowSignificanceFactor(BaseFactor):
    """订单流显著性因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('orderflow_significance', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单流的统计显著性"""
        close = df['close']
        volume = df['volume']
        
        def calc_significance(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            vols = volume[1:]
            
            buy_flow = np.sum(returns[returns > 0] * vols[returns > 0])
            sell_flow = np.sum(np.abs(returns[returns < 0] * vols[returns < 0]))
            
            total_flow = buy_flow + sell_flow
            
            if total_flow > 0:
                buy_ratio = buy_flow / total_flow
                expected_ratio = 0.5
                
                n = len(returns)
                std_expected = np.sqrt(n * 0.5 * 0.5)
                
                z_score = (buy_ratio - expected_ratio) / (std_expected / n + 1e-8)
                
                return z_score
            return 0
        
        significance = close.rolling(window=self.window).apply(calc_significance)
        return significance

class VolumeClusteringFactor(BaseFactor):
    """成交量聚类因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('volume_clustering', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量聚类（高成交量后跟高成交量的概率）"""
        close = df['close']
        volume = df['volume']
        
        def calc_clustering(x):
            if len(x) < 30:
                return 0
            
            vols = volume
            
            avg_vol = np.mean(vols)
            high_vol = vols > avg_vol
            
            high_to_high = 0
            high_count = 0
            
            for i in range(1, len(high_vol)):
                if high_vol[i-1]:
                    high_count += 1
                    if high_vol[i]:
                        high_to_high += 1
            
            if high_count > 0:
                clustering = high_to_high / high_count
                return clustering * 2 - 1
            return 0
        
        clustering = close.rolling(window=self.window).apply(calc_clustering)
        return clustering

class PriceVolumeDecouplingFactor(BaseFactor):
    """价格成交量脱钩因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('price_volume_decoupling', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测价格与成交量的脱钩现象"""
        close = df['close']
        volume = df['volume']
        
        def calc_decoupling(x):
            if len(x) < 20:
                return 0
            
            returns = np.diff(x)
            vol_returns = np.diff(np.log(volume + 1))
            
            correlation = np.corrcoef(returns, vol_returns)[0, 1]
            
            return np.abs(correlation)
        
        decoupling = close.rolling(window=self.window).apply(calc_decoupling)
        return decoupling

class MarketEfficiencyFactor(BaseFactor):
    """市场效率因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('market_efficiency', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算市场效率（可预测性）"""
        close = df['close']
        
        def calc_efficiency(x):
            if len(x) < 50:
                return 0
            
            returns = np.diff(x)
            
            abs_returns = np.abs(returns)
            
            try:
                slope, _ = stats.linregress(np.arange(len(abs_returns)), abs_returns)[:2]
                
                efficiency = 1 / (1 + abs(slope))
                return efficiency
            except:
                return 0.5
        
        efficiency = close.rolling(window=self.window).apply(calc_efficiency)
        return efficiency

class LiquidityMigrationFactor(BaseFactor):
    """流动性迁移因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('liquidity_migration', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流动性迁移（流动性随时间的变化）"""
        close = df['close']
        volume = df['volume']
        
        def calc_migration(x):
            if len(x) < 30:
                return 0
            
            vols = volume
            
            if len(vols) < 10:
                return 0
            
            first_half = np.mean(vols[:len(vols)//2])
            second_half = np.mean(vols[len(vols)//2:])
            
            if first_half > 0:
                migration = (second_half - first_half) / first_half
                return migration
            return 0
        
        migration = close.rolling(window=self.window).apply(calc_migration)
        return migration
