"""
市场微观结构因子（订单流、流动性、订单簿分析等）
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats
from .base import BaseFactor


def _aligned_window_values(
    window: pd.Series, series: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    values = window.to_numpy(dtype=float, copy=False)
    aligned = series.loc[window.index].to_numpy(dtype=float, copy=False)
    return values, aligned


def _full_window_mask(series: pd.Series, window: int) -> pd.Series:
    return series.notna().rolling(window, min_periods=window).sum().eq(window)


def _rolling_std0(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)

class OrderFlowImbalanceFactor(BaseFactor):
    """订单流失衡因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('order_flow_imbalance', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单流失衡（买方/卖方压力）"""
        close = df['close']
        volume = df['volume']

        window_returns = self.window - 1
        valid = _full_window_mask(close, self.window)
        returns = close.diff()
        buy_pressure = (returns.clip(lower=0) * volume).rolling(
            window_returns, min_periods=window_returns
        ).sum()
        sell_pressure = ((-returns.clip(upper=0)) * volume).rolling(
            window_returns, min_periods=window_returns
        ).sum()
        total_pressure = buy_pressure + sell_pressure

        imbalance = (buy_pressure - sell_pressure) / (total_pressure + 1e-8)
        imbalance = imbalance.where(total_pressure > 0, 0.0)
        return imbalance.where(valid)

class LiquidityRatioFactor(BaseFactor):
    """流动性比率因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('liquidity_ratio', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流动性比率（买卖价差和成交量）"""
        close = df['close']
        volume = df['volume']

        window_returns = self.window - 1
        valid = _full_window_mask(close, self.window)
        returns = close.diff()
        volatility = _rolling_std0(returns, window_returns)
        avg_volume = volume.rolling(window_returns, min_periods=window_returns).mean()
        liquidity = avg_volume / (volatility * 100 + 1e-8)
        liquidity = liquidity.where(volatility > 0, 0.0)
        return liquidity.where(valid)

class VolumeWeightedPriceFactor(BaseFactor):
    """成交量加权价格因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('volume_weighted_price', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量加权平均价格"""
        close = df['close']
        volume = df['volume']

        valid = _full_window_mask(close, self.window)
        rolling_pv = (close * volume).rolling(self.window, min_periods=self.window).sum()
        rolling_volume = volume.rolling(self.window, min_periods=self.window).sum()
        vwap = rolling_pv / (rolling_volume + 1e-8)
        result = (vwap - close) / (close + 1e-8)
        return result.where(valid)

class OrderBookPressureFactor(BaseFactor):
    """订单簿压力因子"""
    
    def __init__(self, window: int = 50, levels: int = 5):
        super().__init__('orderbook_pressure', window)
        self.levels = levels
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单簿压力（基于买卖盘口）"""
        close = df['close']

        window_returns = self.window - 1
        valid = _full_window_mask(close, self.window)
        returns = close.diff()
        buy_count = returns.gt(0).rolling(
            window_returns, min_periods=window_returns
        ).sum()
        sell_count = returns.lt(0).rolling(
            window_returns, min_periods=window_returns
        ).sum()
        total = buy_count + sell_count

        pressure = (buy_count - sell_count) / (total + 1e-8)
        pressure = pressure.where(total > 0, 0.0)
        return pressure.where(valid)

class TradeSizeDistributionFactor(BaseFactor):
    """交易规模分布因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('trade_size_distribution', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算交易规模分布特征"""
        close = df['close']
        volume = df['volume']

        valid = _full_window_mask(close, self.window)
        trade_sizes = (close * volume).abs()
        skewness = trade_sizes.rolling(self.window, min_periods=self.window).skew()
        kurtosis = trade_sizes.rolling(self.window, min_periods=self.window).kurt()
        distribution = skewness.abs() + kurtosis / 10
        return distribution.where(valid)

class VolatilityAdjustedVolumeFactor(BaseFactor):
    """波动率调整成交量因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('volatility_adj_volume', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率调整后的成交量"""
        close = df['close']
        volume = df['volume']
        
        valid = _full_window_mask(close, self.window)
        close_values = close.to_numpy(dtype=float, copy=False)
        volume_values = volume.to_numpy(dtype=float, copy=False)
        result = np.full(len(close_values), np.nan, dtype=float)
        chunk_size = 5

        for end in range(self.window - 1, len(close_values)):
            start = end - self.window + 1
            prices = close_values[start : end + 1]
            vols = volume_values[start : end + 1]
            if np.isnan(prices).any() or np.isnan(vols).any():
                continue

            returns = np.diff(prices)
            current_vol = np.std(returns)
            chunk_count = len(returns) // chunk_size
            if chunk_count == 0:
                result[end] = 0.0
                continue

            trimmed_returns = returns[-chunk_count * chunk_size :]
            avg_vol = np.mean(
                np.std(trimmed_returns.reshape(chunk_count, chunk_size), axis=1)
            )
            if avg_vol > 0:
                adj_volume = vols[-1] * (current_vol / avg_vol)
                result[end] = adj_volume / (np.mean(vols) + 1e-8)
            else:
                result[end] = 0.0

        return pd.Series(result, index=close.index).where(valid)

class PriceVelocityFactor(BaseFactor):
    """价格速度因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('price_velocity', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算价格变化速度"""
        close = df['close']
        
        window_returns = self.window - 1
        valid = _full_window_mask(close, self.window)
        returns = close.diff()
        velocity = returns.rolling(window_returns, min_periods=window_returns).mean()
        volatility = _rolling_std0(returns, window_returns)
        result = velocity / (volatility + 1e-8)
        result = result.where(volatility > 0, 0.0)
        return result.where(valid)

class MomentumAccelerationFactor(BaseFactor):
    """动量加速度因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('momentum_acceleration', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算动量加速度（价格变化的加速度）"""
        close = df['close']
        
        valid = _full_window_mask(close, self.window)
        momentum = close.diff().diff()
        acceleration = momentum.rolling(
            self.window - 2, min_periods=self.window - 2
        ).mean()
        momentum_vol = _rolling_std0(momentum, self.window - 2)
        result = acceleration / (momentum_vol + 1e-8)
        result = result.where(momentum_vol > 0, 0.0)
        return result.where(valid)

class VolumeSpikeFactor(BaseFactor):
    """成交量激增因子"""
    
    def __init__(self, window: int = 50, threshold: float = 2.0):
        super().__init__('volume_spike', window)
        self.threshold = threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测成交量激增事件"""
        close = df['close']
        volume = df['volume']
        
        valid = _full_window_mask(close, self.window)
        avg_volume = volume.shift(1).rolling(
            self.window - 1, min_periods=self.window - 1
        ).mean()
        spike_ratio = volume / (avg_volume + 1e-8)
        spike = (spike_ratio - self.threshold).clip(lower=0)
        spike = spike.where(avg_volume > 0, 0.0)
        return spike.where(valid)

class LiquidityShockFactor(BaseFactor):
    """流动性冲击因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('liquidity_shock', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流动性冲击（价格变动与成交量的关系）"""
        close = df['close']
        volume = df['volume']
        
        valid = _full_window_mask(close, self.window)
        price_impact = close.diff().abs() * volume
        prev_mean = price_impact.shift(1).rolling(
            self.window - 2, min_periods=self.window - 2
        ).mean()
        prev_std = _rolling_std0(price_impact.shift(1), self.window - 2)
        shock = (price_impact - prev_mean) / (prev_std + 1e-8)
        shock = shock.where(prev_mean > 0, 0.0)
        return shock.where(valid)

class OrderBookAsymmetryFactor(BaseFactor):
    """订单簿不对称性因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('orderbook_asymmetry', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单簿不对称性"""
        close = df['close']
        
        window_returns = self.window - 1
        valid = _full_window_mask(close, self.window)
        returns = close.diff()
        buy_sum = returns.clip(lower=0).rolling(
            window_returns, min_periods=window_returns
        ).sum()
        buy_count = returns.gt(0).rolling(
            window_returns, min_periods=window_returns
        ).sum()
        sell_mag = (-returns.clip(upper=0))
        sell_sum = sell_mag.rolling(window_returns, min_periods=window_returns).sum()
        sell_count = returns.lt(0).rolling(
            window_returns, min_periods=window_returns
        ).sum()

        buy_mean = buy_sum / (buy_count + 1e-8)
        sell_mean = sell_sum / (sell_count + 1e-8)
        asymmetry = buy_mean / (sell_mean + 1e-8) - 1
        asymmetry = asymmetry.where((buy_count > 0) & (sell_count > 0), 0.0)
        return asymmetry.where(valid)

class TradeDirectionPersistenceFactor(BaseFactor):
    """交易方向持续性因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('trade_direction_persistence', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算交易方向持续性（连续同向交易的概率）"""
        close = df['close']
        
        valid = _full_window_mask(close, self.window)
        directions = np.sign(close.diff())
        same_direction = directions.eq(directions.shift(1)).astype(float)
        persistence = same_direction.rolling(
            self.window - 2, min_periods=self.window - 2
        ).mean()
        persistence = persistence * 2 - 1
        return persistence.where(valid)

class MarketImpactFactor(BaseFactor):
    """市场冲击因子"""
    
    def __init__(self, window: int = 50, alpha: float = 0.5):
        super().__init__('market_impact', window)
        self.alpha = alpha
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算市场冲击（订单流对价格的影响）"""
        close = df['close']
        volume = df['volume']
        
        valid = _full_window_mask(close, self.window)
        order_flow = volume * np.sign(close.diff())
        flow_sum = order_flow.rolling(5, min_periods=5).sum()
        price_std = _rolling_std0(close, 10)
        impact = flow_sum / (price_std * 100 + 1e-8)
        impact = impact.where(price_std > 0, 0.0)
        return impact.where(valid)

class LiquidityDepthFactor(BaseFactor):
    """流动性深度因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('liquidity_depth', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流动性深度（价格变动所需的成交量）"""
        close = df['close']
        volume = df['volume']
        
        window_returns = self.window - 1
        valid = _full_window_mask(close, self.window)
        price_changes = close.diff().abs()
        trade_vols = volume

        mean_x = price_changes.rolling(window_returns, min_periods=window_returns).mean()
        mean_y = trade_vols.rolling(window_returns, min_periods=window_returns).mean()
        mean_xy = (price_changes * trade_vols).rolling(
            window_returns, min_periods=window_returns
        ).mean()
        mean_xx = (price_changes * price_changes).rolling(
            window_returns, min_periods=window_returns
        ).mean()
        cov_xy = mean_xy - mean_x * mean_y
        var_x = mean_xx - mean_x * mean_x
        slope = cov_xy / (var_x + 1e-8)
        depth = slope / (mean_y + 1e-8)
        depth = depth.where(var_x > 0, 0.0)
        return depth.where(valid)

class OrderFlowSignificanceFactor(BaseFactor):
    """订单流显著性因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('orderflow_significance', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单流的统计显著性"""
        close = df['close']
        volume = df['volume']
        
        window_returns = self.window - 1
        valid = _full_window_mask(close, self.window)
        returns = close.diff()
        buy_flow = (returns.clip(lower=0) * volume).rolling(
            window_returns, min_periods=window_returns
        ).sum()
        sell_flow = ((-returns.clip(upper=0)) * volume).rolling(
            window_returns, min_periods=window_returns
        ).sum()
        total_flow = buy_flow + sell_flow
        buy_ratio = buy_flow / (total_flow + 1e-8)
        expected_ratio = 0.5
        n = float(window_returns)
        std_expected = np.sqrt(n * 0.5 * 0.5)
        z_score = (buy_ratio - expected_ratio) / (std_expected / n + 1e-8)
        z_score = z_score.where(total_flow > 0, 0.0)
        return z_score.where(valid)

class VolumeClusteringFactor(BaseFactor):
    """成交量聚类因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('volume_clustering', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量聚类（高成交量后跟高成交量的概率）"""
        close = df['close']
        volume = df['volume']
        
        valid = _full_window_mask(close, self.window)
        volume_values = volume.to_numpy(dtype=float, copy=False)
        result = np.full(len(volume_values), np.nan, dtype=float)

        for end in range(self.window - 1, len(volume_values)):
            start = end - self.window + 1
            vols = volume_values[start : end + 1]
            if np.isnan(vols).any() or len(vols) < 30:
                continue

            avg_vol = np.mean(vols)
            high_vol = vols > avg_vol
            high_to_high = 0
            high_count = 0
            for i in range(1, len(high_vol)):
                if high_vol[i - 1]:
                    high_count += 1
                    if high_vol[i]:
                        high_to_high += 1

            result[end] = (high_to_high / high_count) * 2 - 1 if high_count > 0 else 0.0

        return pd.Series(result, index=close.index).where(valid)

class PriceVolumeDecouplingFactor(BaseFactor):
    """价格成交量脱钩因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('price_volume_decoupling', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测价格与成交量的脱钩现象"""
        close = df['close']
        volume = df['volume']
        
        valid = _full_window_mask(close, self.window)
        returns = close.diff()
        vol_returns = np.log(volume + 1).diff()
        corr = returns.rolling(self.window - 1, min_periods=self.window - 1).corr(
            vol_returns
        )
        decoupling = corr.abs().where(corr.notna(), 0.0)
        return decoupling.where(valid)

class MarketEfficiencyFactor(BaseFactor):
    """市场效率因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('market_efficiency', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算市场效率（可预测性）"""
        close = df['close']
        
        valid = _full_window_mask(close, self.window)
        abs_returns = close.diff().abs()
        m = self.window - 1
        x = np.arange(m, dtype=float)
        mean_x = x.mean()
        denom = np.sum((x - mean_x) ** 2)
        weights = x[::-1]
        abs_values = abs_returns.iloc[1:].to_numpy(dtype=float, copy=False)
        sum_xy = np.convolve(abs_values, weights, mode="valid")
        mean_y = abs_returns.iloc[1:].rolling(m, min_periods=m).mean().to_numpy(
            dtype=float, copy=False
        )

        result = np.full(len(close), np.nan, dtype=float)
        slopes = (sum_xy - m * mean_x * mean_y[m - 1 :]) / (denom + 1e-8)
        result[self.window - 1 :] = 1 / (1 + np.abs(slopes))
        series = pd.Series(result, index=close.index)
        return series.where(valid)

class LiquidityMigrationFactor(BaseFactor):
    """流动性迁移因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('liquidity_migration', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算流动性迁移（流动性随时间的变化）"""
        close = df['close']
        volume = df['volume']
        
        valid = _full_window_mask(close, self.window)
        half_window = self.window // 2
        first_half = volume.shift(half_window).rolling(
            half_window, min_periods=half_window
        ).mean()
        second_half = volume.rolling(half_window, min_periods=half_window).mean()
        migration = (second_half - first_half) / (first_half + 1e-8)
        migration = migration.where(first_half > 0, 0.0)
        return migration.where(valid)
