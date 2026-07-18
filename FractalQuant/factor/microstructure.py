"""
市场微观结构因子（订单流、流动性、订单簿分析等）
"""
import pandas as pd
import numpy as np
import weakref
from typing import List, Dict, Optional
from scipy import stats
from .base import BaseFactor


_MICROSTRUCTURE_CACHE: dict[int, tuple[weakref.ReferenceType[pd.DataFrame], dict]] = {}


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


def _rolling_sum_array(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return result

    clean = np.nan_to_num(values, nan=0.0)
    cumsum = np.empty(len(values) + 1, dtype=float)
    cumsum[0] = 0.0
    cumsum[1:] = np.cumsum(clean, dtype=float)
    result[window - 1 :] = cumsum[window:] - cumsum[:-window]
    return result


def _microstructure_cache(df: pd.DataFrame) -> dict:
    key = id(df)
    entry = _MICROSTRUCTURE_CACHE.get(key)
    if entry is None or entry[0]() is not df:
        ref = weakref.ref(
            df,
            lambda _ref, cache_key=key: _MICROSTRUCTURE_CACHE.pop(cache_key, None),
        )
        entry = (ref, {})
        _MICROSTRUCTURE_CACHE[key] = entry
    return entry[1]


def _valid_close_window(df: pd.DataFrame, window: int) -> pd.Series:
    cache = _microstructure_cache(df)
    key = ("valid_close_window", window)
    if key not in cache:
        cache[key] = _full_window_mask(df["close"], window)
    return cache[key]


def _close_diff(df: pd.DataFrame) -> pd.Series:
    cache = _microstructure_cache(df)
    key = "close_diff"
    if key not in cache:
        cache[key] = df["close"].diff()
    return cache[key]


def _rolling_direction_counts(
    df: pd.DataFrame, window_returns: int
) -> tuple[pd.Series, pd.Series]:
    cache = _microstructure_cache(df)
    key = ("direction_counts", window_returns)
    if key not in cache:
        returns = _close_diff(df).to_numpy(dtype=float, copy=False)
        buy_values = np.where(np.isfinite(returns) & (returns > 0), 1.0, 0.0)
        sell_values = np.where(np.isfinite(returns) & (returns < 0), 1.0, 0.0)
        buy_count = pd.Series(
            _rolling_sum_array(buy_values, window_returns), index=df.index
        )
        sell_count = pd.Series(
            _rolling_sum_array(sell_values, window_returns), index=df.index
        )
        cache[key] = (buy_count, sell_count)
    return cache[key]


def _rolling_flow_sums(
    df: pd.DataFrame, window_returns: int
) -> tuple[pd.Series, pd.Series]:
    cache = _microstructure_cache(df)
    key = ("flow_sums", window_returns)
    if key not in cache:
        returns = _close_diff(df).to_numpy(dtype=float, copy=False)
        volume = df["volume"].to_numpy(dtype=float, copy=False)
        valid = np.isfinite(returns) & np.isfinite(volume)
        buy_values = np.where(valid, np.clip(returns, 0.0, None) * volume, 0.0)
        sell_values = np.where(valid, np.clip(-returns, 0.0, None) * volume, 0.0)
        buy_flow = pd.Series(
            _rolling_sum_array(buy_values, window_returns), index=df.index
        )
        sell_flow = pd.Series(
            _rolling_sum_array(sell_values, window_returns), index=df.index
        )
        cache[key] = (buy_flow, sell_flow)
    return cache[key]


def _rolling_return_stats(
    df: pd.DataFrame, window_returns: int
) -> tuple[pd.Series, pd.Series]:
    cache = _microstructure_cache(df)
    key = ("return_stats", window_returns)
    if key not in cache:
        returns = _close_diff(df).to_numpy(dtype=float, copy=False)
        finite = np.isfinite(returns)
        clean_returns = np.where(finite, returns, 0.0)
        counts = _rolling_sum_array(finite.astype(float), window_returns)
        sums = _rolling_sum_array(clean_returns, window_returns)
        sq_sums = _rolling_sum_array(clean_returns * clean_returns, window_returns)
        mean_values = sums / window_returns
        variance = sq_sums / window_returns - mean_values * mean_values
        variance = np.clip(variance, 0.0, None)
        std_values = np.sqrt(variance)
        mean_values[counts != window_returns] = np.nan
        std_values[counts != window_returns] = np.nan
        mean_returns = pd.Series(mean_values, index=df.index)
        std_returns = pd.Series(std_values, index=df.index)
        cache[key] = (mean_returns, std_returns)
    return cache[key]


def _rolling_return_magnitude_sums(
    df: pd.DataFrame, window_returns: int
) -> tuple[pd.Series, pd.Series]:
    cache = _microstructure_cache(df)
    key = ("return_magnitude_sums", window_returns)
    if key not in cache:
        returns = _close_diff(df).to_numpy(dtype=float, copy=False)
        buy_values = np.where(np.isfinite(returns), np.clip(returns, 0.0, None), 0.0)
        sell_values = np.where(np.isfinite(returns), np.clip(-returns, 0.0, None), 0.0)
        buy_sum = pd.Series(
            _rolling_sum_array(buy_values, window_returns), index=df.index
        )
        sell_sum = pd.Series(
            _rolling_sum_array(sell_values, window_returns), index=df.index
        )
        cache[key] = (buy_sum, sell_sum)
    return cache[key]

class OrderFlowImbalanceFactor(BaseFactor):
    """订单流失衡因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('order_flow_imbalance', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算订单流失衡（买方/卖方压力）"""
        window_returns = self.window - 1
        valid = _valid_close_window(df, self.window)
        buy_pressure, sell_pressure = _rolling_flow_sums(df, window_returns)
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
        volume = df['volume']

        window_returns = self.window - 1
        valid = _valid_close_window(df, self.window)
        _, volatility = _rolling_return_stats(df, window_returns)
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

        valid = _valid_close_window(df, self.window)
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
        window_returns = self.window - 1
        valid = _valid_close_window(df, self.window)
        buy_count, sell_count = _rolling_direction_counts(df, window_returns)
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

        valid = _valid_close_window(df, self.window)
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
        
        valid = _valid_close_window(df, self.window)
        close_values = close.to_numpy(dtype=float, copy=False)
        volume_values = volume.to_numpy(dtype=float, copy=False)
        result = np.full(len(close_values), np.nan, dtype=float)
        chunk_size = 5

        if len(close_values) >= self.window:
            price_windows = np.lib.stride_tricks.sliding_window_view(
                close_values, self.window
            )
            volume_windows = np.lib.stride_tricks.sliding_window_view(
                volume_values, self.window
            )
            valid_windows = np.isfinite(price_windows).all(axis=1) & np.isfinite(
                volume_windows
            ).all(axis=1)
            chunk_count = (self.window - 1) // chunk_size
            window_result = np.full(len(price_windows), np.nan, dtype=float)

            if chunk_count == 0:
                window_result[valid_windows] = 0.0
            elif valid_windows.any():
                valid_price_windows = price_windows[valid_windows]
                valid_volume_windows = volume_windows[valid_windows]
                returns = np.diff(valid_price_windows, axis=1)
                current_vol = returns.std(axis=1)
                trimmed_returns = returns[:, -chunk_count * chunk_size :]
                chunk_vol = np.std(
                    trimmed_returns.reshape(len(trimmed_returns), chunk_count, chunk_size),
                    axis=2,
                )
                avg_vol = chunk_vol.mean(axis=1)
                adjusted = np.zeros(len(valid_price_windows), dtype=float)
                positive_avg = avg_vol > 0
                adjusted[positive_avg] = (
                    valid_volume_windows[positive_avg, -1]
                    * (current_vol[positive_avg] / avg_vol[positive_avg])
                    / (valid_volume_windows[positive_avg].mean(axis=1) + 1e-8)
                )
                window_result[valid_windows] = adjusted

            result[self.window - 1 :] = window_result

        return pd.Series(result, index=close.index).where(valid)

class PriceVelocityFactor(BaseFactor):
    """价格速度因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('price_velocity', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算价格变化速度"""
        window_returns = self.window - 1
        valid = _valid_close_window(df, self.window)
        velocity, volatility = _rolling_return_stats(df, window_returns)
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
        
        valid = _valid_close_window(df, self.window)
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
        
        valid = _valid_close_window(df, self.window)
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
        
        valid = _valid_close_window(df, self.window)
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
        window_returns = self.window - 1
        valid = _valid_close_window(df, self.window)
        buy_sum, sell_sum = _rolling_return_magnitude_sums(df, window_returns)
        buy_count, sell_count = _rolling_direction_counts(df, window_returns)

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
        
        valid = _valid_close_window(df, self.window)
        directions = np.sign(close.diff())
        same_direction = directions.eq(directions.shift(1)).astype(float)
        persistence = same_direction.rolling(
            self.window - 2, min_periods=self.window - 2
        ).mean()
        persistence = persistence * 2 - 1
        return persistence.where(valid)

class MarketImpactFactor(BaseFactor):
    """市场冲击因子"""
    
    def __init__(
        self,
        window: int = 50,
        alpha: float = 0.5,
        flow_window: int = 5,
        volatility_window: int = 10,
    ):
        if min(window, flow_window, volatility_window) <= 0:
            raise ValueError("market impact windows must be positive")
        if window < max(flow_window, volatility_window):
            raise ValueError(
                "window must cover both flow_window and volatility_window"
            )
        super().__init__('market_impact', window)
        self.alpha = alpha
        self.flow_window = flow_window
        self.volatility_window = volatility_window
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算市场冲击（订单流对价格的影响）"""
        close = df['close']
        volume = df['volume']
        
        valid = _valid_close_window(df, self.window)
        order_flow = volume * np.sign(close.diff())
        flow_sum = order_flow.rolling(
            self.flow_window, min_periods=self.flow_window
        ).sum()
        price_std = _rolling_std0(close, self.volatility_window)
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
        valid = _valid_close_window(df, self.window)
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
        volume = df['volume']
        
        window_returns = self.window - 1
        valid = _valid_close_window(df, self.window)
        buy_flow, sell_flow = _rolling_flow_sums(df, window_returns)
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
        
        valid = _valid_close_window(df, self.window)
        volume_values = volume.to_numpy(dtype=float, copy=False)
        result = np.full(len(volume_values), np.nan, dtype=float)

        if len(volume_values) >= self.window and self.window >= 30:
            volume_windows = np.lib.stride_tricks.sliding_window_view(
                volume_values, self.window
            )
            valid_windows = np.isfinite(volume_windows).all(axis=1)
            window_result = np.full(len(volume_windows), np.nan, dtype=float)

            if valid_windows.any():
                valid_volume_windows = volume_windows[valid_windows]
                high_vol = valid_volume_windows > valid_volume_windows.mean(
                    axis=1, keepdims=True
                )
                high_count = high_vol[:, :-1].sum(axis=1)
                high_to_high = np.logical_and(
                    high_vol[:, :-1], high_vol[:, 1:]
                ).sum(axis=1)
                clustered = np.zeros(len(valid_volume_windows), dtype=float)
                positive_count = high_count > 0
                clustered[positive_count] = (
                    (high_to_high[positive_count] / high_count[positive_count]) * 2 - 1
                )
                window_result[valid_windows] = clustered

            result[self.window - 1 :] = window_result

        return pd.Series(result, index=close.index).where(valid)

class PriceVolumeDecouplingFactor(BaseFactor):
    """价格成交量脱钩因子"""
    
    def __init__(self, window: int = 50):
        super().__init__('price_volume_decoupling', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """检测价格与成交量的脱钩现象"""
        close = df['close']
        volume = df['volume']
        
        valid = _valid_close_window(df, self.window)
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
        
        valid = _valid_close_window(df, self.window)
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
        
        valid = _valid_close_window(df, self.window)
        half_window = self.window // 2
        first_half = volume.shift(half_window).rolling(
            half_window, min_periods=half_window
        ).mean()
        second_half = volume.rolling(half_window, min_periods=half_window).mean()
        migration = (second_half - first_half) / (first_half + 1e-8)
        migration = migration.where(first_half > 0, 0.0)
        return migration.where(valid)
