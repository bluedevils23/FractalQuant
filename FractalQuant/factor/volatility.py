"""
波动率因子实现
"""
import pandas as pd
import numpy as np
import weakref
from typing import List, Dict, Optional
from scipy import stats
from .base import VolatilityFactor

# A股每个交易日的分钟bar数（4小时交易时段）。年化时使用 252 个交易日。
# 这些因子运行在分钟频率数据上，因此年化基准为 sqrt(252 * 240)。
TRADING_DAYS_PER_YEAR = 252
BARS_PER_DAY = 240
PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR * BARS_PER_DAY

_VOLATILITY_CACHE: dict[int, tuple[weakref.ReferenceType[pd.DataFrame], dict]] = {}


def _volatility_cache(df: pd.DataFrame) -> dict:
    key = id(df)
    entry = _VOLATILITY_CACHE.get(key)
    if entry is None or entry[0]() is not df:
        ref = weakref.ref(
            df,
            lambda _ref, cache_key=key: _VOLATILITY_CACHE.pop(cache_key, None),
        )
        entry = (ref, {})
        _VOLATILITY_CACHE[key] = entry
    return entry[1]


def _log_returns(df: pd.DataFrame) -> pd.Series:
    cache = _volatility_cache(df)
    key = "log_returns"
    if key not in cache:
        cache[key] = np.log(df["close"] / df["close"].shift(1))
    return cache[key]


def _pct_returns(df: pd.DataFrame) -> pd.Series:
    cache = _volatility_cache(df)
    key = "pct_returns"
    if key not in cache:
        cache[key] = df["close"].pct_change(fill_method=None)
    return cache[key]


def _log_high_low(df: pd.DataFrame) -> pd.Series:
    cache = _volatility_cache(df)
    key = "log_high_low"
    if key not in cache:
        cache[key] = np.log(df["high"] / df["low"])
    return cache[key]


def _log_close_open(df: pd.DataFrame) -> pd.Series:
    cache = _volatility_cache(df)
    key = "log_close_open"
    if key not in cache:
        cache[key] = np.log(df["close"] / df["open"])
    return cache[key]


def _rolling_log_return_std(df: pd.DataFrame, window: int) -> pd.Series:
    cache = _volatility_cache(df)
    key = ("rolling_log_return_std", window)
    if key not in cache:
        cache[key] = _log_returns(df).rolling(window=window).std()
    return cache[key]


def _rolling_pct_return_std(df: pd.DataFrame, window: int) -> pd.Series:
    cache = _volatility_cache(df)
    key = ("rolling_pct_return_std", window)
    if key not in cache:
        cache[key] = _pct_returns(df).rolling(window=window).std()
    return cache[key]


def _true_range(df: pd.DataFrame) -> pd.Series:
    cache = _volatility_cache(df)
    key = "true_range"
    if key not in cache:
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift()
        high_low = (high - low).to_numpy(dtype=float, copy=False)
        high_close = np.abs((high - prev_close).to_numpy(dtype=float, copy=False))
        low_close = np.abs((low - prev_close).to_numpy(dtype=float, copy=False))
        cache[key] = pd.Series(
            np.maximum.reduce([high_low, high_close, low_close]),
            index=df.index,
        )
    return cache[key]

class HistoricalVolatilityFactor(VolatilityFactor):
    """历史波动率因子"""

    def __init__(self, window: int = 20, periods_per_year: int = PERIODS_PER_YEAR):
        super().__init__('historical_volatility', window)
        self.periods_per_year = periods_per_year

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算历史波动率（按分钟频率年化）"""
        volatility = _rolling_log_return_std(df, self.window) * np.sqrt(self.periods_per_year)
        return volatility

class AnnualizedVolatilityFactor(VolatilityFactor):
    """年化波动率因子"""

    def __init__(self, window: int = 20, periods_per_year: int = PERIODS_PER_YEAR):
        super().__init__('annualized_volatility', window)
        self.periods_per_year = periods_per_year

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算年化波动率（按分钟频率年化）"""
        volatility = _rolling_log_return_std(df, self.window) * np.sqrt(self.periods_per_year)
        return volatility

class RealizedVolatilityFactor(VolatilityFactor):
    """实现波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('realized_volatility', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算实现波动率"""
        log_returns = _log_returns(df)
        rv = np.sqrt((log_returns ** 2).rolling(window=self.window).sum())
        return rv

class ParkinsonVolatilityFactor(VolatilityFactor):
    """Parkinson波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('parkinson_volatility', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算Parkinson波动率"""
        high_low = _log_high_low(df)
        volatility = np.sqrt((high_low ** 2).rolling(window=self.window).mean() / (4 * np.log(2)))
        return volatility

class GarmanKlassVolatilityFactor(VolatilityFactor):
    """Garman-Klass波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('garman_klass_volatility', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算Garman-Klass波动率"""
        log_high_low = _log_high_low(df)
        log_close_open = _log_close_open(df)

        variance = (
            0.5 * (log_high_low ** 2).rolling(window=self.window).mean() -
            (2 * np.log(2) - 1) * (log_close_open ** 2).rolling(window=self.window).mean()
        )
        # GK方差在close-open跳动远大于high-low时可能为负，开方前先截断到0，
        # 避免 sqrt(负数) 静默产生NaN而丢失信息。
        volatility = np.sqrt(variance.clip(lower=0.0))
        return volatility

class BollingerBandWidthFactor(VolatilityFactor):
    """布林带宽度因子"""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__('bollinger_band_width', window)
        self.num_std = num_std
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算布林带宽度"""
        middle_band = df['close'].rolling(window=self.window).mean()
        std = df['close'].rolling(window=self.window).std()
        upper_band = middle_band + self.num_std * std
        lower_band = middle_band - self.num_std * std
        width = (upper_band - lower_band) / middle_band
        return width

class ATRFactor(VolatilityFactor):
    """平均真实波动幅度因子"""
    
    def __init__(self, window: int = 14):
        super().__init__('atr', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算ATR"""
        true_range = _true_range(df)
        atr = true_range.rolling(window=self.window).mean()
        return atr

class VolatilityRegimeFactor(VolatilityFactor):
    """波动率 regimes 因子"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__('volatility_regime', short_window)
        self.short_window = short_window
        self.long_window = long_window
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率 regimes"""
        short_vol = _rolling_pct_return_std(df, self.short_window)
        long_vol = _rolling_pct_return_std(df, self.long_window)
        
        regime = short_vol / long_vol
        return regime

class VolatilitySkewFactor(VolatilityFactor):
    """波动率偏度因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('volatility_skew', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率偏度"""
        log_returns = _log_returns(df)
        skew = log_returns.rolling(window=self.window).skew()
        return skew

class VolatilityKurtosisFactor(VolatilityFactor):
    """波动率峰度因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('volatility_kurtosis', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率峰度"""
        log_returns = _log_returns(df)
        kurtosis = log_returns.rolling(window=self.window).kurt()
        return kurtosis
