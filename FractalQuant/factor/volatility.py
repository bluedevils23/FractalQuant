"""
波动率因子实现
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats
from .base import VolatilityFactor

class HistoricalVolatilityFactor(VolatilityFactor):
    """历史波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('historical_volatility', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算历史波动率"""
        log_returns = np.log(df['close'] / df['close'].shift(1))
        volatility = log_returns.rolling(window=self.window).std() * np.sqrt(252 * 24 * 60)
        return volatility

class AnnualizedVolatilityFactor(VolatilityFactor):
    """年化波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('annualized_volatility', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算年化波动率"""
        log_returns = np.log(df['close'] / df['close'].shift(1))
        volatility = log_returns.rolling(window=self.window).std() * np.sqrt(252)
        return volatility

class RealizedVolatilityFactor(VolatilityFactor):
    """实现波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('realized_volatility', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算实现波动率"""
        log_returns = np.log(df['close'] / df['close'].shift(1))
        rv = np.sqrt((log_returns ** 2).rolling(window=self.window).sum())
        return rv

class ParkinsonVolatilityFactor(VolatilityFactor):
    """Parkinson波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('parkinson_volatility', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算Parkinson波动率"""
        high_low = np.log(df['high'] / df['low'])
        volatility = np.sqrt((high_low ** 2).rolling(window=self.window).mean() / (4 * np.log(2)))
        return volatility

class GarmanKlassVolatilityFactor(VolatilityFactor):
    """Garman-Klass波动率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('garman_klass_volatility', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算Garman-Klass波动率"""
        log_high_low = np.log(df['high'] / df['low'])
        log_close_open = np.log(df['close'] / df['open'])
        
        volatility = np.sqrt(
            0.5 * (log_high_low ** 2).rolling(window=self.window).mean() - 
            (2 * np.log(2) - 1) * (log_close_open ** 2).rolling(window=self.window).mean()
        )
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
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
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
        short_vol = df['close'].pct_change().rolling(window=self.short_window).std()
        long_vol = df['close'].pct_change().rolling(window=self.long_window).std()
        
        regime = short_vol / long_vol
        return regime

class VolatilitySkewFactor(VolatilityFactor):
    """波动率偏度因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('volatility_skew', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率偏度"""
        log_returns = np.log(df['close'] / df['close'].shift(1))
        skew = log_returns.rolling(window=self.window).skew()
        return skew

class VolatilityKurtosisFactor(VolatilityFactor):
    """波动率峰度因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('volatility_kurtosis', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算波动率峰度"""
        log_returns = np.log(df['close'] / df['close'].shift(1))
        kurtosis = log_returns.rolling(window=self.window).kurt()
        return kurtosis