"""
趋势和动量因子
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base import TrendFactor, MomentumFactor

class MovingAverageFactor(TrendFactor):
    """移动平均线因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('moving_average', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算移动平均线"""
        ma = df['close'].rolling(window=self.window).mean()
        return ma

class MACDFactor(TrendFactor):
    """MACD因子"""
    
    def __init__(self, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9):
        super().__init__('macd', slow_window)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算MACD"""
        ema_fast = df['close'].ewm(span=self.fast_window, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_window, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_window, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_histogram

class EMAFactor(TrendFactor):
    """指数移动平均线因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('ema', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算指数移动平均线"""
        ema = df['close'].ewm(span=self.window, adjust=False).mean()
        return ema

class ADXFactor(TrendFactor):
    """ADX趋势强度因子"""
    
    def __init__(self, window: int = 14):
        super().__init__('adx', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算ADX"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/self.window).mean() / 
                        (high - low).ewm(alpha=1/self.window).mean())
        minus_di = 100 * (minus_dm.ewm(alpha=1/self.window).mean() / 
                         (high - low).ewm(alpha=1/self.window).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.ewm(alpha=1/self.window).mean()
        
        return adx

class RSIFactor(MomentumFactor):
    """RSI相对强弱指数因子"""
    
    def __init__(self, window: int = 14):
        super().__init__('rsi', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

class StochasticFactor(MomentumFactor):
    """随机指标因子"""
    
    def __init__(self, window: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        super().__init__('stochastic', window)
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算随机指标"""
        high = df['high'].rolling(window=self.window).max()
        low = df['low'].rolling(window=self.window).min()
        
        k = 100 * (df['close'] - low) / (high - low + 1e-8)
        d = k.rolling(window=self.smooth_d).mean()
        
        return d

class CMOFactor(MomentumFactor):
    """Chande Momentum Oscillator因子"""
    
    def __init__(self, window: int = 14):
        super().__init__('cmo', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算CMO"""
        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        total_gain = gain.rolling(window=self.window).sum()
        total_loss = loss.rolling(window=self.window).sum()
        
        cmo = 100 * (total_gain - total_loss) / (total_gain + total_loss + 1e-8)
        return cmo

class WilliamsRFactor(MomentumFactor):
    """Williams %R因子"""
    
    def __init__(self, window: int = 14):
        super().__init__('williams_r', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算Williams %R"""
        high = df['high'].rolling(window=self.window).max()
        low = df['low'].rolling(window=self.window).min()
        
        wr = (high - df['close']) / (high - low + 1e-8) * -100
        return wr

class AOFactor(MomentumFactor):
    """Awesome Oscillator因子"""
    
    def __init__(self, short_window: int = 5, long_window: int = 34):
        super().__init__('awesome_oscillator', long_window)
        self.short_window = short_window
        self.long_window = long_window
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算Awesome Oscillator"""
        median_price = (df['high'] + df['low']) / 2
        
        short_ao = median_price.rolling(window=self.short_window).mean()
        long_ao = median_price.rolling(window=self.long_window).mean()
        
        ao = short_ao - long_ao
        return ao

class CCIFactor(MomentumFactor):
    """商品通道指数因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('cci', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算CCI"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=self.window).mean()
        md = tp.rolling(window=self.window).apply(lambda x: np.mean(np.abs(x - x.iloc[-1])))
        
        cci = (tp - ma) / (0.015 * md + 1e-8)
        return cci

class ROCFactor(MomentumFactor):
    """动量比率因子"""
    
    def __init__(self, window: int = 12):
        super().__init__('roc', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算ROC"""
        roc = (df['close'] - df['close'].shift(self.window)) / (df['close'].shift(self.window) + 1e-8) * 100
        return roc

class TRIXFactor(MomentumFactor):
    """TRIX因子"""
    
    def __init__(self, window: int = 15):
        super().__init__('trix', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算TRIX"""
        ema1 = df['close'].ewm(span=self.window, adjust=False).mean()
        ema2 = ema1.ewm(span=self.window, adjust=False).mean()
        ema3 = ema2.ewm(span=self.window, adjust=False).mean()
        
        trix = ema3.pct_change(periods=1) * 100
        return trix

class LSMAFactor(TrendFactor):
    """线性回归斜率因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('lsma', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算线性回归斜率"""
        def calc_lsma(x):
            if len(x) < 2:
                return x.iloc[-1]
            y = x.values
            x_arr = np.arange(len(y))
            slope, intercept = np.polyfit(x_arr, y, 1)
            return slope
        
        lsma = df['close'].rolling(window=self.window).apply(calc_lsma)
        return lsma