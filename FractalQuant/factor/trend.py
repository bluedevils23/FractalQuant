"""
瓒嬪娍鍜屽姩閲忓洜瀛?
"""
import pandas as pd
import numpy as np
import weakref
from typing import List, Dict, Optional
from .base import TrendFactor, MomentumFactor

_TREND_CACHE: dict[int, tuple[weakref.ReferenceType[pd.DataFrame], dict]] = {}


def _trend_cache(df: pd.DataFrame) -> dict:
    key = id(df)
    entry = _TREND_CACHE.get(key)
    if entry is None or entry[0]() is not df:
        ref = weakref.ref(
            df,
            lambda _ref, cache_key=key: _TREND_CACHE.pop(cache_key, None),
        )
        entry = (ref, {})
        _TREND_CACHE[key] = entry
    return entry[1]


def _close_delta(df: pd.DataFrame) -> pd.Series:
    cache = _trend_cache(df)
    key = "close_delta"
    if key not in cache:
        cache[key] = df["close"].diff()
    return cache[key]


def _gain_loss(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cache = _trend_cache(df)
    key = "gain_loss"
    if key not in cache:
        delta = _close_delta(df)
        cache[key] = (delta.where(delta > 0, 0), -delta.where(delta < 0, 0))
    return cache[key]


def _rolling_gain_loss_mean(
    df: pd.DataFrame, window: int
) -> tuple[pd.Series, pd.Series]:
    cache = _trend_cache(df)
    key = ("gain_loss_mean", window)
    if key not in cache:
        gain, loss = _gain_loss(df)
        cache[key] = (
            gain.rolling(window=window).mean(),
            loss.rolling(window=window).mean(),
        )
    return cache[key]


def _rolling_gain_loss_sum(
    df: pd.DataFrame, window: int
) -> tuple[pd.Series, pd.Series]:
    cache = _trend_cache(df)
    key = ("gain_loss_sum", window)
    if key not in cache:
        gain, loss = _gain_loss(df)
        cache[key] = (
            gain.rolling(window=window).sum(),
            loss.rolling(window=window).sum(),
        )
    return cache[key]


def _rolling_high_low(
    df: pd.DataFrame, window: int
) -> tuple[pd.Series, pd.Series]:
    cache = _trend_cache(df)
    key = ("rolling_high_low", window)
    if key not in cache:
        cache[key] = (
            df["high"].rolling(window=window).max(),
            df["low"].rolling(window=window).min(),
        )
    return cache[key]

class MovingAverageFactor(TrendFactor):
    """绉诲姩骞冲潎绾垮洜瀛?"""
    
    def __init__(self, window: int = 20):
        super().__init__('moving_average', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻绉诲姩骞冲潎绾?"""
        ma = df['close'].rolling(window=self.window).mean()
        return ma

class MACDFactor(TrendFactor):
    """MACD鍥犲瓙"""
    
    def __init__(self, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9):
        super().__init__('macd', slow_window)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻MACD"""
        ema_fast = df['close'].ewm(span=self.fast_window, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_window, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_window, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_histogram

class EMAFactor(TrendFactor):
    """鎸囨暟绉诲姩骞冲潎绾垮洜瀛?"""
    
    def __init__(self, window: int = 20):
        super().__init__('ema', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻鎸囨暟绉诲姩骞冲潎绾?"""
        ema = df['close'].ewm(span=self.window, adjust=False).mean()
        return ema

class ADXFactor(TrendFactor):
    """ADX瓒嬪娍寮哄害鍥犲瓙"""
    
    def __init__(self, window: int = 14):
        super().__init__('adx', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻ADX"""
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
    """RSI鐩稿寮哄急鎸囨暟鍥犲瓙"""
    
    def __init__(self, window: int = 14):
        super().__init__('rsi', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻RSI"""
        gain, loss = _rolling_gain_loss_mean(df, self.window)
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

class StochasticFactor(MomentumFactor):
    """闅忔満鎸囨爣鍥犲瓙"""
    
    def __init__(self, window: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        super().__init__('stochastic', window)
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻闅忔満鎸囨爣"""
        high, low = _rolling_high_low(df, self.window)
        
        k = 100 * (df['close'] - low) / (high - low + 1e-8)
        d = k.rolling(window=self.smooth_d).mean()
        
        return d

class CMOFactor(MomentumFactor):
    """Chande Momentum Oscillator鍥犲瓙"""
    
    def __init__(self, window: int = 14):
        super().__init__('cmo', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻CMO"""
        total_gain, total_loss = _rolling_gain_loss_sum(df, self.window)
        
        cmo = 100 * (total_gain - total_loss) / (total_gain + total_loss + 1e-8)
        return cmo

class WilliamsRFactor(MomentumFactor):
    """Williams %R鍥犲瓙"""
    
    def __init__(self, window: int = 14):
        super().__init__('williams_r', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻Williams %R"""
        high, low = _rolling_high_low(df, self.window)
        
        wr = (high - df['close']) / (high - low + 1e-8) * -100
        return wr

class AOFactor(MomentumFactor):
    """Awesome Oscillator鍥犲瓙"""
    
    def __init__(self, short_window: int = 5, long_window: int = 34):
        super().__init__('awesome_oscillator', long_window)
        self.short_window = short_window
        self.long_window = long_window
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻Awesome Oscillator"""
        median_price = (df['high'] + df['low']) / 2
        
        short_ao = median_price.rolling(window=self.short_window).mean()
        long_ao = median_price.rolling(window=self.long_window).mean()
        
        ao = short_ao - long_ao
        return ao

class CCIFactor(MomentumFactor):
    """鍟嗗搧閫氶亾鎸囨暟鍥犲瓙"""
    
    def __init__(self, window: int = 20):
        super().__init__('cci', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻CCI"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=self.window).mean()
        # 鏍囧噯CCI鐨勫钩鍧囩粷瀵瑰亸宸簲鍥寸粫绐楀彛绉诲姩鍧囧€艰绠楋紝鑰岄潪绐楀彛鏈€鍚庝竴涓€笺€?
        values = tp.to_numpy(dtype=float, copy=False)
        md_values = np.full(len(values), np.nan, dtype=float)
        if len(values) >= self.window:
            windows = np.lib.stride_tricks.sliding_window_view(values, self.window)
            valid = np.isfinite(windows).all(axis=1)
            if valid.any():
                valid_windows = windows[valid]
                mean = valid_windows.mean(axis=1, keepdims=True)
                md_window_values = md_values[self.window - 1 :]
                md_window_values[valid] = np.abs(valid_windows - mean).mean(axis=1)
        md = pd.Series(md_values, index=tp.index)

        cci = (tp - ma) / (0.015 * md + 1e-8)
        return cci

class ROCFactor(MomentumFactor):
    """鍔ㄩ噺姣旂巼鍥犲瓙"""
    
    def __init__(self, window: int = 12):
        super().__init__('roc', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻ROC"""
        roc = (df['close'] - df['close'].shift(self.window)) / (df['close'].shift(self.window) + 1e-8) * 100
        return roc

class TRIXFactor(MomentumFactor):
    """TRIX鍥犲瓙"""
    
    def __init__(self, window: int = 15):
        super().__init__('trix', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻TRIX"""
        ema1 = df['close'].ewm(span=self.window, adjust=False).mean()
        ema2 = ema1.ewm(span=self.window, adjust=False).mean()
        ema3 = ema2.ewm(span=self.window, adjust=False).mean()
        
        trix = ema3.pct_change(periods=1, fill_method=None) * 100
        return trix

class LSMAFactor(TrendFactor):
    """绾挎€у洖褰掓枩鐜囧洜瀛?"""
    
    def __init__(self, window: int = 20):
        super().__init__('lsma', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻绾挎€у洖褰掓枩鐜?"""
        close = df['close']
        values = close.to_numpy(dtype=float, copy=False)
        result = np.full(len(values), np.nan, dtype=float)

        if self.window <= 0:
            return pd.Series(result, index=close.index)

        if self.window == 1:
            return close.astype(float)

        if len(values) < self.window:
            return pd.Series(result, index=close.index)

        windows = np.lib.stride_tricks.sliding_window_view(values, self.window)
        valid = np.isfinite(windows).all(axis=1)

        if valid.any():
            x_arr = np.arange(self.window, dtype=float)
            x_centered = x_arr - x_arr.mean()
            denominator = np.dot(x_centered, x_centered)

            slopes = np.full(len(windows), np.nan, dtype=float)
            valid_windows = windows[valid]
            centered_windows = valid_windows - valid_windows.mean(axis=1, keepdims=True)
            slopes[valid] = centered_windows @ x_centered / denominator
            result[self.window - 1 :] = slopes

        return pd.Series(result, index=close.index)
