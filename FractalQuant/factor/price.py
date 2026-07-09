"""
浠锋牸鍥犲瓙瀹炵幇
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base import PriceFactor, VolumeFactor

class ReturnsFactor(PriceFactor):
    """鏀剁泭鐜囧洜瀛?"""
    
    def __init__(self, window: int = 5):
        super().__init__('returns', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻鏀剁泭鐜?"""
        returns = df['close'].pct_change(periods=self.window, fill_method=None)
        return returns

class LogReturnsFactor(PriceFactor):
    """瀵规暟鏀剁泭鐜囧洜瀛?"""
    
    def __init__(self, window: int = 5):
        super().__init__('log_returns', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻瀵规暟鏀剁泭鐜?"""
        log_returns = np.log(df['close'] / df['close'].shift(self.window))
        return log_returns

class PriceMomentumFactor(PriceFactor):
    """浠锋牸鍔ㄩ噺鍥犲瓙"""
    
    def __init__(self, window: int = 20):
        super().__init__('price_momentum', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻浠锋牸鍔ㄩ噺"""
        momentum = df['close'] / df['close'].shift(self.window) - 1
        return momentum

class PriceRelativeFactor(PriceFactor):
    """浠锋牸鐩稿寮哄害鍥犲瓙"""
    
    def __init__(self, window: int = 20):
        super().__init__('price_relative', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻浠锋牸鐩稿寮哄害"""
        ma_short = df['close'].rolling(window=self.window // 2).mean()
        ma_long = df['close'].rolling(window=self.window).mean()
        relative = ma_short / ma_long - 1
        return relative

class PriceZScoreFactor(PriceFactor):
    """浠锋牸Z-score鍥犲瓙"""
    
    def __init__(self, window: int = 20):
        super().__init__('price_zscore', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻浠锋牸Z-score"""
        mean = df['close'].rolling(window=self.window).mean()
        std = df['close'].rolling(window=self.window).std()
        zscore = (df['close'] - mean) / std
        return zscore

class VolumePriceTrendFactor(VolumeFactor):
    """閲忎环瓒嬪娍鍥犲瓙"""
    
    def __init__(self, window: int = 20):
        super().__init__('volume_price_trend', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻閲忎环瓒嬪娍"""
        vpt = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()
        return vpt

class OBVFactor(VolumeFactor):
    """OBV鑳介噺娼洜瀛?"""
    
    def __init__(self):
        super().__init__('obv', 1)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻OBV

        璇存槑锛歝lose 鍙兘鍥犳暟鎹竻娲楋紙close<=0 缃负 NaN锛夊嚭鐜扮己澶便€傛瘮杈冭繍绠楅亣鍒?
        NaN 浼氬悓鏃惰惤鍏?else 鍒嗘敮锛屽鑷?OBV 琚敊璇湴鎷夊钩銆傝繖閲屾樉寮忔娴?NaN锛?
        鍦ㄧ己澶?bar 涓婁繚鎸佸墠鍊间笉鍙橈紝閬垮厤闈欓粯閿欒銆?
        """
        close = df['close'].to_numpy(dtype=float, copy=False)
        volume = df['volume'].to_numpy(dtype=float, copy=False)

        if len(close) == 0:
            return pd.Series(dtype=float, index=df.index)

        increments = np.zeros(len(close), dtype=float)
        curr = close[1:]
        prev = close[:-1]
        vol = volume[1:]
        valid = np.isfinite(curr) & np.isfinite(prev) & np.isfinite(vol)
        increments[1:] = np.where(valid, np.sign(curr - prev) * vol, 0.0)

        return pd.Series(np.cumsum(increments), index=df.index)

class VolumeMomentumFactor(VolumeFactor):
    """鎴愪氦閲忓姩閲忓洜瀛?"""
    
    def __init__(self, window: int = 5):
        super().__init__('volume_momentum', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻鎴愪氦閲忓姩閲?"""
        volume_ma = df['volume'].rolling(window=self.window).mean()
        momentum = df['volume'] / volume_ma - 1
        return momentum

class VolumePriceConfirmFactor(VolumeFactor):
    """閲忎环纭鍥犲瓙"""
    
    def __init__(self, window: int = 20):
        super().__init__('volume_price_confirm', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """璁＄畻閲忎环纭"""
        price_change = df['close'].pct_change(fill_method=None)
        volume_change = df['volume'].pct_change(fill_method=None)

        confirm = ((price_change > 0) & (volume_change > 0)) | (
            (price_change < 0) & (volume_change < 0)
        )
        return confirm.astype(float)
