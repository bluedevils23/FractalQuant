"""
价格因子实现
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base import PriceFactor, VolumeFactor

class ReturnsFactor(PriceFactor):
    """收益率因子"""
    
    def __init__(self, window: int = 5):
        super().__init__('returns', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算收益率"""
        returns = df['close'].pct_change(periods=self.window, fill_method=None)
        return returns

class LogReturnsFactor(PriceFactor):
    """对数收益率因子"""
    
    def __init__(self, window: int = 5):
        super().__init__('log_returns', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算对数收益率"""
        log_returns = np.log(df['close'] / df['close'].shift(self.window))
        return log_returns

class PriceMomentumFactor(PriceFactor):
    """价格动量因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('price_momentum', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算价格动量"""
        momentum = df['close'] / df['close'].shift(self.window) - 1
        return momentum

class PriceRelativeFactor(PriceFactor):
    """价格相对强度因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('price_relative', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算价格相对强度"""
        ma_short = df['close'].rolling(window=self.window // 2).mean()
        ma_long = df['close'].rolling(window=self.window).mean()
        relative = ma_short / ma_long - 1
        return relative

class PriceZScoreFactor(PriceFactor):
    """价格Z-score因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('price_zscore', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算价格Z-score"""
        mean = df['close'].rolling(window=self.window).mean()
        std = df['close'].rolling(window=self.window).std()
        zscore = (df['close'] - mean) / std
        return zscore

class VolumePriceTrendFactor(VolumeFactor):
    """量价趋势因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('volume_price_trend', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算量价趋势"""
        vpt = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()
        return vpt

class OBVFactor(VolumeFactor):
    """OBV能量潮因子"""
    
    def __init__(self):
        super().__init__('obv', 1)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算OBV

        说明：close 可能因数据清洗（close<=0 置为 NaN）出现缺失。比较运算遇到
        NaN 会同时落入 else 分支，导致 OBV 被错误地拉平。这里显式检测 NaN，
        在缺失 bar 上保持前值不变，避免静默错误。
        """
        close = df['close']
        volume = df['volume']
        obv_values = [0.0]
        for i in range(1, len(df)):
            curr = close.iloc[i]
            prev = close.iloc[i - 1]
            vol = volume.iloc[i]
            if pd.isna(curr) or pd.isna(prev) or pd.isna(vol):
                # 数据缺失，保持前值，不累加
                obv_values.append(obv_values[-1])
            elif curr > prev:
                obv_values.append(obv_values[-1] + vol)
            elif curr < prev:
                obv_values.append(obv_values[-1] - vol)
            else:
                obv_values.append(obv_values[-1])
        return pd.Series(obv_values, index=df.index)

class VolumeMomentumFactor(VolumeFactor):
    """成交量动量因子"""
    
    def __init__(self, window: int = 5):
        super().__init__('volume_momentum', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算成交量动量"""
        volume_ma = df['volume'].rolling(window=self.window).mean()
        momentum = df['volume'] / volume_ma - 1
        return momentum

class VolumePriceConfirmFactor(VolumeFactor):
    """量价确认因子"""
    
    def __init__(self, window: int = 20):
        super().__init__('volume_price_confirm', window)
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算量价确认"""
        price_change = df['close'].pct_change(fill_method=None)
        volume_change = df['volume'].pct_change(fill_method=None)

        confirm = ((price_change > 0) & (volume_change > 0)) | (
            (price_change < 0) & (volume_change < 0)
        )
        return confirm.astype(float)
