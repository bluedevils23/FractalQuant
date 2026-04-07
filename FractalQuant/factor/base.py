"""
因子计算基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

class BaseFactor(ABC):
    """因子基类"""
    
    def __init__(self, name: str, window: int = 20):
        self.name = name
        self.window = window
        
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        pass
    
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.calculate(df)

class PriceFactor(BaseFactor):
    """价格因子基类"""
    pass

class VolumeFactor(BaseFactor):
    """成交量因子基类"""
    pass

class VolatilityFactor(BaseFactor):
    """波动率因子基类"""
    pass

class MomentumFactor(BaseFactor):
    """动量因子基类"""
    pass

class TrendFactor(BaseFactor):
    """趋势因子基类"""
    pass

class OrderBookFactor(BaseFactor):
    """订单簿因子基类"""
    pass