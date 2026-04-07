"""
数据存储模块
"""
import os
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict
import pickle
import json
import logging

from .models import BarData, FactorData

logger = logging.getLogger(__name__)

class DataStore:
    """数据存储器"""
    
    def __init__(self, base_path: str = 'data'):
        self.base_path = base_path
        self.bars_path = os.path.join(base_path, 'bars')
        self.factors_path = os.path.join(base_path, 'factors')
        
        os.makedirs(self.bars_path, exist_ok=True)
        os.makedirs(self.factors_path, exist_ok=True)
        
    def save_bars(self, symbol: str, bars: List[BarData], exchange: str = None):
        """保存K线数据"""
        try:
            data = [bar.to_dict() for bar in bars]
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            filename = f"{symbol.replace('/', '_')}"
            if exchange:
                filename = f"{filename}_{exchange}"
                
            filepath = os.path.join(self.bars_path, f"{filename}.csv")
            df.to_csv(filepath)
            
            logger.info(f"Saved {len(bars)} bars for {symbol} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving bars for {symbol}: {e}")
    
    def load_bars(self, symbol: str, exchange: str = None) -> Optional[pd.DataFrame]:
        """加载K线数据"""
        try:
            filename = f"{symbol.replace('/', '_')}"
            if exchange:
                filename = f"{filename}_{exchange}"
                
            filepath = os.path.join(self.bars_path, f"{filename}.csv")
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
                logger.info(f"Loaded {len(df)} bars for {symbol}")
                return df
            return None
            
        except Exception as e:
            logger.error(f"Error loading bars for {symbol}: {e}")
            return None
    
    def save_factors(self, symbol: str, factor_data: FactorData, exchange: str = None):
        """保存因子数据"""
        try:
            df = factor_data.to_dataframe()
            
            filename = f"{symbol.replace('/', '_')}_factors"
            if exchange:
                filename = f"{filename}_{exchange}"
                
            filepath = os.path.join(self.factors_path, f"{filename}.csv")
            
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
                df = pd.concat([existing_df, df], axis=0)
            
            df.to_csv(filepath)
            logger.info(f"Saved factor data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving factors for {symbol}: {e}")
    
    def load_factors(self, symbol: str, exchange: str = None) -> Optional[pd.DataFrame]:
        """加载因子数据"""
        try:
            filename = f"{symbol.replace('/', '_')}_factors"
            if exchange:
                filename = f"{filename}_{exchange}"
                
            filepath = os.path.join(self.factors_path, f"{filename}.csv")
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
                return df
            return None
            
        except Exception as e:
            logger.error(f"Error loading factors for {symbol}: {e}")
            return None

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_path: str = 'cache'):
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
        
    def save_cache(self, key: str, data):
        """保存缓存"""
        try:
            filepath = os.path.join(self.cache_path, f"{key}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache {key}: {e}")
    
    def load_cache(self, key: str):
        """加载缓存"""
        try:
            filepath = os.path.join(self.cache_path, f"{key}.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading cache {key}: {e}")
            return None
    
    def clear_cache(self, pattern: str = None):
        """清除缓存"""
        try:
            if pattern:
                for filename in os.listdir(self.cache_path):
                    if pattern in filename:
                        os.remove(os.path.join(self.cache_path, filename))
            else:
                for filename in os.listdir(self.cache_path):
                    os.remove(os.path.join(self.cache_path, filename))
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")