"""
数据获取接口
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime, timedelta
import ccxt
import asyncio
import aiohttp
import pandas as pd
import logging

from .models import BarData, TickData, MarketData

logger = logging.getLogger(__name__)

class DataFetcher(ABC):
    """数据获取器基类"""
    
    @abstractmethod
    async def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[BarData]:
        """获取历史数据"""
        pass
    
    @abstractmethod
    async def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时数据"""
        pass
    
    @abstractmethod
    async def fetch_order_book(self, symbol: str) -> Optional[dict]:
        """获取订单簿"""
        pass

class CCXTDataFetcher(DataFetcher):
    """基于CCXT的数据获取器"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1m', 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> List[BarData]:
        """获取历史K线数据"""
        try:
            since = None
            if start_date:
                since = int(start_date.timestamp() * 1000)
            
            all_ohlcv = []
            limit = 1000
            
            while True:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe, 
                    since=since, 
                    limit=limit
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                if end_date and ohlcv[-1][0] >= int(end_date.timestamp() * 1000):
                    break
                    
                if len(ohlcv) < limit:
                    break
                    
                since = ohlcv[-1][0] + 60000
                
            bars = []
            for data in all_ohlcv:
                bar = BarData(
                    timestamp=datetime.fromtimestamp(data[0] / 1000),
                    symbol=symbol,
                    open=data[1],
                    high=data[2],
                    low=data[3],
                    close=data[4],
                    volume=data[5]
                )
                bars.append(bar)
                
            logger.info(f"Fetched {len(bars)} bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    async def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时tick数据"""
        ticks = []
        try:
            for symbol in symbols:
                ticker = await self.exchange.fetch_ticker(symbol)
                tick = TickData(
                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                    symbol=symbol,
                    price=ticker['last'],
                    volume=ticker.get('volume', 0),
                    bid_price=ticker.get('bid'),
                    ask_price=ticker.get('ask')
                )
                ticks.append(tick)
        except Exception as e:
            logger.error(f"Error fetching realtime data: {e}")
        return ticks
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[dict]:
        """获取订单簿数据"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

class ExchangeManager:
    """交易所管理器"""
    
    def __init__(self):
        self.exchanges = {}
        
    def create_exchange(self, exchange_id: str, api_key: str = None, api_secret: str = None):
        """创建交易所实例"""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'timeout': 30000,
                'enableRateLimit': True,
            })
            self.exchanges[exchange_id] = exchange
            return exchange
        except Exception as e:
            logger.error(f"Error creating exchange {exchange_id}: {e}")
            return None
    
    def get_fetcher(self, exchange_id: str, api_key: str = None, api_secret: str = None) -> Optional[CCXTDataFetcher]:
        """获取数据获取器"""
        if exchange_id not in self.exchanges:
            self.create_exchange(exchange_id, api_key, api_secret)
        
        if exchange_id in self.exchanges:
            return CCXTDataFetcher(self.exchanges[exchange_id])
        return None

# 全局交易所管理器
exchange_manager = ExchangeManager()