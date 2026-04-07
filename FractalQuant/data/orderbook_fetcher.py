"""
订单簿数据获取器(深度数据)
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from .models import OrderBookData

class OrderBookFetcher(ABC):
    """订单簿获取器基类"""
    
    @abstractmethod
    async def fetch_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        pass
    
    @abstractmethod
    async def fetch_order_books(self, symbols: List[str], depth: int = 20) -> List[OrderBookData]:
        """批量获取订单簿数据"""
        pass

class CCXTOrderBookFetcher(OrderBookFetcher):
    """基于CCXT的订单簿获取器"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        
    async def fetch_order_book(self, symbol: str, depth: int = 20) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=depth)
            
            bids = [(float(price), float(amount)) for price, amount in orderbook['bids'][:depth]]
            asks = [(float(price), float(amount)) for price, amount in orderbook['asks'][:depth]]
            
            order_book = OrderBookData(
                timestamp=datetime.fromtimestamp(orderbook['timestamp'] / 1000),
                symbol=symbol,
                bids=bids,
                asks=asks,
                exchange=self.exchange.id
            )
            
            return order_book
            
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def fetch_order_books(self, symbols: List[str], depth: int = 20) -> List[OrderBookData]:
        """批量获取订单簿数据"""
        order_books = []
        for symbol in symbols:
            order_book = await self.fetch_order_book(symbol, depth)
            if order_book:
                order_books.append(order_book)
        return order_books

class BinanceOrderBookFetcher(OrderBookFetcher):
    """币安订单簿获取器"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    async def fetch_order_book(self, symbol: str, depth: int = 100) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        try:
            params = {
                'symbol': symbol.replace('/', ''),
                'limit': depth
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/depth", params=params) as response:
                    data = await response.json()
                    
                    if 'lastUpdateId' not in data:
                        return None
                    
                    bids = [(float(price), float(amount)) for price, amount in data['bids'][:depth]]
                    asks = [(float(price), float(amount)) for price, amount in data['asks'][:depth]]
                    
                    order_book = OrderBookData(
                        timestamp=datetime.fromtimestamp(data['lastUpdateId'] / 1000),
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        exchange='binance'
                    )
                    
                    return order_book
                    
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def fetch_order_books(self, symbols: List[str], depth: int = 100) -> List[OrderBookData]:
        """批量获取订单簿数据"""
        order_books = []
        for symbol in symbols:
            order_book = await self.fetch_order_book(symbol, depth)
            if order_book:
                order_books.append(order_book)
        return order_books

class CoinbaseOrderBookFetcher(OrderBookFetcher):
    """Coinbase订单簿获取器"""
    
    def __init__(self):
        self.base_url = "https://api.exchange.coinbase.com"
        
    async def fetch_order_book(self, symbol: str, depth: int = 100) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        try:
            product_id = symbol.replace('/', '-')
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/products/{product_id}/book?level=3") as response:
                    data = await response.json()
                    
                    bids = [(float(level['price']), float(level['size'])) for level in data.get('bids', [])[:depth]]
                    asks = [(float(level['price']), float(level['size'])) for level in data.get('asks', [])[:depth]]
                    
                    order_book = OrderBookData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        exchange='coinbase'
                    )
                    
                    return order_book
                    
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def fetch_order_books(self, symbols: List[str], depth: int = 100) -> List[OrderBookData]:
        """批量获取订单簿数据"""
        order_books = []
        for symbol in symbols:
            order_book = await self.fetch_order_book(symbol, depth)
            if order_book:
                order_books.append(order_book)
        return order_books

class KrakenOrderBookFetcher(OrderBookFetcher):
    """Kraken订单簿获取器"""
    
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
        
    async def fetch_order_book(self, symbol: str, depth: int = 100) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        try:
            pair = symbol.replace('/', '')
            if pair.endswith('T'):
                pair = pair[:-1] + 'Z'
            
            params = {
                'pair': pair,
                'count': depth
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/Depth", params=params) as response:
                    data = await response.json()
                    
                    if data.get('error'):
                        return None
                    
                    result = data['result']
                    pair_key = list(result.keys())[0]
                    pair_data = result[pair_key]
                    
                    bids = [(float(price), float(amount)) for price, amount in pair_data['bids'][:depth]]
                    asks = [(float(price), float(amount)) for price, amount in pair_data['asks'][:depth]]
                    
                    order_book = OrderBookData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        exchange='kraken'
                    )
                    
                    return order_book
                    
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def fetch_order_books(self, symbols: List[str], depth: int = 100) -> List[OrderBookData]:
        """批量获取订单簿数据"""
        order_books = []
        for symbol in symbols:
            order_book = await self.fetch_order_book(symbol, depth)
            if order_book:
                order_books.append(order_book)
        return order_books

class BybitOrderBookFetcher(OrderBookFetcher):
    """Bybit订单簿获取器"""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        
    async def fetch_order_book(self, symbol: str, depth: int = 25) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        try:
            params = {
                'symbol': symbol.replace('/', ''),
                'limit': depth
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v5/market/orderbook", params=params) as response:
                    data = await response.json()
                    
                    if data.get('retCode') != 0:
                        return None
                    
                    result = data['result']
                    bids = [(float(item['price']), float(item['size'])) for item in result.get('b', [])[:depth]]
                    asks = [(float(item['price']), float(item['size'])) for item in result.get('a', [])[:depth]]
                    
                    order_book = OrderBookData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        exchange='bybit'
                    )
                    
                    return order_book
                    
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def fetch_order_books(self, symbols: List[str], depth: int = 25) -> List[OrderBookData]:
        """批量获取订单簿数据"""
        order_books = []
        for symbol in symbols:
            order_book = await self.fetch_order_book(symbol, depth)
            if order_book:
                order_books.append(order_book)
        return order_books

class OKXOrderBookFetcher(OrderBookFetcher):
    """OKX订单簿获取器"""
    
    def __init__(self):
        self.base_url = "https://www.okx.com"
        
    async def fetch_order_book(self, symbol: str, depth: int = 400) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        try:
            params = {
                'instId': symbol
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v5/market/orderbook", params=params) as response:
                    data = await response.json()
                    
                    if data.get('code') != '0':
                        return None
                    
                    result = data['data'][0]
                    bids = [(float(item[0]), float(item[1])) for item in result.get('bids', [])[:depth]]
                    asks = [(float(item[0]), float(item[1])) for item in result.get('asks', [])[:depth]]
                    
                    order_book = OrderBookData(
                        timestamp=datetime.fromtimestamp(int(result['ts']) / 1000),
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        exchange='okx'
                    )
                    
                    return order_book
                    
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def fetch_order_books(self, symbols: List[str], depth: int = 400) -> List[OrderBookData]:
        """批量获取订单簿数据"""
        order_books = []
        for symbol in symbols:
            order_book = await self.fetch_order_book(symbol, depth)
            if order_book:
                order_books.append(order_book)
        return order_books

class KucoinOrderBookFetcher(OrderBookFetcher):
    """Kucoin订单簿获取器"""
    
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        
    async def fetch_order_book(self, symbol: str, depth: int = 100) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        try:
            params = {
                'symbol': symbol.replace('/', '-'),
                'level': 2
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v1/market/orderbook/level2", params=params) as response:
                    data = await response.json()
                    
                    if not data.get('code') == '200000':
                        return None
                    
                    result = data['data']
                    bids = [(float(item[0]), float(item[1])) for item in result.get('bids', [])[:depth]]
                    asks = [(float(item[0]), float(item[1])) for item in result.get('asks', [])[:depth]]
                    
                    order_book = OrderBookData(
                        timestamp=datetime.fromtimestamp(int(result['time']) / 1000),
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        exchange='kucoin'
                    )
                    
                    return order_book
                    
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def fetch_order_books(self, symbols: List[str], depth: int = 100) -> List[OrderBookData]:
        """批量获取订单簿数据"""
        order_books = []
        for symbol in symbols:
            order_book = await self.fetch_order_book(symbol, depth)
            if order_book:
                order_books.append(order_book)
        return order_books

class OrderBookAnalyzer:
    """订单簿分析器"""
    
    def __init__(self):
        self.order_books: Dict[str, OrderBookData] = {}
        
    def calculate_spread(self, order_book: OrderBookData) -> float:
        """计算买卖价差"""
        if not order_book.bids or not order_book.asks:
            return 0
        
        best_bid = order_book.bids[0][0]
        best_ask = order_book.asks[0][0]
        
        return (best_ask - best_bid) / best_bid
    
    def calculate_depth(self, order_book: OrderBookData, levels: int = 5) -> Dict[str, float]:
        """计算订单簿深度"""
        if not order_book.bids or not order_book.asks:
            return {'bid_depth': 0, 'ask_depth': 0}
        
        bid_depth = sum(amount for _, amount in order_book.bids[:levels])
        ask_depth = sum(amount for _, amount in order_book.asks[:levels])
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth
        }
    
    def calculate_imbalance(self, order_book: OrderBookData, levels: int = 10) -> float:
        """计算订单簿失衡"""
        if not order_book.bids or not order_book.asks:
            return 0
        
        bid_volume = sum(amount for _, amount in order_book.bids[:levels])
        ask_volume = sum(amount for _, amount in order_book.asks[:levels])
        
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0
        
        return (bid_volume - ask_volume) / total_volume
    
    def calculate_weighted_price(self, order_book: OrderBookData) -> Tuple[float, float]:
        """计算加权买卖价格"""
        if not order_book.bids or not order_book.asks:
            return 0, 0
        
        bid_volume = sum(amount for _, amount in order_book.bids)
        ask_volume = sum(amount for _, amount in order_book.asks)
        
        bid_weighted = sum(price * amount for price, amount in order_book.bids) / bid_volume if bid_volume > 0 else 0
        ask_weighted = sum(price * amount for price, amount in order_book.asks) / ask_volume if ask_volume > 0 else 0
        
        return bid_weighted, ask_weighted
    
    def analyze_order_book(self, order_book: OrderBookData) -> Dict:
        """全面分析订单簿"""
        spread = self.calculate_spread(order_book)
        depth = self.calculate_depth(order_book)
        imbalance = self.calculate_imbalance(order_book)
        bid_weighted, ask_weighted = self.calculate_weighted_price(order_book)
        
        return {
            'spread': spread,
            'bid_depth': depth['bid_depth'],
            'ask_depth': depth['ask_depth'],
            'imbalance': imbalance,
            'bid_weighted_price': bid_weighted,
            'ask_weighted_price': ask_weighted,
            'best_bid': order_book.bids[0][0] if order_book.bids else 0,
            'best_ask': order_book.asks[0][0] if order_book.asks else 0,
            'mid_price': (order_book.bids[0][0] + order_book.asks[0][0]) / 2 if order_book.bids and order_book.asks else 0
        }

class OrderBookManager:
    """订单簿管理器"""
    
    def __init__(self):
        self.fetchers: Dict[str, OrderBookFetcher] = {}
        self.analyzer = OrderBookAnalyzer()
        self.order_books: Dict[str, OrderBookData] = {}
        
    def register_fetcher(self, exchange: str, fetcher: OrderBookFetcher):
        """注册获取器"""
        self.fetchers[exchange] = fetcher
    
    async def fetch_order_book(self, symbol: str, exchange: str = None, depth: int = 20) -> Optional[OrderBookData]:
        """获取订单簿"""
        if exchange and exchange in self.fetchers:
            order_book = await self.fetchers[exchange].fetch_order_book(symbol, depth)
        else:
            for fetcher in self.fetchers.values():
                order_book = await fetcher.fetch_order_book(symbol, depth)
                if order_book:
                    break
        
        if order_book:
            self.order_books[symbol] = order_book
            
            analysis = self.analyzer.analyze_order_book(order_book)
            order_book.analysis = analysis
            
        return order_book
    
    async def fetch_order_books(self, symbols: List[str], exchange: str = None, depth: int = 20) -> List[OrderBookData]:
        """批量获取订单簿"""
        order_books = []
        for symbol in symbols:
            order_book = await self.fetch_order_book(symbol, exchange, depth)
            if order_book:
                order_books.append(order_book)
        return order_books
    
    def get_order_book(self, symbol: str) -> Optional[OrderBookData]:
        """获取缓存的订单簿"""
        return self.order_books.get(symbol)
    
    def get_all_order_books(self) -> Dict[str, OrderBookData]:
        """获取所有订单簿"""
        return self.order_books
