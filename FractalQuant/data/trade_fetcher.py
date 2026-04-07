"""
交易数据获取器(成交明细)
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class TradeData:
    """交易数据"""
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        price: float,
        volume: float,
        side: str,
        trade_id: str = None,
        exchange: str = None
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.side = side
        self.trade_id = trade_id
        self.exchange = exchange
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'side': self.side,
            'trade_id': self.trade_id,
            'exchange': self.exchange
        }

class TradeFetcher(ABC):
    """交易数据获取器基类"""
    
    @abstractmethod
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """获取交易数据"""
        pass
    
    @abstractmethod
    async def fetch_trades_since(self, symbol: str, since_id: str = None, limit: int = 100) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        pass

class CCXTTradeFetcher(TradeFetcher):
    """基于CCXT的交易数据获取器"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """获取交易数据"""
        try:
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            
            trade_data = []
            for trade in trades:
                side = 'buy' if trade['side'] == 'buy' else 'sell'
                
                trade_obj = TradeData(
                    timestamp=datetime.fromtimestamp(trade['timestamp'] / 1000),
                    symbol=symbol,
                    price=float(trade['price']),
                    volume=float(trade['amount']),
                    side=side,
                    trade_id=trade.get('id'),
                    exchange=self.exchange.id
                )
                trade_data.append(trade_obj)
            
            return trade_data
            
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def fetch_trades_since(self, symbol: str, since_id: str = None, limit: int = 100) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        try:
            params = {}
            if since_id:
                params['sinceId'] = since_id
            
            trades = await self.exchange.fetch_trades(symbol, limit=limit, params=params)
            
            trade_data = []
            for trade in trades:
                side = 'buy' if trade['side'] == 'buy' else 'sell'
                
                trade_obj = TradeData(
                    timestamp=datetime.fromtimestamp(trade['timestamp'] / 1000),
                    symbol=symbol,
                    price=float(trade['price']),
                    volume=float(trade['amount']),
                    side=side,
                    trade_id=trade.get('id'),
                    exchange=self.exchange.id
                )
                trade_data.append(trade_obj)
            
            return trade_data
            
        except Exception as e:
            print(f"Error fetching trades since {since_id} for {symbol}: {e}")
            return []

class BinanceTradeFetcher(TradeFetcher):
    """币安交易数据获取器"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """获取交易数据"""
        try:
            params = {
                'symbol': symbol.replace('/', ''),
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/trades", params=params) as response:
                    data = await response.json()
                    
                    trade_data = []
                    for trade in data:
                        side = 'buy' if trade['isBuyerMaker'] else 'sell'
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromtimestamp(trade['time'] / 1000),
                            symbol=symbol,
                            price=float(trade['price']),
                            volume=float(trade['qty']),
                            side=side,
                            trade_id=str(trade['id']),
                            exchange='binance'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def fetch_trades_since(self, symbol: str, since_id: str = None, limit: int = 100) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        try:
            params = {
                'symbol': symbol.replace('/', ''),
                'limit': limit
            }
            if since_id:
                params['fromId'] = since_id
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/trades", params=params) as response:
                    data = await response.json()
                    
                    trade_data = []
                    for trade in data:
                        side = 'buy' if trade['isBuyerMaker'] else 'sell'
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromtimestamp(trade['time'] / 1000),
                            symbol=symbol,
                            price=float(trade['price']),
                            volume=float(trade['qty']),
                            side=side,
                            trade_id=str(trade['id']),
                            exchange='binance'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades since {since_id} for {symbol}: {e}")
            return []

class CoinbaseTradeFetcher(TradeFetcher):
    """Coinbase交易数据获取器"""
    
    def __init__(self):
        self.base_url = "https://api.exchange.coinbase.com"
        
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """获取交易数据"""
        try:
            product_id = symbol.replace('/', '-')
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/products/{product_id}/trades") as response:
                    data = await response.json()
                    
                    trade_data = []
                    for trade in data[:limit]:
                        side = trade.get('side', 'buy')
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromisoformat(trade['time'].replace('Z', '+00:00')),
                            symbol=symbol,
                            price=float(trade['price']),
                            volume=float(trade['size']),
                            side=side,
                            trade_id=trade['trade_id'],
                            exchange='coinbase'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def fetch_trades_since(self, symbol: str, since_id: str = None, limit: int = 100) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        try:
            product_id = symbol.replace('/', '-')
            
            url = f"{self.base_url}/products/{product_id}/trades"
            if since_id:
                url += f"?after={since_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    
                    trade_data = []
                    for trade in data[:limit]:
                        side = trade.get('side', 'buy')
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromisoformat(trade['time'].replace('Z', '+00:00')),
                            symbol=symbol,
                            price=float(trade['price']),
                            volume=float(trade['size']),
                            side=side,
                            trade_id=trade['trade_id'],
                            exchange='coinbase'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades since {since_id} for {symbol}: {e}")
            return []

class KrakenTradeFetcher(TradeFetcher):
    """Kraken交易数据获取器"""
    
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
        
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """获取交易数据"""
        try:
            pair = symbol.replace('/', '')
            if pair.endswith('T'):
                pair = pair[:-1] + 'Z'
            
            params = {
                'pair': pair,
                'count': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/Trades", params=params) as response:
                    data = await response.json()
                    
                    if data.get('error'):
                        return []
                    
                    result = data['result']
                    pair_key = list(result.keys())[0]
                    trades_data = result[pair_key]
                    
                    trade_data = []
                    for trade in trades_data[:limit]:
                        side = 'buy' if trade[3] == 'b' else 'sell'
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromtimestamp(trade[2]),
                            symbol=symbol,
                            price=float(trade[0]),
                            volume=float(trade[1]),
                            side=side,
                            trade_id=str(trade[2]) + str(trade[0]),
                            exchange='kraken'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def fetch_trades_since(self, symbol: str, since_id: str = None, limit: int = 100) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        return await self.fetch_trades(symbol, limit)

class BybitTradeFetcher(TradeFetcher):
    """Bybit交易数据获取器"""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        
    async def fetch_trades(self, symbol: str, limit: int = 500) -> List[TradeData]:
        """获取交易数据"""
        try:
            params = {
                'symbol': symbol.replace('/', ''),
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v5/market/recent-trade", params=params) as response:
                    data = await response.json()
                    
                    if data.get('retCode') != 0:
                        return []
                    
                    trade_data = []
                    for trade in data['result']['list']:
                        side = trade.get('side', 'Buy')
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromtimestamp(int(trade['time']) / 1000),
                            symbol=symbol,
                            price=float(trade['price']),
                            volume=float(trade['size']),
                            side=side.lower(),
                            trade_id=trade['tradeId'],
                            exchange='bybit'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def fetch_trades_since(self, symbol: str, since_id: str = None, limit: int = 500) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        try:
            params = {
                'symbol': symbol.replace('/', ''),
                'limit': limit
            }
            if since_id:
                params['cursor'] = since_id
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v5/market/recent-trade", params=params) as response:
                    data = await response.json()
                    
                    if data.get('retCode') != 0:
                        return []
                    
                    trade_data = []
                    for trade in data['result']['list']:
                        side = trade.get('side', 'Buy')
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromtimestamp(int(trade['time']) / 1000),
                            symbol=symbol,
                            price=float(trade['price']),
                            volume=float(trade['size']),
                            side=side.lower(),
                            trade_id=trade['tradeId'],
                            exchange='bybit'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades since {since_id} for {symbol}: {e}")
            return []

class OKXTradeFetcher(TradeFetcher):
    """OKX交易数据获取器"""
    
    def __init__(self):
        self.base_url = "https://www.okx.com"
        
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """获取交易数据"""
        try:
            params = {
                'instId': symbol,
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v5/market/trades", params=params) as response:
                    data = await response.json()
                    
                    if data.get('code') != '0':
                        return []
                    
                    trade_data = []
                    for trade in data['data']:
                        side = trade.get('side', 'buy')
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromtimestamp(int(trade['ts']) / 1000),
                            symbol=symbol,
                            price=float(trade['px']),
                            volume=float(trade['sz']),
                            side=side,
                            trade_id=trade['tradeId'],
                            exchange='okx'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def fetch_trades_since(self, symbol: str, since_id: str = None, limit: int = 100) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        return await self.fetch_trades(symbol, limit)

class KucoinTradeFetcher(TradeFetcher):
    """Kucoin交易数据获取器"""
    
    def __init__(self):
        self.base_url = "https://api.kucoin.com"
        
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """获取交易数据"""
        try:
            params = {
                'symbol': symbol.replace('/', '-'),
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v1/market/trades", params=params) as response:
                    data = await response.json()
                    
                    if not data.get('code') == '200000':
                        return []
                    
                    trade_data = []
                    for trade in data['data']['items'][:limit]:
                        side = trade.get('side', 'buy')
                        
                        trade_obj = TradeData(
                            timestamp=datetime.fromtimestamp(int(trade['time']) / 1000),
                            symbol=symbol,
                            price=float(trade['price']),
                            volume=float(trade['size']),
                            side=side,
                            trade_id=trade['tradeId'],
                            exchange='kucoin'
                        )
                        trade_data.append(trade_obj)
                    
                    return trade_data
                    
        except Exception as e:
            print(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def fetch_trades_since(self, symbol: str, since_id: str = None, limit: int = 100) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        return await self.fetch_trades(symbol, limit)

class TradeAnalyzer:
    """交易数据分析器"""
    
    def __init__(self):
        self.trades: List[TradeData] = []
        
    def calculate_volume_profile(self, trades: List[TradeData], num_bins: int = 10) -> Dict:
        """计算成交量分布"""
        if not trades:
            return {}
        
        prices = [trade.price for trade in trades]
        volumes = [trade.volume for trade in trades]
        
        min_price = min(prices)
        max_price = max(prices)
        bin_size = (max_price - min_price) / num_bins
        
        volume_profile = {}
        for i in range(num_bins):
            bin_start = min_price + i * bin_size
            bin_end = bin_start + bin_size
            
            bin_volume = sum(
                volume for price, volume in zip(prices, volumes)
                if bin_start <= price < bin_end
            )
            
            volume_profile[f'{bin_start:.2f}-{bin_end:.2f}'] = bin_volume
        
        return volume_profile
    
    def calculate_buy_sell_ratio(self, trades: List[TradeData]) -> float:
        """计算买卖比例"""
        if not trades:
            return 0
        
        buy_volume = sum(trade.volume for trade in trades if trade.side == 'buy')
        sell_volume = sum(trade.volume for trade in trades if trade.side == 'sell')
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0
        
        return buy_volume / sell_volume if sell_volume > 0 else float('inf')
    
    def calculate_trade_size_distribution(self, trades: List[TradeData]) -> Dict:
        """计算交易规模分布"""
        if not trades:
            return {}
        
        volumes = [trade.volume for trade in trades]
        
        avg_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        min_volume = np.min(volumes)
        max_volume = np.max(volumes)
        
        return {
            'avg': avg_volume,
            'std': std_volume,
            'min': min_volume,
            'max': max_volume,
            'median': np.median(volumes)
        }
    
    def calculate_trade_velocity(self, trades: List[TradeData], window_seconds: int = 60) -> float:
        """计算交易速度"""
        if len(trades) < 2:
            return 0
        
        time_diffs = []
        for i in range(1, len(trades)):
            diff = (trades[i].timestamp - trades[i-1].timestamp).total_seconds()
            time_diffs.append(diff)
        
        avg_diff = np.mean(time_diffs)
        
        if avg_diff == 0:
            return 0
        
        return window_seconds / avg_diff
    
    def analyze_trades(self, trades: List[TradeData]) -> Dict:
        """全面分析交易数据"""
        if not trades:
            return {}
        
        volume_profile = self.calculate_volume_profile(trades)
        buy_sell_ratio = self.calculate_buy_sell_ratio(trades)
        trade_size_dist = self.calculate_trade_size_distribution(trades)
        trade_velocity = self.calculate_trade_velocity(trades)
        
        prices = [trade.price for trade in trades]
        start_price = prices[0]
        end_price = prices[-1]
        price_change = (end_price - start_price) / start_price if start_price > 0 else 0
        
        return {
            'total_trades': len(trades),
            'total_volume': sum(trade.volume for trade in trades),
            'buy_volume': sum(trade.volume for trade in trades if trade.side == 'buy'),
            'sell_volume': sum(trade.volume for trade in trades if trade.side == 'sell'),
            'buy_sell_ratio': buy_sell_ratio,
            'price_change': price_change,
            'trade_velocity': trade_velocity,
            'avg_trade_size': trade_size_dist['avg'],
            'trade_size_std': trade_size_dist['std'],
            'min_trade_size': trade_size_dist['min'],
            'max_trade_size': trade_size_dist['max']
        }

class TradeManager:
    """交易数据管理器"""
    
    def __init__(self):
        self.fetchers: Dict[str, TradeFetcher] = {}
        self.analyzer = TradeAnalyzer()
        self.trades: Dict[str, List[TradeData]] = {}
        
    def register_fetcher(self, exchange: str, fetcher: TradeFetcher):
        """注册获取器"""
        self.fetchers[exchange] = fetcher
    
    async def fetch_trades(self, symbol: str, exchange: str = None, limit: int = 100) -> List[TradeData]:
        """获取交易数据"""
        if exchange and exchange in self.fetchers:
            trades = await self.fetchers[exchange].fetch_trades(symbol, limit)
        else:
            for fetcher in self.fetchers.values():
                trades = await fetcher.fetch_trades(symbol, limit)
                if trades:
                    break
        
        if trades:
            if symbol not in self.trades:
                self.trades[symbol] = []
            self.trades[symbol].extend(trades)
            
            analysis = self.analyzer.analyze_trades(trades)
            self.trades[f'{symbol}_analysis'] = analysis
            
        return trades
    
    async def fetch_trades_since(self, symbol: str, since_id: str = None, exchange: str = None, limit: int = 100) -> List[TradeData]:
        """获取指定ID之后的交易数据"""
        if exchange and exchange in self.fetchers:
            trades = await self.fetchers[exchange].fetch_trades_since(symbol, since_id, limit)
        else:
            for fetcher in self.fetchers.values():
                trades = await fetcher.fetch_trades_since(symbol, since_id, limit)
                if trades:
                    break
        
        if trades:
            if symbol not in self.trades:
                self.trades[symbol] = []
            self.trades[symbol].extend(trades)
            
            analysis = self.analyzer.analyze_trades(trades)
            self.trades[f'{symbol}_analysis'] = analysis
            
        return trades
    
    def get_trades(self, symbol: str) -> List[TradeData]:
        """获取缓存的交易数据"""
        return self.trades.get(symbol, [])
    
    def get_analysis(self, symbol: str) -> Dict:
        """获取交易分析结果"""
        return self.trades.get(f'{symbol}_analysis', {})
    
    def get_all_trades(self) -> Dict[str, List[TradeData]]:
        """获取所有交易数据"""
        return {k: v for k, v in self.trades.items() if not k.endswith('_analysis')}
