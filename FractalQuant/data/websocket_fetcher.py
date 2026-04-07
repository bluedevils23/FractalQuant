"""
WebSocket实时数据获取器
"""
import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Callable, Awaitable
from abc import ABC, abstractmethod
import zlib

from .models import BarData, TickData, OrderBookData

logger = logging.getLogger(__name__)

class WebSocketDataFetcher(ABC):
    """WebSocket数据获取器基类"""
    
    def __init__(self):
        self.subscribed_symbols: Dict[str, List[Callable]] = {}
        self.data_callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        self.websocket = None
        
    @abstractmethod
    async def connect(self):
        """连接WebSocket"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """订阅数据"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        pass
    
    @abstractmethod
    def parse_message(self, message: str) -> Optional[Dict]:
        """解析消息"""
        pass
    
    async def start(self):
        """启动WebSocket监听"""
        self.running = True
        while self.running:
            try:
                await self.connect()
                async with self.websocket as ws:
                    await self.handle_connection(ws)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)
    
    async def stop(self):
        """停止WebSocket"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
    
    async def handle_connection(self, ws):
        """处理连接"""
        async for message in ws:
            try:
                data = self.parse_message(message)
                if data:
                    await self.notify_callbacks(data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def notify_callbacks(self, data: Dict):
        """通知回调函数"""
        symbol = data.get('symbol', '')
        if symbol in self.data_callbacks:
            for callback in self.data_callbacks[symbol]:
                await callback(data)

class BinanceWebSocketFetcher(WebSocketDataFetcher):
    """币安WebSocket数据获取器"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.stream_url = "wss://stream.binance.com:9443/stream?streams="
        
    async def connect(self):
        """连接币安WebSocket"""
        self.websocket = websockets.connect(self.base_url)
        
    async def subscribe(self, symbols: List[str]):
        """订阅币安数据流"""
        streams = []
        for symbol in symbols:
            symbol_lower = symbol.lower().replace('/', '')
            streams.append(f"{symbol_lower}@ticker")
            streams.append(f"{symbol_lower}@depth20")
        
        stream_url = f"{self.stream_url}/".join(streams)
        self.websocket = websockets.connect(stream_url)
        
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        pass
    
    def parse_message(self, message: str) -> Optional[Dict]:
        """解析币安消息"""
        try:
            data = json.loads(message)
            
            if 'stream' in data:
                stream = data['stream']
                payload = data['data']
                
                if '@ticker' in stream:
                    return {
                        'type': 'ticker',
                        'symbol': payload.get('s', ''),
                        'price': float(payload.get('c', 0)),
                        'volume': float(payload.get('v', 0)),
                        'timestamp': datetime.fromtimestamp(payload.get('E', 0) / 1000)
                    }
                elif '@depth' in stream:
                    return {
                        'type': 'orderbook',
                        'symbol': payload.get('s', ''),
                        'bids': payload.get('bids', []),
                        'asks': payload.get('asks', []),
                        'timestamp': datetime.fromtimestamp(payload.get('E', 0) / 1000)
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None

class CoinbaseWebSocketFetcher(WebSocketDataFetcher):
    """Coinbase WebSocket数据获取器"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "wss://ws-feed.exchange.coinbase.com"
        
    async def connect(self):
        """连接Coinbase WebSocket"""
        self.websocket = websockets.connect(self.base_url)
        
    async def subscribe(self, symbols: List[str]):
        """订阅Coinbase数据"""
        channels = ['matches', 'level2']
        products = [symbol.replace('/', '-') for symbol in symbols]
        
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": products,
            "channels": channels
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        products = [symbol.replace('/', '-') for symbol in symbols]
        
        unsubscribe_msg = {
            "type": "unsubscribe",
            "product_ids": products,
            "channels": ['matches', 'level2']
        }
        
        await self.websocket.send(json.dumps(unsubscribe_msg))
    
    def parse_message(self, message: str) -> Optional[Dict]:
        """解析Coinbase消息"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'ticker':
                return {
                    'type': 'ticker',
                    'symbol': data.get('product_id', '').replace('-', '/'),
                    'price': float(data.get('price', 0)),
                    'volume': float(data.get('volume', 0)),
                    'timestamp': datetime.fromisoformat(data.get('time', '').replace('Z', '+00:00'))
                }
            elif data.get('type') == 'l2update':
                return {
                    'type': 'orderbook',
                    'symbol': data.get('product_id', '').replace('-', '/'),
                    'changes': data.get('changes', []),
                    'timestamp': datetime.fromisoformat(data.get('time', '').replace('Z', '+00:00'))
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None

class KrakenWebSocketFetcher(WebSocketDataFetcher):
    """Kraken WebSocket数据获取器"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "wss://ws.kraken.com"
        self.subscription_id = 0
        
    async def connect(self):
        """连接Kraken WebSocket"""
        self.websocket = websockets.connect(self.base_url)
        
    async def subscribe(self, symbols: List[str]):
        """订阅Kraken数据"""
        for symbol in symbols:
            pair = self._convert_symbol(symbol)
            
            subscribe_msg = {
                "event": "subscribe",
                "pair": [pair],
                "subscription": {
                    "name": "ticker",
                    "interval": 1
                }
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
            
            self.subscription_id += 1
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            pair = self._convert_symbol(symbol)
            
            unsubscribe_msg = {
                "event": "unsubscribe",
                "pair": [pair],
                "subscription": {
                    "name": "ticker"
                }
            }
            
            await self.websocket.send(json.dumps(unsubscribe_msg))
    
    def _convert_symbol(self, symbol: str) -> str:
        """转换币种符号"""
        symbol = symbol.replace('/', '')
        if symbol.endswith('T'):
            symbol = symbol[:-1] + 'Z'
        return symbol
    
    def parse_message(self, message: str) -> Optional[Dict]:
        """解析Kraken消息"""
        try:
            data = json.loads(message)
            
            if isinstance(data, list) and len(data) > 1:
                if data[1] == 'ticker':
                    ticker_data = data[2]
                    if isinstance(ticker_data, dict):
                        return {
                            'type': 'ticker',
                            'symbol': data[3],
                            'price': float(ticker_data.get('c', [0])[0]),
                            'volume': float(ticker_data.get('v', [0])[0]),
                            'timestamp': datetime.fromtimestamp(ticker_data.get('t', [0])[0])
                        }
                        
            return None
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None

class BybitWebSocketFetcher(WebSocketDataFetcher):
    """Bybit WebSocket数据获取器"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "wss://stream.bybit.com/v5/public/spot"
        
    async def connect(self):
        """连接Bybit WebSocket"""
        self.websocket = websockets.connect(self.base_url)
        
    async def subscribe(self, symbols: List[str]):
        """订阅Bybit数据"""
        for symbol in symbols:
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"ticker.{symbol}"]
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            unsubscribe_msg = {
                "op": "unsubscribe",
                "args": [f"ticker.{symbol}"]
            }
            
            await self.websocket.send(json.dumps(unsubscribe_msg))
    
    def parse_message(self, message: str) -> Optional[Dict]:
        """解析Bybit消息"""
        try:
            data = json.loads(message)
            
            if data.get('op') == 'subscribe':
                return None
                
            if 'data' in data:
                ticker_data = data['data']
                return {
                    'type': 'ticker',
                    'symbol': ticker_data.get('symbol', ''),
                    'price': float(ticker_data.get('lastPrice', 0)),
                    'volume': float(ticker_data.get('volume24h', 0)),
                    'timestamp': datetime.fromtimestamp(ticker_data.get('timestamp', 0) / 1000)
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None

class OKXWebSocketFetcher(WebSocketDataFetcher):
    """OKX WebSocket数据获取器"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "wss://ws.okx.com:8443/ws/v5/public"
        
    async def connect(self):
        """连接OKX WebSocket"""
        self.websocket = websockets.connect(self.base_url)
        
    async def subscribe(self, symbols: List[str]):
        """订阅OKX数据"""
        for symbol in symbols:
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"channel": "ticker", "instId": symbol}]
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            unsubscribe_msg = {
                "op": "unsubscribe",
                "args": [{"channel": "ticker", "instId": symbol}]
            }
            
            await self.websocket.send(json.dumps(unsubscribe_msg))
    
    def parse_message(self, message: str) -> Optional[Dict]:
        """解析OKX消息"""
        try:
            data = json.loads(message)
            
            if data.get('event') == 'subscribe':
                return None
                
            if 'data' in data:
                ticker_data = data['data'][0]
                return {
                    'type': 'ticker',
                    'symbol': ticker_data.get('instId', ''),
                    'price': float(ticker_data.get('last', 0)),
                    'volume': float(ticker_data.get('vol24h', 0)),
                    'timestamp': datetime.fromtimestamp(int(ticker_data.get('ts', 0)) / 1000)
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None

class KucoinWebSocketFetcher(WebSocketDataFetcher):
    """Kucoin WebSocket数据获取器"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "wss://ws-api.kucoin.com:443/ws/v2"
        
    async def connect(self):
        """连接Kucoin WebSocket"""
        self.websocket = websockets.connect(self.base_url)
        
    async def subscribe(self, symbols: List[str]):
        """订阅Kucoin数据"""
        for symbol in symbols:
            subscribe_msg = {
                "type": "subscribe",
                "topic": f"/market/ticker:{symbol}",
                "privateChannel": False
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            unsubscribe_msg = {
                "type": "unsubscribe",
                "topic": f"/market/ticker:{symbol}",
                "privateChannel": False
            }
            
            await self.websocket.send(json.dumps(unsubscribe_msg))
    
    def parse_message(self, message: str) -> Optional[Dict]:
        """解析Kucoin消息"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'welcome':
                return None
                
            if data.get('type') == 'message':
                ticker_data = data.get('data', {})
                return {
                    'type': 'ticker',
                    'symbol': ticker_data.get('symbol', '').replace('-', '/'),
                    'price': float(ticker_data.get('price', 0)),
                    'volume': float(ticker_data.get('size', 0)),
                    'timestamp': datetime.fromtimestamp(int(ticker_data.get('time', 0)) / 1000)
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None
