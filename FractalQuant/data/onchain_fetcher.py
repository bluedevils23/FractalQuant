"""
链上数据获取器(区块链数据)
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class OnChainData:
    """链上数据"""
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        metric: str,
        value: float,
        unit: str = None,
        exchange: str = None
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.metric = metric
        self.value = value
        self.unit = unit
        self.exchange = exchange
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'metric': self.metric,
            'value': self.value,
            'unit': self.unit,
            'exchange': self.exchange
        }

class OnChainFetcher(ABC):
    """链上数据获取器基类"""
    
    @abstractmethod
    async def fetch_metric(self, symbol: str, metric: str, start_date: datetime = None, end_date: datetime = None) -> List[OnChainData]:
        """获取链上指标数据"""
        pass

class GlassnodeFetcher(OnChainFetcher):
    """Glassnode链上数据获取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.glassnode.com/v1/metrics"
        
    async def fetch_metric(self, symbol: str, metric: str, start_date: datetime = None, end_date: datetime = None) -> List[OnChainData]:
        """获取链上指标数据"""
        try:
            from_date = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'a': symbol,
                'm': metric,
                's': int(datetime.strptime(from_date, '%Y-%m-%d').timestamp()),
                'u': int(datetime.strptime(to_date, '%Y-%m-%d').timestamp()),
                'api_key': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/endpoint", params=params) as response:
                    data = await response.json()
                    
                    on_chain_data = []
                    for item in data:
                        on_chain_obj = OnChainData(
                            timestamp=datetime.fromtimestamp(item['t']),
                            symbol=symbol,
                            metric=metric,
                            value=float(item['v']),
                            exchange='glassnode'
                        )
                        on_chain_data.append(on_chain_obj)
                    
                    return on_chain_data
                    
        except Exception as e:
            print(f"Error fetching {metric} for {symbol}: {e}")
            return []

class CryptoQuantFetcher(OnChainFetcher):
    """CryptoQuant链上数据获取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.cryptoquant.com/v1"
        
    async def fetch_metric(self, symbol: str, metric: str, start_date: datetime = None, end_date: datetime = None) -> List[OnChainData]:
        """获取链上指标数据"""
        try:
            from_date = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'start_date': from_date,
                'end_date': to_date
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/{symbol}/{metric}", params=params, headers=headers) as response:
                    data = await response.json()
                    
                    on_chain_data = []
                    for item in data.get('data', []):
                        on_chain_obj = OnChainData(
                            timestamp=datetime.fromisoformat(item['date'].replace('Z', '+00:00')),
                            symbol=symbol,
                            metric=metric,
                            value=float(item['value']),
                            exchange='cryptoquant'
                        )
                        on_chain_data.append(on_chain_obj)
                    
                    return on_chain_data
                    
        except Exception as e:
            print(f"Error fetching {metric} for {symbol}: {e}")
            return []

class NansenFetcher(OnChainFetcher):
    """Nansen链上数据获取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nansen.ai/v1"
        
    async def fetch_metric(self, symbol: str, metric: str, start_date: datetime = None, end_date: datetime = None) -> List[OnChainData]:
        """获取链上指标数据"""
        try:
            from_date = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'start_date': from_date,
                'end_date': to_date
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/{symbol}/{metric}", params=params, headers=headers) as response:
                    data = await response.json()
                    
                    on_chain_data = []
                    for item in data.get('data', []):
                        on_chain_obj = OnChainData(
                            timestamp=datetime.fromisoformat(item['date'].replace('Z', '+00:00')),
                            symbol=symbol,
                            metric=metric,
                            value=float(item['value']),
                            exchange='nansen'
                        )
                        on_chain_data.append(on_chain_obj)
                    
                    return on_chain_data
                    
        except Exception as e:
            print(f"Error fetching {metric} for {symbol}: {e}")
            return []

class EtherscanFetcher(OnChainFetcher):
    """Etherscan以太坊链上数据获取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        
    async def fetch_metric(self, symbol: str, metric: str, start_date: datetime = None, end_date: datetime = None) -> List[OnChainData]:
        """获取以太坊链上指标数据"""
        try:
            params = {
                'module': 'stats',
                'action': metric,
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    if data.get('status') != '1':
                        return []
                    
                    result = data.get('result', {})
                    
                    on_chain_data = []
                    for key, value in result.items():
                        if key.startswith('ethSupply'):
                            on_chain_obj = OnChainData(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                metric=key,
                                value=float(value),
                                unit='ether',
                                exchange='etherscan'
                            )
                            on_chain_data.append(on_chain_obj)
                    
                    return on_chain_data
                    
        except Exception as e:
            print(f"Error fetching {metric} for {symbol}: {e}")
            return []

class BlockchairFetcher(OnChainFetcher):
    """Blockchair链上数据获取器"""
    
    def __init__(self):
        self.base_url = "https://api.blockchair.com"
        
    async def fetch_metric(self, symbol: str, metric: str, start_date: datetime = None, end_date: datetime = None) -> List[OnChainData]:
        """获取链上指标数据"""
        try:
            from_date = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'start_date': from_date,
                'end_date': to_date
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/{symbol}/stats", params=params) as response:
                    data = await response.json()
                    
                    on_chain_data = []
                    for key, value in data.get('data', {}).items():
                        if isinstance(value, (int, float)):
                            on_chain_obj = OnChainData(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                metric=key,
                                value=float(value),
                                exchange='blockchair'
                            )
                            on_chain_data.append(on_chain_obj)
                    
                    return on_chain_data
                    
        except Exception as e:
            print(f"Error fetching {metric} for {symbol}: {e}")
            return []

class TokenBalanceFetcher(OnChainFetcher):
    """代币余额获取器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.blockcypher.com/v1"
        
    async def fetch_token_balance(self, symbol: str, address: str) -> OnChainData:
        """获取代币余额"""
        try:
            params = {}
            if self.api_key:
                params['token'] = self.api_key
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/btc/main/addrs/{address}/balance", params=params) as response:
                    data = await response.json()
                    
                    on_chain_obj = OnChainData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        metric='balance',
                        value=float(data.get('balance', 0)) / 1e8,
                        unit='btc',
                        exchange='blockcypher'
                    )
                    
                    return on_chain_obj
                    
        except Exception as e:
            print(f"Error fetching balance for {address}: {e}")
            return None
    
    async def fetch_token_supply(self, symbol: str) -> OnChainData:
        """获取代币供应量"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/btc/main") as response:
                    data = await response.json()
                    
                    on_chain_obj = OnChainData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        metric='total_supply',
                        value=float(data.get('total_supply', 0)) / 1e8,
                        unit='btc',
                        exchange='blockcypher'
                    )
                    
                    return on_chain_obj
                    
        except Exception as e:
            print(f"Error fetching supply for {symbol}: {e}")
            return None

class MempoolFetcher(OnChainFetcher):
    """内存池数据获取器"""
    
    def __init__(self):
        self.base_url = "https://mempool.space/api"
        
    async def fetch_mempool_info(self, symbol: str) -> List[OnChainData]:
        """获取内存池信息"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/mempool") as response:
                    data = await response.json()
                    
                    on_chain_data = []
                    
                    for metric, value in data.items():
                        if isinstance(value, (int, float)):
                            on_chain_obj = OnChainData(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                metric=metric,
                                value=float(value),
                                exchange='mempool'
                            )
                            on_chain_data.append(on_chain_obj)
                    
                    return on_chain_data
                    
        except Exception as e:
            print(f"Error fetching mempool info: {e}")
            return []
    
    async def fetch_fee_estimates(self, symbol: str) -> List[OnChainData]:
        """获取费用估算"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v1/fees/recommended") as response:
                    data = await response.json()
                    
                    on_chain_data = []
                    
                    for fee_type, fee_value in data.items():
                        on_chain_obj = OnChainData(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            metric=f'fee_{fee_type}',
                            value=float(fee_value),
                            unit='sats/vB',
                            exchange='mempool'
                        )
                        on_chain_data.append(on_chain_obj)
                    
                    return on_chain_data
                    
        except Exception as e:
            print(f"Error fetching fee estimates: {e}")
            return []

class OnChainAnalyzer:
    """链上数据分析器"""
    
    def __init__(self):
        self.data: Dict[str, List[OnChainData]] = {}
        
    def calculate_moving_average(self, data: List[OnChainData], window: int = 7) -> List[float]:
        """计算移动平均"""
        if len(data) < window:
            return []
        
        values = [d.value for d in data]
        return [np.mean(values[i:i+window]) for i in range(len(values) - window + 1)]
    
    def calculate_volatility(self, data: List[OnChainData], window: int = 7) -> List[float]:
        """计算波动率"""
        if len(data) < window:
            return []
        
        values = [d.value for d in data]
        return [np.std(values[i:i+window]) for i in range(len(values) - window + 1)]
    
    def calculate_growth_rate(self, data: List[OnChainData]) -> List[float]:
        """计算增长率"""
        if len(data) < 2:
            return []
        
        values = [d.value for d in data]
        return [(values[i] - values[i-1]) / values[i-1] if values[i-1] > 0 else 0 for i in range(1, len(values))]
    
    def analyze_on_chain_data(self, data: List[OnChainData]) -> Dict:
        """全面分析链上数据"""
        if not data:
            return {}
        
        values = [d.value for d in data]
        
        return {
            'current_value': values[-1] if values else 0,
            'avg_value': np.mean(values),
            'min_value': np.min(values),
            'max_value': np.max(values),
            'volatility': np.std(values),
            'growth_rate': (values[-1] - values[0]) / values[0] if values[0] > 0 else 0,
            'trend': 'up' if values[-1] > values[0] else 'down' if values[-1] < values[0] else 'stable'
        }

class OnChainManager:
    """链上数据管理器"""
    
    def __init__(self):
        self.fetchers: Dict[str, OnChainFetcher] = {}
        self.analyzer = OnChainAnalyzer()
        self.data: Dict[str, List[OnChainData]] = {}
        
    def register_fetcher(self, source: str, fetcher: OnChainFetcher):
        """注册获取器"""
        self.fetchers[source] = fetcher
    
    async def fetch_metric(self, symbol: str, metric: str, source: str = None, start_date: datetime = None, end_date: datetime = None) -> List[OnChainData]:
        """获取链上指标"""
        if source and source in self.fetchers:
            data = await self.fetchers[source].fetch_metric(symbol, metric, start_date, end_date)
        else:
            for fetcher in self.fetchers.values():
                data = await fetcher.fetch_metric(symbol, metric, start_date, end_date)
                if data:
                    break
        
        if data:
            key = f"{symbol}_{metric}"
            if key not in self.data:
                self.data[key] = []
            self.data[key].extend(data)
            
            analysis = self.analyzer.analyze_on_chain_data(data)
            self.data[f"{key}_analysis"] = analysis
            
        return data
    
    def get_data(self, symbol: str, metric: str) -> List[OnChainData]:
        """获取缓存的链上数据"""
        key = f"{symbol}_{metric}"
        return self.data.get(key, [])
    
    def get_analysis(self, symbol: str, metric: str) -> Dict:
        """获取链上数据分析结果"""
        key = f"{symbol}_{metric}"
        return self.data.get(f"{key}_analysis", {})
    
    def get_all_data(self) -> Dict[str, List[OnChainData]]:
        """获取所有链上数据"""
        return {k: v for k, v in self.data.items() if not k.endswith('_analysis')}
