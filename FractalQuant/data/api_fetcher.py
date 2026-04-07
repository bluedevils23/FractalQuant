"""
API数据获取器(支持多种数据源)
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from abc import ABC, abstractmethod

from .models import BarData, TickData

logger = logging.getLogger(__name__)

class APIDataFetcher(ABC):
    """API数据获取器基类"""
    
    @abstractmethod
    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[BarData]:
        """获取历史数据"""
        pass
    
    @abstractmethod
    def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时数据"""
        pass

class AlphaVantageFetcher(APIDataFetcher):
    """AlphaVantage数据获取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1day', 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> List[BarData]:
        """获取历史数据"""
        try:
            function = 'TIME_SERIES_DAILY_ADJUSTED'
            if timeframe == '1hour':
                function = 'TIME_SERIES_INTRADAY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': '60min',
                    'apikey': self.api_key
                }
            else:
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.api_key
                }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series' not in data:
                logger.error(f"Invalid response for {symbol}")
                return []
            
            time_series_key = [k for k in data.keys() if 'Time Series' in k][0]
            time_series = data[time_series_key]
            
            bars = []
            for date_str, OHLC in time_series.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                
                bar = BarData(
                    timestamp=date,
                    symbol=symbol,
                    open=float(OHLC['1. open']),
                    high=float(OHLC['2. high']),
                    low=float(OHLC['3. low']),
                    close=float(OHLC['4. close']),
                    volume=int(OHLC['6. volume'])
                )
                bars.append(bar)
            
            logger.info(f"Fetched {len(bars)} bars from AlphaVantage for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching from AlphaVantage for {symbol}: {e}")
            return []
    
    def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时数据"""
        ticks = []
        try:
            for symbol in symbols:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    tick = TickData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        price=float(quote.get('05. price', 0)),
                        volume=int(quote.get('06. volume', 0)),
                        bid_price=float(quote.get('08. bid price', 0)),
                        ask_price=float(quote.get('09. ask price', 0))
                    )
                    ticks.append(tick)
                    
        except Exception as e:
            logger.error(f"Error fetching realtime data from AlphaVantage: {e}")
        
        return ticks

class YahooFinanceFetcher(APIDataFetcher):
    """雅虎财经数据获取器"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        
    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1d', 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> List[BarData]:
        """获取历史数据"""
        try:
            period1 = int(start_date.timestamp()) if start_date else int((datetime.now() - timedelta(days=365)).timestamp())
            period2 = int(end_date.timestamp()) if end_date else int(datetime.now().timestamp())
            
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '60m',
                '1d': '1d',
                '1wk': '1wk',
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            params = {
                'symbols': symbol,
                'period1': period1,
                'period2': period2,
                'interval': interval
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                return []
            
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            indicators = result['indicators']['quote'][0]
            
            bars = []
            for i, ts in enumerate(timestamps):
                if ts is None:
                    continue
                    
                bar = BarData(
                    timestamp=datetime.fromtimestamp(ts),
                    symbol=symbol,
                    open=indicators['open'][i] if i < len(indicators['open']) else None,
                    high=indicators['high'][i] if i < len(indicators['high']) else None,
                    low=indicators['low'][i] if i < len(indicators['low']) else None,
                    close=indicators['close'][i] if i < len(indicators['close']) else None,
                    volume=indicators['volume'][i] if i < len(indicators['volume']) else None
                )
                bars.append(bar)
            
            logger.info(f"Fetched {len(bars)} bars from Yahoo Finance for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance for {symbol}: {e}")
            return []
    
    def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时数据"""
        ticks = []
        try:
            symbols_str = ','.join(symbols)
            params = {
                'symbols': symbols_str,
                'fields': 'symbol,regularMarketPrice,regularMarketVolume,bid,ask'
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' in data and data['chart']['result']:
                for result in data['chart']['result']:
                    if result['quote']:
                        quote = result['quote'][0]
                        tick = TickData(
                            timestamp=datetime.now(),
                            symbol=quote.get('symbol', ''),
                            price=quote.get('regularMarketPrice', 0),
                            volume=quote.get('regularMarketVolume', 0),
                            bid_price=quote.get('bid', 0),
                            ask_price=quote.get('ask', 0)
                        )
                        ticks.append(tick)
                        
        except Exception as e:
            logger.error(f"Error fetching realtime data from Yahoo Finance: {e}")
        
        return ticks

class FinnhubFetcher(APIDataFetcher):
    """Finnhub数据获取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        
    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1d', 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> List[BarData]:
        """获取历史数据"""
        try:
            period_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '1d': 'D',
            }
            
            resolution = period_map.get(timeframe, 'D')
            period1 = int(start_date.timestamp()) if start_date else int((datetime.now() - timedelta(days=365)).timestamp())
            period2 = int(end_date.timestamp()) if end_date else int(datetime.now().timestamp())
            
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': period1,
                'to': period2,
                'token': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/stock/candle", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') != 'ok':
                return []
            
            bars = []
            for i in range(len(data['t'])):
                bar = BarData(
                    timestamp=datetime.fromtimestamp(data['t'][i]),
                    symbol=symbol,
                    open=data['o'][i],
                    high=data['h'][i],
                    low=data['l'][i],
                    close=data['c'][i],
                    volume=data['v'][i]
                )
                bars.append(bar)
            
            logger.info(f"Fetched {len(bars)} bars from Finnhub for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching from Finnhub for {symbol}: {e}")
            return []
    
    def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时数据"""
        ticks = []
        try:
            for symbol in symbols:
                params = {
                    'symbol': symbol,
                    'token': self.api_key
                }
                
                response = requests.get(f"{self.base_url}/quote", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data and 'c' in data:
                    tick = TickData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        price=data.get('c', 0),
                        volume=data.get('v', 0),
                        bid_price=data.get('b', 0),
                        ask_price=data.get('a', 0)
                    )
                    ticks.append(tick)
                    
        except Exception as e:
            logger.error(f"Error fetching realtime data from Finnhub: {e}")
        
        return ticks

class PolygonFetcher(APIDataFetcher):
    """Polygon.io数据获取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
        
    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = 'day', 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> List[BarData]:
        """获取历史数据"""
        try:
            from_date = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            agg_map = {
                '1m': 'minute',
                '5m': 'minute',
                '15m': 'minute',
                '30m': 'minute',
                '1h': 'minute',
                'day': 'day',
            }
            
            timespan = agg_map.get(timeframe, 'day')
            
            params = {
                'apiKey': self.api_key,
                'limit': 5000
            }
            
            response = requests.get(
                f"{self.base_url}/aggs/ticker/{symbol}/range/{timespan}/1/1",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('resultsCount', 0) == 0:
                return []
            
            bars = []
            for result in data.get('results', []):
                bar = BarData(
                    timestamp=datetime.fromtimestamp(result['t'] / 1000),
                    symbol=symbol,
                    open=result['o'],
                    high=result['h'],
                    low=result['l'],
                    close=result['c'],
                    volume=result['v']
                )
                bars.append(bar)
            
            logger.info(f"Fetched {len(bars)} bars from Polygon for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching from Polygon for {symbol}: {e}")
            return []
    
    def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时数据"""
        ticks = []
        try:
            for symbol in symbols:
                params = {
                    'apiKey': self.api_key
                }
                
                response = requests.get(f"{self.base_url}/last/trade/{symbol}", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'OK' and 'results' in data:
                    result = data['results']
                    tick = TickData(
                        timestamp=datetime.fromtimestamp(result['t'] / 1000),
                        symbol=symbol,
                        price=result['p'],
                        volume=result['s'],
                        bid_price=result.get('b', 0),
                        ask_price=result.get('a', 0)
                    )
                    ticks.append(tick)
                    
        except Exception as e:
            logger.error(f"Error fetching realtime data from Polygon: {e}")
        
        return ticks

class ExchangeRateAPIFetcher(APIDataFetcher):
    """汇率数据获取器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://v6.exchangerate-api.com/v6"
        
    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = 'day', 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> List[BarData]:
        """获取历史汇率数据"""
        try:
            base_currency = symbol[:3]
            target_currency = symbol[3:]
            
            from_date = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            response = requests.get(
                f"{self.base_url}/{self.api_key}/history/{base_currency}/{target_currency}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('result') != 'success':
                return []
            
            rates = data.get('rates', {})
            
            bars = []
            for date_str, rate_data in rates.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                
                bar = BarData(
                    timestamp=date,
                    symbol=symbol,
                    open=rate_data['open'],
                    high=rate_data['high'],
                    low=rate_data['low'],
                    close=rate_data['close'],
                    volume=1
                )
                bars.append(bar)
            
            logger.info(f"Fetched {len(bars)} bars from ExchangeRateAPI for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching from ExchangeRateAPI for {symbol}: {e}")
            return []
    
    def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时汇率数据"""
        ticks = []
        try:
            for symbol in symbols:
                base_currency = symbol[:3]
                target_currency = symbol[3:]
                
                response = requests.get(
                    f"{self.base_url}/{self.api_key}/latest/{base_currency}",
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('result') == 'success':
                    rates = data.get('conversion_rates', {})
                    if target_currency in rates:
                        tick = TickData(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            price=rates[target_currency],
                            volume=1,
                            bid_price=rates[target_currency],
                            ask_price=rates[target_currency]
                        )
                        ticks.append(tick)
                        
        except Exception as e:
            logger.error(f"Error fetching realtime data from ExchangeRateAPI: {e}")
        
        return ticks

class CryptoCompareFetcher(APIDataFetcher):
    """加密货币历史数据获取器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://min-api.cryptocompare.com/data"
        
    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = 'day', 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> List[BarData]:
        """获取历史加密货币数据"""
        try:
            fsym = symbol[:3]
            tsym = symbol[3:]
            
            limit = 2000
            to_ts = int(end_date.timestamp()) if end_date else int(datetime.now().timestamp())
            
            params = {
                'fsym': fsym,
                'tsym': tsym,
                'limit': limit,
                'toTs': to_ts,
                'api_key': self.api_key
            }
            
            if timeframe == 'minute':
                url = f"{self.base_url}/v2/histominute"
                params['limit'] = 2000
            elif timeframe == 'hour':
                url = f"{self.base_url}/v2/histohour"
                params['limit'] = 2000
            else:
                url = f"{self.base_url}/v2/histoday"
                params['limit'] = 2000
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('Response') != 'Success':
                return []
            
            bars = []
            for item in data.get('Data', []):
                bar = BarData(
                    timestamp=datetime.fromtimestamp(item['time']),
                    symbol=symbol,
                    open=item['open'],
                    high=item['high'],
                    low=item['low'],
                    close=item['close'],
                    volume=item['volumefrom']
                )
                bars.append(bar)
            
            logger.info(f"Fetched {len(bars)} bars from CryptoCompare for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching from CryptoCompare for {symbol}: {e}")
            return []
    
    def fetch_realtime_data(self, symbols: List[str]) -> List[TickData]:
        """获取实时加密货币数据"""
        ticks = []
        try:
            for symbol in symbols:
                fsym = symbol[:3]
                tsym = symbol[3:]
                
                params = {
                    'fsym': fsym,
                    'tsyms': tsym,
                    'api_key': self.api_key
                }
                
                response = requests.get(f"{self.base_url}/price", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data.get('Response') == 'Success':
                    price = data.get('RAW', {}).get(tsym, {}).get('PRICE', 0)
                    volume = data.get('RAW', {}).get(tsym, {}).get('VOLUME24HOURTO', 0)
                    
                    tick = TickData(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        price=price,
                        volume=volume,
                        bid_price=price,
                        ask_price=price
                    )
                    ticks.append(tick)
                    
        except Exception as e:
            logger.error(f"Error fetching realtime data from CryptoCompare: {e}")
        
        return ticks
