"""
数据聚合器(多源数据整合)
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any
from abc import ABC, abstractmethod
import logging

from .models import BarData, TickData, OrderBookData
from .api_fetcher import AlphaVantageFetcher, YahooFinanceFetcher, FinnhubFetcher, PolygonFetcher, ExchangeRateAPIFetcher, CryptoCompareFetcher
from .websocket_fetcher import BinanceWebSocketFetcher, CoinbaseWebSocketFetcher, KrakenWebSocketFetcher
from .orderbook_fetcher import BinanceOrderBookFetcher, CoinbaseOrderBookFetcher, KrakenOrderBookFetcher, OrderBookAnalyzer
from .trade_fetcher import BinanceTradeFetcher, CoinbaseTradeFetcher, KrakenTradeFetcher, TradeAnalyzer
from .onchain_fetcher import GlassnodeFetcher, CryptoQuantFetcher, EtherscanFetcher, MempoolFetcher, OnChainAnalyzer
from .alternative_fetcher import NewsFetcher, TwitterFetcher, RedditFetcher, SentimentAnalyzer

logger = logging.getLogger(__name__)

class DataAggregator:
    """数据聚合器"""
    
    def __init__(self):
        self.api_fetchers: Dict[str, Any] = {}
        self.websocket_fetchers: Dict[str, Any] = {}
        self.orderbook_fetchers: Dict[str, Any] = {}
        self.trade_fetchers: Dict[str, Any] = {}
        self.onchain_fetchers: Dict[str, Any] = {}
        self.alternative_fetchers: Dict[str, Any] = {}
        
        self.api_data: Dict[str, List[BarData]] = {}
        self.websocket_data: Dict[str, List[TickData]] = {}
        self.orderbook_data: Dict[str, List[OrderBookData]] = {}
        self.trade_data: Dict[str, List[Any]] = {}
        self.onchain_data: Dict[str, List[Any]] = {}
        self.alternative_data: Dict[str, List[Any]] = {}
        
        self.data_timestamps: Dict[str, datetime] = {}
        
    def register_api_fetcher(self, name: str, fetcher: Any):
        """注册API获取器"""
        self.api_fetchers[name] = fetcher
    
    def register_websocket_fetcher(self, name: str, fetcher: Any):
        """注册WebSocket获取器"""
        self.websocket_fetchers[name] = fetcher
    
    def register_orderbook_fetcher(self, name: str, fetcher: Any):
        """注册订单簿获取器"""
        self.orderbook_fetchers[name] = fetcher
    
    def register_trade_fetcher(self, name: str, fetcher: Any):
        """注册交易获取器"""
        self.trade_fetchers[name] = fetcher
    
    def register_onchain_fetcher(self, name: str, fetcher: Any):
        """注册链上数据获取器"""
        self.onchain_fetchers[name] = fetcher
    
    def register_alternative_fetcher(self, name: str, fetcher: Any):
        """注册另类数据获取器"""
        self.alternative_fetchers[name] = fetcher
    
    async def fetch_market_data(
        self, 
        symbol: str, 
        timeframe: str = '1d', 
        start_date: datetime = None, 
        end_date: datetime = None,
        sources: List[str] = None
    ) -> Dict[str, List[BarData]]:
        """从多个源获取市场数据"""
        all_data = {}
        
        if sources is None:
            sources = list(self.api_fetchers.keys())
        
        for source in sources:
            if source in self.api_fetchers:
                try:
                    data = await self.api_fetchers[source].fetch_historical_data(symbol, timeframe, start_date, end_date)
                    if data:
                        all_data[source] = data
                        self.api_data[f"{symbol}_{source}"] = data
                        self.data_timestamps[f"{symbol}_{source}"] = datetime.now()
                except Exception as e:
                    logger.error(f"Error fetching from {source}: {e}")
        
        return all_data
    
    async def fetch_order_books(
        self, 
        symbols: List[str], 
        depth: int = 20,
        sources: List[str] = None
    ) -> Dict[str, List[OrderBookData]]:
        """从多个源获取订单簿数据"""
        all_data = {}
        
        if sources is None:
            sources = list(self.orderbook_fetchers.keys())
        
        for source in sources:
            if source in self.orderbook_fetchers:
                try:
                    data = await self.orderbook_fetchers[source].fetch_order_books(symbols, depth)
                    if data:
                        all_data[source] = data
                        for order_book in data:
                            key = f"{order_book.symbol}_{source}"
                            if key not in self.orderbook_data:
                                self.orderbook_data[key] = []
                            self.orderbook_data[key].append(order_book)
                            self.data_timestamps[key] = datetime.now()
                except Exception as e:
                    logger.error(f"Error fetching order books from {source}: {e}")
        
        return all_data
    
    async def fetch_trades(
        self, 
        symbols: List[str], 
        limit: int = 100,
        sources: List[str] = None
    ) -> Dict[str, List[Any]]:
        """从多个源获取交易数据"""
        all_data = {}
        
        if sources is None:
            sources = list(self.trade_fetchers.keys())
        
        for source in sources:
            if source in self.trade_fetchers:
                try:
                    for symbol in symbols:
                        data = await self.trade_fetchers[source].fetch_trades(symbol, limit)
                        if data:
                            if source not in all_data:
                                all_data[source] = []
                            all_data[source].extend(data)
                            
                            key = f"{symbol}_{source}"
                            if key not in self.trade_data:
                                self.trade_data[key] = []
                            self.trade_data[key].extend(data)
                            self.data_timestamps[key] = datetime.now()
                except Exception as e:
                    logger.error(f"Error fetching trades from {source}: {e}")
        
        return all_data
    
    async def fetch_onchain_data(
        self, 
        symbol: str, 
        metrics: List[str],
        sources: List[str] = None
    ) -> Dict[str, List[Any]]:
        """从多个源获取链上数据"""
        all_data = {}
        
        if sources is None:
            sources = list(self.onchain_fetchers.keys())
        
        for source in sources:
            if source in self.onchain_fetchers:
                for metric in metrics:
                    try:
                        data = await self.onchain_fetchers[source].fetch_metric(symbol, metric)
                        if data:
                            key = f"{source}_{metric}"
                            if key not in all_data:
                                all_data[key] = []
                            all_data[key].extend(data)
                            
                            onchain_key = f"{symbol}_{metric}_{source}"
                            if onchain_key not in self.onchain_data:
                                self.onchain_data[onchain_key] = []
                            self.onchain_data[onchain_key].extend(data)
                            self.data_timestamps[onchain_key] = datetime.now()
                    except Exception as e:
                        logger.error(f"Error fetching onchain data from {source}: {e}")
        
        return all_data
    
    async def fetch_alternative_data(
        self, 
        symbols: List[str], 
        sources: List[str] = None
    ) -> Dict[str, List[Any]]:
        """从多个源获取另类数据"""
        all_data = {}
        
        if sources is None:
            sources = list(self.alternative_fetchers.keys())
        
        for source in sources:
            if source in self.alternative_fetchers:
                for symbol in symbols:
                    try:
                        data = await self.alternative_fetchers[source].fetch_data(symbol)
                        if data:
                            key = f"{source}_{symbol}"
                            if key not in all_data:
                                all_data[key] = []
                            all_data[key].extend(data)
                            
                            alt_key = f"{symbol}_alternative_{source}"
                            if alt_key not in self.alternative_data:
                                self.alternative_data[alt_key] = []
                            self.alternative_data[alt_key].extend(data)
                            self.data_timestamps[alt_key] = datetime.now()
                    except Exception as e:
                        logger.error(f"Error fetching alternative data from {source}: {e}")
        
        return all_data
    
    def get_latest_data(self, symbol: str, source: str = None) -> Dict[str, Any]:
        """获取最新数据"""
        data = {}
        
        if source:
            key = f"{symbol}_{source}"
            if key in self.api_data:
                data['api'] = self.api_data[key]
            if key in self.websocket_data:
                data['websocket'] = self.websocket_data[key]
            if key in self.orderbook_data:
                data['orderbook'] = self.orderbook_data[key]
            if key in self.trade_data:
                data['trades'] = self.trade_data[key]
        else:
            for key, value in self.api_data.items():
                if symbol in key:
                    data['api'] = value
            for key, value in self.websocket_data.items():
                if symbol in key:
                    data['websocket'] = value
            for key, value in self.orderbook_data.items():
                if symbol in key:
                    data['orderbook'] = value
            for key, value in self.trade_data.items():
                if symbol in key:
                    data['trades'] = value
        
        return data
    
    def get_data_timestamps(self) -> Dict[str, datetime]:
        """获取所有数据的时间戳"""
        return self.data_timestamps
    
    def get_data_age(self, symbol: str, source: str = None) -> Dict[str, timedelta]:
        """获取数据年龄"""
        ages = {}
        
        if source:
            key = f"{symbol}_{source}"
            if key in self.data_timestamps:
                ages[source] = datetime.now() - self.data_timestamps[key]
        else:
            for key, timestamp in self.data_timestamps.items():
                if symbol in key:
                    source_name = key.split('_')[-1]
                    ages[source_name] = datetime.now() - timestamp
        
        return ages

class MultiSourceDataMerger:
    """多源数据合并器"""
    
    def __init__(self):
        self.aggregator = DataAggregator()
    
    async def merge_market_data(
        self, 
        symbols: List[str], 
        timeframe: str = '1d',
        start_date: datetime = None, 
        end_date: datetime = None,
        sources: List[str] = None
    ) -> pd.DataFrame:
        """合并市场数据"""
        all_data = {}
        
        for symbol in symbols:
            data = await self.aggregator.fetch_market_data(symbol, timeframe, start_date, end_date, sources)
            
            for source, bar_data in data.items():
                if bar_data:
                    df = pd.DataFrame([bar.to_dict() for bar in bar_data])
                    df.set_index('timestamp', inplace=True)
                    df.columns = [f"{col}_{source}" for col in df.columns]
                    
                    if symbol not in all_data:
                        all_data[symbol] = df
                    else:
                        all_data[symbol] = all_data[symbol].join(df, how='outer')
        
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()
    
    async def merge_order_book_data(
        self, 
        symbols: List[str], 
        depth: int = 20,
        sources: List[str] = None
    ) -> pd.DataFrame:
        """合并订单簿数据"""
        all_data = {}
        
        for symbol in symbols:
            data = await self.aggregator.fetch_order_books([symbol], depth, sources)
            
            for source, order_books in data.items():
                if order_books:
                    for order_book in order_books:
                        analysis = self._analyze_order_book(order_book)
                        analysis_df = pd.DataFrame([analysis])
                        analysis_df['timestamp'] = order_book.timestamp
                        
                        if symbol not in all_data:
                            all_data[symbol] = analysis_df
                        else:
                            all_data[symbol] = pd.concat([all_data[symbol], analysis_df], ignore_index=True)
        
        return pd.concat(all_data, axis=0) if all_data else pd.DataFrame()
    
    async def merge_onchain_data(
        self, 
        symbols: List[str], 
        metrics: List[str],
        sources: List[str] = None
    ) -> pd.DataFrame:
        """合并链上数据"""
        all_data = {}
        
        for symbol in symbols:
            data = await self.aggregator.fetch_onchain_data(symbol, metrics, sources)
            
            for key, onchain_data in data.items():
                if onchain_data:
                    df = pd.DataFrame([d.to_dict() for d in onchain_data])
                    if not df.empty:
                        df.set_index('timestamp', inplace=True)
                        
                        if symbol not in all_data:
                            all_data[symbol] = df
                        else:
                            all_data[symbol] = all_data[symbol].join(df, how='outer')
        
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()
    
    async def merge_alternative_data(
        self, 
        symbols: List[str],
        sources: List[str] = None
    ) -> pd.DataFrame:
        """合并另类数据"""
        all_data = {}
        
        for symbol in symbols:
            data = await self.aggregator.fetch_alternative_data([symbol], sources)
            
            for key, alternative_data in data.items():
                if alternative_data:
                    df = pd.DataFrame([d.to_dict() for d in alternative_data])
                    if not df.empty:
                        df.set_index('timestamp', inplace=True)
                        
                        if symbol not in all_data:
                            all_data[symbol] = df
                        else:
                            all_data[symbol] = all_data[symbol].join(df, how='outer')
        
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()
    
    def _analyze_order_book(self, order_book: Any) -> Dict:
        """分析订单簿"""
        if not order_book.bids or not order_book.asks:
            return {}
        
        best_bid = order_book.bids[0][0]
        best_ask = order_book.asks[0][0]
        
        bid_depth = sum(amount for _, amount in order_book.bids[:5])
        ask_depth = sum(amount for _, amount in order_book.asks[:5])
        
        spread = (best_ask - best_bid) / best_bid
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': (best_bid + best_ask) / 2,
            'spread': spread,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': imbalance
        }

class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        self.missing_values: Dict[str, int] = {}
        self.outliers: Dict[str, List] = {}
        self.gaps: Dict[str, List] = {}
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """检查缺失值"""
        missing = df.isnull().sum().to_dict()
        self.missing_values = missing
        return missing
    
    def check_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, List]:
        """检查异常值"""
        outliers = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            series = df[column].dropna()
            if len(series) > 0:
                mean = series.mean()
                std = series.std()
                
                if std > 0:
                    z_scores = np.abs((series - mean) / std)
                    outlier_indices = series[z_scores > threshold].index.tolist()
                    
                    if outlier_indices:
                        outliers[column] = outlier_indices
        
        self.outliers = outliers
        return outliers
    
    def check_gaps(self, df: pd.DataFrame, timeframe: str = '1d') -> Dict[str, List]:
        """检查数据 gaps"""
        gaps = {}
        
        for column in df.columns:
            if 'timestamp' in column.lower() or df[column].dtype == 'datetime64[ns]':
                series = df[column].dropna()
                if len(series) > 1:
                    time_diffs = series.diff().dropna()
                    
                    expected_interval = self._get_expected_interval(timeframe)
                    threshold = expected_interval * 2
                    
                    gap_indices = series[time_diffs > threshold].index.tolist()
                    
                    if gap_indices:
                        gaps[column] = gap_indices
        
        self.gaps = gaps
        return gaps
    
    def _get_expected_interval(self, timeframe: str) -> timedelta:
        """获取预期时间间隔"""
        intervals = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1wk': timedelta(weeks=1)
        }
        return intervals.get(timeframe, timedelta(days=1))
    
    def check_data_quality(self, df: pd.DataFrame, timeframe: str = '1d') -> Dict:
        """全面检查数据质量"""
        missing = self.check_missing_values(df)
        outliers = self.check_outliers(df)
        gaps = self.check_gaps(df, timeframe)
        
        total_rows = len(df)
        total_missing = sum(missing.values())
        outlier_count = sum(len(v) for v in outliers.values())
        
        return {
            'total_rows': total_rows,
            'total_missing': total_missing,
            'missing_percentage': total_missing / (total_rows * len(df.columns)) * 100 if total_rows > 0 else 0,
            'outlier_count': outlier_count,
            'gaps_count': sum(len(v) for v in gaps.values()),
            'missing_by_column': missing,
            'outliers_by_column': outliers,
            'gaps_by_column': gaps
        }

class DataNormalizer:
    """数据归一化器"""
    
    def __init__(self):
        self.stats: Dict[str, Dict] = {}
    
    def normalize(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """归一化数据"""
        normalized_df = df.copy()
        
        for column in df.select_dtypes(include=[np.number]).columns:
            series = df[column].dropna()
            
            if len(series) > 0:
                if method == 'zscore':
                    mean = series.mean()
                    std = series.std()
                    normalized_df[column] = (df[column] - mean) / std if std > 0 else df[column]
                    
                    self.stats[column] = {'mean': mean, 'std': std, 'method': 'zscore'}
                    
                elif method == 'minmax':
                    min_val = series.min()
                    max_val = series.max()
                    normalized_df[column] = (df[column] - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else df[column]
                    
                    self.stats[column] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                    
                elif method == 'log':
                    normalized_df[column] = np.log1p(df[column] - df[column].min() + 1)
                    
                    self.stats[column] = {'min': df[column].min(), 'method': 'log'}
        
        return normalized_df
    
    def denormalize(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """反归一化数据"""
        denormalized_df = df.copy()
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if column in self.stats:
                stats = self.stats[column]
                
                if stats['method'] == 'zscore':
                    denormalized_df[column] = df[column] * stats['std'] + stats['mean']
                    
                elif stats['method'] == 'minmax':
                    denormalized_df[column] = df[column] * (stats['max'] - stats['min']) + stats['min']
                    
                elif stats['method'] == 'log':
                    denormalized_df[column] = np.expm1(df[column]) + stats['min']
        
        return denormalized_df

class DataEnricher:
    """数据增强器"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        enriched_df = df.copy()
        
        if 'close' in df.columns:
            close = df['close']
            
            # 移动平均线
            enriched_df['ma_5'] = close.rolling(window=5).mean()
            enriched_df['ma_20'] = close.rolling(window=20).mean()
            enriched_df['ma_50'] = close.rolling(window=50).mean()
            
            # 指数移动平均线
            enriched_df['ema_12'] = close.ewm(span=12, adjust=False).mean()
            enriched_df['ema_26'] = close.ewm(span=26, adjust=False).mean()
            
            # MACD
            enriched_df['macd'] = enriched_df['ema_12'] - enriched_df['ema_26']
            enriched_df['macd_signal'] = enriched_df['macd'].ewm(span=9, adjust=False).mean()
            enriched_df['macd_hist'] = enriched_df['macd'] - enriched_df['macd_signal']
            
            # 波动率
            enriched_df['volatility_20'] = close.pct_change().rolling(window=20).std()
            
            # RSI
            enriched_df['rsi'] = self._calculate_rsi(close)
            
            # 布林带
            enriched_df['bb_middle'] = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            enriched_df['bb_upper'] = enriched_df['bb_middle'] + 2 * bb_std
            enriched_df['bb_lower'] = enriched_df['bb_middle'] - 2 * bb_std
        
        return enriched_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def add_sentiment_score(self, df: pd.DataFrame, alternative_data: List[Any] = None) -> pd.DataFrame:
        """添加情感分数"""
        if alternative_data:
            sentiments = [d.sentiment for d in alternative_data if d.sentiment is not None]
            if sentiments:
                df['sentiment_score'] = np.mean(sentiments)
        
        return df
    
    def add_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量分布"""
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            close = df['close']
            
            df['volume_ma_5'] = volume.rolling(window=5).mean()
            df['volume_ratio'] = volume / df['volume_ma_5']
            
            price_change = close.pct_change()
            df['volume_price_trend'] = price_change * volume
        
        return df
    
    def enrich_data(self, df: pd.DataFrame, alternative_data: List[Any] = None) -> pd.DataFrame:
        """全面增强数据"""
        enriched_df = df.copy()
        
        enriched_df = self.add_technical_indicators(enriched_df)
        enriched_df = self.add_volume_profile(enriched_df)
        enriched_df = self.add_sentiment_score(enriched_df, alternative_data)
        
        return enriched_df

class MultiSourceDataManager:
    """多源数据管理器"""
    
    def __init__(self):
        self.aggregator = DataAggregator()
        self.merger = MultiSourceDataMerger()
        self.quality_checker = DataQualityChecker()
        self.normalizer = DataNormalizer()
        self.enricher = DataEnricher()
        
        self.raw_data: Dict[str, Any] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
    
    def register_fetchers(
        self,
        api_fetchers: Dict[str, Any] = None,
        websocket_fetchers: Dict[str, Any] = None,
        orderbook_fetchers: Dict[str, Any] = None,
        trade_fetchers: Dict[str, Any] = None,
        onchain_fetchers: Dict[str, Any] = None,
        alternative_fetchers: Dict[str, Any] = None
    ):
        """注册所有获取器"""
        if api_fetchers:
            for name, fetcher in api_fetchers.items():
                self.aggregator.register_api_fetcher(name, fetcher)
        
        if websocket_fetchers:
            for name, fetcher in websocket_fetchers.items():
                self.aggregator.register_websocket_fetcher(name, fetcher)
        
        if orderbook_fetchers:
            for name, fetcher in orderbook_fetchers.items():
                self.aggregator.register_orderbook_fetcher(name, fetcher)
        
        if trade_fetchers:
            for name, fetcher in trade_fetchers.items():
                self.aggregator.register_trade_fetcher(name, fetcher)
        
        if onchain_fetchers:
            for name, fetcher in onchain_fetchers.items():
                self.aggregator.register_onchain_fetcher(name, fetcher)
        
        if alternative_fetchers:
            for name, fetcher in alternative_fetchers.items():
                self.aggregator.register_alternative_fetcher(name, fetcher)
    
    async def fetch_and_process(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        start_date: datetime = None,
        end_date: datetime = None,
        sources: List[str] = None,
        normalize: bool = True,
        enrich: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """获取并处理数据"""
        processed_data = {}
        
        for symbol in symbols:
            # 获取数据
            market_data = await self.aggregator.fetch_market_data(symbol, timeframe, start_date, end_date, sources)
            
            # 合并数据
            merged_df = await self.merger.merge_market_data([symbol], timeframe, start_date, end_date, sources)
            
            # 检查数据质量
            quality = self.quality_checker.check_data_quality(merged_df, timeframe)
            
            # 归一化
            if normalize:
                merged_df = self.normalizer.normalize(merged_df)
            
            # 增强
            if enrich:
                alternative_key = f"{symbol}_alternative"
                alternative_data = self.aggregator.alternative_data.get(alternative_key, [])
                merged_df = self.enricher.enrich_data(merged_df, alternative_data)
            
            processed_data[symbol] = merged_df
            
            self.raw_data[symbol] = market_data
            self.processed_data[symbol] = merged_df
        
        return processed_data
    
    def get_data(self, symbol: str) -> pd.DataFrame:
        """获取处理后的数据"""
        return self.processed_data.get(symbol, pd.DataFrame())
    
    def get_quality_report(self, symbol: str) -> Dict:
        """获取数据质量报告"""
        if symbol in self.raw_data:
            merged_df = pd.concat([
                pd.DataFrame([bar.to_dict() for bar in bars]) 
                for bars in self.raw_data[symbol].values()
            ])
            return self.quality_checker.check_data_quality(merged_df)
        return {}
    
    def get_all_processed_data(self) -> Dict[str, pd.DataFrame]:
        """获取所有处理后的数据"""
        return self.processed_data
