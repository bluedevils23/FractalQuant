"""
另类数据获取器(新闻、社交媒体)
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import re
from collections import Counter

class AlternativeData:
    """另类数据"""
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        source: str,
        content: str,
        sentiment: float = None,
        volume: int = 0,
        metadata: Dict = None
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.source = source
        self.content = content
        self.sentiment = sentiment
        self.volume = volume
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'source': self.source,
            'content': self.content,
            'sentiment': self.sentiment,
            'volume': self.volume,
            'metadata': self.metadata
        }

class AlternativeFetcher(ABC):
    """另类数据获取器基类"""
    
    @abstractmethod
    async def fetch_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> List[AlternativeData]:
        """获取另类数据"""
        pass

class NewsFetcher(AlternativeFetcher):
    """新闻数据获取器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    async def fetch_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> List[AlternativeData]:
        """获取新闻数据"""
        try:
            from_date = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'q': symbol,
                'sortBy': 'publishedAt',
                'from': from_date,
                'to': to_date,
                'language': 'en',
                'apiKey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/everything", params=params) as response:
                    data = await response.json()
                    
                    alternative_data = []
                    for article in data.get('articles', []):
                        sentiment = self._analyze_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                        
                        alt_data = AlternativeData(
                            timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                            symbol=symbol,
                            source='newsapi',
                            content=article.get('title', ''),
                            sentiment=sentiment,
                            metadata={
                                'description': article.get('description'),
                                'url': article.get('url'),
                                'author': article.get('author'),
                                'source': article.get('source', {}).get('name')
                            }
                        )
                        alternative_data.append(alt_data)
                    
                    return alternative_data
                    
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """分析情感"""
        positive_words = ['buy', 'bullish', 'up', 'gain', 'profit', 'increase', 'strong', 'good', 'excellent', 'great']
        negative_words = ['sell', 'bearish', 'down', 'loss', 'decrease', 'weak', 'bad', 'poor', 'terrible', 'worst']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return 0
        
        return (positive_count - negative_count) / total

class TwitterFetcher(AlternativeFetcher):
    """Twitter数据获取器"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, access_token: str = None, access_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.base_url = "https://api.twitter.com/2"
        
    async def fetch_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> List[AlternativeData]:
        """获取Twitter数据"""
        try:
            query = f"{symbol} -is:retweet lang:en"
            
            params = {
                'query': query,
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics'
            }
            
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/tweets/search/recent", params=params, headers=headers) as response:
                    data = await response.json()
                    
                    alternative_data = []
                    for tweet in data.get('data', []):
                        sentiment = self._analyze_sentiment(tweet.get('text', ''))
                        
                        alt_data = AlternativeData(
                            timestamp=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                            symbol=symbol,
                            source='twitter',
                            content=tweet.get('text', ''),
                            sentiment=sentiment,
                            volume=tweet.get('public_metrics', {}).get('retweet_count', 0) + 
                                   tweet.get('public_metrics', {}).get('reply_count', 0) +
                                   tweet.get('public_metrics', {}).get('like_count', 0),
                            metadata={
                                'tweet_id': tweet.get('id'),
                                'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                                'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                                'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                                'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0)
                            }
                        )
                        alternative_data.append(alt_data)
                    
                    return alternative_data
                    
        except Exception as e:
            print(f"Error fetching Twitter data for {symbol}: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """分析情感"""
        positive_words = ['buy', 'bullish', 'up', 'gain', 'profit', 'increase', 'strong', 'good', 'excellent', 'great']
        negative_words = ['sell', 'bearish', 'down', 'loss', 'decrease', 'weak', 'bad', 'poor', 'terrible', 'worst']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return 0
        
        return (positive_count - negative_count) / total

class RedditFetcher(AlternativeFetcher):
    """Reddit数据获取器"""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        
    async def fetch_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> List[AlternativeData]:
        """获取Reddit数据"""
        try:
            query = f"{symbol} crypto"
            
            params = {
                'q': query,
                'sort': 'relevance',
                'limit': 100,
                'syntax': 'cloudsearch'
            }
            
            headers = {
                'User-Agent': 'TradingBot/1.0'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/search.json", params=params, headers=headers) as response:
                    data = await response.json()
                    
                    alternative_data = []
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        
                        sentiment = self._analyze_sentiment(post_data.get('title', '') + ' ' + post_data.get('selftext', ''))
                        
                        alt_data = AlternativeData(
                            timestamp=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                            symbol=symbol,
                            source='reddit',
                            content=post_data.get('title', ''),
                            sentiment=sentiment,
                            volume=post_data.get('ups', 0) - post_data.get('downs', 0),
                            metadata={
                                'subreddit': post_data.get('subreddit'),
                                'score': post_data.get('score'),
                                'num_comments': post_data.get('num_comments'),
                                'url': f"https://reddit.com{post_data.get('permalink', '')}"
                            }
                        )
                        alternative_data.append(alt_data)
                    
                    return alternative_data
                    
        except Exception as e:
            print(f"Error fetching Reddit data for {symbol}: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """分析情感"""
        positive_words = ['buy', 'bullish', 'up', 'gain', 'profit', 'increase', 'strong', 'good', 'excellent', 'great']
        negative_words = ['sell', 'bearish', 'down', 'loss', 'decrease', 'weak', 'bad', 'poor', 'terrible', 'worst']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return 0
        
        return (positive_count - negative_count) / total

class GoogleTrendsFetcher(AlternativeFetcher):
    """Google Trends数据获取器"""
    
    def __init__(self):
        self.base_url = "https://trends.google.com/trends"
        
    async def fetch_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> List[AlternativeData]:
        """获取Google Trends数据"""
        try:
            from_date = start_date.strftime('%Y-%m-%d') if start_date else (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'q': symbol,
                'date': f'{from_date} {to_date}'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/trendsdata", params=params) as response:
                    data = await response.json()
                    
                    alternative_data = []
                    for item in data.get('default', {}).get('timelineData', []):
                        alt_data = AlternativeData(
                            timestamp=datetime.fromtimestamp(item[0] / 1000),
                            symbol=symbol,
                            source='google_trends',
                            content=f"Interest: {item[1][0]}",
                            sentiment=item[1][0] / 100,
                            metadata={
                                'interest': item[1][0],
                                'date': item[0]
                            }
                        )
                        alternative_data.append(alt_data)
                    
                    return alternative_data
                    
        except Exception as e:
            print(f"Error fetching Google Trends data for {symbol}: {e}")
            return []

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self):
        self.positive_words = {
            'buy', 'bullish', 'up', 'gain', 'profit', 'increase', 'strong', 'good', 'excellent', 'great',
            'positive', 'optimistic', 'surge', 'breakout', 'rally', 'hooray', 'win', 'success', 'better'
        }
        self.negative_words = {
            'sell', 'bearish', 'down', 'loss', 'decrease', 'weak', 'bad', 'poor', 'terrible', 'worst',
            'negative', 'pessimistic', 'plunge', 'crash', 'sell-off', 'fail', 'loss', 'worse', 'fall'
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """分析情感"""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        
        total = positive_count + negative_count
        
        if total == 0:
            return 0
        
        return (positive_count - negative_count) / total
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """批量分析情感"""
        return [self.analyze_sentiment(text) for text in texts]

class AlternativeDataAnalyzer:
    """另类数据分析器"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data: Dict[str, List[AlternativeData]] = {}
        
    def calculate_sentiment_score(self, data: List[AlternativeData]) -> Dict:
        """计算情感分数"""
        if not data:
            return {}
        
        sentiments = [d.sentiment for d in data if d.sentiment is not None]
        
        if not sentiments:
            return {}
        
        return {
            'avg_sentiment': np.mean(sentiments),
            'min_sentiment': np.min(sentiments),
            'max_sentiment': np.max(sentiments),
            'sentiment_volatility': np.std(sentiments),
            'positive_ratio': sum(1 for s in sentiments if s > 0) / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s < 0) / len(sentiments)
        }
    
    def calculate_volume_score(self, data: List[AlternativeData]) -> Dict:
        """计算Volume分数"""
        if not data:
            return {}
        
        volumes = [d.volume for d in data]
        
        return {
            'total_volume': sum(volumes),
            'avg_volume': np.mean(volumes),
            'max_volume': np.max(volumes),
            'volume_volatility': np.std(volumes)
        }
    
    def calculate_attention_score(self, data: List[AlternativeData]) -> Dict:
        """计算关注度分数"""
        if not data:
            return {}
        
        return {
            'total_mentions': len(data),
            'unique_sources': len(set(d.source for d in data)),
            'avg_sentiment': np.mean([d.sentiment for d in data if d.sentiment is not None]) if any(d.sentiment is not None for d in data) else 0,
            'sentiment_trend': 'increasing' if len(data) > 1 and data[-1].sentiment > data[0].sentiment else 'decreasing' if len(data) > 1 and data[-1].sentiment < data[0].sentiment else 'stable'
        }
    
    def analyze_alternative_data(self, data: List[AlternativeData]) -> Dict:
        """全面分析另类数据"""
        if not data:
            return {}
        
        sentiment_score = self.calculate_sentiment_score(data)
        volume_score = self.calculate_volume_score(data)
        attention_score = self.calculate_attention_score(data)
        
        return {
            **sentiment_score,
            **volume_score,
            **attention_score
        }

class AlternativeDataManager:
    """另类数据管理器"""
    
    def __init__(self):
        self.fetchers: Dict[str, AlternativeFetcher] = {}
        self.analyzer = AlternativeDataAnalyzer()
        self.data: Dict[str, List[AlternativeData]] = {}
        
    def register_fetcher(self, source: str, fetcher: AlternativeFetcher):
        """注册获取器"""
        self.fetchers[source] = fetcher
    
    async def fetch_data(self, symbol: str, source: str = None, start_date: datetime = None, end_date: datetime = None) -> List[AlternativeData]:
        """获取另类数据"""
        if source and source in self.fetchers:
            data = await self.fetchers[source].fetch_data(symbol, start_date, end_date)
        else:
            for fetcher in self.fetchers.values():
                data = await fetcher.fetch_data(symbol, start_date, end_date)
                if data:
                    break
        
        if data:
            key = f"{symbol}_alternative"
            if key not in self.data:
                self.data[key] = []
            self.data[key].extend(data)
            
            analysis = self.analyzer.analyze_alternative_data(data)
            self.data[f"{key}_analysis"] = analysis
            
        return data
    
    def get_data(self, symbol: str) -> List[AlternativeData]:
        """获取缓存的另类数据"""
        key = f"{symbol}_alternative"
        return self.data.get(key, [])
    
    def get_analysis(self, symbol: str) -> Dict:
        """获取另类数据分析结果"""
        key = f"{symbol}_alternative"
        return self.data.get(f"{key}_analysis", {})
    
    def get_all_data(self) -> Dict[str, List[AlternativeData]]:
        """获取所有另类数据"""
        return {k: v for k, v in self.data.items() if not k.endswith('_analysis')}
