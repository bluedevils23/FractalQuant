# 市场数据获取指南

## 概述

本系统支持多种数据源获取市场数据，包括：

### 数据类型
- **K线数据** (OHLCV)
- **Tick数据** (实时价格)
- **订单簿数据** (深度数据)
- **交易数据** (成交明细)
- **链上数据** (区块链数据)
- **另类数据** (新闻、社交媒体)

### 数据源
- **API数据源**: AlphaVantage, Yahoo Finance, Finnhub, Polygon.io, ExchangeRateAPI, CryptoCompare
- **WebSocket数据源**: Binance, Coinbase, Kraken, Bybit, OKX, Kucoin
- **订单簿数据源**: Binance, Coinbase, Kraken, Bybit, OKX, Kucoin
- **交易数据源**: Binance, Coinbase, Kraken, Bybit, OKX, Kucoin
- **链上数据源**: Glassnode, CryptoQuant, Nansen, Etherscan, Blockchair, Mempool
- **另类数据源**: NewsAPI, Twitter, Reddit, Google Trends

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

```python
from dotenv import load_dotenv
import os

load_dotenv()

# API密钥
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
FINNHUB_KEY = os.getenv('FINNHUB_KEY')
POLYGON_KEY = os.getenv('POLYGON_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
```

### 3. 基本使用

#### 获取K线数据

```python
import asyncio
from data import YahooFinanceFetcher

async def fetch_kline_data():
    fetcher = YahooFinanceFetcher()
    bars = await fetcher.fetch_historical_data(
        symbol="BTC-USD",
        timeframe="1d",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    for bar in bars:
        print(f"{bar.timestamp}: ${bar.close}")

asyncio.run(fetch_kline_data())
```

#### 获取实时数据

```python
import asyncio
from data import BinanceWebSocketFetcher

async def fetch_realtime_data():
    fetcher = BinanceWebSocketFetcher()
    await fetcher.connect()
    await fetcher.subscribe(["BTC/USDT", "ETH/USDT"])
    
    # 处理实时数据
    async def handle_data(data):
        print(f"{data['symbol']}: ${data['price']}")
    
    fetcher.data_callbacks["BTC/USDT"].append(handle_data)
    
    await asyncio.sleep(10)
    await fetcher.stop()

asyncio.run(fetch_realtime_data())
```

#### 获取订单簿数据

```python
import asyncio
from data import BinanceOrderBookFetcher

async def fetch_orderbook():
    fetcher = BinanceOrderBookFetcher()
    order_book = await fetcher.fetch_order_book("BTC/USDT", depth=20)
    
    print(f"买价: ${order_book.bids[0][0]}")
    print(f"卖价: ${order_book.asks[0][0]}")
    print(f"买卖价差: {(order_book.asks[0][0] - order_book.bids[0][0]) / order_book.asks[0][0] * 100:.4f}%")

asyncio.run(fetch_orderbook())
```

#### 获取交易数据

```python
import asyncio
from data import BinanceTradeFetcher

async def fetch_trades():
    fetcher = BinanceTradeFetcher()
    trades = await fetcher.fetch_trades("BTC/USDT", limit=100)
    
    for trade in trades:
        print(f"{trade.timestamp}: ${trade.price} x {trade.volume} ({trade.side})")

asyncio.run(fetch_trades())
```

#### 获取链上数据

```python
import asyncio
from data import GlassnodeFetcher

async def fetch_onchain():
    fetcher = GlassnodeFetcher(api_key="YOUR_API_KEY")
    data = await fetcher.fetch_metric("BTC", "market_price_usd")
    
    for d in data:
        print(f"{d.timestamp}: ${d.value}")

asyncio.run(fetch_onchain())
```

#### 获取另类数据

```python
import asyncio
from data import NewsFetcher

async def fetch_news():
    fetcher = NewsFetcher(api_key="YOUR_API_KEY")
    data = await fetcher.fetch_data("BTC")
    
    for d in data:
        print(f"{d.timestamp}: {d.content}")
        print(f"  情感分数: {d.sentiment:.4f}")

asyncio.run(fetch_news())
```

### 4. 高级使用

#### 多源数据聚合

```python
import asyncio
from data import DataAggregator

async def multi_source_aggregation():
    aggregator = DataAggregator()
    
    # 注册多个获取器
    aggregator.register_api_fetcher('binance', BinanceOrderBookFetcher())
    aggregator.register_api_fetcher('coinbase', CoinbaseOrderBookFetcher())
    
    # 从多个源获取数据
    data = await aggregator.fetch_order_books(
        ["BTC/USDT", "ETH/USDT"],
        depth=20,
        sources=['binance', 'coinbase']
    )
    
    for source, order_books in data.items():
        print(f"{source}: {len(order_books)} 个订单簿")

asyncio.run(multi_source_aggregation())
```

#### 多源数据合并

```python
import asyncio
from data import MultiSourceDataMerger

async def merge_data():
    merger = MultiSourceDataMerger()
    
    # 合并多个源的数据
    df = await merger.merge_market_data(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1d",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        sources=['binance', 'coinbase']
    )
    
    print(df.head())
    print(f"\n数据形状: {df.shape}")

asyncio.run(merge_data())
```

#### 数据质量检查

```python
import asyncio
from data import DataQualityChecker

async def check_quality():
    merger = MultiSourceDataMerger()
    quality_checker = DataQualityChecker()
    
    df = await merger.merge_market_data(
        symbols=["BTC/USDT"],
        timeframe="1d",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    quality = quality_checker.check_data_quality(df, timeframe="1d")
    
    print(f"总行数: {quality['total_rows']}")
    print(f"缺失值: {quality['total_missing']}")
    print(f"缺失百分比: {quality['missing_percentage']:.2f}%")
    print(f"异常值: {quality['outlier_count']}")
    print(f"gaps数量: {quality['gaps_count']}")

asyncio.run(check_quality())
```

#### 数据归一化

```python
import asyncio
from data import DataNormalizer

async def normalize_data():
    normalizer = DataNormalizer()
    
    # Z-score归一化
    normalized_df = normalizer.normalize(df, method='zscore')
    
    # Min-Max归一化
    normalized_df = normalizer.normalize(df, method='minmax')
    
    # 反归一化
    denormalized_df = normalizer.denormalize(normalized_df, method='zscore')

asyncio.run(normalize_data())
```

#### 数据增强

```python
import asyncio
from data import DataEnricher

async def enrich_data():
    enricher = DataEnricher()
    
    # 添加技术指标
    enriched_df = enricher.add_technical_indicators(df)
    
    # 添加成交量分布
    enriched_df = enricher.add_volume_profile(enriched_df)
    
    # 添加情感分数
    enriched_df = enricher.add_sentiment_score(enriched_df, alternative_data)
    
    # 全面增强
    enriched_df = enricher.enrich_data(df, alternative_data)

asyncio.run(enrich_data())
```

### 5. 完整示例

```python
import asyncio
from datetime import datetime, timedelta
from data import (
    MultiSourceDataManager,
    DataQualityChecker,
    DataNormalizer,
    DataEnricher
)

async def full_pipeline():
    # 初始化管理器
    manager = MultiSourceDataManager()
    quality_checker = DataQualityChecker()
    normalizer = DataNormalizer()
    enricher = DataEnricher()
    
    # 注册获取器
    manager.register_fetchers(
        api_fetchers={
            'alpha_vantage': AlphaVantageFetcher(api_key="YOUR_API_KEY"),
            'yahoo_finance': YahooFinanceFetcher()
        },
        orderbook_fetchers={
            'binance': BinanceOrderBookFetcher(),
            'coinbase': CoinbaseOrderBookFetcher()
        },
        trade_fetchers={
            'binance': BinanceTradeFetcher(),
            'coinbase': CoinbaseTradeFetcher()
        },
        onchain_fetchers={
            'glassnode': GlassnodeFetcher(api_key="YOUR_API_KEY")
        },
        alternative_fetchers={
            'news': NewsFetcher(api_key="YOUR_API_KEY"),
            'twitter': TwitterFetcher(access_token="YOUR_ACCESS_TOKEN")
        }
    )
    
    # 获取并处理数据
    processed_data = await manager.fetch_and_process(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe='1d',
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        sources=['binance', 'coinbase', 'alpha_vantage'],
        normalize=True,
        enrich=True
    )
    
    # 获取数据质量报告
    for symbol in processed_data.keys():
        quality = manager.get_quality_report(symbol)
        print(f"\n{symbol} 数据质量:")
        print(f"  总行数: {quality.get('total_rows', 0)}")
        print(f"  缺失值: {quality.get('total_missing', 0)}")
        print(f"  异常值: {quality.get('outlier_count', 0)}")
    
    # 获取处理后的数据
    for symbol, df in processed_data.items():
        print(f"\n{symbol} 数据:")
        print(df.head())
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

asyncio.run(full_pipeline())
```

## 数据模型

### BarData (K线数据)
```python
@dataclass
class BarData:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: Optional[float] = None
    exchange: Optional[str] = None
    trades_count: Optional[int] = None
    vwap: Optional[float] = None
```

### TickData (Tick数据)
```python
@dataclass
class TickData:
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_price: Optional[float] = None
    ask_volume: Optional[float] = None
    exchange: Optional[str] = None
    trade_id: Optional[str] = None
```

### OrderBookData (订单簿数据)
```python
@dataclass
class OrderBookData:
    timestamp: datetime
    symbol: str
    bids: List[tuple]  # [(price, volume), ...]
    asks: List[tuple]
    exchange: Optional[str] = None
    analysis: Optional[Dict] = None
    order_count: Optional[int] = None
```

### TradeData (交易数据)
```python
@dataclass
class TradeData:
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None
    exchange: Optional[str] = None
```

### OnChainData (链上数据)
```python
@dataclass
class OnChainData:
    timestamp: datetime
    symbol: str
    metric: str
    value: float
    unit: Optional[str] = None
    exchange: Optional[str] = None
    metadata: Optional[Dict] = None
```

### AlternativeData (另类数据)
```python
@dataclass
class AlternativeData:
    timestamp: datetime
    symbol: str
    source: str
    content: str
    sentiment: Optional[float] = None
    volume: int = 0
    metadata: Optional[Dict] = None
```

## 支持的数据源

### API数据源

| 数据源 | 支持的数据类型 | 需要API密钥 | 备注 |
|--------|--------------|-----------|------|
| AlphaVantage | K线、实时数据 | 是 | 免费版有限制 |
| Yahoo Finance | K线、实时数据 | 否 | 免费 |
| Finnhub | K线、实时数据 | 是 | 免费版有限制 |
| Polygon.io | K线、实时数据 | 是 | 免费版有限制 |
| ExchangeRateAPI | 汇率数据 | 是 | 免费版有限制 |
| CryptoCompare | 加密货币数据 | 是 | 免费版有限制 |

### WebSocket数据源

| 数据源 | 支持的交易对 | 数据类型 | 备注 |
|--------|------------|---------|------|
| Binance | 500+ | 实时价格、订单簿 | 免费 |
| Coinbase | 30+ | 实时价格、订单簿 | 免费 |
| Kraken | 100+ | 实时价格、订单簿 | 免费 |
| Bybit | 100+ | 实时价格、订单簿 | 免费 |
| OKX | 200+ | 实时价格、订单簿 | 免费 |
| Kucoin | 200+ | 实时价格、订单簿 | 免费 |

### 订单簿数据源

| 数据源 | 最大深度 | 更新频率 | 备注 |
|--------|---------|---------|------|
| Binance | 100 | 实时 | 免费 |
| Coinbase | 100 | 实时 | 免费 |
| Kraken | 100 | 实时 | 免费 |
| Bybit | 25 | 实时 | 免费 |
| OKX | 400 | 实时 | 免费 |
| Kucoin | 100 | 实时 | 免费 |

### 链上数据源

| 数据源 | 支持的链 | 数据类型 | 需要API密钥 | 备注 |
|--------|---------|---------|-----------|------|
| Glassnode | 比特币、以太坊 | 链上指标 | 是 | 免费版有限制 |
| CryptoQuant | 多链 | 链上指标 | 是 | 付费 |
| Nansen | 以太坊 | 链上分析 | 是 | 付费 |
| Etherscan | 以太坊 | 以太坊数据 | 是 | 免费 |
| Blockchair | 多链 | 链上数据 | 否 | 免费 |
| Mempool | 比特币 | 内存池数据 | 否 | 免费 |

### 另类数据源

| 数据源 | 数据类型 | 需要API密钥 | 备注 |
|--------|---------|-----------|------|
| NewsAPI | 新闻 | 是 | 免费版有限制 |
| Twitter | 推文 | 是 | 需要开发者账号 |
| Reddit | 帖子 | 否 | 免费 |
| Google Trends | 搜索趋势 | 否 | 免费 |

## 技术指标

系统支持以下技术指标：

- **移动平均线**: MA(5), MA(20), MA(50)
- **指数移动平均线**: EMA(12), EMA(26)
- **MACD**: MACD线、信号线、柱状图
- **波动率**: 20日波动率
- **RSI**: 相对强弱指数
- **布林带**: 上轨、中轨、下轨
- **成交量**: 成交量移动平均、成交量比率、量价趋势

## 数据质量检查

系统提供以下数据质量检查功能：

- **缺失值检查**: 检测数据中的缺失值
- **异常值检测**: 使用Z-score检测异常值
- **数据Gap检测**: 检测数据时间间隔异常
- **数据质量报告**: 生成详细的质量报告

## 数据归一化

系统支持以下归一化方法：

- **Z-score**: 标准化到均值为0，标准差为1
- **Min-Max**: 缩放到[0, 1]区间
- **Log**: 对数变换

## 数据增强

系统提供以下数据增强功能：

- **技术指标**: 添加多种技术指标
- **成交量分布**: 添加成交量相关指标
- **情感分数**: 添加另类数据的情感分数
- **综合增强**: 一键添加所有增强功能

## 性能优化

- **异步处理**: 使用asyncio进行异步数据获取
- **批量处理**: 支持批量获取数据
- **数据缓存**: 自动缓存获取的数据
- **连接池**: 复用HTTP连接

## 错误处理

- **重试机制**: 自动重试失败的请求
- **超时控制**: 设置请求超时
- **异常捕获**: 捕获并记录异常
- **日志记录**: 详细记录操作日志

## 最佳实践

1. **使用多源数据**: 从多个数据源获取数据以提高数据质量
2. **定期更新**: 定期更新数据以保持数据新鲜度
3. **数据验证**: 在使用数据前进行验证
4. **错误处理**: 妥善处理错误和异常
5. **性能监控**: 监控数据获取性能

## 常见问题

### Q: 如何处理API限流？
A: 系统内置了限流处理机制，自动处理API调用频率限制。

### Q: 如何处理数据缺失？
A: 系统提供数据缺失检测和处理功能，可以填充缺失值或删除缺失数据。

### Q: 如何提高数据获取性能？
A: 使用异步处理、批量获取、数据缓存等技术提高性能。

### Q: 如何选择合适的数据源？
A: 根据数据类型、数据质量、API限制等因素选择合适的数据源。

