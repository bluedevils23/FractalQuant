"""
市场数据获取示例
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta

from data import (
    # API获取器
    AlphaVantageFetcher, YahooFinanceFetcher, FinnhubFetcher,
    PolygonFetcher, ExchangeRateAPIFetcher, CryptoCompareFetcher,
    
    # WebSocket获取器
    BinanceWebSocketFetcher, CoinbaseWebSocketFetcher,
    KrakenWebSocketFetcher, BybitWebSocketFetcher, OKXWebSocketFetcher, KucoinWebSocketFetcher,
    
    # 订单簿获取器
    BinanceOrderBookFetcher, CoinbaseOrderBookFetcher,
    KrakenOrderBookFetcher, BybitOrderBookFetcher, OKXOrderBookFetcher, KucoinOrderBookFetcher,
    
    # 交易获取器
    BinanceTradeFetcher, CoinbaseTradeFetcher,
    KrakenTradeFetcher, BybitTradeFetcher, OKXTradeFetcher, KucoinTradeFetcher,
    
    # 链上数据获取器
    GlassnodeFetcher, CryptoQuantFetcher, EtherscanFetcher,
    MempoolFetcher,
    
    # 另类数据获取器
    NewsFetcher, TwitterFetcher, RedditFetcher, GoogleTrendsFetcher,
    
    # 数据聚合器
    DataAggregator, MultiSourceDataMerger, DataQualityChecker,
    DataNormalizer, DataEnricher, MultiSourceDataManager
)

async def example_api_fetchers():
    """示例1：API数据获取器"""
    print("=" * 80)
    print("示例1：API数据获取器")
    print("=" * 80)
    
    # AlphaVantage
    print("\n1. AlphaVantage API:")
    try:
        fetcher = AlphaVantageFetcher(api_key="YOUR_API_KEY")
        bars = await fetcher.fetch_historical_data("AAPL", timeframe='1day')
        print(f"  获取到 {len(bars)} 条K线数据")
        if bars:
            print(f"  最新价格: ${bars[-1].close}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Yahoo Finance
    print("\n2. Yahoo Finance API:")
    try:
        fetcher = YahooFinanceFetcher()
        bars = await fetcher.fetch_historical_data("BTC-USD", timeframe='1d')
        print(f"  获取到 {len(bars)} 条K线数据")
        if bars:
            print(f"  最新价格: ${bars[-1].close}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Finnhub
    print("\n3. Finnhub API:")
    try:
        fetcher = FinnhubFetcher(api_key="YOUR_API_KEY")
        bars = await fetcher.fetch_historical_data("AAPL", timeframe='1d')
        print(f"  获取到 {len(bars)} 条K线数据")
        if bars:
            print(f"  最新价格: ${bars[-1].close}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Polygon.io
    print("\n4. Polygon.io API:")
    try:
        fetcher = PolygonFetcher(api_key="YOUR_API_KEY")
        bars = await fetcher.fetch_historical_data("AAPL", timeframe='day')
        print(f"  获取到 {len(bars)} 条K线数据")
        if bars:
            print(f"  最新价格: ${bars[-1].close}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # ExchangeRateAPI
    print("\n5. ExchangeRateAPI:")
    try:
        fetcher = ExchangeRateAPIFetcher(api_key="YOUR_API_KEY")
        bars = await fetcher.fetch_historical_data("USDEUR", timeframe='day')
        print(f"  获取到 {len(bars)} 条汇率数据")
        if bars:
            print(f"  最新汇率: {bars[-1].close}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # CryptoCompare
    print("\n6. CryptoCompare API:")
    try:
        fetcher = CryptoCompareFetcher(api_key="YOUR_API_KEY")
        bars = await fetcher.fetch_historical_data("BTCUSD", timeframe='day')
        print(f"  获取到 {len(bars)} 条加密货币数据")
        if bars:
            print(f"  最新价格: ${bars[-1].close}")
    except Exception as e:
        print(f"  错误: {e}")

async def example_websocket_fetchers():
    """示例2：WebSocket实时数据获取器"""
    print("\n" + "=" * 80)
    print("示例2：WebSocket实时数据获取器")
    print("=" * 80)
    
    # Binance WebSocket
    print("\n1. Binance WebSocket:")
    try:
        fetcher = BinanceWebSocketFetcher()
        await fetcher.connect()
        await fetcher.subscribe(["BTC/USDT", "ETH/USDT"])
        print("  已订阅 BTC/USDT 和 ETH/USDT")
        print("  正在接收实时数据...")
        
        # 运行5秒后停止
        await asyncio.sleep(5)
        await fetcher.stop()
        print("  已停止")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Coinbase WebSocket
    print("\n2. Coinbase WebSocket:")
    try:
        fetcher = CoinbaseWebSocketFetcher()
        await fetcher.connect()
        await fetcher.subscribe(["BTC/USD", "ETH/USD"])
        print("  已订阅 BTC/USD 和 ETH/USD")
        print("  正在接收实时数据...")
        
        await asyncio.sleep(5)
        await fetcher.stop()
        print("  已停止")
    except Exception as e:
        print(f"  错误: {e}")

async def example_orderbook_fetchers():
    """示例3：订单簿数据获取器"""
    print("\n" + "=" * 80)
    print("示例3：订单簿数据获取器")
    print("=" * 80)
    
    # Binance OrderBook
    print("\n1. Binance OrderBook:")
    try:
        fetcher = BinanceOrderBookFetcher()
        order_book = await fetcher.fetch_order_book("BTC/USDT", depth=10)
        if order_book:
            print(f"  买价: ${order_book.bids[0][0]}")
            print(f"  卖价: ${order_book.asks[0][0]}")
            print(f"  买卖价差: {(order_book.asks[0][0] - order_book.bids[0][0]) / order_book.asks[0][0] * 100:.4f}%")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Coinbase OrderBook
    print("\n2. Coinbase OrderBook:")
    try:
        fetcher = CoinbaseOrderBookFetcher()
        order_book = await fetcher.fetch_order_book("BTC/USD", depth=10)
        if order_book:
            print(f"  买价: ${order_book.bids[0][0]}")
            print(f"  卖价: ${order_book.asks[0][0]}")
    except Exception as e:
        print(f"  错误: {e}")

async def example_trade_fetchers():
    """示例4：交易数据获取器"""
    print("\n" + "=" * 80)
    print("示例4：交易数据获取器")
    print("=" * 80)
    
    # Binance Trades
    print("\n1. Binance Trades:")
    try:
        fetcher = BinanceTradeFetcher()
        trades = await fetcher.fetch_trades("BTC/USDT", limit=10)
        print(f"  获取到 {len(trades)} 笔交易")
        if trades:
            print(f"  最新交易: ${trades[-1].price}, {trades[-1].volume} BTC")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Coinbase Trades
    print("\n2. Coinbase Trades:")
    try:
        fetcher = CoinbaseTradeFetcher()
        trades = await fetcher.fetch_trades("BTC/USD", limit=10)
        print(f"  获取到 {len(trades)} 笔交易")
        if trades:
            print(f"  最新交易: ${trades[-1].price}, {trades[-1].volume} BTC")
    except Exception as e:
        print(f"  错误: {e}")

async def example_onchain_fetchers():
    """示例5：链上数据获取器"""
    print("\n" + "=" * 80)
    print("示例5：链上数据获取器")
    print("=" * 80)
    
    # Glassnode
    print("\n1. Glassnode 链上数据:")
    try:
        fetcher = GlassnodeFetcher(api_key="YOUR_API_KEY")
        data = await fetcher.fetch_metric("BTC", "market_price_usd")
        print(f"  获取到 {len(data)} 条链上数据")
        if data:
            print(f"  最新价格: ${data[-1].value}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Mempool
    print("\n2. Mempool 内存池数据:")
    try:
        fetcher = MempoolFetcher()
        data = await fetcher.fetch_mempool_info("BTC")
        print(f"  获取到 {len(data)} 条内存池数据")
        if data:
            for d in data[:3]:
                print(f"    {d.metric}: {d.value}")
    except Exception as e:
        print(f"  错误: {e}")

async def example_alternative_fetchers():
    """示例6：另类数据获取器"""
    print("\n" + "=" * 80)
    print("示例6：另类数据获取器")
    print("=" * 80)
    
    # News
    print("\n1. 新闻数据:")
    try:
        fetcher = NewsFetcher(api_key="YOUR_API_KEY")
        data = await fetcher.fetch_data("BTC")
        print(f"  获取到 {len(data)} 条新闻")
        if data:
            print(f"  最新新闻: {data[0].content[:100]}...")
            print(f"  情感分数: {data[0].sentiment:.4f}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Twitter
    print("\n2. Twitter数据:")
    try:
        fetcher = TwitterFetcher(access_token="YOUR_ACCESS_TOKEN")
        data = await fetcher.fetch_data("BTC")
        print(f"  获取到 {len(data)} 条推文")
        if data:
            print(f"  最新推文: {data[0].content[:100]}...")
            print(f"  情感分数: {data[0].sentiment:.4f}")
            print(f"  互动量: {data[0].volume}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # Reddit
    print("\n3. Reddit数据:")
    try:
        fetcher = RedditFetcher()
        data = await fetcher.fetch_data("BTC")
        print(f"  获取到 {len(data)} 条帖子")
        if data:
            print(f"  最新帖子: {data[0].content[:100]}...")
            print(f"  情感分数: {data[0].sentiment:.4f}")
    except Exception as e:
        print(f"  错误: {e}")

async def example_data_aggregator():
    """示例7：数据聚合器"""
    print("\n" + "=" * 80)
    print("示例7：数据聚合器")
    print("=" * 80)
    
    aggregator = DataAggregator()
    
    # 注册多个获取器
    aggregator.register_api_fetcher('binance', BinanceOrderBookFetcher())
    aggregator.register_api_fetcher('coinbase', CoinbaseOrderBookFetcher())
    
    # 从多个源获取数据
    print("\n1. 从多个源获取订单簿数据:")
    try:
        data = await aggregator.fetch_order_books(
            ["BTC/USDT", "ETH/USDT"],
            depth=20,
            sources=['binance', 'coinbase']
        )
        for source, order_books in data.items():
            print(f"  {source}: {len(order_books)} 个订单簿")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 从多个源获取市场数据
    print("\n2. 从多个源获取市场数据:")
    try:
        data = await aggregator.fetch_market_data(
            "BTC/USDT",
            timeframe='1d',
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            sources=['binance', 'coinbase']
        )
        for source, bars in data.items():
            print(f"  {source}: {len(bars)} 条K线")
    except Exception as e:
        print(f"  错误: {e}")

async def example_multi_source_manager():
    """示例8：多源数据管理器"""
    print("\n" + "=" * 80)
    print("示例8：多源数据管理器")
    print("=" * 80)
    
    manager = MultiSourceDataManager()
    
    # 注册获取器
    manager.register_fetchers(
        api_fetchers={
            'alpha_vantage': AlphaVantageFetcher(api_key="YOUR_API_KEY"),
            'yahoo_finance': YahooFinanceFetcher()
        },
        orderbook_fetchers={
            'binance': BinanceOrderBookFetcher(),
            'coinbase': CoinbaseOrderBookFetcher()
        }
    )
    
    # 获取并处理数据
    print("\n1. 获取并处理市场数据:")
    try:
        processed_data = await manager.fetch_and_process(
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframe='1d',
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            normalize=True,
            enrich=True
        )
        
        for symbol, df in processed_data.items():
            print(f"  {symbol}: {len(df)} 条数据")
            print(f"    列名: {list(df.columns)[:5]}...")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 获取数据质量报告
    print("\n2. 数据质量报告:")
    try:
        quality = manager.get_quality_report("BTC/USDT")
        print(f"  总行数: {quality.get('total_rows', 0)}")
        print(f"  缺失值: {quality.get('total_missing', 0)}")
        print(f"  异常值: {quality.get('outlier_count', 0)}")
    except Exception as e:
        print(f"  错误: {e}")

async def example_data_normalizer():
    """示例9：数据归一化"""
    print("\n" + "=" * 80)
    print("示例9：数据归一化")
    print("=" * 80)
    
    # 创建示例数据
    df = pd.DataFrame({
        'open': [100, 102, 98, 105, 103],
        'high': [105, 108, 100, 110, 108],
        'low': [95, 98, 95, 100, 98],
        'close': [102, 100, 99, 108, 105],
        'volume': [1000, 1200, 800, 1500, 1100]
    })
    
    print("\n1. 原始数据:")
    print(df)
    
    # Z-score归一化
    normalizer = DataNormalizer()
    normalized_df = normalizer.normalize(df, method='zscore')
    
    print("\n2. Z-score归一化后:")
    print(normalized_df.round(4))
    
    # 反归一化
    denormalized_df = normalizer.denormalize(normalized_df, method='zscore')
    
    print("\n3. 反归一化后:")
    print(denormalized_df.round(2))

async def example_data_enricher():
    """示例10：数据增强"""
    print("\n" + "=" * 80)
    print("示例10：数据增强")
    print("=" * 80)
    
    # 创建示例数据
    df = pd.DataFrame({
        'open': [100, 102, 98, 105, 103, 101, 104, 106, 102, 105],
        'high': [105, 108, 100, 110, 108, 106, 109, 111, 107, 110],
        'low': [95, 98, 95, 100, 98, 99, 102, 104, 98, 102],
        'close': [102, 100, 99, 108, 105, 104, 107, 109, 105, 108],
        'volume': [1000, 1200, 800, 1500, 1100, 900, 1300, 1400, 1000, 1200]
    })
    
    enricher = DataEnricher()
    
    print("\n1. 原始数据:")
    print(df)
    
    # 增强数据
    enriched_df = enricher.enrich_data(df)
    
    print("\n2. 增强后数据 (包含技术指标):")
    print(enriched_df.round(4))
    
    print("\n3. 技术指标说明:")
    print("  - ma_5: 5日移动平均")
    print("  - ma_20: 20日移动平均")
    print("  - ma_50: 50日移动平均")
    print("  - ema_12: 12日指数移动平均")
    print("  - ema_26: 26日指数移动平均")
    print("  - macd: MACD线")
    print("  - macd_signal: MACD信号线")
    print("  - macd_hist: MACD柱状图")
    print("  - volatility_20: 20日波动率")
    print("  - rsi: 相对强弱指数")
    print("  - bb_upper: 布林带上轨")
    print("  - bb_middle: 布林带中轨")
    print("  - bb_lower: 布林带下轨")
    print("  - volume_ma_5: 5日平均成交量")
    print("  - volume_ratio: 成交量比率")
    print("  - volume_price_trend: 量价趋势")

async def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("市场数据获取示例")
    print("=" * 80)
    
    # 运行示例
    await example_api_fetchers()
    await example_websocket_fetchers()
    await example_orderbook_fetchers()
    await example_trade_fetchers()
    await example_onchain_fetchers()
    await example_alternative_fetchers()
    await example_data_aggregator()
    await example_multi_source_manager()
    await example_data_normalizer()
    await example_data_enricher()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成")
    print("=" * 80)

if __name__ == '__main__':
    asyncio.run(main())
