"""
市场数据获取测试
"""
import asyncio
import sys
from datetime import datetime, timedelta

# 测试导入
def test_imports():
    """测试模块导入"""
    print("=" * 80)
    print("测试模块导入")
    print("=" * 80)
    
    try:
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
        
        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

async def test_yahoo_finance():
    """测试Yahoo Finance获取器"""
    print("\n" + "=" * 80)
    print("测试Yahoo Finance获取器")
    print("=" * 80)
    
    try:
        from data import YahooFinanceFetcher
        
        fetcher = YahooFinanceFetcher()
        
        # 测试获取K线数据
        bars = await fetcher.fetch_historical_data(
            symbol="BTC-USD",
            timeframe="1d",
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now()
        )
        
        print(f"✓ 获取到 {len(bars)} 条K线数据")
        if bars:
            print(f"  最新价格: ${bars[-1].close}")
            print(f"  数据时间范围: {bars[0].timestamp} 至 {bars[-1].timestamp}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

async def test_binance_orderbook():
    """测试Binance订单簿获取器"""
    print("\n" + "=" * 80)
    print("测试Binance订单簿获取器")
    print("=" * 80)
    
    try:
        from data import BinanceOrderBookFetcher
        
        fetcher = BinanceOrderBookFetcher()
        
        # 测试获取订单簿
        order_book = await fetcher.fetch_order_book("BTC/USDT", depth=10)
        
        if order_book:
            print(f"✓ 获取到订单簿数据")
            print(f"  买价: ${order_book.bids[0][0]}")
            print(f"  卖价: ${order_book.asks[0][0]}")
            print(f"  买卖价差: {(order_book.asks[0][0] - order_book.bids[0][0]) / order_book.asks[0][0] * 100:.4f}%")
            print(f"  买盘深度: {sum(amount for _, amount in order_book.bids[:5])} BTC")
            print(f"  卖盘深度: {sum(amount for _, amount in order_book.asks[:5])} BTC")
            return True
        else:
            print("✗ 未能获取订单簿数据")
            return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

async def test_binance_trades():
    """测试Binance交易获取器"""
    print("\n" + "=" * 80)
    print("测试Binance交易获取器")
    print("=" * 80)
    
    try:
        from data import BinanceTradeFetcher
        
        fetcher = BinanceTradeFetcher()
        
        # 测试获取交易数据
        trades = await fetcher.fetch_trades("BTC/USDT", limit=10)
        
        print(f"✓ 获取到 {len(trades)} 笔交易")
        if trades:
            print(f"  最新交易: ${trades[-1].price}, {trades[-1].volume} BTC")
            print(f"  交易方向分布:")
            buy_count = sum(1 for t in trades if t.side == 'buy')
            sell_count = sum(1 for t in trades if t.side == 'sell')
            print(f"    买入: {buy_count} 笔")
            print(f"    卖出: {sell_count} 笔")
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

async def test_data_aggregator():
    """测试数据聚合器"""
    print("\n" + "=" * 80)
    print("测试数据聚合器")
    print("=" * 80)
    
    try:
        from data import DataAggregator, BinanceOrderBookFetcher, CoinbaseOrderBookFetcher
        
        aggregator = DataAggregator()
        
        # 注册获取器
        aggregator.register_orderbook_fetcher('binance', BinanceOrderBookFetcher())
        aggregator.register_orderbook_fetcher('coinbase', CoinbaseOrderBookFetcher())
        
        # 测试从多个源获取数据
        data = await aggregator.fetch_order_books(
            ["BTC/USDT"],
            depth=20,
            sources=['binance', 'coinbase']
        )
        
        print(f"✓ 从 {len(data)} 个源获取数据")
        for source, order_books in data.items():
            print(f"  {source}: {len(order_books)} 个订单簿")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

async def test_data_normalizer():
    """测试数据归一化器"""
    print("\n" + "=" * 80)
    print("测试数据归一化器")
    print("=" * 80)
    
    try:
        import pandas as pd
        from data import DataNormalizer
        
        # 创建示例数据
        df = pd.DataFrame({
            'open': [100, 102, 98, 105, 103],
            'high': [105, 108, 100, 110, 108],
            'low': [95, 98, 95, 100, 98],
            'close': [102, 100, 99, 108, 105],
            'volume': [1000, 1200, 800, 1500, 1100]
        })
        
        normalizer = DataNormalizer()
        
        # 测试Z-score归一化
        normalized_df = normalizer.normalize(df, method='zscore')
        print(f"✓ Z-score归一化成功")
        print(f"  原始数据形状: {df.shape}")
        print(f"  归一化后形状: {normalized_df.shape}")
        
        # 测试反归一化
        denormalized_df = normalizer.denormalize(normalized_df, method='zscore')
        print(f"✓ 反归一化成功")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_enricher():
    """测试数据增强器"""
    print("\n" + "=" * 80)
    print("测试数据增强器")
    print("=" * 80)
    
    try:
        import pandas as pd
        from data import DataEnricher
        
        # 创建示例数据
        df = pd.DataFrame({
            'open': [100, 102, 98, 105, 103, 101, 104, 106, 102, 105],
            'high': [105, 108, 100, 110, 108, 106, 109, 111, 107, 110],
            'low': [95, 98, 95, 100, 98, 99, 102, 104, 98, 102],
            'close': [102, 100, 99, 108, 105, 104, 107, 109, 105, 108],
            'volume': [1000, 1200, 800, 1500, 1100, 900, 1300, 1400, 1000, 1200]
        })
        
        enricher = DataEnricher()
        
        # 测试添加技术指标
        enriched_df = enricher.add_technical_indicators(df)
        print(f"✓ 添加技术指标成功")
        print(f"  原始列数: {len(df.columns)}")
        print(f"  增强后列数: {len(enriched_df.columns)}")
        print(f"  新增列: {list(set(enriched_df.columns) - set(df.columns))}")
        
        # 测试添加成交量分布
        enriched_df = enricher.add_volume_profile(enriched_df)
        print(f"✓ 添加成交量分布成功")
        
        # 测试全面增强
        enriched_df = enricher.enrich_data(df)
        print(f"✓ 全面增强成功")
        print(f"  最终列数: {len(enriched_df.columns)}")
        print(f"  最终列名: {list(enriched_df.columns)}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multi_source_manager():
    """测试多源数据管理器"""
    print("\n" + "=" * 80)
    print("测试多源数据管理器")
    print("=" * 80)
    
    try:
        from data import MultiSourceDataManager, YahooFinanceFetcher
        
        manager = MultiSourceDataManager()
        
        # 注册获取器
        manager.register_fetchers(
            api_fetchers={
                'yahoo_finance': YahooFinanceFetcher()
            }
        )
        
        # 测试获取并处理数据
        processed_data = await manager.fetch_and_process(
            symbols=["BTC-USD"],
            timeframe='1d',
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            normalize=True,
            enrich=True
        )
        
        print(f"✓ 获取并处理数据成功")
        for symbol, df in processed_data.items():
            print(f"  {symbol}: {len(df)} 条数据")
            print(f"    列名: {list(df.columns)[:5]}...")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("市场数据获取测试")
    print("=" * 80)
    
    results = {}
    
    # 运行导入测试
    results['imports'] = test_imports()
    
    # 运行功能测试
    results['yahoo_finance'] = await test_yahoo_finance()
    results['binance_orderbook'] = await test_binance_orderbook()
    results['binance_trades'] = await test_binance_trades()
    results['data_aggregator'] = await test_data_aggregator()
    results['data_normalizer'] = await test_data_normalizer()
    results['data_enricher'] = await test_data_enricher()
    results['multi_source_manager'] = await test_multi_source_manager()
    
    # 打印测试结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
        return True
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return False

if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
