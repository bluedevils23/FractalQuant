"""
验证预开盘数据过滤
检查上海股票的09:14-09:15数据是否被正确排除
"""

import sys
from pathlib import Path
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.auction_tick_cache import AuctionTickCache
from scripts.generate_auction_factors import load_auction_event_frame


def check_raw_file():
    """检查原始文件中的预开盘数据"""
    print("=" * 80)
    print("1. 检查原始文件")
    print("=" * 80)

    raw_file = Path(r"E:\逐笔数据\2025\202501\20250102\600000.SH\逐笔委托.csv")

    if not raw_file.exists():
        print(f"\n[X] 文件不存在: {raw_file}")
        return

    print(f"\n读取文件: {raw_file}")

    try:
        df = pd.read_csv(raw_file, encoding='gbk')
        print(f"总记录数: {len(df)}")

        # 统计各时段
        pre_open = df[(df['时间'] >= 91400000) & (df['时间'] < 91500000)]
        stage1 = df[(df['时间'] >= 91500000) & (df['时间'] < 92000000)]
        stage2 = df[(df['时间'] >= 92000000) & (df['时间'] < 92500000)]

        print(f"\n时段统计:")
        print(f"  预开盘(09:14-09:15): {len(pre_open):>6} 条 ({len(pre_open)/len(df)*100:>5.2f}%)")
        print(f"  阶段1(09:15-09:20):  {len(stage1):>6} 条 ({len(stage1)/len(df)*100:>5.2f}%)")
        print(f"  阶段2(09:20-09:25):  {len(stage2):>6} 条 ({len(stage2)/len(df)*100:>5.2f}%)")

        if len(pre_open) > 0:
            print(f"\n预开盘数据样本（前5条）:")
            print(pre_open[['时间', '委托类型', '委托代码', '委托价格', '委托数量']].head())

            print(f"\n预开盘委托类型分布:")
            print(pre_open['委托类型'].value_counts())

            print(f"\n预开盘委托代码分布:")
            print(pre_open['委托代码'].value_counts())
        else:
            print("\n[INFO] 原始文件中无预开盘数据")

    except Exception as e:
        print(f"\n[X] 读取失败: {e}")
        import traceback
        traceback.print_exc()


def check_cache_loading():
    """检查AuctionTickCache加载后的数据"""
    print("\n" + "=" * 80)
    print("2. 检查AuctionTickCache加载")
    print("=" * 80)

    symbol_dir = Path(r"E:\逐笔数据\2025\202501\20250102\600000.SH")

    if not symbol_dir.exists():
        print(f"\n[X] 目录不存在: {symbol_dir}")
        return

    print(f"\n加载目录: {symbol_dir}")

    try:
        cache = AuctionTickCache(None)
        orders = cache.load_orders(symbol_dir)

        print(f"\n加载后的记录数: {len(orders)}")
        print(f"列名: {orders.columns.tolist()}")

        if 'trade_time' in orders.columns:
            min_time = orders['trade_time'].min()
            max_time = orders['trade_time'].max()
            print(f"\n时间范围:")
            print(f"  最小时间: {min_time}")
            print(f"  最大时间: {max_time}")

            # 检查是否有09:15之前的数据
            target_date = pd.Timestamp('2025-01-02')
            start_915 = target_date + pd.Timedelta(hours=9, minutes=15)
            start_914 = target_date + pd.Timedelta(hours=9, minutes=14)

            pre_open_count = ((orders['trade_time'] >= start_914) &
                             (orders['trade_time'] < start_915)).sum()

            if pre_open_count > 0:
                print(f"\n[X] 警告: 发现 {pre_open_count} 条预开盘数据（09:14-09:15）！")
                pre_open_data = orders[(orders['trade_time'] >= start_914) &
                                      (orders['trade_time'] < start_915)]
                print("\n预开盘数据样本:")
                print(pre_open_data[['trade_time', 'order_type', 'side', 'price', 'quantity']].head())
            else:
                print(f"\n[OK] 确认: 无预开盘数据，过滤正确")
                print(f"     所有数据 >= {start_915}")
        else:
            print("\n[WARN] 未找到trade_time字段")
            print(f"实际列名: {orders.columns.tolist()}")

    except Exception as e:
        print(f"\n[X] 加载失败: {e}")
        import traceback
        traceback.print_exc()


def check_event_reconstruction():
    """检查事件重建后的数据"""
    print("\n" + "=" * 80)
    print("3. 检查事件重建")
    print("=" * 80)

    symbol_dir = Path(r"E:\逐笔数据\2025\202501\20250102\600000.SH")

    if not symbol_dir.exists():
        print(f"\n[X] 目录不存在: {symbol_dir}")
        return

    print(f"\n重建事件: {symbol_dir}")

    try:
        cache = AuctionTickCache(None)
        events, ok = load_auction_event_frame(
            symbol_dir,
            "600000.SH",
            expected_trade_date="2025-01-02",
            cache=cache,
        )

        print(f"\n重建状态: {ok}")
        print(f"事件数量: {len(events)}")

        if not events.empty:
            print(f"\n事件类型分布:")
            print(events['event_type'].value_counts())

            print(f"\n买卖方向分布:")
            print(events['side'].value_counts())

            if 'trade_time' in events.columns:
                min_time = events['trade_time'].min()
                max_time = events['trade_time'].max()
                print(f"\n时间范围:")
                print(f"  最小时间: {min_time}")
                print(f"  最大时间: {max_time}")

                # 检查是否有09:15之前的数据
                target_date = pd.Timestamp('2025-01-02')
                start_915 = target_date + pd.Timedelta(hours=9, minutes=15)
                start_914 = target_date + pd.Timedelta(hours=9, minutes=14)

                pre_open_count = ((events['trade_time'] >= start_914) &
                                 (events['trade_time'] < start_915)).sum()

                if pre_open_count > 0:
                    print(f"\n[X] 警告: 发现 {pre_open_count} 条预开盘事件（09:14-09:15）！")
                    pre_open_events = events[(events['trade_time'] >= start_914) &
                                            (events['trade_time'] < start_915)]
                    print("\n预开盘事件样本:")
                    print(pre_open_events[['trade_time', 'event_type', 'side', 'price', 'quantity']].head())
                else:
                    print(f"\n[OK] 确认: 无预开盘事件，过滤正确")
                    print(f"     所有事件 >= {start_915}")
            else:
                print("\n[WARN] 未找到trade_time字段")
        else:
            print("\n[WARN] 事件数据为空")

    except Exception as e:
        print(f"\n[X] 重建失败: {e}")
        import traceback
        traceback.print_exc()


def compare_sh_sz():
    """对比上海和深圳的时间范围"""
    print("\n" + "=" * 80)
    print("4. 对比上海和深圳")
    print("=" * 80)

    sh_dir = Path(r"E:\逐笔数据\2025\202501\20250102\600000.SH")
    sz_dir = Path(r"E:\逐笔数据\2025\202501\20250102\000001.SZ")

    cache = AuctionTickCache(None)

    # 上海
    if sh_dir.exists():
        print("\n上海股票（600000.SH）:")
        try:
            events_sh, ok_sh = load_auction_event_frame(sh_dir, "600000.SH", cache=cache)
            if not events_sh.empty and 'trade_time' in events_sh.columns:
                print(f"  事件数量: {len(events_sh)}")
                print(f"  最小时间: {events_sh['trade_time'].min()}")
                print(f"  最大时间: {events_sh['trade_time'].max()}")
                print(f"  重建状态: {ok_sh}")
        except Exception as e:
            print(f"  [X] 加载失败: {e}")
    else:
        print(f"\n[X] 上海目录不存在")

    # 深圳
    if sz_dir.exists():
        print("\n深圳股票（000001.SZ）:")
        try:
            events_sz, ok_sz = load_auction_event_frame(sz_dir, "000001.SZ", cache=cache)
            if not events_sz.empty and 'trade_time' in events_sz.columns:
                print(f"  事件数量: {len(events_sz)}")
                print(f"  最小时间: {events_sz['trade_time'].min()}")
                print(f"  最大时间: {events_sz['trade_time'].max()}")
                print(f"  重建状态: {ok_sz}")
        except Exception as e:
            print(f"  [X] 加载失败: {e}")
    else:
        print(f"\n[X] 深圳目录不存在")


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print(" 预开盘数据过滤验证")
    print("*" * 80)

    # 1. 检查原始文件
    check_raw_file()

    # 2. 检查Cache加载
    check_cache_loading()

    # 3. 检查事件重建
    check_event_reconstruction()

    # 4. 对比上海和深圳
    compare_sh_sz()

    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
关键验证点:
1. [CHECK] 原始文件是否包含09:14-09:15的数据
2. [CHECK] AuctionTickCache加载是否过滤了预开盘数据
3. [CHECK] 事件重建后是否排除了预开盘数据
4. [CHECK] 上海和深圳的时间范围是否一致（都从09:15开始）

预期结果:
- 原始文件: 上海有预开盘数据，深圳无
- Cache加载: 两者都无预开盘数据（被过滤）
- 事件重建: 两者都从09:15:00开始
    """)


if __name__ == "__main__":
    main()
