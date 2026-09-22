"""验证量比因子所需数据的可用性"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 数据路径配置
STOCK_MINUTE_ROOT = Path(r"D:\workspace\stockdata\stock-data\行情数据\stock_1min")
ETF_MINUTE_ROOT = Path(r"D:\workspace\stockdata\etf-data\etf_1min")
STOCK_DAILY_PATH = Path(r"D:\workspace\stockdata\stock-data\行情数据\stock_daily.parquet")
ETF_DAILY_PATH = Path(r"D:\workspace\stockdata\etf-data\etf_daily.parquet")
TICK_ROOT = Path(r"E:\逐笔数据")


def check_minute_data_volume():
    """检查分钟数据中的成交量字段"""
    print("=" * 80)
    print("1. 检查分钟数据中的成交量字段")
    print("=" * 80)

    # 测试股票
    test_files = [
        STOCK_MINUTE_ROOT / "000001.SZ.parquet",
        STOCK_MINUTE_ROOT / "600000.SH.parquet",
        STOCK_MINUTE_ROOT / "000858.SZ.parquet",
    ]

    for file_path in test_files:
        if not file_path.exists():
            print(f"\n[X] 文件不存在: {file_path}")
            continue

        try:
            # 读取文件
            df = pd.read_parquet(file_path)
            print(f"\n[OK] {file_path.name}")
            print(f"   数据形状: {df.shape}")
            print(f"   列名: {df.columns.tolist()}")

            # 检查成交量字段
            volume_cols = [col for col in df.columns if 'vol' in col.lower()]
            if volume_cols:
                print(f"   找到成交量相关字段: {volume_cols}")

                for col in volume_cols:
                    vol_data = pd.to_numeric(df[col], errors='coerce')
                    print(f"\n   {col} 统计:")
                    print(f"     - 非空数量: {vol_data.notna().sum()} / {len(vol_data)}")
                    print(f"     - 最小值: {vol_data.min():.0f}")
                    print(f"     - 最大值: {vol_data.max():.0f}")
                    print(f"     - 平均值: {vol_data.mean():.0f}")
                    print(f"     - 中位数: {vol_data.median():.0f}")

                    # 显示前几行
                    print(f"\n   前5行数据:")
                    print(df[[col]].head())
            else:
                print("   [WARN] 未找到成交量相关字段")

            # 检查索引结构
            print(f"\n   索引类型: {type(df.index)}")
            if isinstance(df.index, pd.MultiIndex):
                print(f"   索引名称: {df.index.names}")
            else:
                print(f"   索引名称: {df.index.name}")

        except Exception as e:
            print(f"\n[X] 读取失败: {file_path}")
            print(f"   错误: {e}")


def check_daily_data_volume():
    """检查日线数据中的成交量字段"""
    print("\n" + "=" * 80)
    print("2. 检查日线数据中的成交量字段")
    print("=" * 80)

    for data_path in [STOCK_DAILY_PATH, ETF_DAILY_PATH]:
        if not data_path.exists():
            print(f"\n[X] 文件不存在: {data_path}")
            continue

        try:
            print(f"\n[OK] {data_path.name}")

            # 读取schema
            import pyarrow.parquet as pq
            schema = pq.read_schema(data_path)
            print(f"   字段列表: {schema.names}")

            # 读取部分数据
            df = pd.read_parquet(
                data_path,
                columns=['vol', 'amount'] if 'vol' in schema.names else ['amount']
            )

            if 'vol' in df.columns:
                vol_data = pd.to_numeric(df['vol'], errors='coerce')
                print(f"\n   vol 统计:")
                print(f"     - 非空数量: {vol_data.notna().sum()} / {len(vol_data)}")
                print(f"     - 最小值: {vol_data.min():.0f}")
                print(f"     - 最大值: {vol_data.max():.0f}")
                print(f"     - 平均值: {vol_data.mean():.0f}")
                print(f"     - 中位数: {vol_data.median():.0f}")

                # 检查单位（可能是手或股）
                print(f"\n   单位推测:")
                if vol_data.median() < 1e6:
                    print(f"     [WARN] 中位数较小，可能是'手'（需要×100转为股）")
                else:
                    print(f"     [OK] 中位数较大，可能已是'股'")
            else:
                print("   [X] 未找到vol字段")

            # 检查amount字段
            if 'amount' in df.columns:
                amt_data = pd.to_numeric(df['amount'], errors='coerce')
                print(f"\n   amount 统计:")
                print(f"     - 平均值: {amt_data.mean():.0f}")
                print(f"     - 中位数: {amt_data.median():.0f}")

        except Exception as e:
            print(f"\n[X] 读取失败: {data_path}")
            print(f"   错误: {e}")


def check_auction_data_volume():
    """检查集合竞价数据中的成交量"""
    print("\n" + "=" * 80)
    print("3. 检查集合竞价数据中的成交量")
    print("=" * 80)

    if not TICK_ROOT.exists():
        print(f"\n[X] 逐笔数据根目录不存在: {TICK_ROOT}")
        return

    # 查找一个测试日期
    test_dirs = list(TICK_ROOT.glob("*/*/*"))[:3]

    for test_dir in test_dirs:
        if not test_dir.is_dir():
            continue

        # 查找第一个股票目录
        symbol_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if not symbol_dirs:
            continue

        symbol_dir = symbol_dirs[0]
        print(f"\n测试目录: {symbol_dir}")

        # 查找快照文件
        snapshot_file = symbol_dir / "snapshot.csv"
        if not snapshot_file.exists():
            print(f"   [WARN] 未找到snapshot.csv")
            continue

        try:
            # 读取快照数据
            df = pd.read_csv(snapshot_file, encoding='gbk')
            print(f"   [OK] 快照数据: {df.shape}")
            print(f"   列名: {df.columns.tolist()}")

            # 检查成交量字段
            if '成交量' in df.columns:
                print(f"\n   成交量统计:")
                print(df['成交量'].describe())

                # 检查时间范围
                if '时间' in df.columns:
                    print(f"\n   时间范围: {df['时间'].min()} - {df['时间'].max()}")

            # 检查是否有逐笔成交文件
            transaction_file = symbol_dir / "transaction.csv"
            if transaction_file.exists():
                df_trans = pd.read_csv(transaction_file, encoding='gbk', nrows=10)
                print(f"\n   [OK] 逐笔成交数据存在")
                print(f"   列名: {df_trans.columns.tolist()}")

        except Exception as e:
            print(f"   [X] 读取失败: {e}")

        break  # 只测试一个


def simulate_volume_ratio_calculation():
    """模拟量比因子计算流程"""
    print("\n" + "=" * 80)
    print("4. 模拟量比因子计算流程")
    print("=" * 80)

    # 测试文件
    test_file = STOCK_MINUTE_ROOT / "000001.SZ.parquet"
    if not test_file.exists():
        print(f"[X] 测试文件不存在: {test_file}")
        return

    try:
        print(f"\n测试股票: {test_file.name}")

        # 1. 读取分钟数据
        df = pd.read_parquet(test_file, columns=['vol', 'amount'])
        print(f"[OK] 读取分钟数据: {df.shape}")

        # 2. 提取交易日期
        if isinstance(df.index, pd.MultiIndex):
            level = "trade_date" if "trade_date" in df.index.names else 0
            trade_dates = pd.to_datetime(df.index.get_level_values(level))
        elif df.index.name == "trade_date":
            trade_dates = pd.to_datetime(df.index)
        elif "trade_date" in df.columns:
            trade_dates = pd.to_datetime(df["trade_date"])
        else:
            print("[X] 无法找到trade_date字段")
            return

        print(f"[OK] 提取交易日期")

        # 3. 转换成交量（可能需要×100）
        volumes = pd.to_numeric(df["vol"], errors="coerce")

        # 判断单位
        median_vol = volumes.median()
        if median_vol < 1e6:
            print(f"[WARN] 检测到成交量中位数较小 ({median_vol:.0f})，乘以100转换为股")
            volumes = volumes * 100
        else:
            print(f"[OK] 成交量单位似乎已是股 (中位数: {median_vol:.0f})")

        # 4. 按日聚合
        daily_volumes = volumes.groupby(trade_dates.normalize()).sum()
        print(f"\n[OK] 聚合为日成交量:")
        print(f"   交易日数量: {len(daily_volumes)}")
        print(f"   日期范围: {daily_volumes.index.min()} - {daily_volumes.index.max()}")
        print(f"\n   最近5个交易日:")
        print(daily_volumes.tail())

        # 5. 计算量比（模拟）
        print(f"\n[OK] 模拟量比计算:")
        if len(daily_volumes) >= 6:
            # 假设最后一天是集合竞价日
            recent_5d = daily_volumes.iloc[-6:-1]
            avg_5d = recent_5d.mean()

            # 假设集合竞价成交量是日成交量的30%
            simulated_auction_volume = daily_volumes.iloc[-1] * 0.3
            volume_ratio = simulated_auction_volume / avg_5d

            print(f"   前5日平均成交量: {avg_5d:,.0f} 股")
            print(f"   模拟集合竞价成交量: {simulated_auction_volume:,.0f} 股")
            print(f"   量比: {volume_ratio:.4f}")

            if 0.1 < volume_ratio < 10:
                print(f"   [OK] 量比在合理范围内")
            else:
                print(f"   [WARN] 量比异常，需检查数据单位")

    except Exception as e:
        print(f"[X] 模拟失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print(" 量比因子数据验证脚本")
    print("*" * 80)

    # 1. 检查分钟数据
    check_minute_data_volume()

    # 2. 检查日线数据
    check_daily_data_volume()

    # 3. 检查集合竞价数据
    check_auction_data_volume()

    # 4. 模拟计算流程
    simulate_volume_ratio_calculation()

    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
[CHECK] 需要确认的关键点：

1. 分钟数据中是否有 'vol' 字段
2. 'vol' 的单位是'手'还是'股'（需要×100转换吗？）
3. 日线数据中是否有 'vol' 字段
4. 集合竞价最终成交量从哪里获取（现有代码中的 auction_matched_volume）

[ACTION] 下一步行动：

1. 如果vol字段存在且单位正确 -> 直接实现量比因子
2. 如果vol字段需要单位转换 -> 在代码中添加×100
3. 如果vol字段不存在 -> 需要从其他数据源获取或重新采集数据
    """)


if __name__ == "__main__":
    main()
