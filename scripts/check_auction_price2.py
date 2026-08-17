#!/usr/bin/env python3
"""检查集合竞价阶段 bid_price2/ask_price2 的典型值"""

import pandas as pd
from pathlib import Path
import numpy as np

# 配置
TICK_ROOT = Path(r"E:\逐笔数据")
SAMPLE_SIZE = 5  # 检查几个标的
SAMPLE_DATES = 3  # 每个标的检查几个日期

def parse_trade_time(trade_date: pd.Series, raw_time: pd.Series) -> pd.Series:
    """解析交易时间"""
    date_text = trade_date.astype(str).str.zfill(8)
    time_text = raw_time.astype(str).str.zfill(9)
    return pd.to_datetime(
        date_text + time_text,
        format="%Y%m%d%H%M%S%f",
        errors="coerce",
    )

def load_quote_sample(csv_path: Path) -> pd.DataFrame:
    """加载行情快照并筛选集合竞价时段"""
    try:
        # 读取CSV
        raw = pd.read_csv(
            csv_path,
            encoding="gbk",
            low_memory=False,
        )

        # 重命名列
        column_map = {
            "自然日": "raw_trade_date",
            "时间": "raw_time",
            "申卖价1": "ask_price1",
            "申卖量1": "ask_qty1",
            "申买价1": "bid_price1",
            "申买量1": "bid_qty1",
            "申卖价2": "ask_price2",
            "申卖量2": "ask_qty2",
            "申买价2": "bid_price2",
            "申买量2": "bid_qty2",
            "申卖价3": "ask_price3",
            "申卖量3": "ask_qty3",
            "申买价3": "bid_price3",
            "申买量3": "bid_qty3",
        }

        available_renames = {k: v for k, v in column_map.items() if k in raw.columns}
        raw = raw.rename(columns=available_renames)

        # 解析时间
        raw["trade_time"] = parse_trade_time(raw["raw_trade_date"], raw["raw_time"])

        # 筛选集合竞价时段 09:15-09:25
        mask = (
            raw["trade_time"].dt.time >= pd.Timestamp("09:15").time()
        ) & (
            raw["trade_time"].dt.time < pd.Timestamp("09:25").time()
        )

        auction = raw.loc[mask].copy()

        # 转换数值列
        for col in ["bid_price1", "ask_price1", "bid_price2", "ask_price2",
                    "bid_price3", "ask_price3", "bid_qty1", "ask_qty1",
                    "bid_qty2", "ask_qty2", "bid_qty3", "ask_qty3"]:
            if col in auction.columns:
                auction[col] = pd.to_numeric(auction[col], errors="coerce")

        return auction

    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        return pd.DataFrame()

def analyze_price2_levels(df: pd.DataFrame) -> dict:
    """分析price2层级的特征"""
    if df.empty:
        return {}

    stats = {}

    # 检查price1 == price2的比例
    if all(col in df.columns for col in ["bid_price1", "bid_price2", "ask_price1", "ask_price2"]):
        price1_eq = (df["bid_price1"] == df["ask_price1"]).sum()
        price2_eq = (df["bid_price2"] == df["ask_price2"]).sum()

        bid12_eq = (df["bid_price1"] == df["bid_price2"]).sum()
        ask12_eq = (df["ask_price1"] == df["ask_price2"]).sum()

        stats["total_snapshots"] = len(df)
        stats["price1_equal_ratio"] = price1_eq / len(df)  # bid_price1 == ask_price1
        stats["price2_equal_ratio"] = price2_eq / len(df)  # bid_price2 == ask_price2
        stats["bid_price1_eq_price2_ratio"] = bid12_eq / len(df)
        stats["ask_price1_eq_price2_ratio"] = ask12_eq / len(df)

        # price2的有效性
        bid2_valid = df["bid_price2"].notna() & (df["bid_price2"] > 0)
        ask2_valid = df["ask_price2"].notna() & (df["ask_price2"] > 0)
        stats["bid_price2_valid_ratio"] = bid2_valid.sum() / len(df)
        stats["ask_price2_valid_ratio"] = ask2_valid.sum() / len(df)

        # 计算price2相对price1的偏离
        if bid2_valid.sum() > 0:
            bid_deviation = (df.loc[bid2_valid, "bid_price2"] /
                           df.loc[bid2_valid, "bid_price1"] - 1).abs()
            stats["bid_price2_deviation_median"] = bid_deviation.median()
            stats["bid_price2_deviation_mean"] = bid_deviation.mean()

        if ask2_valid.sum() > 0:
            ask_deviation = (df.loc[ask2_valid, "ask_price2"] /
                           df.loc[ask2_valid, "ask_price1"] - 1).abs()
            stats["ask_price2_deviation_median"] = ask_deviation.median()
            stats["ask_price2_deviation_mean"] = ask_deviation.mean()

        # 计算bid_price2 vs ask_price2的spread
        both_valid = bid2_valid & ask2_valid
        if both_valid.sum() > 0:
            spread = (df.loc[both_valid, "ask_price2"] - df.loc[both_valid, "bid_price2"]) / df.loc[both_valid, "bid_price1"]
            stats["price2_spread_median"] = spread.median()
            stats["price2_spread_mean"] = spread.mean()
            stats["price2_spread_positive_ratio"] = (spread > 0).sum() / both_valid.sum()

    return stats

def main():
    print("=" * 80)
    print("检查集合竞价阶段 bid_price2/ask_price2 的典型值")
    print("=" * 80)

    if not TICK_ROOT.exists():
        print(f"\n✗ 逐笔数据目录不存在: {TICK_ROOT}")
        return

    # 发现可用的日期目录
    date_dirs = sorted([
        d for d in TICK_ROOT.rglob("*/*/*")
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit()
    ])

    if not date_dirs:
        print(f"\n✗ 未找到日期目录")
        return

    print(f"\n找到 {len(date_dirs)} 个日期目录")

    # 随机选择一些日期
    sample_dates = date_dirs[-SAMPLE_DATES:] if len(date_dirs) >= SAMPLE_DATES else date_dirs

    all_stats = []

    for date_dir in sample_dates:
        print(f"\n日期: {date_dir.name}")
        print("-" * 80)

        # 找到该日期下的标的
        symbol_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()])[:SAMPLE_SIZE]

        for symbol_dir in symbol_dirs:
            symbol = symbol_dir.name
            print(f"\n  标的: {symbol}")

            # 找到行情文件
            quote_file = symbol_dir / "行情.csv"
            if not quote_file.exists():
                print(f"    ✗ 行情文件不存在")
                continue

            print(f"    ✓ 加载 {quote_file}")
            df = load_quote_sample(quote_file)

            if df.empty:
                continue

            stats = analyze_price2_levels(df)
            if stats:
                stats["date"] = date_dir.name
                stats["symbol"] = symbol
                all_stats.append(stats)

                print(f"    快照数: {stats['total_snapshots']}")
                print(f"    bid_price1 == ask_price1: {stats['price1_equal_ratio']:.2%}")
                print(f"    bid_price2 == ask_price2: {stats['price2_equal_ratio']:.2%}")
                print(f"    bid_price1 == bid_price2: {stats['bid_price1_eq_price2_ratio']:.2%}")
                print(f"    ask_price1 == ask_price2: {stats['ask_price1_eq_price2_ratio']:.2%}")
                print(f"    bid_price2有效率: {stats['bid_price2_valid_ratio']:.2%}")
                print(f"    ask_price2有效率: {stats['ask_price2_valid_ratio']:.2%}")

                if "bid_price2_deviation_median" in stats:
                    print(f"    bid_price2偏离price1 (中位): {stats['bid_price2_deviation_median']:.4%}")
                if "ask_price2_deviation_median" in stats:
                    print(f"    ask_price2偏离price1 (中位): {stats['ask_price2_deviation_median']:.4%}")
                if "price2_spread_median" in stats:
                    print(f"    price2 spread (中位): {stats['price2_spread_median']:.4%}")
                    print(f"    price2 spread > 0: {stats['price2_spread_positive_ratio']:.2%}")

    # 汇总统计
    if all_stats:
        print("\n" + "=" * 80)
        print("汇总统计")
        print("=" * 80)

        summary_df = pd.DataFrame(all_stats)

        print(f"\n样本数: {len(summary_df)}")
        print(f"\nbid_price1 == ask_price1 (均值): {summary_df['price1_equal_ratio'].mean():.2%}")
        print(f"bid_price2 == ask_price2 (均值): {summary_df['price2_equal_ratio'].mean():.2%}")
        print(f"bid_price1 == bid_price2 (均值): {summary_df['bid_price1_eq_price2_ratio'].mean():.2%}")
        print(f"ask_price1 == ask_price2 (均值): {summary_df['ask_price1_eq_price2_ratio'].mean():.2%}")

        if "price2_spread_median" in summary_df.columns:
            valid_spread = summary_df["price2_spread_median"].dropna()
            if not valid_spread.empty:
                print(f"\nprice2 spread 分布:")
                print(f"  中位数: {valid_spread.median():.4%}")
                print(f"  均值: {valid_spread.mean():.4%}")
                print(f"  最小值: {valid_spread.min():.4%}")
                print(f"  最大值: {valid_spread.max():.4%}")

        if "price2_spread_positive_ratio" in summary_df.columns:
            valid_positive = summary_df["price2_spread_positive_ratio"].dropna()
            if not valid_positive.empty:
                print(f"\nprice2 spread > 0 的比例 (均值): {valid_positive.mean():.2%}")

        print("\n结论:")
        print("-" * 80)

        # 判断price2是否有意义
        bid2_valid_avg = summary_df["bid_price2_valid_ratio"].mean()
        ask2_valid_avg = summary_df["ask_price2_valid_ratio"].mean()
        price2_eq_avg = summary_df["price2_equal_ratio"].mean()
        bid12_eq_avg = summary_df["bid_price1_eq_price2_ratio"].mean()

        if bid2_valid_avg < 0.5 or ask2_valid_avg < 0.5:
            print("✗ price2数据有效率过低（<50%），不适合用于计算spread")
        elif price2_eq_avg > 0.9:
            print("✗ bid_price2 == ask_price2 比例过高（>90%），price2无法形成有效spread")
        elif bid12_eq_avg > 0.9:
            print("✗ bid_price1 == bid_price2 比例过高（>90%），price2与price1重复，无额外信息")
        else:
            if "price2_spread_median" in summary_df.columns:
                spread_median = summary_df["price2_spread_median"].median()
                spread_positive = summary_df["price2_spread_positive_ratio"].mean()

                if spread_median > 0.0001 and spread_positive > 0.7:
                    print("✓ price2可以用于计算spread:")
                    print(f"  - bid_price2/ask_price2 形成有效价差 (中位 {spread_median:.4%})")
                    print(f"  - {spread_positive:.1%} 的快照中 ask_price2 > bid_price2")
                    print("\n建议：采用方案1，使用 (ask_price2 - bid_price2) / bid_price1 计算spread")
                else:
                    print("△ price2 spread信号较弱，但仍有一定区分度")
                    print("  建议：可尝试方案1，但需在因子生成后检查因子分布")
            else:
                print("? 数据不足以判断，建议检查更多样本")

if __name__ == "__main__":
    main()
