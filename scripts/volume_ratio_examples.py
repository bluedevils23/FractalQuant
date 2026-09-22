"""
量比因子计算示例
演示不同量比计算方法的差异
"""

import pandas as pd
import numpy as np


def example_auction_volume_ratio():
    """示例：集合竞价量比计算（方法一）"""
    print("=" * 80)
    print("示例1：集合竞价量比计算")
    print("=" * 80)

    # 模拟数据
    historical_volumes = pd.Series(
        data=[10000000, 12000000, 9000000, 11000000, 10500000],  # 单位：股
        index=pd.date_range("2024-01-01", periods=5),
        name="日成交量"
    )

    auction_volume = 3000000  # 集合竞价成交量：300万股

    print("\n历史日成交量（前5日）:")
    print(historical_volumes)

    # 计算5日平均
    avg_5d = historical_volumes.mean()
    print(f"\n前5日平均成交量: {avg_5d:,.0f} 股")
    print(f"集合竞价成交量: {auction_volume:,.0f} 股")

    # 计算量比
    volume_ratio = auction_volume / avg_5d
    print(f"\n量比 = {auction_volume:,.0f} / {avg_5d:,.0f} = {volume_ratio:.4f}")
    print(f"      = {volume_ratio * 100:.2f}%")

    # 解读
    print("\n解读:")
    if volume_ratio > 2.5:
        print(f"  ✓ 量比 {volume_ratio:.2f} > 2.5，显著放量！")
    elif volume_ratio > 1.0:
        print(f"  ○ 量比 {volume_ratio:.2f} > 1.0，温和放量")
    elif volume_ratio > 0.5:
        print(f"  - 量比 {volume_ratio:.2f}，接近正常水平")
    else:
        print(f"  ▼ 量比 {volume_ratio:.2f} < 0.5，缩量")

    return volume_ratio


def example_intraday_volume_ratio():
    """示例：盘中实时量比计算（方法二）"""
    print("\n" + "=" * 80)
    print("示例2：盘中实时量比计算（对比）")
    print("=" * 80)

    # 历史数据
    avg_daily_volume = 10500000  # 前5日平均日成交量
    minutes_per_day = 240  # 每日交易分钟数
    avg_volume_per_minute = avg_daily_volume / minutes_per_day

    print(f"\n历史基准:")
    print(f"  前5日平均日成交量: {avg_daily_volume:,.0f} 股")
    print(f"  平均每分钟成交量: {avg_volume_per_minute:,.0f} 股/分钟")

    # 模拟不同时刻的量比
    time_points = [
        ("10:00", 30, 5000000),   # 开盘30分钟，累计成交500万
        ("11:30", 120, 15000000), # 上午收盘，累计成交1500万
        ("14:00", 150, 18000000), # 下午1小时，累计成交1800万
        ("15:00", 210, 25000000), # 收盘前，累计成交2500万
    ]

    print(f"\n不同时刻的量比:")
    print(f"{'时刻':<8} {'已交易(分钟)':<12} {'累计成交量':<15} {'当前速度':<18} {'量比':<8}")
    print("-" * 80)

    for time, minutes, cumulative_vol in time_points:
        current_speed = cumulative_vol / minutes
        ratio = current_speed / avg_volume_per_minute

        print(f"{time:<8} {minutes:<12} {cumulative_vol:>12,}股  "
              f"{current_speed:>12,.0f}股/分  {ratio:>6.2f}")


def example_volume_ratio_zscore():
    """示例：量比Z-score标准化"""
    print("\n" + "=" * 80)
    print("示例3：量比Z-score标准化")
    print("=" * 80)

    # 模拟过去20日的量比数据
    np.random.seed(42)
    historical_ratios = pd.Series(
        data=np.random.normal(0.3, 0.1, 20),  # 均值0.3，标准差0.1
        index=pd.date_range("2024-01-01", periods=20),
        name="量比"
    )

    current_ratio = 0.85  # 今日量比

    print("\n过去20日量比分布:")
    print(f"  均值: {historical_ratios.mean():.4f}")
    print(f"  标准差: {historical_ratios.std():.4f}")
    print(f"  最小值: {historical_ratios.min():.4f}")
    print(f"  最大值: {historical_ratios.max():.4f}")

    print(f"\n今日量比: {current_ratio:.4f}")

    # 计算Z-score
    mean = historical_ratios.mean()
    std = historical_ratios.std()
    zscore = (current_ratio - mean) / std

    print(f"\nZ-score计算:")
    print(f"  Z = ({current_ratio:.4f} - {mean:.4f}) / {std:.4f}")
    print(f"  Z = {zscore:.4f}")

    print("\n解读:")
    if zscore > 2:
        print(f"  ✓ Z-score {zscore:.2f} > 2，极端放量（超过95%的历史水平）")
    elif zscore > 1:
        print(f"  ○ Z-score {zscore:.2f} > 1，显著放量（超过84%的历史水平）")
    elif zscore > 0:
        print(f"  - Z-score {zscore:.2f} > 0，高于均值")
    else:
        print(f"  ▼ Z-score {zscore:.2f} < 0，低于均值")


def example_comparison():
    """示例：不同量比方法的对比"""
    print("\n" + "=" * 80)
    print("示例4：集合竞价量比 vs 盘中量比的区别")
    print("=" * 80)

    # 假设场景
    avg_daily_volume = 10000000  # 前5日平均日成交量：1000万股
    auction_volume = 3000000     # 集合竞价成交量：300万股

    # 方法一：集合竞价量比
    auction_ratio = auction_volume / avg_daily_volume

    # 方法二：如果按盘中量比来算（假设15分钟后）
    # 集合竞价后，9:30开盘，到9:45是15分钟
    # 假设15分钟内又成交了200万股
    cumulative_vol_945 = auction_volume + 2000000  # 总共500万
    minutes_945 = 15
    current_speed = cumulative_vol_945 / minutes_945
    avg_speed = avg_daily_volume / 240
    intraday_ratio = current_speed / avg_speed

    print("\n场景设定:")
    print(f"  前5日平均日成交量: {avg_daily_volume:,.0f} 股")
    print(f"  集合竞价成交量(9:25): {auction_volume:,.0f} 股")
    print(f"  9:45累计成交量: {cumulative_vol_945:,.0f} 股")

    print("\n方法一：集合竞价量比")
    print(f"  量比 = {auction_volume:,.0f} / {avg_daily_volume:,.0f}")
    print(f"      = {auction_ratio:.4f} ({auction_ratio*100:.2f}%)")

    print("\n方法二：盘中实时量比（9:45时刻）")
    print(f"  平均速度 = {avg_daily_volume:,.0f} / 240分钟 = {avg_speed:,.0f} 股/分钟")
    print(f"  当前速度 = {cumulative_vol_945:,.0f} / {minutes_945}分钟 = {current_speed:,.0f} 股/分钟")
    print(f"  量比 = {current_speed:,.0f} / {avg_speed:,.0f}")
    print(f"      = {intraday_ratio:.4f}")

    print("\n对比:")
    print(f"  集合竞价量比: {auction_ratio:.4f} (只看集合竞价成交)")
    print(f"  盘中量比(9:45): {intraday_ratio:.4f} (看开盘15分钟的成交速度)")
    print("\n区别:")
    print("  - 集合竞价量比：衡量集合竞价阶段的参与热度")
    print("  - 盘中量比：衡量开盘后的持续交易活跃度")
    print("  - 两者关注的是不同的市场信号！")


def example_research_threshold():
    """示例：研报中的量比阈值应用"""
    print("\n" + "=" * 80)
    print("示例5：研报策略中的量比阈值")
    print("=" * 80)

    # 模拟不同股票的量比
    stocks = pd.DataFrame({
        '股票': ['股票A', '股票B', '股票C', '股票D', '股票E'],
        '集合竞价量比': [3.2, 1.8, 0.5, 5.1, 2.3],
        '换手率': [0.35, 0.25, 0.15, 0.42, 0.28],
        '集合竞价金额(万)': [1200, 800, 300, 2500, 950],
    })

    print("\n候选股票池:")
    print(stocks.to_string(index=False))

    # 应用研报中的筛选条件
    print("\n应用《集合竞价量比策略V1》的筛选条件:")
    print("  1. 量比 > 2.5")
    print("  2. 换手率 > 0.3")
    print("  3. 集合竞价金额 > 1000万")

    filtered = stocks[
        (stocks['集合竞价量比'] > 2.5) &
        (stocks['换手率'] > 0.3) &
        (stocks['集合竞价金额(万)'] > 1000)
    ]

    print("\n筛选结果:")
    if len(filtered) > 0:
        print(filtered.to_string(index=False))
        print(f"\n✓ 共筛选出 {len(filtered)} 只股票")
    else:
        print("  无符合条件的股票")


def main():
    """运行所有示例"""
    print("\n")
    print("*" * 80)
    print(" 量比计算方法详解与示例")
    print("*" * 80)

    # 示例1：基本计算
    example_auction_volume_ratio()

    # 示例2：盘中量比（对比）
    example_intraday_volume_ratio()

    # 示例3：Z-score标准化
    example_volume_ratio_zscore()

    # 示例4：方法对比
    example_comparison()

    # 示例5：研报阈值
    example_research_threshold()

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
在集合竞价因子中，我们使用的是：

  量比 = 集合竞价成交量 / 过去N日平均日成交量

关键特点：
  1. 只关注集合竞价阶段（9:15-9:25）的成交
  2. 对比历史日均量，评估开盘参与热度
  3. 量比 > 2.5 被认为是显著放量信号
  4. 结合换手率、成交金额等指标一起使用效果更佳

与盘中实时量比的区别：
  - 集合竞价量比：看的是"开盘前"的市场预期
  - 盘中量比：看的是"开盘后"的持续活跃度
  - 两者都是重要指标，但关注点不同
    """)


if __name__ == "__main__":
    main()
