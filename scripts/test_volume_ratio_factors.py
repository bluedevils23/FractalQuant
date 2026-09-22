"""
测试量比因子实现
测试单只股票的量比因子计算
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_auction_factors import (
    load_daily_volume_history,
    _apply_volume_ratio_factors,
    _apply_volume_ratio_zscore,
)
import pandas as pd
import numpy as np


def check_load_daily_volume_history():
    """测试日成交量历史加载"""
    print("=" * 80)
    print("测试1: 加载日成交量历史")
    print("=" * 80)

    minute_path = Path(r"D:\workspace\stockdata\stock-data\行情数据\stock_1min\000001.SZ.parquet")

    if not minute_path.exists():
        print(f"[X] 测试文件不存在: {minute_path}")
        return None

    try:
        daily_volumes = load_daily_volume_history(minute_path)
        print(f"\n[OK] 成功加载日成交量历史")
        print(f"  交易日数量: {len(daily_volumes)}")
        print(f"  日期范围: {daily_volumes.index.min()} - {daily_volumes.index.max()}")
        print(f"\n  统计信息:")
        print(f"    最小值: {daily_volumes.min():,.0f} 股")
        print(f"    最大值: {daily_volumes.max():,.0f} 股")
        print(f"    平均值: {daily_volumes.mean():,.0f} 股")
        print(f"    中位数: {daily_volumes.median():,.0f} 股")

        print(f"\n  最近10个交易日:")
        print(daily_volumes.tail(10))

        return daily_volumes

    except Exception as e:
        print(f"[X] 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_volume_ratio_calculation(daily_volumes):
    """测试量比因子计算"""
    print("\n" + "=" * 80)
    print("测试2: 量比因子计算")
    print("=" * 80)

    if daily_volumes is None or len(daily_volumes) < 25:
        print("[X] 历史数据不足，无法测试")
        return

    # 构造测试数据框
    test_dates = daily_volumes.index[-25:].tolist()
    test_data = []

    for date in test_dates:
        # 模拟集合竞价成交量为日成交量的30%
        day_volume = daily_volumes.loc[date]
        auction_volume = day_volume * 0.3

        test_data.append({
            "trade_date": date.strftime("%Y-%m-%d"),
            "auction_matched_volume": auction_volume,
            "auction_volume_ratio_5d": np.nan,
            "auction_volume_ratio_20d": np.nan,
            "auction_volume_ratio_5d_zscore": np.nan,
        })

    result = pd.DataFrame(test_data)

    # 计算量比因子
    print("\n[INFO] 计算量比因子...")
    for index, row in result.iterrows():
        _apply_volume_ratio_factors(result, index, daily_volumes)

    # 计算Z-score
    print("[INFO] 计算Z-score...")
    _apply_volume_ratio_zscore(result)

    # 显示结果
    print("\n[OK] 计算完成")
    print("\n最近5个交易日的量比因子:")
    display_columns = [
        "trade_date",
        "auction_matched_volume",
        "auction_volume_ratio_5d",
        "auction_volume_ratio_20d",
        "auction_volume_ratio_5d_zscore",
    ]

    tail_data = result[display_columns].tail()
    print("\n" + tail_data.to_string(index=False))

    # 统计分析
    print("\n量比因子统计:")
    print("\nauction_volume_ratio_5d:")
    ratio_5d = result["auction_volume_ratio_5d"].dropna()
    if len(ratio_5d) > 0:
        print(f"  数量: {len(ratio_5d)}")
        print(f"  均值: {ratio_5d.mean():.4f}")
        print(f"  标准差: {ratio_5d.std():.4f}")
        print(f"  最小值: {ratio_5d.min():.4f}")
        print(f"  最大值: {ratio_5d.max():.4f}")

    print("\nauction_volume_ratio_20d:")
    ratio_20d = result["auction_volume_ratio_20d"].dropna()
    if len(ratio_20d) > 0:
        print(f"  数量: {len(ratio_20d)}")
        print(f"  均值: {ratio_20d.mean():.4f}")
        print(f"  标准差: {ratio_20d.std():.4f}")
        print(f"  最小值: {ratio_20d.min():.4f}")
        print(f"  最大值: {ratio_20d.max():.4f}")

    print("\nauction_volume_ratio_5d_zscore:")
    zscore = result["auction_volume_ratio_5d_zscore"].dropna()
    if len(zscore) > 0:
        print(f"  数量: {len(zscore)}")
        print(f"  均值: {zscore.mean():.4f}")
        print(f"  标准差: {zscore.std():.4f}")
        print(f"  最小值: {zscore.min():.4f}")
        print(f"  最大值: {zscore.max():.4f}")

    # 验证逻辑正确性
    print("\n" + "=" * 80)
    print("验证逻辑正确性")
    print("=" * 80)

    last_row = result.iloc[-1]
    print(f"\n最后一个交易日: {last_row['trade_date']}")
    print(f"  集合竞价成交量: {last_row['auction_matched_volume']:,.0f} 股")

    # 手工计算验证
    last_date = pd.Timestamp(last_row['trade_date'])
    prior_volumes = daily_volumes.loc[daily_volumes.index < last_date]

    if len(prior_volumes) >= 5:
        recent_5d = prior_volumes.iloc[-5:]
        avg_5d = recent_5d.mean()
        expected_ratio_5d = last_row['auction_matched_volume'] / avg_5d

        print(f"\n  前5日成交量: {recent_5d.values}")
        print(f"  前5日平均: {avg_5d:,.0f} 股")
        print(f"  预期量比(5日): {expected_ratio_5d:.4f}")
        print(f"  实际量比(5日): {last_row['auction_volume_ratio_5d']:.4f}")

        if abs(expected_ratio_5d - last_row['auction_volume_ratio_5d']) < 0.0001:
            print("  [OK] 5日量比计算正确")
        else:
            print("  [WARN] 5日量比计算可能有误")

    if len(prior_volumes) >= 20:
        recent_20d = prior_volumes.iloc[-20:]
        avg_20d = recent_20d.mean()
        expected_ratio_20d = last_row['auction_matched_volume'] / avg_20d

        print(f"\n  前20日平均: {avg_20d:,.0f} 股")
        print(f"  预期量比(20日): {expected_ratio_20d:.4f}")
        print(f"  实际量比(20日): {last_row['auction_volume_ratio_20d']:.4f}")

        if abs(expected_ratio_20d - last_row['auction_volume_ratio_20d']) < 0.0001:
            print("  [OK] 20日量比计算正确")
        else:
            print("  [WARN] 20日量比计算可能有误")


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print(" 量比因子实现测试")
    print("*" * 80)

    # 测试1: 加载日成交量历史
    daily_volumes = check_load_daily_volume_history()

    if daily_volumes is not None:
        # 测试2: 量比因子计算
        check_volume_ratio_calculation(daily_volumes)

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
