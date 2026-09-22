"""
测试成交委托比率因子实现
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_auction_factors import (
    load_auction_event_frame,
    _apply_event_factors,
    _apply_transaction_order_ratio_factors,
)
from scripts.auction_tick_cache import AuctionTickCache


def check_transaction_order_ratio_basic():
    """测试基本的成交委托比率计算逻辑"""
    print("=" * 80)
    print("测试1: 基本逻辑验证")
    print("=" * 80)

    # 模拟数据
    stage1_adds = pd.DataFrame({
        'quantity': [1000, 1500, 800, 1200],
        'side': ['B', 'B', 'S', 'S'],
        'notional': [100000, 150000, 80000, 120000],
    })

    stage1_cancels = pd.DataFrame({
        'quantity': [200, 300],
        'side': ['B', 'S'],
        'notional': [20000, 30000],
    })

    stage2_adds = pd.DataFrame({
        'quantity': [500, 600],
        'side': ['B', 'S'],
        'notional': [50000, 60000],
    })

    matched_volume = 2500.0  # 最终成交量

    # 创建row字典
    row = {
        'auction_matched_volume': matched_volume,
    }

    # 调用函数
    _apply_transaction_order_ratio_factors(
        row,
        pd.Timestamp('2025-01-02'),
        pd.DataFrame(),  # events暂时不用
        stage1_adds,
        stage1_cancels,
        stage2_adds,
    )

    # 验证计算结果
    print("\n输入数据:")
    print(f"  阶段1买入委托: {stage1_adds[stage1_adds['side']=='B']['quantity'].sum():.0f} 股")
    print(f"  阶段1卖出委托: {stage1_adds[stage1_adds['side']=='S']['quantity'].sum():.0f} 股")
    print(f"  阶段1买入撤单: {stage1_cancels[stage1_cancels['side']=='B']['quantity'].sum():.0f} 股")
    print(f"  阶段1卖出撤单: {stage1_cancels[stage1_cancels['side']=='S']['quantity'].sum():.0f} 股")
    print(f"  阶段2买入委托: {stage2_adds[stage2_adds['side']=='B']['quantity'].sum():.0f} 股")
    print(f"  阶段2卖出委托: {stage2_adds[stage2_adds['side']=='S']['quantity'].sum():.0f} 股")
    print(f"  最终成交量: {matched_volume:.0f} 股")

    # 手工计算验证
    total_add = stage1_adds['quantity'].sum() + stage2_adds['quantity'].sum()
    stage1_net = stage1_adds['quantity'].sum() - stage1_cancels['quantity'].sum()
    stage2_total = stage2_adds['quantity'].sum()
    net_order = stage1_net + stage2_total

    print("\n中间计算:")
    print(f"  总委托量: {total_add:.0f} 股")
    print(f"  阶段1净委托: {stage1_net:.0f} 股")
    print(f"  阶段2委托: {stage2_total:.0f} 股")
    print(f"  总净委托: {net_order:.0f} 股")

    expected_submitted_ratio = matched_volume / total_add
    expected_net_ratio = matched_volume / net_order

    print("\n预期结果:")
    print(f"  成交/总委托比率: {expected_submitted_ratio:.4f}")
    print(f"  成交/净委托比率: {expected_net_ratio:.4f}")

    print("\n实际计算结果:")
    print(f"  auction_transaction_submitted_ratio: {row.get('auction_transaction_submitted_ratio', np.nan):.4f}")
    print(f"  auction_transaction_net_order_ratio: {row.get('auction_transaction_net_order_ratio', np.nan):.4f}")
    print(f"  auction_stage1_net_order_qty: {row.get('auction_stage1_net_order_qty', np.nan):.0f}")
    print(f"  auction_stage2_order_qty: {row.get('auction_stage2_order_qty', np.nan):.0f}")
    print(f"  auction_buy_order_imbalance: {row.get('auction_buy_order_imbalance', np.nan):.4f}")
    print(f"  auction_stage1_order_participation: {row.get('auction_stage1_order_participation', np.nan):.4f}")
    print(f"  auction_stage2_order_participation: {row.get('auction_stage2_order_participation', np.nan):.4f}")

    # 验证正确性
    if abs(row['auction_transaction_submitted_ratio'] - expected_submitted_ratio) < 0.0001:
        print("\n[OK] 成交/总委托比率计算正确")
    else:
        print(f"\n[X] 成交/总委托比率计算错误")

    if abs(row['auction_transaction_net_order_ratio'] - expected_net_ratio) < 0.0001:
        print("[OK] 成交/净委托比率计算正确")
    else:
        print(f"[X] 成交/净委托比率计算错误")


def check_real_data():
    """测试真实数据"""
    print("\n" + "=" * 80)
    print("测试2: 真实数据验证")
    print("=" * 80)

    # 测试路径
    test_symbol_dir = Path(r"E:\逐笔数据\2025\202501\20250102\000001.SZ")

    if not test_symbol_dir.exists():
        print(f"\n[X] 测试目录不存在: {test_symbol_dir}")
        return

    print(f"\n测试目录: {test_symbol_dir}")

    try:
        # 加载事件数据
        cache = AuctionTickCache(None)
        events, reconstruction_ok = load_auction_event_frame(
            test_symbol_dir,
            "000001.SZ",
            expected_trade_date="2025-01-02",
            cache=cache,
        )

        print(f"\n事件重建状态: {reconstruction_ok}")
        print(f"事件数量: {len(events)}")

        if events.empty:
            print("\n[X] 事件数据为空")
            return

        print(f"\n事件类型分布:")
        print(events['event_type'].value_counts())
        print(f"\n买卖方向分布:")
        print(events['side'].value_counts())

        # 统计委托和撤单
        adds = events[events['event_type'] == 'A']
        cancels = events[events['event_type'] == 'C']

        print(f"\n委托统计:")
        print(f"  总委托数: {len(adds)}")
        print(f"  总撤单数: {len(cancels)}")
        print(f"  买入委托: {len(adds[adds['side']=='B'])}")
        print(f"  卖出委托: {len(adds[adds['side']=='S'])}")
        print(f"  买入撤单: {len(cancels[cancels['side']=='B'])}")
        print(f"  卖出撤单: {len(cancels[cancels['side']=='S'])}")

        print(f"\n委托量统计:")
        print(f"  总委托量: {adds['quantity'].sum():,.0f} 股")
        print(f"  总撤单量: {cancels['quantity'].sum():,.0f} 股")
        print(f"  净委托量: {adds['quantity'].sum() - cancels['quantity'].sum():,.0f} 股")

        # 按阶段统计
        trade_day = pd.Timestamp('2025-01-02')
        split_time = trade_day + pd.Timedelta(hours=9, minutes=20)

        stage1_adds = adds[adds['trade_time'] < split_time]
        stage2_adds = adds[adds['trade_time'] >= split_time]
        stage1_cancels = cancels[cancels['trade_time'] < split_time]

        print(f"\n阶段统计:")
        print(f"  阶段1委托: {len(stage1_adds)} 笔, {stage1_adds['quantity'].sum():,.0f} 股")
        print(f"  阶段1撤单: {len(stage1_cancels)} 笔, {stage1_cancels['quantity'].sum():,.0f} 股")
        print(f"  阶段2委托: {len(stage2_adds)} 笔, {stage2_adds['quantity'].sum():,.0f} 股")

        print("\n[OK] 真实数据加载成功")
        print("\n提示: 可以基于这些数据计算成交委托比率因子")
        print("      但需要从快照数据或其他来源获取最终成交量")

    except Exception as e:
        print(f"\n[X] 加载失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print(" 成交委托比率因子实现测试")
    print("*" * 80)

    # 测试1: 基本逻辑
    check_transaction_order_ratio_basic()

    # 测试2: 真实数据
    check_real_data()

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

    print("\n新增因子说明:")
    print("  1. auction_transaction_submitted_ratio    - 成交/总委托比率")
    print("  2. auction_transaction_net_order_ratio    - 成交/净委托比率（扣除撤单）")
    print("  3. auction_stage1_net_order_qty           - 阶段1净委托量")
    print("  4. auction_stage2_order_qty               - 阶段2委托量")
    print("  5. auction_buy_order_imbalance            - 买入委托不平衡度")
    print("  6. auction_stage1_order_participation     - 阶段1委托参与度")
    print("  7. auction_stage2_order_participation     - 阶段2委托参与度")


if __name__ == "__main__":
    main()
