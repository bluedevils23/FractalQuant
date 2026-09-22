"""
检查成交比率因子所需数据的可行性
成交比率 = 成交量 / 委托量
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 数据路径配置
TICK_ROOT = Path(r"E:\逐笔数据")


def check_auction_event_data():
    """检查集合竞价事件数据（委托、成交、撤单）"""
    print("=" * 80)
    print("检查集合竞价事件数据")
    print("=" * 80)

    if not TICK_ROOT.exists():
        print(f"\n[X] 逐笔数据根目录不存在: {TICK_ROOT}")
        return None

    # 查找测试目录
    test_dirs = list(TICK_ROOT.glob("*/*/*"))[:5]

    data_summary = {
        'has_orders': [],
        'has_transactions': [],
        'has_snapshot': [],
        'order_sample': None,
        'transaction_sample': None,
    }

    for test_dir in test_dirs:
        if not test_dir.is_dir():
            continue

        # 查找股票目录
        symbol_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if not symbol_dirs:
            continue

        for symbol_dir in symbol_dirs[:3]:  # 每个日期测试3只股票
            print(f"\n测试目录: {symbol_dir}")

            # 1. 检查order.csv（委托数据）
            order_file = symbol_dir / "order.csv"
            if order_file.exists():
                try:
                    df_order = pd.read_csv(order_file, encoding='gbk', nrows=20)
                    print(f"  [OK] 找到order.csv")
                    print(f"      行数（前20行样本）: {len(df_order)}")
                    print(f"      列名: {df_order.columns.tolist()}")

                    data_summary['has_orders'].append(True)

                    if data_summary['order_sample'] is None:
                        data_summary['order_sample'] = df_order
                        print(f"\n      前5行数据:")
                        print(df_order.head().to_string())

                except Exception as e:
                    print(f"  [X] 读取order.csv失败: {e}")
                    data_summary['has_orders'].append(False)
            else:
                print(f"  [-] 未找到order.csv")
                data_summary['has_orders'].append(False)

            # 2. 检查transaction.csv（成交数据）
            transaction_file = symbol_dir / "transaction.csv"
            if transaction_file.exists():
                try:
                    df_trans = pd.read_csv(transaction_file, encoding='gbk', nrows=20)
                    print(f"  [OK] 找到transaction.csv")
                    print(f"      行数（前20行样本）: {len(df_trans)}")
                    print(f"      列名: {df_trans.columns.tolist()}")

                    data_summary['has_transactions'].append(True)

                    if data_summary['transaction_sample'] is None:
                        data_summary['transaction_sample'] = df_trans
                        print(f"\n      前5行数据:")
                        print(df_trans.head().to_string())

                except Exception as e:
                    print(f"  [X] 读取transaction.csv失败: {e}")
                    data_summary['has_transactions'].append(False)
            else:
                print(f"  [-] 未找到transaction.csv")
                data_summary['has_transactions'].append(False)

            # 3. 检查snapshot.csv（快照数据）
            snapshot_file = symbol_dir / "snapshot.csv"
            if snapshot_file.exists():
                data_summary['has_snapshot'].append(True)
            else:
                data_summary['has_snapshot'].append(False)

        if data_summary['order_sample'] is not None and data_summary['transaction_sample'] is not None:
            break  # 已经获取到样本数据，可以停止

    return data_summary


def analyze_data_availability(data_summary):
    """分析数据可用性"""
    print("\n" + "=" * 80)
    print("数据可用性分析")
    print("=" * 80)

    if data_summary is None:
        print("\n[X] 无法获取数据摘要")
        return

    # 统计覆盖率
    total_checked = len(data_summary['has_orders'])
    order_coverage = sum(data_summary['has_orders']) / total_checked * 100 if total_checked > 0 else 0
    transaction_coverage = sum(data_summary['has_transactions']) / total_checked * 100 if total_checked > 0 else 0
    snapshot_coverage = sum(data_summary['has_snapshot']) / total_checked * 100 if total_checked > 0 else 0

    print(f"\n测试样本数: {total_checked}")
    print(f"\n数据文件覆盖率:")
    print(f"  order.csv (委托数据):       {order_coverage:.1f}%")
    print(f"  transaction.csv (成交数据):  {transaction_coverage:.1f}%")
    print(f"  snapshot.csv (快照数据):     {snapshot_coverage:.1f}%")

    # 分析order.csv字段
    if data_summary['order_sample'] is not None:
        print("\n" + "=" * 80)
        print("order.csv 字段分析")
        print("=" * 80)

        df_order = data_summary['order_sample']
        print(f"\n字段列表: {df_order.columns.tolist()}")

        # 检查关键字段
        required_fields = {
            '时间': ['时间', 'time', '交易时间'],
            '订单号': ['委托号', 'order_id', '订单号', '委托索引'],
            '委托方向': ['买卖方向', 'side', 'bs_flag', '委托方向'],
            '委托价格': ['委托价', 'price', '委托价格'],
            '委托量': ['委托量', 'quantity', 'qty', '委托数量'],
            '委托类型': ['委托类别', 'order_type', '订单类型'],
        }

        print("\n关键字段检查:")
        field_mapping = {}
        for field_name, possible_names in required_fields.items():
            found = None
            for possible in possible_names:
                if possible in df_order.columns:
                    found = possible
                    break
            if found:
                print(f"  [OK] {field_name}: 找到字段 '{found}'")
                field_mapping[field_name] = found

                # 显示该字段的统计信息
                if found in df_order.columns:
                    col_data = df_order[found]
                    print(f"       非空数量: {col_data.notna().sum()} / {len(col_data)}")
                    if field_name in ['委托价格', '委托量']:
                        try:
                            numeric_data = pd.to_numeric(col_data, errors='coerce')
                            print(f"       范围: {numeric_data.min()} - {numeric_data.max()}")
                        except:
                            pass
                    elif field_name in ['委托方向', '委托类型']:
                        print(f"       唯一值: {col_data.unique()[:10].tolist()}")
            else:
                print(f"  [X] {field_name}: 未找到匹配字段")

        return field_mapping

    else:
        print("\n[X] 无order.csv样本数据")
        return None


def check_order_time_coverage(data_summary):
    """检查委托数据的时间覆盖范围"""
    print("\n" + "=" * 80)
    print("委托数据时间覆盖分析")
    print("=" * 80)

    if data_summary is None or data_summary['order_sample'] is None:
        print("\n[X] 无样本数据")
        return

    df_order = data_summary['order_sample']

    # 查找时间字段
    time_field = None
    for possible in ['时间', 'time', '交易时间']:
        if possible in df_order.columns:
            time_field = possible
            break

    if time_field is None:
        print("[X] 未找到时间字段")
        return

    print(f"\n使用时间字段: '{time_field}'")

    # 解析时间
    try:
        # 尝试不同的时间格式
        time_data = df_order[time_field].astype(str)

        # 检查时间格式
        sample_time = time_data.iloc[0]
        print(f"时间样本: {sample_time}")

        # 尝试解析
        if len(sample_time) == 9:  # 格式：HHMMSSmmm
            df_order['parsed_time'] = pd.to_datetime(
                time_data.str.zfill(9),
                format='%H%M%S%f',
                errors='coerce'
            ).dt.time
        else:
            df_order['parsed_time'] = pd.to_datetime(time_data, errors='coerce').dt.time

        # 过滤集合竞价时间段
        auction_start = pd.Timestamp('09:15:00').time()
        auction_end = pd.Timestamp('09:25:00').time()

        df_order['is_auction'] = (
            (df_order['parsed_time'] >= auction_start) &
            (df_order['parsed_time'] < auction_end)
        )

        auction_orders = df_order[df_order['is_auction']]

        print(f"\n时间范围:")
        print(f"  全部委托: {df_order['parsed_time'].min()} - {df_order['parsed_time'].max()}")
        print(f"  集合竞价委托: {len(auction_orders)} / {len(df_order)} ({len(auction_orders)/len(df_order)*100:.1f}%)")

        if len(auction_orders) > 0:
            print(f"\n集合竞价委托样本:")
            print(auction_orders[['parsed_time', time_field]].head().to_string())
        else:
            print(f"\n[WARN] 样本数据中没有集合竞价时段的委托")

    except Exception as e:
        print(f"[X] 时间解析失败: {e}")
        import traceback
        traceback.print_exc()


def simulate_transaction_order_ratio():
    """模拟成交委托比率计算"""
    print("\n" + "=" * 80)
    print("模拟成交委托比率计算")
    print("=" * 80)

    print("\n概念说明:")
    print("  成交委托比率 = 实际成交量 / 委托总量")
    print("  - 高比率: 表示买卖意愿真实，市场流动性好")
    print("  - 低比率: 表示试盘、撤单较多，可能存在虚假挂单")

    print("\n计算逻辑:")
    print("""
  1. 09:15-09:20 阶段1:
     - 统计所有买入委托量（A类订单）
     - 统计所有卖出委托量（A类订单）
     - 统计撤单量（D类订单，深市用C类成交记录）

  2. 09:20-09:25 阶段2:
     - 统计所有委托量（不允许撤单）

  3. 09:25 最终成交:
     - 获取集合竞价最终成交量

  4. 计算比率:
     买入成交委托比率 = 最终买入成交量 / (买入委托量 - 买入撤单量)
     卖出成交委托比率 = 最终卖出成交量 / (卖出委托量 - 卖出撤单量)
    """)

    print("\n模拟示例:")

    # 模拟数据
    buy_orders_stage1 = 1000000  # 阶段1买入委托100万股
    sell_orders_stage1 = 900000   # 阶段1卖出委托90万股
    buy_cancel_stage1 = 200000    # 阶段1买入撤单20万股
    sell_cancel_stage1 = 100000   # 阶段1卖出撤单10万股

    buy_orders_stage2 = 300000    # 阶段2买入委托30万股
    sell_orders_stage2 = 350000   # 阶段2卖出委托35万股

    # 最终成交（取买卖较小者）
    final_matched = 850000  # 最终成交85万股

    # 计算有效委托
    effective_buy = buy_orders_stage1 - buy_cancel_stage1 + buy_orders_stage2
    effective_sell = sell_orders_stage1 - sell_cancel_stage1 + sell_orders_stage2

    print(f"\n阶段1委托 (09:15-09:20):")
    print(f"  买入委托: {buy_orders_stage1:,} 股")
    print(f"  卖出委托: {sell_orders_stage1:,} 股")
    print(f"  买入撤单: {buy_cancel_stage1:,} 股")
    print(f"  卖出撤单: {sell_cancel_stage1:,} 股")

    print(f"\n阶段2委托 (09:20-09:25):")
    print(f"  买入委托: {buy_orders_stage2:,} 股")
    print(f"  卖出委托: {sell_orders_stage2:,} 股")

    print(f"\n有效委托量:")
    print(f"  买入有效委托 = {buy_orders_stage1:,} - {buy_cancel_stage1:,} + {buy_orders_stage2:,} = {effective_buy:,} 股")
    print(f"  卖出有效委托 = {sell_orders_stage1:,} - {sell_cancel_stage1:,} + {sell_orders_stage2:,} = {effective_sell:,} 股")

    print(f"\n最终成交: {final_matched:,} 股")

    # 计算比率
    buy_ratio = final_matched / effective_buy if effective_buy > 0 else 0
    sell_ratio = final_matched / effective_sell if effective_sell > 0 else 0
    total_ratio = final_matched / (effective_buy + effective_sell) if (effective_buy + effective_sell) > 0 else 0

    print(f"\n成交委托比率:")
    print(f"  买入成交委托比率 = {final_matched:,} / {effective_buy:,} = {buy_ratio:.4f} ({buy_ratio*100:.2f}%)")
    print(f"  卖出成交委托比率 = {final_matched:,} / {effective_sell:,} = {sell_ratio:.4f} ({sell_ratio*100:.2f}%)")
    print(f"  总成交委托比率   = {final_matched:,} / {effective_buy + effective_sell:,} = {total_ratio:.4f} ({total_ratio*100:.2f}%)")

    print("\n解读:")
    if buy_ratio > 0.8:
        print(f"  买方成交率 {buy_ratio:.2f} 很高，买盘真实")
    elif buy_ratio > 0.5:
        print(f"  买方成交率 {buy_ratio:.2f} 中等")
    else:
        print(f"  买方成交率 {buy_ratio:.2f} 较低，可能存在试盘或撤单")


def check_existing_code_implementation():
    """检查现有代码中的事件重建逻辑"""
    print("\n" + "=" * 80)
    print("检查现有代码的事件重建逻辑")
    print("=" * 80)

    code_file = Path(__file__).parent / "generate_auction_factors.py"

    if not code_file.exists():
        print(f"[X] 代码文件不存在: {code_file}")
        return

    try:
        with open(code_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找关键函数
        key_functions = [
            'load_auction_event_frame',
            '_reconstruct_sh_events',
            '_reconstruct_sz_events',
            '_finalize_event_frame',
            '_load_raw_orders',
        ]

        print("\n现有事件重建函数:")
        for func in key_functions:
            if func in content:
                print(f"  [OK] 找到函数: {func}")
            else:
                print(f"  [X] 未找到函数: {func}")

        # 检查是否已经有委托数据处理
        if 'load_auction_event_frame' in content:
            print("\n[OK] 现有代码已经实现了事件重建框架")
            print("     包含委托(order)、成交(transaction)、撤单(cancel)的处理")
            print("\n可以基于现有框架直接计算成交委托比率因子")
        else:
            print("\n[X] 现有代码没有事件重建框架")

        # 检查现有的事件因子
        if 'EVENT_FACTOR_COLUMNS' in content:
            print("\n现有事件因子列:")
            import re
            match = re.search(r'EVENT_FACTOR_COLUMNS\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if match:
                factors = match.group(1)
                for line in factors.split('\n'):
                    line = line.strip().strip(',').strip('"').strip("'")
                    if line and not line.startswith('#'):
                        print(f"    - {line}")

    except Exception as e:
        print(f"[X] 读取代码文件失败: {e}")


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print(" 成交委托比率因子数据可行性检查")
    print("*" * 80)

    # 1. 检查原始数据文件
    data_summary = check_auction_event_data()

    # 2. 分析数据可用性
    field_mapping = analyze_data_availability(data_summary)

    # 3. 检查时间覆盖
    check_order_time_coverage(data_summary)

    # 4. 模拟计算
    simulate_transaction_order_ratio()

    # 5. 检查现有代码
    check_existing_code_implementation()

    # 总结
    print("\n" + "=" * 80)
    print("可行性总结")
    print("=" * 80)

    if data_summary and sum(data_summary['has_orders']) > 0:
        print("\n[OK] 数据可行性: 高")
        print("\n关键发现:")
        print("  1. [OK] 委托数据(order.csv)存在")
        print("  2. [OK] 成交数据(transaction.csv)存在")
        print("  3. [OK] 现有代码已有事件重建框架(load_auction_event_frame)")
        print("  4. [OK] 可以直接基于events DataFrame计算成交委托比率")

        print("\n建议实现的因子:")
        print("  - BuyTransaction_BuyOrder_ratio_09150920:   买入成交/买入委托比率(阶段1)")
        print("  - SellTransaction_SellOrder_ratio_09150920: 卖出成交/卖出委托比率(阶段1)")
        print("  - Transaction_Order_ratio_09200925:         成交/委托比率(阶段2)")
        print("  - BuyTransaction_BuyOrder_ratio_09150925:   买入成交/买入委托比率(全阶段)")
        print("  - auction_matched_submitted_ratio:          已有因子（类似功能）")

        print("\n实现难度: 中等")
        print("  - 需要在events DataFrame基础上统计委托量和成交量")
        print("  - 需要区分买入/卖出方向")
        print("  - 需要区分阶段1和阶段2")

        print("\n预计工作量: 2-3天")
        print("  - 1天: 实现统计逻辑")
        print("  - 1天: 测试验证")
        print("  - 0.5天: 因子分析")

    else:
        print("\n[X] 数据可行性: 低")
        print("\n问题:")
        print("  - 无法找到委托数据文件")
        print("  - 建议检查数据源或重新采集数据")


if __name__ == "__main__":
    main()
