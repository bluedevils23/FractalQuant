# CICC因子生成器：从分钟回填改为日级输出

修改日期：2026-08-18

## 1. 修改背景

根据研报分析（`report_475_factor_and_strategy_recommendations.md`），当前项目中：

- ETF侧已有CICC分钟因子实现（58个因子）
- 原始实现将日级因子回填到每个分钟bar，造成存储冗余
- 研报建议的用法是"日级选股因子"，不需要分钟粒度
- 后续需要支持20日滚动平滑（如 `pv_corr_avg_20d`），日级输出更适合

## 2. 主要修改

### 2.1 脚本名称与路径

- **脚本**：`scripts/generate_etf_cicc_minute_factors.py`（名称未改，保持兼容）
- **日志名称**：`generate_etf_cicc_daily_factors`
- **输出目录**：`D:\workspace\stockdata\etf-data\etf_cicc_daily_factors`
  - 原输出：`etf_1min_cicc_factors`（分钟级，每个文件几万行）
  - 新输出：`etf_cicc_daily_factors`（日级，每个文件几百行）

### 2.2 代码修改

**删除的函数**：
```python
def merge_daily_factors_back(minute_df, daily_exposure) -> pd.DataFrame:
    # 此函数将日级因子回填到分钟数据，已删除
```

**修改的函数**：
```python
def process_file(input_path, output_root, overwrite):
    # 原：计算daily_exposure后merge回minute_df
    # 新：直接输出daily_exposure，不再回填
    
    daily_exposure, factor_columns = calculate_daily_factor_exposures(cicc_df)
    
    # Output daily factors directly, no merge back to minute data
    result_df = daily_exposure.rename(columns={"code": "ts_code", "date": "trade_date"})
    result_df = result_df.set_index("trade_date")
```

**输出schema变化**：

| 字段类型 | 原输出（分钟级） | 新输出（日级） |
|---|---|---|
| Index | `trade_time` (分钟时间戳) | `trade_date` (交易日期) |
| OHLCV列 | ✓ 包含 open, high, low, close, volume | ✗ 不包含 |
| 因子列 | ✓ 58个CICC因子（每分钟重复相同值） | ✓ 58个CICC因子（每日唯一值） |
| 行数 | ~240行/天 × N天 | N行（N=交易日数） |

### 2.3 存储效率

以一个ETF 2年历史数据为例：

- **原输出**：~480天 × 240分钟 = 115,200行，每行包含OHLCV + 58因子
- **新输出**：~480行，每行只包含ts_code + 58因子
- **压缩比**：约 240:1（单文件）

对于100个ETF，存储节省约 **95%**。

## 3. 使用方式

### 3.1 生成因子

```bash
# 生成所有ETF的日级因子
python -m scripts.generate_etf_cicc_minute_factors

# 只生成指定ETF
python -m scripts.generate_etf_cicc_minute_factors --symbols 510300.SH 159001.SZ

# 并行处理（8个worker）
python -m scripts.generate_etf_cicc_minute_factors --workers 8 --overwrite
```

### 3.2 读取因子

```python
import pandas as pd

# 读取单个ETF的日级因子
df = pd.read_parquet("D:/workspace/stockdata/etf-data/etf_cicc_daily_factors/510300.SH.parquet")
# Index: trade_date (datetime)
# Columns: ts_code, mmt_pm, mmt_last30, corr_pv, shape_skew, ...

# 示例：计算20日滚动平均
df['corr_pv_avg_20d'] = df['corr_pv'].rolling(20, min_periods=10).mean()
df['shape_skew_avg_20d'] = df['shape_skew'].rolling(20, min_periods=10).mean()
```

### 3.3 合并多个ETF

```python
from pathlib import Path
import pandas as pd

factor_root = Path("D:/workspace/stockdata/etf-data/etf_cicc_daily_factors")
files = sorted(factor_root.glob("*.parquet"))

# 读取所有ETF并拼接
dfs = [pd.read_parquet(f) for f in files]
panel = pd.concat(dfs, ignore_index=False)
panel = panel.reset_index()  # trade_date变为列
panel = panel.sort_values(['ts_code', 'trade_date'])

print(f"Panel shape: {panel.shape}")
# 预期：(N_etfs × N_days, 1 + 58) 约 (48000, 59)
```

## 4. 下游影响

### 4.1 需要更新的脚本

以下脚本如果依赖原分钟级CICC因子，需要更新：

```bash
# 检查依赖
grep -r "etf_1min_cicc_factors" FractalQuant/
```

预期结果：应该没有脚本依赖（原实现主要用于研究验证）。

### 4.2 回测与特征工程

- **日级选股模型**：直接使用新输出，更高效
- **分钟级回测**：如需要在分钟bar上做决策，可以用 `pd.merge(minute_df, daily_df, left_on='trade_date', right_index=True, how='left')`
- **20日平滑特征**：在日级因子上直接 `.rolling(20).mean()`，避免在分钟级数据上重复计算

### 4.3 股票侧扩展

当前修改只影响ETF。下一步扩展到股票侧时，建议：

```python
# 新建独立脚本
scripts/generate_stock_cicc_daily_factors.py
  ├─ 读取 stock_1min/*.parquet
  ├─ 调用相同的 cicc_methods 函数
  └─ 输出到 stock_cicc_daily_factors/*.parquet

# 共用的日级聚合逻辑可抽取为
FractalQuant/factor/cicc_daily_aggregator.py
```

## 5. 验证步骤

### 5.1 功能验证

```bash
# 1. 生成单个ETF
python -m scripts.generate_etf_cicc_minute_factors --symbols 510300.SH --overwrite

# 2. 检查输出
python test_cicc_daily_output.py
```

预期输出：
```
Output shape: (480, 59)  # ~2年交易日
Index: trade_date
Columns: ['ts_code', 'mmt_pm', 'mmt_last30', ...]
✓ Output is daily-level (480 rows, likely trading days)
✓ ts_code column present: 510300.SH
```

### 5.2 因子值一致性验证

确认修改前后同一交易日的因子值不变：

```python
# 读取旧版本（如果还存在）
old_df = pd.read_parquet("etf_1min_cicc_factors/510300.SH.parquet")
old_daily = old_df.groupby(old_df.index.normalize()).first()

# 读取新版本
new_df = pd.read_parquet("etf_cicc_daily_factors/510300.SH.parquet")

# 比对因子列
factor_cols = [c for c in new_df.columns if c != 'ts_code']
comparison = old_daily[factor_cols].compare(new_df[factor_cols])
assert len(comparison) == 0, "Factor values changed!"
```

## 6. 后续工作

### P0：立即可做
- ✓ 修改CICC生成器为日级输出（已完成）
- ☐ 对10个ETF做完整回测验证输出正确性
- ☐ 添加单元测试验证日级聚合逻辑

### P1：中期规划
- ☐ 实现20日滚动平滑特征生成器
  ```python
  # 输入：etf_cicc_daily_factors/*.parquet
  # 输出：etf_cicc_daily_smoothed/*.parquet（额外58列）
  corr_pv_avg_20d, corr_pv_std_20d, corr_pv_trend_20d, ...
  ```
- ☐ 创建股票侧CICC日级因子生成器
- ☐ 抽取ETF/股票共用的聚合逻辑

### P2：长期优化
- ☐ 实现Benford检验因子（`benford_volume_deviation_5000`）
- ☐ 在主选股模型中测试20日平滑特征的增量IC
- ☐ 对比滚动分钟特征（如`volatility_skew_20`）与日级聚合特征的信息正交性

## 7. 注意事项

1. **因果性**：日级因子在 `trade_date` 收盘后才可用，使用时确保 `available_time = trade_date + 1个交易日开盘`
2. **缺失值处理**：某些交易日可能因数据质量导致因子为NaN，下游应统一使用 `.fillna()` 或 `.dropna()`
3. **复权处理**：当前脚本未处理ETF复权，如需要应在读取分钟数据时预先调整
4. **内存优化**：如果单个ETF历史过长（>10年），可考虑按年份分片输出

## 8. Changelog

| 日期 | 版本 | 修改内容 |
|---|---|---|
| 2026-08-18 | v2.0 | 改为日级输出，删除分钟回填逻辑 |
| (原始) | v1.0 | 日级因子回填到分钟数据 |
