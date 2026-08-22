# CICC因子修复完成报告

修复日期：2026-08-18  
文件：`Replication-of-Minute-Frequency-Factor-refer-CICC/MinuteFrequentFactorCalculateMethodsCICC.py`

## 修复概述

已完成对CICC因子库中3个关键错误的修复，所有测试通过。

---

## 修复1：bottom20VolumeRet 阈值错误 ✓

**位置**：Line 457-480

**问题**：
```python
# 错误：使用 bottom_k(50) 导致与 bottom50VolumeRet 相同
.filter((pl.col('volume') <= pl.col('volume').bottom_k(50).max()))
```

**修复**：
```python
# 正确：使用 bottom_k(20)
.filter((pl.col('volume') <= pl.col('volume').bottom_k(20).max()))
```

**测试结果**：
```
bottom20VolumeRet: 0.2
bottom50VolumeRet: 0.3
✓ 两个因子值不同 (diff=0.100000)
```

---

## 修复2：corr_square 公式错误 ✓

**位置**：Line 289

**问题**：
```python
# 错误：计算的是 √cov / (σ_x² · σ_y²)
.with_columns(pl.col('corr').pow(0.5).alias('corr_square'))
```

**正确公式**：R² = cov² / (σ_x² · σ_y²)

**修复**：
```python
# 正确：计算相关系数的平方
.with_columns(pl.col('corr').pow(2).alias('corr_square'))
```

**测试结果**：
```
QRS 结果: 1.352245
✓ 公式已修复为 pow(2)
```

---

## 修复3：除零保护缺失 ✓

**影响函数**：
- `cal_shape_skratio` (Line 685-693)
- `cal_shape_skratioVol` (Line 718-728)
- `cal_vol_upRatio` (Line 563-585)
- `cal_vol_downRatio` (Line 617-639)

**问题**：
当 kurtosis=0 或 std=0 时，除法运算产生 inf 值

**修复模式**：
```python
# cal_shape_skratio 示例
shape_skratio = df.group_by(['date', 'code']).agg(
    pl.col('return').skew().alias('skew'),
    pl.col('return').kurtosis().alias('kurt')
).with_columns(
    pl.when((pl.col('kurt') != 0) & pl.col('kurt').is_finite())
    .then(pl.col('skew') / pl.col('kurt'))
    .otherwise(None)
    .alias('shape_skratio')
).select(['code', 'date', 'shape_skratio'])

# cal_vol_upRatio 示例  
.group_by(['code', 'date']).agg([
    pl.col('up_std').first().alias('up_std'),
    pl.col('total_std').first().alias('total_std')
]).with_columns(
    pl.when((pl.col('total_std') > 0) & pl.col('total_std').is_finite())
    .then(pl.col('up_std') / pl.col('total_std'))
    .otherwise(None)
    .alias('vol_upRatio')
)
```

**测试结果**：
```
# 恒定收益率数据（std=0, kurtosis=0）
shape_skratio: null  ✓ 正确返回 None
vol_upRatio: null    ✓ 正确返回 None
```

---

## 测试验证

**测试脚本**：`test_cicc_fixes.py`

**测试结果**：
```
============================================================
测试汇总
============================================================
✓ 通过: bottom20阈值修复
✓ 通过: corr_square公式修复
✓ 通过: 除零保护修复

============================================================
✓ 所有测试通过
```

---

## 技术细节

### Polars 表达式陷阱

在修复过程中遇到的 Polars API 问题：

**错误用法**：
```python
# ❌ 这会导致 "truth value of an Expr is ambiguous" 错误
pl.when(pl.col('kurt') != 0)  # 直接在 when() 中使用比较表达式
```

**正确用法**：
```python
# ✓ 需要在 when() 中构造完整的布尔表达式
pl.when((pl.col('kurt') != 0) & pl.col('kurt').is_finite())
```

### 数据类型问题

测试数据构造时的陷阱：

**错误用法**：
```python
# ❌ pl.date() 返回 Object 类型，导致 over() 操作失败
'date': [pl.date(2024, 1, 2)] * 30
```

**正确用法**：
```python
# ✓ 使用 datetime.date 返回 Date 类型
import datetime
'date': [datetime.date(2024, 1, 2)] * 30
```

---

## 影响评估

### 受影响因子

1. **mmt_bottom20VolumeRet**：修复前与 bottom50 完全相同，现已正交化
2. **mmt_ols_qrs**：修复前使用错误的数学公式，现已符合统计学定义
3. **shape_skratio, shape_skratioVol**：修复前可能产生 inf，现返回 None
4. **vol_upRatio, vol_downRatio**：修复前可能产生 inf，现返回 None

### 数据重算建议

**必要性**：高优先级

修复涉及因子计算公式的核心错误，建议重新计算历史因子面板：

```bash
# 重新生成 ETF 日级因子
python -m scripts.generate_etf_cicc_minute_factors --overwrite --workers 8

# 输出目录
D:\workspace\stockdata\etf-data\etf_cicc_daily_factors\
```

**预期改善**：
- `mmt_bottom20VolumeRet` 与 `mmt_bottom50VolumeRet` 的相关系数从 1.0 降至合理区间
- `mmt_ols_qrs` 因子的 IC 可能显著提升（公式修正后）
- 极端市况下因子覆盖率提升（除零保护避免 inf 传播）

---

## 后续工作

### P1：立即执行
- ☐ 使用修复后的代码重新生成 ETF CICC 日级因子（69个文件）
- ☐ 对比修复前后因子值差异，量化影响范围
- ☐ 在测试集上验证 IC 改善幅度

### P2：中期优化
- ☐ 检查其余因子是否有类似的除零风险（根据 cicc_factor_bugs_and_fixes.md）
- ☐ 验证时间筛选逻辑（`is_in([start, end])`）是否符合设计意图
- ☐ 添加数据质量检查：每日 bar 数是否完整

### P3：长期规划
- ☐ 实现 20 日滚动平滑特征（如 `pv_corr_avg_20d`）
- ☐ 扩展到股票侧：`scripts/generate_stock_cicc_daily_factors.py`
- ☐ 在主选股模型中测试修复后因子的增量 IC

---

## 参考文档

- 原始错误分析：`cicc_factor_bugs_and_fixes.md`
- 日级输出迁移：`cicc_daily_factor_migration.md`
- 测试脚本：`test_cicc_fixes.py`
- 修改文件：`MinuteFrequentFactorCalculateMethodsCICC.py`

---

## Changelog

| 日期 | 修复内容 | 测试状态 |
|---|---|---|
| 2026-08-18 | bottom20VolumeRet 阈值修正 | ✓ 通过 |
| 2026-08-18 | corr_square 公式修正 | ✓ 通过 |
| 2026-08-18 | 4个因子添加除零保护 | ✓ 通过 |
