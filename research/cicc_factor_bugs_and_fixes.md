# CICC因子实现缺陷与修复建议

检查日期：2026-08-18  
文件：`Replication-of-Minute-Frequency-Factor-refer-CICC/MinuteFrequentFactorCalculateMethodsCICC.py`

## 1. 严重错误：bottom20VolumeRet 使用错误阈值

**位置**：Line 457-480

**问题**：
```python
def cal_mmt_bottom20VolumeRet(df: pl.DataFrame):
    """20底量成交动量 - 最低成交量的20根k线收益率动量"""
    .filter((pl.col('volume') <= pl.col('volume').bottom_k(50).max())  # ❌ 应为 bottom_k(20)
```

**影响**：
- `mmt_bottom20VolumeRet` 和 `mmt_bottom50VolumeRet` 产生相同的值
- 因子完全冗余，浪费计算资源和存储空间

**修复**：
```python
def cal_mmt_bottom20VolumeRet(df: pl.DataFrame):
    return (
        df.lazy().with_columns(
            (pl.col('close') / pl.col('open')).alias('ret')
        )
        .filter(
            (pl.col('volume') <= (
                pl.col('volume')
                .bottom_k(20)  # ✓ 修正为 20
                .max()
            )).over(['code', 'date'])
        )
        .group_by(['code', 'date']).agg(
            (pl.col('ret').product() - 1).alias('mmt_bottom20VolumeRet')
        ).collect()
    )
```

---

## 2. 公式错误：corr_square 计算不一致

**位置**：Line 137 vs Line 212

**问题**：
```python
# cal_mmt_ols_qrs (line 137)
pl.col('cov').pow(0.5) / (pl.col('var_x') * pl.col('var_y'))  # ❌ 错误公式

# cal_mmt_ols_corr_square_mean (line 212)
pl.col('cov').pow(2) / (pl.col('var_x') * pl.col('var_y'))    # ✓ 正确公式
```

**数学原理**：
- 相关系数平方：`R² = cov² / (σ_x² · σ_y²)`
- Line 137 计算的是 `√cov / (σ_x² · σ_y²)`，无统计学意义

**修复**：
```python
# Line 137
.with_columns(
    pl.when(pl.col('var_x') * pl.col('var_y') != 0)
    .then(
        pl.col('cov').pow(2) / (pl.col('var_x') * pl.col('var_y'))  # ✓ 改为 pow(2)
    )
    .otherwise(None)
    .alias('corr_square')
)
```

---

## 3. 时间筛选逻辑可能有误

**位置**：Lines 18, 33, 69, 84

**问题**：
```python
# cal_mmt_pm: 下午盘动量
df.filter(pl.col('time').is_in([130000000, 145900000]))
```

`.is_in([a, b])` 只匹配**精确等于** a 或 b 的行，不包括 a 和 b 之间的值。

**可能的设计意图**：

### 情况A：故意只用首尾价格（当前实现正确）
如果因子定义是"用下午首个bar的open和末个bar的close计算收益"，则当前实现正确。

### 情况B：想用整个时间段（当前实现错误）
如果因子定义是"用下午所有bar计算VWAP或其他聚合统计"，应改为：
```python
df.filter(
    (pl.col('time') >= 130000000) & 
    (pl.col('time') <= 145900000)
)
```

**影响因子**：
- `mmt_pm`（下午盘动量）
- `mmt_last30`（尾盘半小时动量）
- `mmt_am`（上午盘动量）
- `mmt_between`（去头尾动量）

**判断依据**：
查看因子后续计算：
```python
.group_by(['code', 'date']).agg(
    (pl.col('close').last() / pl.col('open').first()).alias('mmt_pm')
)
```

使用 `.first()` 和 `.last()`，说明**设计意图是只用首尾价格**。因此当前实现**可能是正确的**，但代码语义不清晰。

**建议改进**（提升可读性）：
```python
# 方案1：显式注释说明只用首尾
df.filter(pl.col('time').is_in([130000000, 145900000]))  # Only first and last bars

# 方案2：改用更清晰的逻辑
df.filter(
    (pl.col('time') == 130000000) |  # First bar of PM session
    (pl.col('time') == 145900000)     # Last bar of PM session
)
```

---

## 4. 除零保护不一致

**问题分布**：

### 有完善保护的因子
```python
# cal_liq_amihud_1min (line 751-753)
pl.when(pl.col('volume') > 0)
.then(pl.col('pct_change_abs') / pl.col('volume'))
.otherwise(0)
```

### 缺少保护的因子
```python
# cal_shape_skratio (line 684)
(pl.col('return').skew() / pl.col('return').kurtosis())
# 当 kurtosis=0 时产生 inf

# cal_vol_upRatio (line 581-586)
pl.col('up_return').std().fill_null(0) / pl.col('return').std()
# 当 return.std()=0 时产生 inf
```

**修复示例**：
```python
# cal_shape_skratio
.group_by(['date', 'code']).agg(
    pl.when(pl.col('return').kurtosis() != 0)
    .then(pl.col('return').skew() / pl.col('return').kurtosis())
    .otherwise(None)
    .alias('shape_skratio')
)

# cal_vol_upRatio
.group_by(['code', 'date']).agg(
    pl.when(pl.col('return').std() > 0)
    .then(pl.col('up_return').std().fill_null(0) / pl.col('return').std())
    .otherwise(None)
    .alias('vol_upRatio')
)
```

---

## 5. 时间格式转换潜在问题

**位置**：`generate_etf_cicc_minute_factors.py` Line 289-294

```python
def convert_time_to_cicc_int(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        index.hour * 10000000 + index.minute * 100000,
        index=index,
        dtype="int64",
    )
```

**问题**：
- 输入：`09:30:00` → 输出：`93000000`
- 输入：`14:59:00` → 输出：`145900000`
- **秒信息被丢弃**

如果原始分钟数据包含秒信息（如 `09:30:15`），转换后会统一为 `09:30:00`。

**判断是否有问题**：
- 如果输入数据保证是整分钟（`:00`秒），则无问题
- 如果输入包含非零秒数，需要确认CICC因子库的时间语义

**验证方法**：
```python
# 检查输入数据的秒分布
df = pd.read_parquet("etf_1min/510300.SH.parquet")
print(df.index.second.value_counts())
# 如果全是0，则无问题；如果有非0值，需要决定是否四舍五入
```

---

## 6. 数据质量假设未验证

CICC因子假设：
1. **每日有完整的240根分钟bar**（09:30-11:30, 13:00-15:00）
2. **没有停牌或缺失分钟**
3. **volume/amount 非负且有限**

**潜在风险**：
- 停牌日可能导致因子计算失败或返回NaN
- 部分因子使用 `bottom_k(50)` / `top_k(50)`，如果某日只有<50根bar会报错
- `product()` 计算累积收益时，如果有NaN会传播

**建议**：
在 `generate_etf_cicc_minute_factors.py` 的 `normalize_minute_frame` 中添加：
```python
def normalize_minute_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    # ... 现有逻辑 ...
    
    # 添加数据质量检查
    daily_counts = df.groupby(df.index.normalize()).size()
    if (daily_counts < 200).any():  # 少于200根bar可能有问题
        LOGGER.warning(
            f"{input_path.name}: {(daily_counts < 200).sum()} days with <200 bars"
        )
    
    # 确保 volume/close 非负
    df['volume'] = df['volume'].clip(lower=0)
    df['close'] = df['close'].clip(lower=1e-6)  # 避免除零
    
    return df
```

---

## 修复优先级

### P0：立即修复（影响因子正确性）
1. ✅ **bottom20VolumeRet 阈值错误** → 改为 `bottom_k(20)`
2. ✅ **corr_square 公式错误** → Line 137 改为 `.pow(2)`

### P1：中期修复（提升稳健性）
3. ☐ 统一添加除零保护（`shape_skratio`, `vol_upRatio`, `vol_downRatio`）
4. ☐ 在生成器中添加数据质量检查和日志

### P2：低优先级（代码可读性）
5. ☐ 时间筛选逻辑添加注释说明设计意图
6. ☐ 验证时间格式转换是否需要处理秒

---

## 验证方法

### 1. 验证 bottom20 修复
```python
import polars as pl

# 构造测试数据：30根bar
test_df = pl.DataFrame({
    'code': ['510300.SH'] * 30,
    'date': ['2024-01-02'] * 30,
    'open': range(100, 130),
    'close': range(101, 131),
    'volume': list(range(1, 31))  # volume从1到30
})

# 修复前：bottom_k(50) 会选中全部30根bar
# 修复后：bottom_k(20) 只选中volume<=20的bar

result = cal_mmt_bottom20VolumeRet(test_df)
print(result)
```

### 2. 验证 corr_square 修复
```python
# 构造完美线性关系数据
test_df = pl.DataFrame({
    'code': ['510300.SH'] * 50,
    'date': ['2024-01-02'] * 50,
    'high': range(100, 150),
    'low': range(90, 140),  # low = high - 10 (完美线性)
})

result = cal_mmt_ols_qrs(test_df)
# corr_square 应该非常接近 1.0
# 修复前：会得到错误的值
# 修复后：应该 ≈ 0.999+
```

---

## 是否需要修改

**最终决策**取决于：
1. **CICC官方定义**：如果有原始文档，应以官方公式为准
2. **回测结果**：如果当前实现已在生产环境运行且效果稳定，修复可能导致因子值突变
3. **因子相关性**：先检查 `mmt_bottom20VolumeRet` 和 `mmt_bottom50VolumeRet` 的相关系数，如果≈1.0则确认有bug

**保守做法**：
1. 修复明确的bug（bottom20, corr_square）
2. 生成修复前后的因子值对比
3. 在测试集上验证IC是否改善
4. 如果IC提升，重新计算历史因子面板
