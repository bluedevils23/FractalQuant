# 单因子回测模块使用指南

基于 Alphalens 方法论的完整单因子回测框架，整合了 TEJ 教程和方正证券多因子系列的最佳实践。

## 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [详细文档](#详细文档)
- [教程参考](#教程参考)
- [实战案例](#实战案例)

## 功能特性

### 核心功能

1. **因子预处理**
   - MAD去极值（中位数绝对偏差）
   - 百分位去极值
   - Z-score标准化
   - 按日期分组处理

2. **因子中性化**
   - 行业中性化
   - 市值中性化
   - OLS回归残差法

3. **IC分析**
   - Pearson IC（线性相关）
   - Spearman RankIC（秩相关）
   - IC时间序列
   - IR（信息比率）计算

4. **分组回测**
   - 10分组（可自定义）
   - 分组累积收益
   - 多空组合（Q1 - Q10）
   - 多头-平均组合（Q1 - 其余组平均）

5. **性能指标**
   - 年化收益率
   - 最大回撤
   - 夏普比率
   - 胜率
   - 波动率

6. **可视化报告**
   - IC时间序列图
   - 分组收益图
   - 多空组合净值曲线
   - 性能指标表格

## 快速开始

### 安装依赖

```bash
cd FractalQuant
uv sync
```

或使用 pip：

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels tqdm
```

### 最简单的例子

```python
from pathlib import Path
import pandas as pd
from FractalQuant.factor.single_factor_backtest import quick_backtest

# 准备数据
factor_df = pd.DataFrame({
    'trade_date': [...],  # 日期
    'stock_code': [...],  # 股票代码
    'factor_value': [...]  # 因子值
})

price_df = pd.DataFrame({
    'trade_date': [...],  # 日期
    'stock_code': [...],  # 股票代码
    'close': [...]  # 收盘价
})

# 运行回测
results = quick_backtest(
    factor_df=factor_df,
    price_df=price_df,
    output_dir='./backtest_output',
    factor_name='my_factor'
)

# 查看结果
print(results['performance'])
```

### 运行示例

```bash
# 进入项目目录
cd FractalQuant

# 运行示例
uv run python examples/single_factor_backtest_example.py
```

## 详细文档

### 1. 数据格式要求

#### 因子数据格式

```python
factor_df = pd.DataFrame({
    'trade_date': pd.to_datetime,  # 交易日期
    'stock_code': str,              # 股票代码（如 '600000.SH'）
    'factor_value': float           # 因子值
})
```

示例：
```
   trade_date  stock_code  factor_value
0  2020-01-02  600000.SH      1.234567
1  2020-01-02  600001.SH     -0.543210
2  2020-01-03  600000.SH      0.987654
...
```

#### 价格数据格式

```python
price_df = pd.DataFrame({
    'trade_date': pd.to_datetime,  # 交易日期
    'stock_code': str,              # 股票代码
    'close': float                  # 收盘价
})
```

### 2. 配置参数

```python
from FractalQuant.factor.single_factor_backtest import FactorBacktestConfig

config = FactorBacktestConfig(
    # 分组参数
    n_quantiles=10,              # 分组数量
    
    # 预处理参数
    winsorize_method='mad',      # 去极值方法：'mad' 或 'percentile'
    winsorize_std=3.0,           # MAD标准差倍数
    winsorize_percentile=(0.025, 0.975),  # 百分位截断点
    standardize=True,            # 是否标准化
    
    # 中性化参数
    neutralize_industry=False,   # 是否行业中性化
    neutralize_market_cap=False, # 是否市值中性化
    
    # 前向收益期
    forward_periods=[1, 5, 10, 20],  # 前向1/5/10/20日收益
    
    # 交易成本
    commission=0.0003,           # 单边手续费（万3）
    slippage=0.0001              # 滑点（万1）
)
```

### 3. 完整使用示例

```python
from pathlib import Path
import pandas as pd
from FractalQuant.factor.single_factor_backtest import (
    FactorBacktestConfig,
    SingleFactorBacktest
)

# 1. 读取数据
factor_df = pd.read_csv('factor_data.csv')
factor_df['trade_date'] = pd.to_datetime(factor_df['trade_date'])

price_df = pd.read_csv('price_data.csv')
price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])

# 2. 配置参数
config = FactorBacktestConfig(
    n_quantiles=10,
    forward_periods=[1, 5, 10, 20],
    winsorize_method='mad',
    standardize=True
)

# 3. 创建回测实例
backtest = SingleFactorBacktest(config)

# 4. 运行回测
results = backtest.run_backtest(
    factor_df=factor_df,
    price_df=price_df,
    date_col='trade_date',
    stock_col='stock_code',
    factor_col='factor_value',
    price_col='close'
)

# 5. 生成报告
backtest.generate_report(
    output_dir=Path('./backtest_output'),
    factor_name='my_alpha_factor'
)

# 6. 查看结果
print("=" * 80)
print("回测结果摘要")
print("=" * 80)

for period_name, perf_dict in results['performance'].items():
    print(f"\n{period_name}:")
    for strategy, metrics in perf_dict.items():
        print(f"  {strategy}:")
        print(f"    年化收益: {metrics['annualized_return']:.2%}")
        print(f"    夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"    最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"    胜率: {metrics['win_rate']:.2%}")
```

### 4. 输出文件说明

运行回测后，会在输出目录生成以下文件：

```
backtest_output/
├── my_factor_ic_analysis.png        # IC时间序列图
├── my_factor_quantile_returns.png   # 分组收益图
├── my_factor_long_short.png         # 多空组合图
└── my_factor_performance.csv        # 性能指标表
```

#### IC分析图
- 展示IC和RankIC的时间序列
- 评估因子预测能力的稳定性

#### 分组收益图
- 10组累积收益曲线
- 验证因子单调性

#### 多空组合图
- 多空组合（Q1-Q10）净值曲线
- 多头-平均组合净值曲线

#### 性能指标表
包含以下指标：
- Period: 收益期（1d, 5d, 10d, 20d）
- Strategy: 策略类型（long_short, long_average, q1）
- total_return: 总收益率
- annualized_return: 年化收益率
- volatility: 波动率
- sharpe_ratio: 夏普比率
- max_drawdown: 最大回撤
- win_rate: 胜率
- n_periods: 交易次数

## 教程参考

本模块基于以下教程和最佳实践开发：

### 1. TEJ Alphalens 教程

**链接**: https://www.tejwin.com/insight/alphalens-價值因子篇/

**核心要点**:
- 价值因子构建方法（BPROE_quantile）
- 标准化的因子分析流程
- IC/RankIC的计算和解读
- 分组回测的实施方法

**示例因子**:
```python
# BPROE因子 = PB倒数排名 + ROE排名
def calculate_bproe(df):
    # PB倒数排名
    df['pb_inv_rank'] = df.groupby('date')['pb'].transform(
        lambda x: (1/x).rank(pct=True)
    )
    
    # ROE排名
    df['roe_rank'] = df.groupby('date')['roe'].rank(pct=True)
    
    # 组合因子
    df['bproe'] = df['pb_inv_rank'] + df['roe_rank']
    
    return df
```

### 2. 外资因子篇

**链接**: https://medium.com/tej-api-金融資料分析/實戰應用-用-alphalens-剖析因子表現-外資因子篇-b0010522c5b0

**核心要点**:
- 外资持股变动因子
- 因子与未来收益的相关性分析
- 月度调仓回测

### 3. 知乎 Alphalens 实战

**链接**: https://zhuanlan.zhihu.com/p/1935835160911275301

**核心要点**:
- 详细的数据准备流程
- 因子去极值和标准化方法
- 行业中性化处理
- 完整的回测框架

## 实战案例

### 案例1：价值因子回测（参考TEJ教程）

```python
import numpy as np
import pandas as pd
from FractalQuant.factor.single_factor_backtest import quick_backtest, FactorBacktestConfig

# 构建BPROE价值因子
def build_bproe_factor(fundamental_df):
    """
    构建BPROE因子
    
    Args:
        fundamental_df: 包含 trade_date, stock_code, pb, roe 的DataFrame
    
    Returns:
        因子DataFrame
    """
    result = fundamental_df.copy()
    
    # 按日期分组计算排名
    result['pb_inv_rank'] = result.groupby('trade_date')['pb'].transform(
        lambda x: (1/x).rank(pct=True)
    )
    
    result['roe_rank'] = result.groupby('trade_date')['roe'].rank(pct=True)
    
    # 组合因子（等权）
    result['factor_value'] = (result['pb_inv_rank'] + result['roe_rank']) / 2
    
    return result[['trade_date', 'stock_code', 'factor_value']]

# 读取基本面数据
fundamental_df = pd.read_csv('fundamental_data.csv')
fundamental_df['trade_date'] = pd.to_datetime(fundamental_df['trade_date'])

# 读取价格数据
price_df = pd.read_csv('price_data.csv')
price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])

# 构建因子
factor_df = build_bproe_factor(fundamental_df)

# 配置（月度调仓）
config = FactorBacktestConfig(
    n_quantiles=10,
    forward_periods=[1],  # 月度只看下个月收益
    winsorize_method='mad',
    winsorize_std=3.0,
    standardize=True
)

# 运行回测
results = quick_backtest(
    factor_df=factor_df,
    price_df=price_df,
    output_dir='./backtest_output/bproe_value_factor',
    factor_name='BPROE_value_factor',
    config=config
)

print("BPROE价值因子回测完成！")
```

### 案例2：动量因子回测

```python
# 构建动量因子
def build_momentum_factor(price_df, window=20):
    """
    构建动量因子（过去N日收益率）
    
    Args:
        price_df: 价格数据
        window: 回看窗口
    
    Returns:
        因子DataFrame
    """
    result = price_df.copy()
    result = result.sort_values(['stock_code', 'trade_date'])
    
    # 计算过去N日收益
    result['past_return'] = result.groupby('stock_code')['close'].transform(
        lambda x: x.pct_change(periods=window)
    )
    
    result = result.rename(columns={'past_return': 'factor_value'})
    
    return result[['trade_date', 'stock_code', 'factor_value']].dropna()

# 读取价格数据
price_df = pd.read_csv('price_data.csv')
price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])

# 构建因子
factor_df = build_momentum_factor(price_df, window=20)

# 配置（日频调仓）
config = FactorBacktestConfig(
    n_quantiles=10,
    forward_periods=[1, 5, 10, 20],  # 1/5/10/20日收益
    winsorize_method='mad',
    standardize=True
)

# 运行回测
results = quick_backtest(
    factor_df=factor_df,
    price_df=price_df,
    output_dir='./backtest_output/momentum_20d',
    factor_name='momentum_20d',
    config=config
)

print("动量因子回测完成！")
```

### 案例3：多因子对比分析

```python
from pathlib import Path

# 定义多个因子
factors_to_test = {
    'momentum_20d': lambda df: build_momentum_factor(df, 20),
    'momentum_60d': lambda df: build_momentum_factor(df, 60),
    'reversal_5d': lambda df: -build_momentum_factor(df, 5),  # 反转因子
}

# 读取价格数据
price_df = pd.read_csv('price_data.csv')
price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])

# 配置
config = FactorBacktestConfig(
    n_quantiles=10,
    forward_periods=[5, 10, 20],
    standardize=True
)

# 对每个因子运行回测
all_results = {}

for factor_name, factor_builder in factors_to_test.items():
    print(f"\n回测因子: {factor_name}")
    
    # 构建因子
    factor_df = factor_builder(price_df)
    
    # 运行回测
    output_dir = Path(f'./backtest_output/factor_comparison/{factor_name}')
    results = quick_backtest(
        factor_df=factor_df,
        price_df=price_df,
        output_dir=output_dir,
        factor_name=factor_name,
        config=config
    )
    
    all_results[factor_name] = results

# 生成对比报告
comparison_data = []

for factor_name, results in all_results.items():
    # 取10日收益的指标
    if '10d' in results['performance']:
        metrics = results['performance']['10d']['long_short']
        ic_mean = results['ic_results']['10d']['IC'].mean()
        rank_ic_mean = results['ic_results']['10d']['RankIC'].mean()
        
        comparison_data.append({
            '因子名称': factor_name,
            'IC均值': f"{ic_mean:.4f}",
            'RankIC均值': f"{rank_ic_mean:.4f}",
            '年化收益': f"{metrics['annualized_return']:.2%}",
            '夏普比率': f"{metrics['sharpe_ratio']:.3f}",
            '最大回撤': f"{metrics['max_drawdown']:.2%}",
            '胜率': f"{metrics['win_rate']:.2%}"
        })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "=" * 80)
print("因子对比结果（10日收益）")
print("=" * 80)
print(comparison_df.to_string(index=False))

# 保存对比结果
comparison_df.to_csv('./backtest_output/factor_comparison/comparison_summary.csv', index=False)
```

## 方法论说明

### 1. MAD去极值

使用中位数绝对偏差（MAD）进行去极值，相比简单截断更稳健：

```python
median = factor_values.median()
mad = |factor_values - median|.median()
std = 1.4826 * mad  # MAD转标准差

lower = median - 3 * std
upper = median + 3 * std

# 线性压缩而非截断
values[values < lower] = linspace(median - 3.5*std, median - 3*std)
values[values > upper] = linspace(median + 3*std, median + 3.5*std)
```

### 2. 因子标准化

按截面（每个交易日）进行Z-score标准化：

```python
factor_zscore = (factor - factor.mean()) / factor.std()
```

### 3. IC计算

- **Pearson IC**: 衡量因子值与未来收益的线性相关性
- **Spearman RankIC**: 衡量排序的单调性关系（更稳健）

```python
IC = corr(factor_values, forward_returns)
RankIC = corr(rank(factor_values), rank(forward_returns))

IR = IC_mean / IC_std * sqrt(252)  # 信息比率（年化）
```

### 4. 分组回测

每个调仓日：
1. 按因子值排序，分为10组（Q1为因子值最大组）
2. 计算下期各组平均收益
3. 累积收益，生成净值曲线

多空组合：Q1 - Q10（做多高因子值，做空低因子值）

### 5. 性能指标

- **年化收益**: `(1 + 总收益)^(252/交易日数) - 1`
- **最大回撤**: `max(历史最高 - 当前值) / 历史最高`
- **夏普比率**: `均值 / 标准差 * sqrt(252)`
- **胜率**: `盈利次数 / 总交易次数`

## 常见问题

### Q1: 为什么IC值很小但收益还可以？

A: IC衡量的是线性相关性，即使IC较小（如0.02-0.05），通过分组做多做空也能放大收益。RankIC通常比IC更重要，因为它衡量的是排序关系。

### Q2: 如何判断因子是否有效？

A: 综合考虑以下指标：
- IC均值 > 0.02（或 < -0.02）
- RankIC均值 > 0.03（或 < -0.03）
- IR（信息比率）> 0.5
- 分组收益单调性好
- 多空组合夏普比率 > 1.0

### Q3: 月度因子如何回测？

A: 设置 `forward_periods=[1]`，数据按月采样，回测时下期收益就是下个月收益。

### Q4: 如何添加行业中性化？

A: 准备行业哑变量数据，设置 `neutralize_industry=True`，并传入 `industry_df` 参数。

### Q5: 因子值是越大越好还是越小越好？

A: 取决于因子定义。如果因子值大代表好股票，Q1（最大组）收益应该最高。如果相反，可以对因子值取负。

## 进阶使用

### 自定义因子处理

```python
from FractalQuant.factor.single_factor_backtest import FactorPreprocessor

preprocessor = FactorPreprocessor()

# 自定义去极值
factor_processed = preprocessor.mad_winsorize(
    factor_series,
    n_std=3.0
)

# 自定义标准化
factor_standardized = preprocessor.standardize(factor_processed)
```

### 自定义IC分析

```python
from FractalQuant.factor.single_factor_backtest import ICAnalyzer

ic_analyzer = ICAnalyzer()

# 计算单期IC
ic_dict = ic_analyzer.calculate_ic(
    factor_values=factor_series,
    forward_returns=return_series
)

print(f"IC: {ic_dict['IC']:.4f}")
print(f"RankIC: {ic_dict['RankIC']:.4f}")
```

### 自定义分组数量

```python
config = FactorBacktestConfig(
    n_quantiles=5,  # 改为5分组
    # ... 其他参数
)
```

## 贡献与反馈

如有问题或建议，请联系项目维护者。

## 参考资料

1. TEJ台湾经济新报 - Alphalens价值因子篇
2. TEJ - Alphalens外资因子实战应用
3. 知乎 - Alphalens因子分析实战
4. 方正证券 - 多因子系列研报
5. alphalens-reloaded 开源项目

---

**版本**: v1.0.0  
**更新日期**: 2026-08-19  
**作者**: FractalQuant Team
