# 高频交易策略系统

一个完整的Python高频交易策略系统，支持多因子、多交易所、多资产交易。

## 功能特性

### 核心功能
- **多因子模型**：支持100+种高级因子
- **多交易所支持**：支持CCXT支持的所有交易所
- **多资产交易**：支持多种加密货币交易对
- **本地运行**：完全本地运行，无需服务器
- **回测系统**：完善的回测和性能分析

### 高级因子类型

#### 1. 高级统计因子（分形、混沌理论）
- 李雅普诺夫指数（Lyapunov Exponent）
- Hurst指数（长程相关性）
- 关联维度（Correlation Dimension）
- 科尔莫哥洛夫熵（Kolmogorov Entropy）
- 多尺度谱（Multifractal Spectrum）
- 去趋势波动分析（DFA）
- 小波熵（Wavelet Entropy）
- 相空间体积（Phase Space Volume）
- 庞加莱截面（Poincare Section）
- 分岔图（Bifurcation Diagram）
- 混沌指示器（Chaos Indicator）
- 时间反演不对称性
- 非线性自相关

#### 2. 机器学习因子
- 机器学习预测（线性、岭回归、Lasso、随机森林、GBM）
- 异常检测（基于统计和机器学习）
- 聚类状态识别（K-Means）
- 降维技术（PCA、ICA）
- 集成预测器
- 神经网络预测
- 支持向量机预测
- 特征重要性分析
- 自编码器异常检测
- 高斯过程预测
- 分位数回归

#### 3. 市场微观结构因子
- 订单流失衡
- 流动性比率
- 成交量加权价格
- 订单簿压力
- 交易规模分布
- 波动率调整成交量
- 价格速度
- 动量加速度
- 成交量激增
- 流动性冲击
- 订单簿不对称性
- 交易方向持续性
- 市场冲击
- 流动性深度
- 订单流显著性
- 成交量聚类
- 价格成交量脱钩
- 市场效率
- 流动性迁移

#### 4. 跨市场因子
- 跨市场相关性
- 套利机会
- 市场联动性
- 相对强度
- 协整关系
- 跨市场波动率
- 市场状态切换
- 跨市场熵
- 跨市场相干性
- 格兰杰因果性
- 联合分布
- Copula相关性
- 相位同步
- 信息流
- 多尺度相关性
- 动态相关性

### 因子组合方法
- 等权组合
- 排名组合
- Z-score标准化
- PCA降权
- 风险平价
- 自适应权重

### 权重优化方法
- 等权优化
- 风险平价优化
- 均值-方差优化
- 贝叶斯优化
- 自适应优化
- 因子集成

### 回测引擎特性
- 多资产回测
- 滚动窗口回测
- 详细性能分析
- 交易成本模拟
- 杠杆支持
- 多种风险指标

## 项目结构

```
1/
├── config/               # 配置文件
│   ├── config.py         # 配置管理
│   └── __init__.py
├── data/                 # 数据模块
│   ├── fetcher.py        # 数据获取
│   ├── models.py         # 数据模型
│   ├── store.py          # 数据存储
│   └── __init__.py
├── factor/               # 因子模块
│   ├── base.py           # 基础因子类
│   ├── price.py          # 价格因子
│   ├── volatility.py     # 波动率因子
│   ├── trend.py          # 趋势因子
│   ├── orderbook.py      # 订单簿因子
│   ├── advanced.py       # 高级统计因子
│   ├── ml.py             # 机器学习因子
│   ├── microstructure.py # 微观结构因子
│   ├── crossmarket.py    # 跨市场因子
│   ├── combiner.py       # 因子组合器
│   ├── selector.py       # 因子选择器
│   └── __init__.py
├── signal/               # 信号模块
│   ├── generator.py      # 信号生成
│   ├── optimizer.py      # 信号优化
│   └── __init__.py
├── risk/                 # 风险管理模块
│   ├── manager.py        # 风险管理器
│   ├── stoploss.py       # 止损管理
│   └── __init__.py
├── execution/            # 执行模块
│   ├── executor.py       # 订单执行器
│   └── __init__.py
├── backtest/             # 回测模块
│   ├── engine.py         # 回测引擎
│   └── __init__.py
├── strategy/             # 策略模块
│   ├── strategy.py       # 主策略
│   ├── optimizer.py      # 策略优化
│   └── __init__.py
├── examples.py           # 使用示例
├── run_strategy.py       # 策略运行脚本
├── requirements.txt      # 依赖列表
└── README.md             # 项目文档
```

## 安装

### 系统要求
- Python 3.8+
- pip 20.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用 uv

```powershell
uv venv
.venv\Scripts\activate
uv sync
```

运行示例：

```powershell
uv run python FractalQuant/scripts/generate_etf_minute_factors.py --input-root F:\stock-data\etf-data\etf_1min --output-root F:\stock-data\etf-data\etf_1min_factors --workers 30
```

### 环境配置

复制 `.env.example` 到 `.env` 并填写您的API密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 交易所API密钥
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# 回测配置
BACKTEST_INITIAL_CAPITAL=100000
BACKTEST_COMMISSION=0.001
BACKTEST_SLIPPAGE=0.0005

# 因子配置
FACTOR_WINDOW_SHORT=5
FACTOR_WINDOW_MEDIUM=20
FACTOR_WINDOW_LONG=60
VOLATILITY_WINDOW=20
LYAPUNOV_WINDOW=50
HURST_WINDOW=50
ML_FORECAST_STEPS=5
ML_HIDDEN_SIZE=10
ORDERFLOW_WINDOW=50
LIQUIDITY_WINDOW=50
CROSSMARKET_WINDOW=50
ARBITRAGE_THRESHOLD=0.001
```

## 使用方法

### 1. 运行示例

```bash
python examples.py
```

### 2. 运行回测

```bash
python run_strategy.py --mode backtest --symbols BTC/USDT ETH/USDT --days 30
```

### 3. 因子分析

```bash
python run_strategy.py --mode factor_analysis --symbols BTC/USDT --days 30
```

### 4. 实盘交易（模拟）

```bash
python run_strategy.py --mode live --symbols BTC/USDT --exchange binance
```

### 5. 命令行参数

```bash
python run_strategy.py --help
```

参数说明：
- `--mode`: 运行模式 (backtest/live/factor_analysis)
- `--symbols`: 交易对列表
- `--timeframe`: 时间框架 (1m/5m/15m/1h/4h/1d)
- `--days`: 数据天数
- `--exchange`: 交易所名称
- `--initial_capital`: 初始资金
- `--strategy_type`: 策略类型 (hf/arbitrage/mean_reversion/momentum)

## 策略类型

### 1. 高频交易策略 (hf)
- 默认策略
- 基于多因子组合
- 适合高频交易

### 2. 套利策略 (arbitrage)
- 跨市场套利
- 价差检测
- 自动执行

### 3. 均值回归策略 (mean_reversion)
- Z-score均值回归
- 波动率检测
- 反向交易

### 4. 动量策略 (momentum)
- 价格动量
- 成交量确认
- 趋势跟踪

## 因子使用示例

### 基础因子

```python
from factor.price import ReturnsFactor, PriceMomentumFactor
from factor.volatility import HistoricalVolatilityFactor
from factor.trend import MACDFactor, RSIFactor

# 创建因子
returns = ReturnsFactor(window=5)
momentum = PriceMomentumFactor(window=20)
volatility = HistoricalVolatilityFactor(window=20)
macd = MACDFactor()
rsi = RSIFactor()

# 计算因子
factor_value = factor.calculate(df)
```

### 高级因子

```python
from factor.advanced import LyapunovExponentFactor, HurstExponentFactor
from factor.ml import MLForecastFactor, MLAnomalyDetectionFactor

# 李雅普诺夫指数（混沌理论）
lyapunov = LyapunovExponentFactor(window=50)
lyapunov_value = lyapunov.calculate(df)

# Hurst指数（长程相关性）
hurst = HurstExponentFactor(window=50)
hurst_value = hurst.calculate(df)

# 机器学习预测
ml_forecast = MLForecastFactor(model_type='linear')
ml_value = ml_forecast.calculate(df)

# 异常检测
anomaly = MLAnomalyDetectionFactor()
anomaly_value = anomaly.calculate(df)
```

### 因子组合

```python
from factor.combiner import FactorCombiner

# 创建因子列表
factors = [
    ReturnsFactor(window=5),
    PriceMomentumFactor(window=20),
    HistoricalVolatilityFactor(window=20),
    MACDFactor(),
    LyapunovExponentFactor(window=50),
    MLForecastFactor(model_type='linear'),
]

# 创建组合器
combiner = FactorCombiner(factors=factors, method='pca')

# 组合因子
combined = combiner.combine_factors(df)
```

### 因子选择

```python
from factor.selector import FactorSelector

# 创建选择器
selector = FactorSelector()

# 选择因子
selected_factors = selector.select_factors(
    factor_df, 
    returns, 
    method='all', 
    target_count=10
)
```

### 权重优化

```python
from factor.selector import WeightOptimizer

# 创建优化器
optimizer = WeightOptimizer(method='risk_parity')

# 优化权重
weights = optimizer.optimize(factor_df, returns)
```

## 回测示例

```python
from backtest.engine import BacktestEngine
from strategy.strategy import HighFrequencyTradingStrategy

# 创建策略
strategy = HighFrequencyTradingStrategy()

# 创建回测引擎
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    leverage=10
)

# 运行回测
result = engine.run(data, signal_generator, symbols)

# 查看结果
print(f"总收益率: {result.total_return*100:.2f}%")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown*100:.2f}%")
```

## 性能指标

### 回测指标
- 总收益率
- 年化收益率
- 最大回撤
- 夏普比率
- 索提诺比率
- 卡玛比率
- 信息比率
- 胜率
- 盈亏比
- 总交易数
- 平均持仓时间
- 平均盈利/亏损

### 详细指标
- 连续盈利/亏损次数
- 盈亏分布
- 波动率分解
- 收益偏度/峰度
- 日度指标

## 风险管理

### 风控参数
- 最大仓位大小
- 止损阈值
- 止盈阈值
- 杠杆倍数
- 最大回撤限制
- 最大持仓数量
- 日最大亏损

### 风控功能
- 自动止损
- 跟踪止损
- 波动率调整仓位
- 风险平价分配



## 免责声明

本软件仅供学习和研究使用。加密货币交易存在高风险，请谨慎使用。作者不对任何交易损失负责。

