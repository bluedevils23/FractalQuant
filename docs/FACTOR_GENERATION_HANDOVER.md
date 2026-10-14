# 因子生成项目交接文档

更新时间：2026-09-23
适用仓库：`D:\workspace\stock-fractalquant\FractalQuant`

## 1. 项目概览

项目的因子计算代码主要位于 `FractalQuant/factor/`，批量生成入口位于
`scripts/`。生成器通常按“一个标的一份 parquet”输出，并使用 `uv run`
启动。

当前已实现的生成管线如下：

| 管线 | 入口脚本 | 资产/频率 | 因子数量 | 默认输出 |
| --- | --- | --- | ---: | --- |
| 标准分钟因子 | `generate_etf_minute_factors.py` | ETF/分钟 | base 58 / multi 162 | `D:\workspace\stockdata\etf-data\etf_1min_factors*` |
| Session Path | `generate_etf_minute_factors.py --write-session-path-factors` | 股票 + ETF/分钟 | 3 | `*_intraday_session_path_factors` |
| FZ 日频因子 | `generate_fz_daily_factors.py` | 股票 + ETF/日 | 38 | 按 `--asset-type` 选择默认目录 |
| CICC 日频因子 | `generate_etf_cicc_daily_factors.py` | ETF/日 | 58 | `D:\workspace\stockdata\etf-factors\etf_cicc_daily_factor` |
| Advanced 因子 | `generate_stock_advanced_factor.py` / `generate_etf_advanced_factor.py` | 股票 / ETF/分钟 | 46 | `stock_advanced_factors` / `etf_1min_advanced_factors` |
| Orderbook 因子 | `generate_stock_orderbook_factors.py` | 股票 + ETF/分钟 | base 84 / multi 156 | `stock_1min_orderbook_factors*` / `etf_1min_orderbook_factors*` |
| ETF Orderbook 专用入口 | `generate_etf_orderbook_factors.py` | ETF/分钟 | base 84 / multi 156 | `etf_1min_orderbook_factors*` |
| 日内策略因子 | `generate_intraday_strategy_factors.py` | 股票 + ETF/分钟 | P0 21 / P0+P1 34 | `*_intraday_strategy_p0*` |
| 开盘集合竞价 | `generate_auction_factors.py` | 股票 + ETF/日 | 94 | `*_auction_factors` |
| 收盘集合竞价 | `generate_close_auction_factors.py` | 股票 + ETF/日 | 7 | `*_close_auction_factors` |
| Tail30 | `generate_tail30_factors.py` | ETF/日 | 13 | `etf_tail30_factors` |
| FZ 噪声边界 | `generate_etf_noise_bound_ratio_factors.py` | ETF + 指数/分钟 | 9 个派生字段 | `etf_noise_bound_ratio_factors` |
| CrossMarket 综合 | `generate_etf_crossmarket_factors.py` | ETF + 映射指数/分钟 | 35 | `etf_1min_crossmarket_factors` |
| CrossMarket Beta/残差 | `generate_etf_crossmarket_minute_factors.py` | ETF + 映射指数/分钟 | 4 | `etf_1min_crossmarket_factors_rqdata` |
| CrossMarket OFI | `generate_etf_crossmarket_ofi_factors.py` | ETF Orderbook/分钟 | 8 | 追加至 `etf_1min_crossmarket_factors` |
| 开源分钟研究因子 | `generate_open_source_factors.py` | ETF/股票分钟源、日频输出 | 19 | `etf_open_source_factors` |
| 开源 APM/OVP | `generate_open_source_apm_ovp.py` | ETF + 映射指数/日 | 4 | `etf_open_source_apm_ovp` |
| 开源 Tick 因子 | `generate_open_source_tick_factors.py` | A股逐笔/日 | 33 | `open_source_tick_factors` |
| Report78 技术因子 | `generate_etf_report78_factors.py` | ETF/日 | 4 | `etf_report78_factors` |
| 共享市场状态 | `generate_etf_regime_daily_factors.py` | ETF共享/日 | 12 | `etf_regime_daily_factors\market_regime_daily.parquet` |
| 前一交易日上下文 | `generate_previous_day_factors.py` | 股票 + ETF/日 | 11 | `*_daily_context_factors` |

因子数量是当前代码注册表中的因子字段数量，不包含行情原始字段、键字段和诊断字段。
同一公式的多窗口版本分别计数；不同管线间可能存在语义相近字段，不能把各行数量直接
相加后视为去重后的全项目因子数。当前 `scripts/` 下共有 20 个
`generate_*factor*.py` 入口，另有 `generate_open_source_apm_ovp.py`；对应上表 19 条独立生成管线。Session Path 是标准分钟入口的
附属输出模式。

## 2. 数据目录约定

项目默认依赖以下外部数据目录：

| 数据 | 默认位置 | 说明 |
| --- | --- | --- |
| ETF 1 分钟 | `D:\workspace\stockdata\etf-data\etf_1min` | 每个 ETF 一个 parquet，通常含 `trade_time/open/high/low/close/volume 或 vol/amount/adj_factor` |
| 股票 1 分钟 | `D:\workspace\stockdata\stock-data\行情数据\stock_1min` | auction 生成器的默认股票分钟目录 |
| ETF 日线 | `D:\workspace\stockdata\etf-data\etf_daily.parquet` | FZ、日内策略、竞价上下文使用 |
| 股票日线 | `D:\workspace\stockdata\stock-data\行情数据\stock_daily.parquet` | 日内策略、竞价上下文使用 |
| 逐笔数据 | `E:\逐笔数据` | 目录结构为 `YYYY\YYYYMM\YYYYMMDD\symbol`，orderbook、advanced、竞价使用 |
| 指数 1 分钟 | `D:\workspace\stockdata\index-data\index_1min` | noise-bound 因子使用 |

### 2.1 复权口径

分钟内收益、振幅、VWAP 和盘口结构使用当日原始价格；只要分子分母属于同一交易日，
累计复权因子会相消，不应重复复权。跨交易日价格比较必须使用
`adjusted_price = raw_price * adj_factor`，且 `adj_factor` 必须是累计复权因子，不能用
QMT 的单次除权事件倍率 `dr` 代替。缺失、非数值或非正的 `adj_factor` 不得默认为 1；
依赖它的跨日因子应保持 `NaN`，同日因子可继续计算。

当前明确使用复权价的跨日路径包括：Session Path 的前收盘收益；FZ 的
`FeiEPuHuo`、`CaoMuJieBing`、`XieTongJiaCha`；日内策略的日线背景；开源分钟因子的
日收益/隔夜收益和价格状态排序；APM/OVP 的隔夜收益与 `ret20`；开源 Tick 的
收盘收益与 `ret20`；Report78 的长期动量；开盘竞价的历史日线上下文；以及昨日上下文
因子的跨日收益与排名。CICC 的 `liq_amihud_1min`、`corr_prvr` 已按交易日重置，不会把
次日第一分钟与前日最后一分钟连接，因此无需跨日复权。

| 管线 | `adj_factor` 结论 |
| --- | --- |
| 标准分钟、Advanced、Orderbook、收盘竞价、Tail30 | 公式在单一交易日内计算，不需要跨日复权 |
| Session Path | `intraday_return_from_prev_close` 使用当日及前日复权收盘 |
| FZ | `FeiEPuHuo`、`CaoMuJieBing`、`XieTongJiaCha` 使用复权价；其余同日分钟公式保持原价 |
| CICC | 分钟收益按 `code/date` 重置，不跨日连接 |
| 日内策略 | 分钟路径及历史日线背景使用复权 OHLC；缺失因子时相关输出为空 |
| 开盘竞价 | 交易所竞价前收盘字段保持其撮合口径；另取的历史日线上下文使用复权价 |
| Noise-bound、Regime | 输入是指数价格；指数序列不使用股票/ETF `adj_factor` |
| CrossMarket | 分钟滚动在日内处理；上一日 NAV/规模由独立日状态侧车提供 |
| 开源分钟、APM/OVP、开源 Tick | 跨日收益、`ret20` 和跨日价格状态使用复权价 |
| Report78 | `technical_long_mom` 使用复权日收盘，其余字段为同日统计 |
| Previous-day context | 跨日收益、滚动统计和截面排名使用复权价 |

逐笔数据目录名可能带有错误的交易所后缀。orderbook 入口默认只使用六位数字代码匹配，
例如 `501001.SZ` 可以匹配到分钟文件 `501001.SH.parquet`；只有显式传入
`--strict-suffix` 时才要求完整后缀匹配。

CICC 复现管线仍依赖项目同级目录中的外部复现代码：

```text
D:\workspace\stock-fractalquant\Replication-of-Minute-Frequency-Factor-refer-CICC
```

## 3. 各生成脚本说明

### 3.1 标准 ETF 分钟因子

入口：`scripts/generate_etf_minute_factors.py`  
实现：`FractalQuant/factor/price.py`、`trend.py`、`volatility.py`、
`microstructure.py`、`fractional.py`

`base` 是兼容版注册表，包含收益、动量、价格位置、波动率、趋势、成交量、
流动性和基础订单流/订单簿因子。`multi` 在此基础上增加多窗口版本，输出目录默认分别为：

```text
D:\workspace\stockdata\etf-data\etf_1min_factors
D:\workspace\stockdata\etf-data\etf_1min_factors_multiwindow
```

计算按交易日分组，滚动窗口不会跨夜间或跨交易日污染；每个交易日开头的窗口预热
`NaN` 是预期行为。已有输出默认跳过，`--overwrite` 才会重算。

常用命令：

```powershell
uv run python scripts/generate_etf_minute_factors.py --window-profile base --workers 5
uv run python scripts/generate_etf_minute_factors.py --window-profile multi --workers 5
```

### 3.2 FZ 日频因子

统一入口：`scripts/generate_fz_daily_factors.py`
实现来源：`FractalQuant/factor/fz_methods.py`、`fz_factor.py`、`fz_base.py`

脚本默认针对 ETF 资产（`--asset-type etf`），也可传入 `--asset-type stock` 处理股票。
运行机制是先把分钟行情拆成交易日面板，计算原始 FZ 因子，再计算组合因子，最后按 `code/factor_date` 输出
每个标的一份日频 parquet。输入必须是 orderbook 同口径的
241 根分钟：`09:30–11:30`、`13:01–15:00`；缺行、重复或异常时间网格的
`code/date` 会跳过且记录告警。`GaoDiECha` 等因子还需要对应资产的日线数据。

潮汐因子在通过241分钟输入校验后，按研报剔除 `09:30` 和 `15:00`，将其余239分钟
重新编号；9分钟邻域成交量在完整序列上计算，涨潮低点使用 `5..t-1`，退潮低点使用
`t+1..233`。边界分钟参与低点选择，缺少任一有效半潮汐时保持空值。`ChaoXi` 要求
最近20个交易日的两个原始半潮汐值均有效，不会删除缺失日后跨更长区间凑满20个样本。

`factor_date=d` 的暴露包含 d 日完整分钟信息，只能在 d+1 或之后用于回测和交易；
月频回测应选取月末构造日，而不是把日频因子复制到分钟行。

`CaoMuJieBing` 依赖 `retail_trade_ratio` 和中证全指 `000985.CSI` 日收益
`csi_all_share_return`。`retail_trade_ratio` 是单笔成交额小于4万元的个人投资者买卖额
均值除以当日总成交额，必须来自逐笔成交数据，不得用零值、普通成交量或分钟 OHLCV
推算。股票可传入 `--build-cao-mu-source`，从 `E:\逐笔数据` 的逐笔成交 CSV 构造并
增量缓存到 `D:\workspace\stockdata\stock-factors\stock_cao_mu_jie_bing_source.parquet`；
中证全指收益率由 `D:\workspace\stockdata\指数数据\index_daily\000985.CSI.parquet`
的收盘价计算。生成器会合并该侧车文件但不会修改原始股票日线。逐笔文件缺失的日期，
该因子仍为空；传入 `--strict-source-fields` 会在分钟分片前终止。

日线输入还必须包含累计 `adj_factor`。`FeiEPuHuo` 的跨日高低价、
`CaoMuJieBing` 的日收益和 `XieTongJiaCha` 的当日/前日收盘均使用复权价；
`GaoDiECha` 的同日开收到开盘收益仍可使用原价。缺失复权因子时，上述三个跨日因子
保持空值，不得以 1 补齐。

生成成功后，输出目录会写入 `_fz_generation_manifest.json`，记录 schema、因子源字段
可用性、异常交易日统计和时点语义。当前生成器的截面标准化和市场相对计算仅使用本次
选中的目标标的；使用静态 CSI 名单会保留幸存者偏差，不能将结果声明为全 A 研报复现。

默认输出目录：

```text
股票：D:\workspace\stockdata\stock-factors\stock_fz_daily_factors
ETF： D:\workspace\stockdata\etf-data\etf_daily_fz_factors
```

常用命令：

```powershell
uv run python scripts/generate_fz_daily_factors.py --workers 5
uv run python scripts/generate_fz_daily_factors.py --asset-type stock --workers 5
uv run python scripts/generate_fz_daily_factors.py --asset-type stock --build-cao-mu-source --overwrite --workers 5
uv run python scripts/generate_fz_daily_factors.py --asset-type stock --strict-source-fields
```

该入口是文件级跳过：只要目标 parquet 已存在且未传 `--overwrite`，整个标的都会跳过。

共38个FZ因子。新增的 `ChongJian` 与 `ZaiHouChongJian` 分别是日协方差和20日
“灾后重建”暴露；`YueYaoYanBoDongLv`、`YueYaoYanShouYiLv` 是可独立回测的
20日耀眼子因子。已有输出文件不会自动补列，需显式传入 `--overwrite` 重算。

研报映射：

- [适度冒险](https://mp.weixin.qq.com/s/2pObrtp3V0dv50MFGg_fhw)、[潮汐](https://mp.weixin.qq.com/s/_2xWXM8iyNzYDYolT9vVsw)、[勇攀高峰与灾后重建](https://mp.weixin.qq.com/s/-IxH3n-uR0BwOIbOyzqPrw)、[云开雾散](https://mp.weixin.qq.com/s/vX4I9SpKRF3_HKQOGhtZcQ)；
- [飞蛾扑火](https://mp.weixin.qq.com/s/V8r1Xbz5J0A9D5J_9GD_IA)、[草木皆兵](https://mp.weixin.qq.com/s/hdtzQaCF2h6ZjZd-0HAftQ)、[水中行舟](https://mp.weixin.qq.com/s/Zv6XX8ddUvBCtG-4LM1MKQ)、[花隐林间](https://mp.weixin.qq.com/s/4G8Fm2x1Bv61idbbQS2BlA)；
- [待著而救](https://mp.weixin.qq.com/s/vCK2QIi0W19WWTyfiWpAyQ)、[多空博弈](https://mp.weixin.qq.com/s/04H8mVTVxa6OznMSj3JKgg)、[协同效应](https://mp.weixin.qq.com/s/_L3kv76mgoxctgW45iwIBA)。

### 3.3 CICC 日频因子

入口：`scripts/generate_etf_cicc_daily_factors.py`  
实现来源：项目同级 `Replication-of-Minute-Frequency-Factor-refer-CICC`

CICC 函数从分钟行情计算日级暴露，直接输出一行一个 `trade_date/ts_code` 的日频结果，
不再合并回原始分钟行。脚本文件名保留 `minute` 是历史兼容命名，当前默认输出目录为
`D:\workspace\stockdata\etf-factors\etf_cicc_daily_factor`。该脚本注册 58 个 CICC
因子，按名称前缀分为动量、波动、分布形态、流动性、价量相关、日内分布和交易时段。

```powershell
uv run python scripts/generate_etf_cicc_daily_factors.py --workers 5
```

同样是文件级跳过；需要补充或修复已有标的时必须使用 `--overwrite`。
`liq_amihud_1min` 和 `corr_prvr` 的分钟变化率按 `code/date` 分组，交易日首分钟不会
错误引用上一交易日的末分钟。

### 3.4 Advanced 因子

入口：

```text
scripts/generate_stock_advanced_factor.py
scripts/generate_etf_advanced_factor.py
```

共同实现：`FractalQuant/factor/advanced_runtime.py` 和 `advanced.py`。当前通过
反射自动构造 `BaseFactor` 子类，注册 46 个 advanced 因子。计算按交易日分组，
默认排除 `future_returns`，因此默认输出不包含未来收益标签。

默认输出：

```text
D:\workspace\stockdata\a-share-data\stock_advanced_factors
D:\workspace\stockdata\etf-data\etf_1min_advanced_factors
```

`--include-future-returns` 只应在明确生成标签或研究数据集时使用，不能把该列当作
模型特征。`--skip-tick-check` 可以跳过逐笔覆盖检查，但不会改变 advanced 因子的
分钟输入计算。

```powershell
uv run python scripts/generate_etf_advanced_factor.py --workers 5
uv run python scripts/generate_stock_advanced_factor.py --workers 5
```

advanced 入口在没有 `--overwrite` 时按标的文件跳过，不支持只更新已有 parquet 的部分日期。

### 3.5 Orderbook 因子

核心实现：`FractalQuant/factor/stock_orderbook.py`  
两种入口：

```text
scripts/generate_stock_orderbook_factors.py
scripts/generate_etf_orderbook_factors.py
```

两者复用同一套计算逻辑，主要包括：

- L1/L5 盘口快照：mid、spread、depth imbalance、weighted imbalance、盘口压力、斜率、深度集中度等；
- OFI/MLOFI：标准 OFI、归一化 MLOFI、事件强度、深度分歧、impact beta；
- 委托流：数量、金额、笔数不平衡及多窗口版本；
- 成交流：成交量不平衡、VWAP gap、成交方向持续性、成交量分布；
- 流动性/冲击：liquidity shock、market impact、liquidity depth、price velocity、market efficiency 等；
- VPIN 和 `adverse_selection_markout_30s`；
- 上下文异常和分段异常因子。

`base` 输出 84 个 orderbook 因子，`multi` 输出 156 个因子。因子先在逐笔盘口/成交时间轴
上计算，再通过不向前看的时间对齐映射到分钟行情。

ETF 专用入口默认配置：

```text
逐笔：E:\逐笔数据
分钟：D:\workspace\stockdata\etf-data\etf_1min
输出：D:\workspace\stockdata\etf-data\etf_1min_orderbook_factors
```

常用命令：

```powershell
uv run python scripts/generate_etf_orderbook_factors.py --workers 5
uv run python scripts/generate_etf_orderbook_factors.py --window-profile multi --workers 5
```

orderbook 入口按交易日增量跳过：已有 parquet 中已经存在的 `trade_date` 不会重复计算；
新日期会追加并按 `trade_date/trade_time` 排序。`--overwrite` 只替换本次请求日期，
不会删除请求范围外的历史日期。

注意：`adverse_selection_markout_30s` 是成交后 30 秒成熟的已实现 markout，成熟前为
`NaN`，不能在成交原始时刻当作即时信号使用。它没有未来信息泄露，但在研究中要保持
“可用时间晚于成交时间 30 秒”的语义。

### 3.6 日内策略因子

入口：`scripts/generate_intraday_strategy_factors.py`  
实现：`FractalQuant/factor/intraday_strategy.py` 和 `intraday_strategy_p1.py`

P0 包含 21 个因子，覆盖开盘路径/多项式、GFTD、突破、前 30 日尾部统计和延迟极值；
P1 增加 13 个因子，覆盖市场资金流、路径 KNN、波动率收敛和早盘区间，因此 P0+P1
共 34 个因子。

默认输出：

```text
D:\workspace\stockdata\a-share-data\stock_1min_intraday_strategy_p0_factors
D:\workspace\stockdata\etf-data\etf_1min_intraday_strategy_p0_factors
```

使用 `--priority-profile p0_p1` 时，输出切换到对应的 `*_p0_p1_factors` 目录。日线
上下文会为日期过滤额外读取约 180 天历史；不能把 `--date-from` 的读取预热区误删。
分钟和日线输入都应提供累计 `adj_factor`；缺失时依赖复权价的路径、突破和日线背景
保持空值，而不是静默按未复权价格计算。

```powershell
uv run python scripts/generate_intraday_strategy_factors.py --asset-type etf --workers 5
uv run python scripts/generate_intraday_strategy_factors.py --asset-type etf --priority-profile p0_p1 --workers 5
```

该入口支持按日期替换已有输出；没有日期过滤时会按全量结果处理。P1 的市场资金流
还依赖 `--pool-membership-path`、`--target-pool-path`（如果研究配置启用了动态池）。

### 3.7 开盘集合竞价因子

入口：`scripts/generate_auction_factors.py`

该脚本生成每日开盘竞价因子，当前有 94 个因子字段，分为：

- overnight/stage1/stage2 收益和金额比例；
- 委托撤单、虚假压力和大单行为；
- stage2 路径、波动、反转和效率；
- 鲁棒 imbalance/fisher 变化；
- 基于前 5/20 日 ADV 的参与度；
- 相对基准 ETF 的超额收益和市场上下文。
- 区间振幅、阶段端点收益、上下行步数、L3 买方占比等研报补充字段；
- 20 日历史平滑、5/20 日量比和竞价换手率字段。

默认 ETF 输出：

```text
D:\workspace\stockdata\etf-factors\etf_auction_factors
```

默认 benchmark 为 `510300.SH`。竞价入口中的 `--write-session-path-factors` 和
`--session-path-only` 已弃用；分钟级 Session Path 应通过
`generate_etf_minute_factors.py` 的同名参数生成，避免把开盘竞价和全日分钟路径混成
同一生成职责。

```powershell
uv run python scripts/generate_auction_factors.py --asset-type etf --workers 5
uv run python scripts/generate_etf_minute_factors.py --asset-type etf --session-path-only --workers 5
```

竞价输出按日期合并；`--overwrite` 替换请求日期，同时保留范围外日期。竞价因子使用
前一交易日/前 5 或 20 日上下文时，必须保证日线数据不含未来日期。

### 3.8 ETF noise-bound ratio 因子

入口：`scripts/generate_etf_noise_bound_ratio_factors.py`  
实现：`FractalQuant/factor/noise_area.py`

脚本使用 `WINDOW=14` 的 ETF/指数配对，计算上下噪声边界、边界比率、强度、净强度和
主导信号等字段。默认不是对所有 ETF 直接计算，而是从 `etf_basic_data.parquet` 中
筛选同时存在 ETF 和指数分钟文件的代表 ETF；如果存在候选文件，会优先使用候选 ETF，
并按指数每个选择一个代表 ETF。

```powershell
uv run python scripts/generate_etf_noise_bound_ratio_factors.py --workers 5
```

已有输出是文件级跳过。修改指数映射、候选 ETF 或 `NoiseArea` 参数后必须显式使用
`--overwrite`。

### 3.9 收盘集合竞价与 Tail30

收盘集合竞价入口为 `scripts/generate_close_auction_factors.py`，从 14:57--15:00 的逐笔
委托和成交重建 7 个日频因子：委托失衡、净委托失衡、撤单比例与方向，以及买方、卖方、
总体成交相对提交金额的三个代理字段。代理字段必须保留 `_proxy` 后缀；订单窗口为
`[14:57,15:00)`，成交窗口为 `[14:57,15:00]`。

Tail30 入口为 `scripts/generate_tail30_factors.py`，从 ETF 14:31--15:00 的 30 根分钟线
生成 13 个日频因子，包括收益、VWAP 偏离、反转、日内/残差波动率和 MaxILLIQ 的
单分钟、3 分钟、5 分钟代理。`tail30_return` 使用 14:30 close 到 15:00 close；完整日
应有 30 根尾盘分钟。两套输出通常在下一交易日可用。

```powershell
uv run python scripts/generate_close_auction_factors.py --asset-type etf --workers 5
uv run python scripts/generate_tail30_factors.py --workers 5
```

### 3.10 CrossMarket 因子

CrossMarket 有三个互补入口：

- `generate_etf_crossmarket_factors.py`：35 个字段，由 16 个通用跨市场因子、15 个
  ETF/映射指数相对价值字段和 4 个同指数 ETF 截面字段组成；
- `generate_etf_crossmarket_minute_factors.py`：4 个 benchmark-relative 字段，分别是
  60 分钟市场 beta、20 分钟残差动量、60 分钟残差 ZScore 和 60 分钟基准相关性；
- `generate_etf_crossmarket_ofi_factors.py`：8 个 L1/MLOFI 截面 OFI 字段，依赖已生成的
  ETF orderbook。该入口默认禁用，必须显式传 `--enable-orderbook-factors`。

综合入口还依赖 `exchange_fund_index_mapping.csv`、映射指数分钟数据，以及由
`generate_etf_daily_states.py` 生成的上一日 NAV/规模/份额状态。状态文件是输入侧车，
不属于独立因子管线。

```powershell
uv run python scripts/generate_etf_crossmarket_factors.py --workers 5
uv run python scripts/generate_etf_crossmarket_minute_factors.py --workers 5
uv run python scripts/generate_etf_crossmarket_ofi_factors.py --enable-orderbook-factors --workers 5
```

### 3.11 开源研报因子

`generate_open_source_factors.py` 从分钟 OHLCV 构造 19 个日频开源研报代理，覆盖理想
振幅、聪明钱、理想反转、ERR/TGD、成交量峰谷和价格跳跃等主题。默认输入是 ETF 分钟
目录，但可通过输入/输出路径用于同结构数据。跨日 `daily_return`、`overnight_return`
以及理想振幅/反转/换手和 TGD 的价格状态使用复权收盘价；纯日内分钟路径保持原价。

该入口的计算版本为 `3`。聪明钱跨日比较使用 `close * adj_factor` 和
`volume / adj_factor`，保持价量乘积不变；缺少有效复权因子的日期不参与其计算。
09:31 收益以同日 09:30 收盘价为参考，没有 09:30 行时使用首分钟开盘价。
振幅切割先剔除无效收益/振幅再分组，不依赖成交量历史是否有效；成交量与价格状态
分别判断历史有效性。爆发后续金额比只使用同一上午/下午时段内前后金额均有效的事件。
常规时段须严格匹配 09:31–11:30、13:01–15:00 的 240 根分钟网格，且 OHLCV 有效；
不完整日期保留审计行，但不输出因子，也不提供日收盘价、振幅和成交额参与后续计算。
日成交额要求所有参与统计的分钟（含实际存在的 09:30 行）均有效，零成交额有效。
波峰/山脊事件成交额缺失保留为空，不再将缺失聚合成零。

`--symbols`、`--symbols-file` 和 `--limit` **只筛选输出**，ERR/TGD 始终使用输入
目录中的全部标的计算，或使用 `--calculation-symbols-file` 指定的计算池，因此限制输出数量不会降低提取工作量。
输出池必须属于计算池，指定日期范围内每个输出标的都须有记录。任一标的提取失败或为空
都会阻止写出。输出目录的 `manifest.json` 记录版本、计算池及其哈希、TGD 最小样本数
与日历版本；旧目录没有清单或计算约定不同，须使用新的输出目录，避免混用历史结果。
同一约定下仍可通过 `--overwrite` 重算目标文件，日期过滤仍只发生在写出阶段。
未指定 `--overwrite` 时，已有目标文件须与本次结果一致才能跳过。输出由独占写锁保护，
整批暂存并读回验证后切换目录；写入异常不发布半成品，切换中断由下次运行自动恢复。
暂存会复制已有目录，需预留相应磁盘空间；目录切换期间路径可能短暂不可用。

`available_time` 使用 `exchange-calendars` 的 XSHG 日历取下一交易日 09:30，
包括输入最后一天，不再依赖个股下一条行情。该日历用于境内 ETF/A 股；超出已知
节假日覆盖范围时明确失败，需更新日历依赖。2026 年中秋、国庆边界已按
[上交所休市安排](https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml)
验证。旧因子文件不会随代码更新自动修复，应在新目录重新生成并重跑下游分析。

`generate_open_source_apm_ovp.py` 生成 `kaiyuan_apmnew_m20`、
`kaiyuan_apmraw_m20`、`kaiyuan_ovp_m20`、`kaiyuan_avp_m20`。标的隔夜收益和
`ret20` 使用复权价，映射指数使用指数原价；同日 AM/PM 收益的复权因子会相消。

`generate_open_source_tick_factors.py` 从 A 股逐笔成交与委托构造 33 个日频因子，包括
单笔金额分位、MTS/MTE、大小单资金流、NIR/CNIR、主动交易行为和 EVL/EVM/EVS。
20 日滚动项必须保留 `--date-from` 之前的预热历史；日期下界只应在写出时过滤。该入口
通过 `--daily-path`（默认股票日线 parquet）合并累计 `adj_factor`，收盘到收盘收益和
`ret20` 使用复权收盘；缺失复权因子的相关行保持空值并记录告警。

逐笔入口计算版本为 `2`，旧因子目录需要换新目录重新生成。原始归档的中文字段/
位置字段（`raw_trade_date`）使用万分之一元、股，标准化的 `price/quantity` 字段使用
元、股；适配器按字段契约转换，不能通过价格中位数判断单位。直接调用标准化函数时
可用 `price_divisor` 显式指定其他来源。15:00 分钟的收盘成交保留在日收盘价、金额和
分钟统计中，首个交易分钟的收益以当日第一笔成交价为参考；分钟统计按实际有成交的
钟表分钟分组。数量版理想反转以 `daily_mean_trade_volume / adj_factor` 做跨日排序，
不改变原始成交量、成交额或其他日内输入。

EVL/EVM/EVS 已按第 6 篇研报第 6 页修正：超大单占比依次对大单、中单、小单占比
做带截距的一元截面回归，分别取残差。条件资金流沿用项目四档定义：大单为
`[20万, 100万)`、小单为 `<4万`，二者不重叠；`>=2万` 仅用于 NIR 家族。
`flow_bucket_schema` 保留字符串。仅有委托文件时可以生成委托记忆因子，无需日线
复权文件，成交和资金流因子保持空值。

`--symbols`、`--symbols-file`、`--limit` 只限制输出，计算池为截止 `--date-to` 的
完整归档发现结果，因此限制输出不会降低提取工作量。任一标的提取失败或无有效数据
会阻止整批写出。`manifest.json` 记录计算版本、完整标的池与哈希、数据路径、单位
契约和日历版本，禁止将无清单旧结果或不同约定的结果混入同一目录。
`available_time` 使用 XSHG 日历的下一交易日 09:30，包括输入末日；超出已知日历
范围时失败，不猜测下一个工作日。以上修正不自动回写旧文件或更新下游回测。

### 3.12 Report78、共享市场状态与昨日上下文

- `generate_etf_report78_factors.py`：4 个 ETF 日频因子，分别为日内量比、CVILLIQ、
  QUA 和长期动量；长期动量使用复权日收盘，完整日信息映射到下一可用日；
- `generate_etf_regime_daily_factors.py`：基于配置指数生成一份全市场共享文件，含当前/
  下一状态、三状态概率、预期波动、置信度、熵和转移分数，共 12 个输出字段；
- `generate_previous_day_factors.py`：从严格早于目标日的日线/分钟上下文生成 11 个字段，
  包括昨日会话路径、2/20 日截面收益排名、60/10 日偏度及相对偏度、市场宽度和市场动量。

这些日频结果均带显式 `available_time` 或源日期信息；下游不能把完整交易日信息回填到
同日更早的交易时点。

## 4. 增量、覆盖和并行规则

通用规则：

1. 默认不传 `--overwrite`，优先保护已有 parquet。
2. `--workers N` 使用 `ProcessPoolExecutor`，建议按内存和磁盘吞吐逐步调高；orderbook 和 advanced 对逐笔数据读取较重，通常从 5 开始。
3. `--symbols-file` 文件建议使用 UTF-8/UTF-8 BOM，每行一个标准 `ts_code`；脚本通常支持 `#` 行尾注释。
4. 日期参数通常是闭区间，支持 `YYYYMMDD` 或带分隔符的日期。
5. 运行前先做 1 个标的的 smoke test，再启动全量任务；全量任务应将 stdout/stderr 重定向到日志文件。

不同脚本的跳过粒度不同：

| 类型 | 跳过粒度 | 需要补日期时的做法 |
| --- | --- | --- |
| 标准分钟、FZ、CICC、advanced、noise-bound、Report78、开源分钟/Tick | 标的文件 | 传 `--overwrite` 重算整个目标文件 |
| orderbook | 标的 + 交易日 | 不传 `--overwrite` 可增量补新日期 |
| intraday strategy | 标的；指定日期时按请求日期替换 | 使用 `--date-from/--date-to` 加 `--overwrite` |
| opening/close auction、Tail30 | 标的 + 请求日期 | 使用日期范围增量补算；`--overwrite` 替换请求日期 |
| CrossMarket 综合/Beta/OFI | 标的文件或已有 CrossMarket 文件 | 补列或修复时显式使用 `--overwrite`；OFI 另需启用开关 |
| regime、previous-day context | 共享文件或资产级输出 | 写前核对源日期和 `available_time`，不要并发写同一路径 |

## 5. 推荐运行流程

### 5.1 单标的 smoke test

```powershell
uv run python scripts/generate_etf_minute_factors.py --symbols 510300.SH --workers 1
uv run python scripts/generate_etf_orderbook_factors.py --symbols 510300.SH --workers 1
uv run python scripts/generate_etf_advanced_factor.py --symbols 510300.SH --workers 1 --skip-tick-check
```

检查输出 parquet 的行数、`trade_time` 范围、`ts_code`、因子列数量和非空比例，再进行全量生成。

### 5.2 全量生成

```powershell
uv run python scripts/generate_etf_minute_factors.py --workers 5
uv run python scripts/generate_fz_daily_factors.py --asset-type etf --workers 5
uv run python scripts/generate_etf_cicc_daily_factors.py --workers 5
uv run python scripts/generate_etf_advanced_factor.py --workers 5
uv run python scripts/generate_etf_orderbook_factors.py --workers 5
uv run python scripts/generate_intraday_strategy_factors.py --asset-type etf --workers 5
uv run python scripts/generate_auction_factors.py --asset-type etf --workers 5
uv run python scripts/generate_close_auction_factors.py --asset-type etf --workers 5
uv run python scripts/generate_tail30_factors.py --workers 5
uv run python scripts/generate_etf_noise_bound_ratio_factors.py --workers 5
uv run python scripts/generate_etf_crossmarket_factors.py --workers 5
uv run python scripts/generate_etf_crossmarket_minute_factors.py --workers 5
uv run python scripts/generate_open_source_factors.py --workers 5
uv run python scripts/generate_open_source_apm_ovp.py --workers 5
uv run python scripts/generate_open_source_tick_factors.py --workers 4
uv run python scripts/generate_etf_report78_factors.py
uv run python scripts/generate_etf_regime_daily_factors.py
uv run python scripts/generate_previous_day_factors.py --asset-type etf
```

建议先生成基础分钟、日线及 ETF 日状态，再生成 orderbook；CrossMarket OFI 必须排在
orderbook 和 CrossMarket 综合输出之后。开盘/收盘竞价和开源 Tick 都依赖逐笔数据。
不同输出目录彼此独立，可以分批运行，但不要让同一个脚本的两个实例同时写同一目录。
上面没有默认执行 CrossMarket OFI，因为其生成器默认关闭；确认 orderbook 覆盖后再显式
传入 `--enable-orderbook-factors`。

## 6. 质量检查和常见问题

### 6.1 快速检查输出

```powershell
uv run python -c "import pandas as pd; p=r'D:\workspace\stockdata\etf-data\etf_1min_orderbook_factors\510300.SH.parquet'; df=pd.read_parquet(p); print(df.shape); print(df.columns.tolist()); print(df[['trade_date','trade_time']].head())"
```

至少检查：

- `trade_time` 是否单调、是否重复；
- 原始行情字段是否完整；
- 因子是否全为 `NaN` 或常数；
- orderbook 的 `trade_date` 是否连续且没有重复日期；
- `adverse_selection_markout_30s` 是否在成交后 30 秒成熟；
- advanced 输出是否误包含 `future_returns`；
- auction 的可用时间是否晚于对应的竞价信息时间。

### 6.2 典型错误

- 找不到逐笔文件：先确认日期层级和六位数字代码，再考虑 `--strict-suffix` 是否被误启用。
- 输出只有原始行情没有因子：检查外部复现仓库、列名和依赖包，查看日志中的单因子 warning。
- 全量任务长时间没有写入：orderbook 首先会扫描 `E:\逐笔数据` 并建立日期/标的任务索引，这是正常的前置阶段。
- 日期过滤后窗口变短：advanced 和 intraday strategy 需要历史预热；不要把预热数据误当成输出范围删除。
- 修改因子注册表后输出未变化：文件级跳过或旧 parquet schema 仍在生效，使用 `--overwrite` 并记录新输出列集合。
- 拆分/分红日跨日收益异常：确认源字段是累计 `adj_factor`，抽取事件日前后做
  “原价按比例变化、复权因子反向变化后结果不变”的回归检查；不得将 QMT `dr` 当作
  累计复权因子。

## 7. 代码维护边界

- 标准分钟因子的注册和窗口组合在 `generate_etf_minute_factors.py`；
  通用因子实现位于 `FractalQuant/factor/` 对应模块。
- 股票和 ETF orderbook 的计算真值在 `FractalQuant/factor/stock_orderbook.py`，不要只修改某个生成脚本中的列名列表。
- advanced 的自动发现机制依赖 `advanced.py` 中的 `BaseFactor` 子类；新增类后要确认构造函数可无参数实例化，并确认是否属于未来标签。
- 日内策略和 P1 的列清单分别由 `intraday_strategy.py`、`intraday_strategy_p1.py` 维护；生成脚本只负责资产、日期、缓存和输出。
- 开盘竞价注册表在 `generate_auction_factors.py`，收盘竞价和 Tail30 分别维护自己的
  `FACTOR_COLUMNS`；修改代理公式时必须同步字段说明、单位和 `available_time` 语义。
- CrossMarket 的 35 个综合字段、4 个 beta/残差字段和 8 个 OFI 字段属于三个独立注册表；
  OFI 不应在缺少 orderbook 覆盖时用零值代替。
- 开源分钟、开源 Tick、Report78、regime 和 previous-day 的列清单分别位于对应的
  `factor/` 模块或生成脚本；文档数量应从注册表读取，不要依赖旧报告中的历史计数。
- CICC 外部复现公式来自同级 Replication 目录；升级外部代码后应重新做一标的日频输出
  对照验证。

## 8. 交接后的第一步

接手人应先执行：

```powershell
Set-Location D:\workspace\stock-fractalquant\FractalQuant
git rev-parse --show-toplevel
uv run python -m compileall FractalQuant scripts
uv run pytest -q tests/test_stock_orderbook_first_batch_factors.py
```

然后核对 `D:\workspace\stockdata` 的实际数据覆盖日期、输出目录更新时间和最近一次生成日志，
再决定是增量补算还是使用 `--overwrite` 全量重算。
