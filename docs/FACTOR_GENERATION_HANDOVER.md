# 因子生成项目交接文档

更新时间：2026-07-21  
适用仓库：`D:\workspace\stock-fractalquant\FractalQuant`

## 1. 项目概览

项目的因子计算代码主要位于 `FractalQuant/factor/`，批量生成入口位于
`scripts/`。生成器通常按“一个标的一份 parquet”输出，并使用 `uv run`
启动。

当前已实现的生成管线如下：

| 管线 | 入口脚本 | 资产 | 因子数量 | 默认输出 |
| --- | --- | --- | ---: | --- |
| 标准分钟因子 | `generate_etf_minute_factors.py` | ETF | base 58 / multi 162 | `D:\workspace\stockdata\etf-data\etf_1min_factors*` |
| FZ 日频因子 | `generate_fz_daily_factors.py` | 股票 + ETF | 38 | 按 `--asset-type` 选择默认目录 |
| CICC 分钟因子 | `generate_etf_cicc_minute_factors.py` | ETF | 58 | `D:\workspace\stockdata\etf-data\etf_1min_cicc_factors` |
| Advanced 因子 | `generate_stock_advanced_factor.py` / `generate_etf_advanced_factor.py` | 股票 / ETF | 46 | `stock_advanced_factors` / `etf_1min_advanced_factors` |
| Orderbook 因子 | `generate_stock_orderbook_factors.py` | 股票 + ETF | base 63 / multi 135 | `stock_1min_orderbook_factors*` / `etf_1min_orderbook_factors*` |
| ETF Orderbook 专用入口 | `generate_etf_orderbook_factors.py` | ETF | base 63 / multi 135 | `etf_1min_orderbook_factors*` |
| 日内策略因子 | `generate_intraday_strategy_factors.py` | 股票 + ETF | P0 21 / P0+P1 34 | `*_intraday_strategy_p0*` |
| 开盘集合竞价 | `generate_auction_factors.py` | 股票 + ETF | 51 | `*_auction_factors` |
| FZ 噪声边界 | `generate_etf_noise_bound_ratio_factors.py` | ETF + 指数 | 9 个主要边界/强度字段 | `etf_noise_bound_ratio_factors` |

因子数量是当前代码注册表中的因子字段数量，不包含行情原始字段、键字段和诊断字段。

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
ETF 兼容入口：`scripts/generate_etf_fz_daily_factors.py`
实现来源：`FractalQuant/factor/fz_methods.py`、`fz_factor.py`、`fz_base.py`

脚本通过 `--asset-type stock|etf` 选择股票或 ETF 默认数据目录，先把分钟行情拆成
交易日面板，计算原始 FZ 因子，再计算组合因子，最后按 `code/factor_date` 输出
每个标的一份日频 parquet。输入必须是 orderbook 同口径的
241 根分钟：`09:30–11:30`、`13:01–15:00`；缺行、重复或异常时间网格的
`code/date` 会跳过且记录告警。`GaoDiECha` 等因子还需要对应资产的日线数据。

潮汐因子在通过241分钟输入校验后，按研报剔除 `09:30` 和 `15:00`，将其余239分钟
重新编号；9分钟邻域成交量在完整序列上计算，涨潮低点使用 `5..t-1`，退潮低点使用
`t+1..233`。边界分钟参与低点选择，缺少任一有效半潮汐时保持空值。`ChaoXi` 要求
最近20个交易日的两个原始半潮汐值均有效，不会删除缺失日后跨更长区间凑满20个样本。

`factor_date=d` 的暴露包含 d 日完整分钟信息，只能在 d+1 或之后用于回测和交易；
月频回测应选取月末构造日，而不是把日频因子复制到分钟行。旧
`generate_etf_fz_minute_factors.py` 仅保留兼容入口，运行时会输出弃用提示并生成同样的
日频文件。

`CaoMuJieBing` 依赖 `retail_trade_ratio` 和中证全指 `000985.CSI` 日收益
`csi_all_share_return`。`retail_trade_ratio` 是单笔成交额小于4万元的个人投资者买卖额
均值除以当日总成交额，必须来自逐笔成交数据，不得用零值、普通成交量或分钟 OHLCV
推算。股票可传入 `--build-cao-mu-source`，从 `E:\逐笔数据` 的逐笔成交 CSV 构造并
增量缓存到 `D:\workspace\stockdata\stock-factors\stock_cao_mu_jie_bing_source.parquet`；
中证全指收益率由 `D:\workspace\stockdata\指数数据\index_daily\000985.CSI.parquet`
的收盘价计算。生成器会合并该侧车文件但不会修改原始股票日线。逐笔文件缺失的日期，
该因子仍为空；传入 `--strict-source-fields` 会在分钟分片前终止。

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
uv run python scripts/generate_etf_fz_daily_factors.py --workers 5
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

### 3.3 CICC 分钟因子

入口：`scripts/generate_etf_cicc_minute_factors.py`  
实现来源：项目同级 `Replication-of-Minute-Frequency-Factor-refer-CICC`

CICC 函数先生成日级暴露，再按交易日合并回原始分钟行。输出保留分钟键和行情字段，
日级因子在同一交易日的分钟行中重复。该脚本当前注册 58 个 CICC 因子。

```powershell
uv run python scripts/generate_etf_cicc_minute_factors.py --workers 5
```

同样是文件级跳过；需要补充或修复已有标的时必须使用 `--overwrite`。

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

`base` 输出 63 个 orderbook 因子，`multi` 输出 135 个因子。因子先在逐笔盘口/成交时间轴
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

```powershell
uv run python scripts/generate_intraday_strategy_factors.py --asset-type etf --workers 5
uv run python scripts/generate_intraday_strategy_factors.py --asset-type etf --priority-profile p0_p1 --workers 5
```

该入口支持按日期替换已有输出；没有日期过滤时会按全量结果处理。P1 的市场资金流
还依赖 `--pool-membership-path`、`--target-pool-path`（如果研究配置启用了动态池）。

### 3.7 开盘集合竞价因子

入口：`scripts/generate_auction_factors.py`

该脚本生成每日开盘竞价因子，当前有 51 个核心字段，分为：

- overnight/stage1/stage2 收益和金额比例；
- 委托撤单、虚假压力和大单行为；
- stage2 路径、波动、反转和效率；
- 鲁棒 imbalance/fisher 变化；
- 基于前 5/20 日 ADV 的参与度；
- 相对基准 ETF 的超额收益和市场上下文。

默认 ETF 输出：

```text
D:\workspace\stockdata\etf-data\etf_auction_factors
```

默认 benchmark 为 `510300.SH`。使用 `--write-session-path-factors` 时，还会输出
分钟级 session path companion parquet，默认目录为
`D:\workspace\stockdata\etf-factors\etf_intraday_session_path_factors`。

```powershell
uv run python scripts/generate_auction_factors.py --asset-type etf --workers 5
uv run python scripts/generate_auction_factors.py --asset-type etf --write-session-path-factors --workers 5
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
| 标准分钟、FZ、CICC、advanced、noise-bound | 标的文件 | 传 `--overwrite` 重算整个目标文件 |
| orderbook | 标的 + 交易日 | 不传 `--overwrite` 可增量补新日期 |
| intraday strategy | 标的；指定日期时按请求日期替换 | 使用 `--date-from/--date-to` 加 `--overwrite` |
| auction | 标的 + 请求日期 | 使用 `--date-from/--date-to` 加 `--overwrite` |

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
uv run python scripts/generate_etf_cicc_minute_factors.py --workers 5
uv run python scripts/generate_etf_advanced_factor.py --workers 5
uv run python scripts/generate_etf_orderbook_factors.py --workers 5
uv run python scripts/generate_intraday_strategy_factors.py --asset-type etf --workers 5
uv run python scripts/generate_auction_factors.py --asset-type etf --workers 5
uv run python scripts/generate_etf_noise_bound_ratio_factors.py --workers 5
```

建议先生成基础分钟、日线依赖，再生成需要逐笔数据的 advanced/orderbook/竞价管线。
不同输出目录彼此独立，可以分批运行，但不要让同一个脚本的两个实例同时写同一目录。

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

## 7. 代码维护边界

- 标准分钟因子的注册和窗口组合在 `generate_etf_minute_factors.py`；
  通用因子实现位于 `FractalQuant/factor/` 对应模块。
- 股票和 ETF orderbook 的计算真值在 `FractalQuant/factor/stock_orderbook.py`，不要只修改某个生成脚本中的列名列表。
- advanced 的自动发现机制依赖 `advanced.py` 中的 `BaseFactor` 子类；新增类后要确认构造函数可无参数实例化，并确认是否属于未来标签。
- 日内策略和 P1 的列清单分别由 `intraday_strategy.py`、`intraday_strategy_p1.py` 维护；生成脚本只负责资产、日期、缓存和输出。
- 外部复现管线的公式来自同级 Replication 目录；升级外部代码后应重新做一标的对照验证。

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
