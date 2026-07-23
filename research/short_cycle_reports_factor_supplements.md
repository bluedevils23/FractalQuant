# 短周期交易策略研报对当前因子的补充建议

研究日期：2026-07-19

## 1. 研究范围

阅读并核对了以下 4 篇海通证券短周期交易策略研报：

1. `131_短周期交易策略研究之一——基于集合竞价分时走势的A股T+0策略.pdf`（17 页，2019-07-14）
2. `152_短周期交易策略研究之二——基于日内收益分布特征的股指期货交易策略.pdf`（26 页，2019-12-25）
3. `156_短周期交易策略研究之三——日内价格异动个股的短期收益表现.pdf`（20 页，2020-02-13）
4. `161_短周期交易策略研究之四——基于周内效应和市场状态的A股择时策略.pdf`（27 页，2020-03-11）

原始目录：`D:\BaiduNetdiskDownload\日内交易&T0资料合集\短周期交易策略`

当前代码覆盖基线：

- `scripts/generate_auction_factors.py`：37 个集合竞价因子，包括两阶段收益、路径、L3 不平衡、撤单、巨单、虚假压力和成交参与度。
- `scripts/generate_stock_orderbook_factors.py`：63 个基础订单簿因子，包括 OFI/MLOFI、深度、冲击、韧性、订单流和成交流。
- `scripts/generate_etf_minute_factors.py`：价格动量、技术指标、波动率、量价和微观结构因子，并有多窗口版本。

因此，研报的真实增量不在普通动量、波动率、OFI 或瞬时盘口不平衡，而在：

1. 固定交易阶段的原文口径；
2. 从开盘起累计的会话路径；
3. 标的相对市场、现货或基准的状态；
4. 日历效应与市场状态的交互；
5. 事件发生后才启动的非对称条件因子。

## 2. 结论摘要

### 2.1 最值得优先补充的因子

第一优先级：现有数据即可实现、与当前因子不重复。

| 因子 | 主要来源 | 用途 | 最早可用时间 |
|---|---|---|---|
| `auction_amount_to_prev5d_adv_240` | 研报 131 pp.6-8 | 原文集合竞价量比 | 开盘撮合结果到达后 |
| `auction_final_vs_stage2_twap` | 研报 152 的定价偏差逻辑 | 最终撮合价相对阶段二价格中枢 | 09:25 后 |
| `auction_l3_imbalance_twap_stage2` | 研报 152 的长窗口委托不平衡 | 阶段二绝对买卖压力水平 | 09:25 后 |
| `auction_relative_spread_twap_stage2` | 研报 152 的长窗口价差 | 阶段二流动性状态 | 09:25 后 |
| `intraday_drawdown_from_session_high` | 研报 156 p.5 | 当前价格相对当日高点的回撤 | 当前分钟结束后 |
| `intraday_rebound_from_session_low` | 研报 156 p.5 | 当前价格相对当日低点的反弹 | 当前分钟结束后 |
| `intraday_return_from_prev_close` | 研报 156 | 昨收锚定的当日累计收益 | 当前分钟结束后 |
| `market_return_from_prev_close` | 研报 156 pp.6-8 | 同分钟市场状态 | 全市场该分钟到齐后 |
| `prev_2d_return_rank_cs` | 研报 156 p.8 | 异动前短期状态 | 当日开盘前 |
| `prev_20d_return_rank_cs` | 研报 156 p.5 | 异动样本的月度位置 | 当日开盘前 |
| `market_above_ma20_prevclose` | 研报 161 pp.7-8 | 日级上涨/下跌市场状态 | 当日开盘前 |
| `market_momentum_2d_prevclose` | 研报 161 pp.7-9 | 更短周期市场状态 | 当日开盘前 |

第二优先级：需要官方涨跌停、流通股本、基准映射或更长窗口聚合。

| 因子 | 主要来源 | 额外依赖 |
|---|---|---|
| `auction_stage1_touched_limit_up/down` | 研报 131 p.8 | 当日官方涨跌停价、ST/上市状态 |
| `auction_stage2_monotonic_up/down` | 研报 131 p.8 | 可由现有阶段二收益和反转次数派生，主要用于原文复现 |
| `cum_turnover_rate`、`cum_turnover_rank_cs` | 研报 156 pp.7-8 | 当时可知的流通股本、全市场截面 |
| `prev_20d_bottom10_flag` | 研报 156 p.5 | 完整历史股票池与截面排名 |
| `preclose_return_15m/30m` | 研报 152 pp.12-14 | 固定尾盘窗口 |
| `preclose_l1_imbalance_mean_30m` | 研报 152 pp.11-12 | 14:27-14:57 盘口时间加权 |
| `preclose_excess_return_benchmark_15m` | 研报 152 | 股票/ETF 到基准映射 |
| `auction_gap_excess_benchmark` | 研报 152 的期现逻辑迁移 | 基准开盘撮合结果 |
| `effective_weekday`、`is_post_long_holiday` | 研报 161 p.17 | 交易日历 |

### 2.2 不应重复增加的内容

- 普通短窗动量、收益率、RSI、MACD 和波动率：当前分钟因子已覆盖。
- 60 秒订单流、成交方向、冲击和盘口不平衡：当前订单簿因子已覆盖。
- 集合竞价两阶段收益、阶段二斜率/效率/反转次数、撤单和虚假压力：当前 37 个竞价因子已覆盖。
- 单独的星期几哑变量：在股票截面同一时刻是常数，不能直接产生截面排序价值，应作为市场状态交互或模型门控。

## 3. 研报一：集合竞价路径

### 3.1 研究发现

样本为 2015-01 至 2019-05 A 股，预测目标为 `收盘价/开盘价-1`（pp.6-8）。

- 隔夜涨幅 rank IC 均值 -0.15。
- 09:15-09:20 阶段涨幅 rank IC 均值 -0.11。
- 09:20-09:25 阶段涨幅 rank IC 均值 0.03。
- 集合竞价量比 rank IC 均值 0.01。
- 第一阶段触及涨停的股票日内平均收益 -0.16%，触及跌停为 0.57%。
- 第二阶段持续上行的股票日内平均收益 0.57%，持续下行为 -0.12%。

报告的集合竞价量比是：

```text
240 * 当日集合竞价成交额 / 过去5个完整交易日的平均全天成交额
```

当前 `auction_amount_ratio_5d` 是“当日竞价成交额/过去 5 个竞价日的平均竞价成交额”，不是同一指标；`auction_amount_to_prev20d_adv` 的窗口和缩放也不同。

### 3.2 建议字段

#### `auction_amount_to_prev5d_adv_240`

```text
240 * auction_amount[d] / mean(daily_amount[d-1:d-5])
```

- 历史日必须完整，严格排除当日。
- 5 日不足或任一历史日缺失时返回 NaN。
- 用于 09:26 以后或开盘到收盘预测；不能假定完整信号仍能以开盘撮合价成交。

#### `auction_stage2_monotonic_up/down`（派生复现字段）

对 `[09:20,09:25)` 按时间排序、去重后的有效指示价 `P[0:n]`：

```text
n >= 2 and P[-1] > P[0] and all(diff(P) >= 0)
```

对应的严格持续下行定义为：

```text
n >= 2 and P[-1] < P[0] and all(diff(P) <= 0)
```

当前 `auction_stage2_up_step_ratio` 会把平价 tick 计入分母，不能精确替代“总体上涨且没有 tick 下跌”。

不过当前已经同时输出 `auction_return_stage2` 和 `auction_stage2_reversal_count`。在快照完整、反转次数忽略平价步的前提下，可直接派生：

```text
monotonic_up   = return_stage2 > 0 and reversal_count == 0
monotonic_down = return_stage2 < 0 and reversal_count == 0
```

因此这两个标记主要用于忠实复现研报，不构成新的连续信息维度，优先级低于成交量口径、阶段一触限和会话累计路径。

#### `auction_stage1_touched_limit_up/down`

```text
exists indicative_price[t] >= official_limit_up[d]
exists indicative_price[t] <= official_limit_down[d]
```

- 必须读取当日官方限价，不能统一使用前收盘正负 10%。
- 需要处理 ST、科创板/创业板、北交所和上市初期无涨跌停等规则。
- 缺少官方限价时应为 NaN。

### 3.3 研报回测不可直接照搬

- 最终阶段路径和成交额在撮合后才完整，却假设按同一开盘价买入，存在同撮合时点偏差。
- A 股 T+0 需要底仓或融券。
- 407.9% 是单利年化，样本内参数筛选明显，且缺乏独立样本外验证。
- 报告自身显示开盘后 15/30/60 秒成交容量有限；这些数据只能作为执行诊断，不能回填为 09:25 因子。

## 4. 研报二：日内分布、尾盘与期现状态

### 4.1 研究发现

样本为 2016-01 至 2019-11 的 IF/IH/IC 主力合约（pp.6-21）。报告所谓“隔夜收益”实际是前收盘至次日 10:00，必须拆为纯隔夜和次日早盘两段。

主要有效信号：

- 收盘价低于结算价；
- 尾盘 30 分钟委买量大于委卖量；
- 尾盘 15 分钟下跌；
- 尾盘 15 分钟基差下降。

报告存在口径冲突：单因子与三因子章节写“收盘低于结算价做多”，双因子章节却写成“高于结算价做多”；结合相关系数方向，低于结算价更合理，但无法确认原始回测代码。

### 4.2 对当前项目的真正增量

当前已有大量 10-300 秒盘口和成交流因子，研报带来的增量是固定尾盘长窗口和分段标签。

建议字段：

```text
preclose_return_15m
    = log(mid[14:57-) / mid[14:42])

preclose_return_30m
    = log(mid[14:57-) / mid[14:27])

preclose_to_vwap_60m
    = mid[14:57-) / VWAP[14:00,14:57) - 1

preclose_l1_imbalance_mean_30m
    = TWAP((bid_qty1-ask_qty1)/(bid_qty1+ask_qty1), [14:27,14:57))

preclose_relative_spread_mean_15m
    = TWAP((ask1-bid1)/mid, [14:42,14:57))

preclose_excess_return_benchmark_15m
    = asset_log_return_15m - benchmark_log_return_15m
```

A 股/ETF 14:57 后进入收盘集合竞价。若信号用于参加收盘竞价，只能使用 14:57 前数据；若使用实际收盘价，因子只能用于次日。

标签应拆为：

```text
target_close_to_next_open
target_next_open_to_1000
target_close_to_next_1000
```

这能判断收益来自隔夜风险补偿还是次日早盘日内效应。

### 4.3 对集合竞价层的合理迁移

以下是从期现逻辑推导出的研究假设，不是研报直接验证结果：

```text
auction_gap_excess_benchmark
    = log(asset_open/asset_prev_close)
      - log(benchmark_open/benchmark_prev_close)

auction_stage2_excess_return_benchmark
    = asset_stage2_log_return - benchmark_stage2_log_return

auction_final_vs_stage2_twap
    = final_match_price / TWAP(stage2_indicative_price) - 1

auction_l3_imbalance_twap_stage2
    = TWAP(stage2_l3_imbalance)

auction_relative_spread_twap_stage2
    = TWAP((ask1-bid1)/mid)
```

相对价差只使用 `ask1>bid1>0` 的有效快照；交叉盘口不能当成正常负价差。股票基准可选宽基或行业指数；ETF 优先使用跟踪指数。所有跨基准字段等待标的和基准中较晚的数据到达时间。

## 5. 研报三：日内极值异动后的非对称收益

### 5.1 研究定义与结论

样本为 2015-2019 A 股，且先限定“前 1 个月涨幅最低 10%”，再剔除 ST、上市不足 3 个月及当日涨跌停股票（p.5）。

```text
日内大幅下跌 = close/session_high - 1 < -5%
日内大幅上涨 = close/session_low  - 1 >  5%
```

核心不是异动本身，而是方向非对称交互：

- 市场下跌时，冲高回落且换手不高、前 2 日收益较高的股票更像过度反应。
- 市场上涨时，从低点大幅反弹且前 2 日收益较低的股票更像补涨。
- 异动股次日普遍低开，收益主要集中在随后约 2 日；部分方向的中长期超额不持续。

最终策略 2015-2019 年化 49.9%，但最大回撤 49%，且阈值与规则均在同一时期选择。

### 5.2 分钟因子原子层

```text
intraday_drawdown_from_session_high[m]
    = close[m] / max(high[session_start:m]) - 1

intraday_rebound_from_session_low[m]
    = close[m] / min(low[session_start:m]) - 1

intraday_return_from_prev_close[m]
    = close[m] / previous_close - 1

intraday_down_excess_5pct[m]
    = max(-intraday_drawdown_from_session_high[m] - 0.05, 0)

intraday_up_excess_5pct[m]
    = max(intraday_rebound_from_session_low[m] - 0.05, 0)
```

必须按交易日重置，只使用当前分钟及此前数据；分钟 `m` 生成的因子最早在 `m+1` 成交。

历史与截面状态：

```text
prev_2d_return = close[d-1] / close[d-3] - 1
prev_2d_return_rank_cs = 当日开盘前的横截面百分位
prev_20d_return = close[d-1] / close[d-21] - 1
prev_20d_bottom10_flag = prev_20d_return 横截面排名 <= 10%
cum_turnover_rate[m] = cumulative_volume[m] / float_shares[d-1]
cum_turnover_rank_cs[m] = 同分钟横截面百分位
```

若缺少流通股本，应改名为同时间进度成交量/金额比，不应错误称作换手率。

### 5.3 建议派生交互

策略层或模型交互层再构造：

```text
down_overreaction_score
    = 1[market_return<0]
      * 1[drawdown<-5%]
      * 1[turnover_rank<=2/3]
      * 1[prev_2d_return_rank>1/3]
      * (-drawdown-5%)

up_catchup_score
    = 1[market_return>0]
      * 1[rebound>5%]
      * 1[prev_2d_return_rank<=1/3]
      * (rebound-5%)
```

应同时保留连续原子因子，避免只依赖样本内发现的 5% 和三分位硬阈值。

建议检验与现有微观结构因子的交互：

- `down_overreaction_score_x_ofi_reversal`
- `up_catchup_score_x_ofi_confirmation`
- `prevday_intraday_event_x_auction_overnight_return`

最后一个可检验研报所述“异动后次日低开”是否已被现有竞价隔夜收益吸收。

## 6. 研报四：周内效应与市场状态

### 6.1 研究发现

样本为 2005-01 至 2020-02（pp.7-22）。沪深 300 全样本周一收益最高、周四最低，但区分市场状态后差异更明显：

- 上涨市：周一和周五较强；
- 下跌市：周一和周四较弱；
- 下跌市的周二、周三相对较好，常表现为对周一的修正。

市场状态有两种定义：

```text
前一交易日收盘价 > 前一交易日 MA20
过去 2 个完整交易日累计收益 > 0
```

报告使用 5 年滚动窗口、每 20 日重新拟合 GARCH-M。沪深 300 上涨市周一效应在滚动窗口中显著占比 100%，下跌市周一和周四的负向效应较弱但仍有状态差异。

### 6.2 建议字段与交互

```text
market_above_ma20_prevclose
    = sign(index_close[d-1] / index_ma20[d-1] - 1)

market_momentum_2d_prevclose
    = index_close[d-1] / index_close[d-3] - 1

calendar_gap_days
    = date[d] - date[d-1]

is_post_long_holiday
    = 1[calendar_gap_days >= 3 and actual_weekday != Monday]

effective_weekday
    = Monday if calendar_gap_days >= 3 else actual_weekday
```

模型层交互：

```text
effective_monday_x_market_state
thursday_x_down_market
friday_x_up_market
tue_wed_x_prior_monday_return
```

对于 09:25 或开盘预测，市场状态只能使用前一日收盘信息；对于收盘生成次日信号，可使用 14:55 的当日指数状态，但交易必须在之后发生。

星期几在同一股票截面是常数，不应作为独立选股因子。它适合：

- 时间序列择时；
- 对个股 alpha 的门控；
- 与 beta、行业暴露、市场状态或异动事件交互。

## 7. 推荐实现顺序

### P0：低依赖、高增量

1. 在集合竞价生成器补 `auction_amount_to_prev5d_adv_240`。
2. 补阶段二 TWAP 价格中枢、L3 不平衡均值和相对价差均值。
3. 在分钟因子层补会话累计高点回撤、低点反弹和昨收累计收益。
4. 建立严格滞后的市场指数状态：MA20、2 日动量、同分钟市场收益。
5. 补前 2 日、前 20 日收益及横截面排名。

### P1：跨截面和跨市场

6. 接入官方涨跌停价后补阶段一触及涨跌停。
7. 接入流通股本后补累计换手率及同分钟截面排名。
8. 建立股票/ETF 到宽基、行业或跟踪指数的映射，补相对收益与竞价 gap excess。
9. 增加尾盘 15/30/60 分钟固定窗口聚合。

### P2：交互与策略研究

10. 增加有效星期、长假后首日和市场状态交互。
11. 构造异动过度反应/补涨分数，并与 OFI、盘口韧性做增量检验。
12. 分离隔夜、开盘至 10:00、开盘至收盘等不同标签，避免把不同经济来源混成一个目标。

## 8. 实现前必须修正或明确的数据契约

独立审计当前代码后，以下问题比继续扩列更优先，否则新增因子可能带入隐性泄漏或错误时间语义：

1. 无开盘撮合时，竞价事件因子使用了完整 `[09:20,09:25)` 事件，但 `available_time` 可能只是最后一条行情时间，早于最后事件。应取全部输入的最大到达时间，或将完整阶段二因子统一设为 09:25 后可用。
2. 当前竞价指示价使用 `(ask1+bid1)/2`，不一定等于交易所理论撮合价；集合竞价还可能出现交叉盘口。触限、价差和 TWAP 因子必须先确认原始行情字段语义。
3. 行情阶段把 09:20 快照同时用作阶段一终点和阶段二起点，事件阶段则采用阶段一 `<09:20`、阶段二 `>=09:20`。应统一并测试 09:20:00 边界。
4. 分钟 bar 必须明确是开始标记还是结束标记；任何使用当前 OHLC/close 的因子最早只能在该 bar 完成后交易。15:00 行还可能包含收盘集合竞价，不能假设按 15:00 收盘价再次成交。
5. 固定 bar 数滚动会跨越午休，不等于相同长度的连续交易时间。尾盘、早盘和会话累计字段应按交易时段切分。
6. 同时间戳的委托、成交和行情若缺少交易所序号，不能默认事件一定先于快照；需要固定同时间戳排序契约。
7. `factor/crossmarket.py` 的部分 rolling 回调直接取参考序列末尾窗口；若接入历史时点会使用未来参考数据。新的 benchmark excess 应按时间戳 point-in-time join 单独实现，不应直接复用该逻辑。
8. `FutureReturnsFactor` 只能作为标签，训练特征清单必须硬性排除，即使 CLI 允许显式生成。
9. 横截面排名的可用时间是该分钟全部入选标的数据到齐后的最大时间戳；股票池、ST、上市期、停牌、流通股本和复权因子都必须 point-in-time。
10. 深沪成交方向字段需逐所验证；若使用 tick rule 或价格变化代理，只能命名为 proxy，不能与“净主动成交金额”混称。

## 9. 验证要求

四篇研报的样本都截止于 2020 年以前，不能把历史回测结果当成当前有效性的证明。新增因子应统一采用：

1. 2021-2026 独立样本外；
2. 按年份和注册制前后分段；
3. purged walk-forward，历史统计严格 `shift(1)`；
4. 对现有 37 个竞价因子、63 个订单簿因子做增量 IC 和消融；
5. 同时报告 Rank IC、ICIR、分组单调性、换手率、容量、交易成本后收益；
6. 对硬阈值做 4%-8%、窗口和分位数敏感性，不在全样本挑最佳参数；
7. 股票截面与时间序列任务分开评估；
8. 每个字段记录 `available_time`，标签从下一可成交时点开始；
9. A 股、ETF、宽基指数分别验证，不能把股指期货风险溢价直接外推；
10. 所有相对市场字段必须等待标的和基准中较晚的数据时间戳。

## 10. 总体判断

四篇研报对当前因子的最大价值，不是提供四套可直接复制的策略，而是提示当前体系还缺少三个结构层：

1. `session path`：从开盘起累计的高低点路径、固定尾盘窗口和阶段单调性；
2. `market context`：同分钟市场收益、MA20/2 日市场状态、标的相对基准偏离；
3. `event memory`：前一日异动、前 2/20 日状态、日历和长假后的条件效应。

优先补原子变量和严格可用时间，再让模型学习交互；研报中的硬信号只适合作为复现与消融基准。
