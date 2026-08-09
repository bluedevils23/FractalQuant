# 集合竞价研报与策略补充建议

研究日期：2026-08-09

## 1. 研究范围

本次核对 `D:\研报\集合竞价` 下全部 11 个文件：4 份券商研报和 7 份 JoinQuant 策略文本。其中 `4.集合竞价摸奖策略1.0-致敬2022.txt` 与 `43.集合竞价摸奖策略1.0-致敬2022.txt` 的 SHA256 完全相同，实际只有 10 份独立内容。

券商研报：

1. 广发证券 2014《基于集合竞价试盘者行为的 A 股 T+0 策略》
2. 海通证券 2019《基于集合竞价分时走势的 A 股 T+0 策略》
3. 广发证券 2024《集合竞价相关因子》
4. 长江证券 2024《开盘集合竞价因子》

当前比较基线为：

- `scripts/generate_auction_factors.py` 原有的 51 个 auction 因子，以及本次新增的 25 个可实现因子、诊断列和参考列；
- 同脚本的 3 个 session-path 分钟因子；
- `stock-miniqmt/AUCTION_FACTOR_STRATEGY.md` 的海通策略设计；
- `stock-alphalens-reloaded/stock_auction_t1_backtest.py` 的可执行 T+1 基准；
- 现有 auction LGBM 的 09:30 或 09:32 后入场实现。

## 2. 总体结论

原有 51 因子已经较完整地覆盖海通 2019 的核心信号，以及撤单、巨单、阶段二路径、竞价参与度、基准超额和前日市场状态。本次实现了其中数据源已具备、且公式可明确验证的 25 个字段；其余待补内容集中在广发成交委托比率、末分钟虚拟匹配量语义和收盘集合竞价三类。

1. 长江 2024 的精确振幅、总快照数和 L3 买量委比端点；
2. 广发 2014 和海通 2019 的阶段一触及涨跌停及冲高回落程度；
3. JoinQuant 策略反复使用的竞价量占昨量、竞价金额占流通市值和 09:24-09:25 末分钟变化；
4. 广发 2024 的成交委托比率，但必须先确认未披露的分子、分母和逐所订单方向语义。

推荐优先级：

| 优先级 | 内容 | 原因 |
|---|---|---|
| P0（已实现） | 长江 13 特征中原 51 因子缺失或口径不一致的字段 | 原始行情可计算，定义公开，报告给出长期分年结果 |
| P0（已实现） | 阶段一最高/最低相对昨收、触及涨跌停、最终开盘相对阶段一极值 | 同时服务广发 2014、海通 2019 和摸奖策略 |
| P1（部分实现） | 竞价量占昨量、竞价额占流通市值和末 60 秒价格已实现；虚拟匹配量变化待核 | 多份策略重复出现，但匹配量字段语义尚未确认 |
| P1 | 广发 2024 开盘成交委托比率 | RankIC 较强，但报告未披露公式，当前缓存也缺少完整匹配链 |
| P2 | 收盘集合竞价成交委托比率 | 有研究价值，但应进入独立 closing-auction 管线 |
| P2 | 10 秒重采样序列模型 | 数据和训练成本高，先用新增原子因子做增量检验 |

## 3. 长江 2024 的 13 个特征映射

报告使用每只股票过去 20 日原始特征均值做单因子回测，预测未来 5 日收益。最强的三个特征是：

- 快照数量：RankIC `-6.59%`；
- 二阶段振幅：RankIC `-6.19%`；
- 二阶段涨跌幅：RankIC `+5.20%`。

现有实现与原文的逐项关系如下：

| 原文特征 | 原文定义 | 当前实现 | 判断 |
|---|---|---|---|
| 振幅 | `(最高价-最低价)/最低价` | `auction_range_ratio` | 本次精确实现 |
| 一阶段振幅 | 阶段一的上述振幅 | `auction_stage1_range_ratio` | 本次精确实现 |
| 二阶段振幅 | 阶段二的上述振幅 | `auction_stage2_range_ratio`；原 `auction_stage2_range_bps` 仍保留 | 本次精确实现 |
| 隔夜收益率 | 开盘成交价/昨收-1 | `auction_overnight_return` | 精确覆盖 |
| 一阶段涨跌幅 | 一阶段价格/昨收-1 | `auction_stage1_end_return_from_prev_close`；原 `auction_return_stage1` 仍保留 | 本次按报告口径实现 |
| 二阶段涨跌幅 | 二阶段价格/一阶段价格-1 | `auction_stage2_end_return_from_stage1_end`；原 `auction_return_stage2` 仍保留 | 本次统一端点实现 |
| 上行比例 | 上行快照数/快照数 | `auction_up_step_ratio` | 本次精确实现 |
| 下行比例 | 下行快照数/快照数 | `auction_down_step_ratio` | 本次精确实现 |
| 快照数量 | 09:15-09:25 有效价格快照总数 | `auction_snapshot_count_total` | 本次实现为模型因子 |
| 开盘成交量 | 最终开盘成交量 | `auction_matched_volume` | 精确参考列 |
| 买量委比 | 三档买量/六档买卖量 | `auction_l3_buy_share_final` | 本次实现 |
| 一阶段买量委比 | 09:20 买量委比 | `auction_l3_buy_share_stage1_end` | 本次实现 |
| 二阶段买量委比变化 | 09:25 买量委比-09:20 买量委比 | `auction_l3_buy_share_change_stage2` | 本次统一端点实现 |

### 3.1 已实现的 P0 原子字段

```text
auction_range_ratio
auction_stage1_range_ratio
auction_stage2_range_ratio
auction_stage1_end_return_from_prev_close
auction_stage2_end_return_from_stage1_end
auction_up_step_ratio
auction_down_step_ratio
auction_snapshot_count_total
auction_l3_buy_share_final
auction_l3_buy_share_stage1_end
auction_l3_buy_share_change_stage2
```

其中：

```text
l3_buy_share = (bid_qty1 + bid_qty2 + bid_qty3) /
               (bid_qty1 + bid_qty2 + bid_qty3 + ask_qty1 + ask_qty2 + ask_qty3)
```

`l3_buy_share = (l3_imbalance + 1) / 2`，因此不需要重复计算底层量，只需输出原文要求的端点。阶段边界统一采用阶段一 `[09:15,09:20)`、阶段二 `[09:20,09:25)`；实际端点时间写入 `auction_stage1_end_time` 和 `auction_stage2_end_time` 参考列。

20 日均值建议先放在研究变换层，而不是继续扩张原始 auction 文件。报告没有明确“过去 20 日”是否包含当日，验证时固定比较两种口径：

```text
inclusive20 = rolling(20).mean()              # 含当日，09:25 后可用
prior20     = shift(1).rolling(20).mean()     # 仅历史日，开盘前可用
```

## 4. 阶段一试盘与触限因子

### 4.1 广发 2014

报告策略条件为：

- 09:15-09:20 阶段指示价触及涨停；
- 最终开盘涨幅在 2%-6%；
- 假设按开盘价买入；
- 盘中止盈 `+1.5%`、止损 `-1.5%`，未触发则尾盘卖出；
- 2010-01 至 2014-05 共 249 个交易日样本，报告年化 19.72%，双边成本 0.3%。

报告还把触及涨停后的回落分为小幅回落且高开、部分回落且小幅高开、大幅回落甚至低开三类。相比单一触限标记，连续变量更适合模型。

### 4.2 海通 2019

海通报告显示，阶段一触及涨停股票的平均日内收益为 `-0.16%`，触及跌停为 `+0.57%`。当前海通基准中的其他核心字段均已存在：

- `auction_overnight_return`；
- `auction_return_stage2`；
- `auction_stage2_efficiency_ratio`；
- `auction_amount_to_prev5d_adv_240`。

阶段二无下降价格变化（允许平价快照）可由 `auction_return_stage2 > 0` 且 `auction_stage2_efficiency_ratio` 接近 1 表达；实现时应使用数值容差，不必增加信息重复的布尔因子。

### 4.3 已实现字段

本次已实现不依赖官方限价的四个连续变量：

```text
auction_stage1_max_return_from_prev_close
auction_stage1_min_return_from_prev_close
auction_open_pullback_from_stage1_max
auction_open_rebound_from_stage1_min
```

股票日线已提供目标交易日官方涨跌停价，因此以下四项也已实现；ETF 日线缺少官方限价时保持 `NaN`：

```text
auction_stage1_touched_limit_up
auction_stage1_touched_limit_down
auction_stage1_limit_up_distance_bps
auction_stage1_limit_down_distance_bps
```

触限标记读取当日官方涨跌停价，没有用统一的昨收正负 10% 推算。若缺少官方限价，返回 `NaN`，不会写 `False`。距离定义为 `(涨停价-阶段一最高价)/涨停价` 和 `(阶段一最低价-跌停价)/跌停价`，并转换为 bps；触及时为 0 或负值。

当前脚本以买一卖一中点作为 `indicative_price`。只有供应商数据确认买一等于卖一，或字段确实代表虚拟撮合价时，触限和极值字段才可视为交易所口径；否则字段名应保留 `proxy`。

## 5. 广发 2024 Level-2 因子

报告列出 15 个因子：

- 09:15-09:20：买、卖、总成交委托比，以及买、卖、总撤单委托比；
- 09:20-09:25：买、卖、总成交委托比；
- 09:15-09:25：买、卖、总成交委托比；
- 14:57-15:00：买、卖、总成交委托比。

报告明确写着“详细定义请联系作者”，因此不能把当前撤单量/新增量直接命名为原文因子。当前最接近的是：

- `auction_bid_cancel_qty_ratio_stage1`；
- `auction_ask_cancel_qty_ratio_stage1`；
- `auction_cancel_notional_ratio_stage1`。

它们只属于代理，不是复现。

报告中较强的 20 日平滑因子为：

| 因子 | 20 日平滑 RankIC | 胜率 |
|---|---:|---:|
| `BuyTransaction_BuyOrder_ratio_09150925` | -10.10% | 28% |
| `Transaction_Order_ratio_09200925` | -9.20% | 28% |
| `BuyTransaction_BuyOrder_ratio_09150920` | -9.20% | 22% |
| `SellTransaction_SellOrder_ratio_14571500` | -9.60% | 23% |
| `BuyWithdrew_BuyOrder_ratio_09150920` | -5.00% | 27% |

### 5.1 实现前的数据缺口

当前 transaction 缓存只保留 `trade_code`、数量和买卖委托序号，并截到 `<09:25`。原始文件还包含 `BS标志`、成交价格和 09:25 开盘撮合记录。若报告的“成交委托比”是“某时段提交且最终在开盘撮合成交的订单占比”，则必须：

1. 保留 09:25 实际撮合记录；
2. 保留成交价格和 `BS标志`；
3. 通过买卖委托序号回连委托的提交时间和方向；
4. 分别验证上交所、深交所的编号和方向字段；
5. 明确分子分母使用订单数、股数还是金额。

在拿不到作者定义时，建议以独立命名实现三个候选口径并做敏感性检验，不要使用原报告字段名冒充精确复现。

收盘集合竞价因子需要单独缓存 `[14:57,15:00]` 委托和 15:00 撮合记录。它只能用于次日信号，不应塞入当前 09:25 可用的 opening-auction 文件。

## 6. JoinQuant 策略中可复用的原子变量

### 6.1 反复出现且值得研究

| 原子变量 | 来源 | 当前状态 | 建议层级 |
|---|---|---|---|
| `auction_volume_to_prevday_volume` | 16、21，阈值常为 3% | 本次实现，日线 `vol × 100` 转股数 | auction + prior-day context |
| `auction_amount_to_float_mcap_prevclose` | 96 | 本次对有 `circ_mv` 的股票实现，换算为人民币元 | context |
| `auction_last60s_price_return` | 99 的 09:24-09:25 比较 | 本次按虚拟撮合价实现 | auction |
| `auction_last60s_match_volume_growth` | 99 | 原始字段语义待核 | auction proxy |
| `auction_final_to_full_max` | 4/43 要求最终价等于竞价最高价 | 本次实现为最终虚拟撮合价/全程最高价-1 | auction |
| `auction_open_to_prev_high` | 99 | 本次实现 | context |
| `auction_open_to_prev7d_close_max` | 46 | 本次实现，要求连续 7 个有效交易日 | context |
| `auction_amount_log` 或横截面分位数 | 46、99 | 有 `auction_amount` 参考列 | 下游派生 |

`96.集合竞价量比策略V1.txt` 的所谓竞价换手率公式为 `money / (circulating_market_cap * 1,000,000)`。JoinQuant 流通市值通常以亿元计，该分母与亿元换算存在明显疑点，不能照抄阈值 `0.3`。应先统一成：

```text
auction_amount_to_float_mcap_prevclose = auction_amount / float_market_cap_cny[d-1]
```

### 6.2 不应进入通用 auction 因子层

以下内容属于策略资格或前日状态，不是竞价微观结构：

- 昨日涨停、首板/一进二/三板、连板次数；
- ST、退市、上市天数、停牌和开盘涨跌停；
- 前日成交额、市值、流通市值和换手率；
- 20/60 日相对位置、左侧压力、均线多头排列和跳空缺口；
- 前 3 日涨幅、昨日 K 线形态、历史涨跌停次数；
- 盘中止盈、止损、尾盘是否封板。

这些字段应放进 `eligibility/context/execution` 三层，避免让 auction 生成器依赖全套策略主数据。

## 7. 策略实现建议

### 7.1 海通持续上行策略：保留现有实现

现有 `stock_auction_t1_backtest.py` 已实现主要过滤、09:25:59 信号截止、09:30 分钟 VWAP 入场和下一交易日 14:50-14:56 VWAP 退出。无需再写一套重复策略。

继续同时保留三条收益线：

1. `paper_oracle`：按开盘价买入，仅用于论文复现；
2. `executable_t0_overlay`：要求开盘前已有可卖底仓，09:30 后买入，当日卖出等量旧仓；
3. `executable_t1`：09:30 后买入，下一交易日退出。

### 7.2 广发试盘者策略：新增规则基准

建议新增独立回测入口，不修改通用 LGBM：

```text
candidate =
    eligible_stock
    and auction_stage1_touched_limit_up
    and 0.02 <= auction_overnight_return <= 0.06
```

同时按 `auction_open_pullback_from_stage1_max` 分组，以检验报告三种回落类型。执行版本使用 09:30 后 15/30/60 秒 VWAP，不假设开盘撮合价成交。

止盈止损必须用分钟内高低价时处理同一分钟同时触及两条线的顺序不确定性，至少报告保守、乐观两种成交顺序。普通 A 股当天新买股份不可卖，T+0 版本必须绑定已有底仓，否则改成 T+1。

### 7.3 长江人工特征增量模型

先做低成本、可解释的增量实验，不直接复刻 TCN：

1. 基线：当前 51 auction 因子；
2. 实验 A：基线 + 11 个 P0 原子字段；
3. 实验 B：实验 A + 11 个 inclusive20 均值；
4. 实验 C：实验 A + prior20 均值；
5. 标签分别用下一交易日和未来 5 日收益；
6. 使用现有 walk-forward LGBM，固定相同股票池、时间切分、交易成本和持仓数。

只有 A/B/C 在独立样本外对 IC、分组单调性和成本后收益有稳定增量，才值得投入 10 秒重采样 TCN。报告本身指出开盘因子与分钟因子的秩相关达到 60.72%，增量测试必须同时控制技术和 orderbook 因子。

### 7.4 广发成交委托比率研究

先实现命名明确的候选口径，再对 09:15-09:20、09:20-09:25、全开盘竞价分段做单因子 IC。若候选定义无法在 2019-2024 大致复现报告方向，不进入模型。

收盘 14:57-15:00 卖方成交委托比率单独做 next-day 因子研究，不能和 09:25 决策混用。

### 7.5 JoinQuant 事件策略：只做模块化基准

建议保留四类独立基准，不合并成一个“大杂烩”策略：

- `weak_to_strong`：昨日首板/曾涨停 + 开盘缺口 + 竞价量占昨量；
- `last_minute_abnormal`：昨日涨停 + 09:24-09:25 价格、虚拟匹配量和成交额变化；
- `auction_turnover`：历史量比 Top-N + 竞价额占流通市值；
- `lottery_limit_up`：首板/三板 + 最终竞价价接近全程最高价 + 3.82%-6.18% 缺口。

原文本的阈值和卖出规则均为样本内经验规则，只作为复现分支。生产研究应保留连续变量，让模型或预先固定的敏感性矩阵决定阈值。

## 8. JoinQuant 代码的主要不可直接复用问题

1. 多个策略在 09:25 或 09:26 观察完整竞价结果，却假设按 `day_open` 成交，存在同撮合价格偏差。
2. 9:25-9:30 不接受普通连续竞价委托，完整信号应在 09:30 后按真实盘口成交。
3. 普通 A 股当日买入不可卖出；代码中的 `closeable_amount == 0` 实际会让所谓 T+0 逻辑变成隔夜持仓。
4. `46` 注释掉 `avoid_future_data`，且部分全市场/历史过滤没有 point-in-time 股票池证明。
5. 4/43 完全重复，不能当成两条独立证据。
6. 多数回测窗口短、参数多、缺少样本外、滑点、涨跌停排队和容量约束。
7. `99` 的 `v_ratio` 是最终 `volume` 相对 09:24 初始卖一量，不是通常意义的成交量增长率，必须改名并验证供应商字段。
8. 多个策略以 `a1_p` 替代虚拟撮合价，只有字段语义验证后才可复现。

## 9. 不建议新增的字段或实现

- 不新增 `stage2_monotonic_up/down` 到模型特征：可由现有收益和效率精确派生，信息重复。
- 不把固定阈值布尔条件写入生成器，如开盘 2%-6%、竞价量占昨量 3%、金额 1000 万。
- 不把 ST、首板、连板、市值和均线状态混入 auction 原始文件。
- 不把收盘集合竞价因子放入 09:25 可用的 opening-auction schema。
- 不直接复刻长江 TCN：先证明公开的 13 个人工特征相对原有 51 因子的增量。
- 不把现有撤单比例改名为广发成交/委托比例，二者经济含义不同。

## 10. 数据与验证要求

### 10.1 数据契约

- 阶段一 `[09:15,09:20)`，阶段二 `[09:20,09:25)`；
- 完整 auction 信号最早在 09:25 撮合记录到达后可用；
- `available_time` 取所有输入中最晚到达时间；
- 官方涨跌停价、ST、上市状态、流通股本和成分股池必须 point-in-time；
- 逐所验证订单号、撤单号、成交方向和 `BS标志`；
- 所有历史均值明确是否含当日，滚动历史不能使用未来日；
- 缺少源字段时返回 `NaN`，不写伪造的零或 False。

### 10.2 因子验收

每个新增字段至少验证：

1. 09:15、09:20、09:25 边界样例；
2. 平价快照、重复时间戳、空盘口和交叉盘口；
3. 上交所和深交所各至少 20 个手工回放日；
4. 公式与原始 CSV 独立重算一致；
5. `trade_date` 唯一、`available_time` 因果正确；
6. 与现有近似字段的相关性、差值分布和不一致样本解释。

### 10.3 回测验收

- 使用 2021-2026 独立样本外和 purged walk-forward；
- 分年度、板块、市值、行业、牛熊和高低波动状态；
- 同时报告 RankIC、ICIR、分组单调性、换手、容量和成本后收益；
- 入场使用 09:30 后 15/30/60 秒 VWAP，并模拟未成交和涨跌停；
- 比较 auction-only、auction + technical、auction + technical + orderbook；
- T+0 底仓收益与日内增量收益分开核算；
- 所有 JoinQuant 阈值做预先固定的邻域敏感性，不在样本外挑最优点。

## 11. 推荐实施顺序

1. 已完成：增加长江 P0 的振幅、总快照数、上下行比例和 L3 买量委比端点。
2. 已完成阶段一极值、开盘回撤、官方限价距离和触限字段。
3. 在现有 LGBM 上做 51 因子基线与 P0 增量消融。
4. 已完成末 60 秒价格收益、最终价相对全程最高价、竞价量占昨量、竞价额占流通市值、开盘相对昨高和前 7 日收盘最高。
5. 扩充 opening transaction 缓存并完成逐所订单匹配语义验证。
6. 只有成交委托比率复现方向通过后，再增加广发 2024 候选因子。
7. 独立建设 closing-auction 管线，研究 14:57-15:00 卖方成交委托比率。
8. 最后评估 10 秒重采样序列模型，避免在原子特征尚未验证前增加模型复杂度。

## 12. 最终判断

最有把握的新增价值来自长江 2024 的公开人工特征和广发 2014/海通 2019 的阶段一试盘极值。本次已将其中 25 个口径明确、数据源具备的字段加入生成器，使 auction 因子数由 51 增至 76，且没有改变 opening-auction 时间窗口。

广发 2024 的成交委托比率可能有更强选股能力，但其公式未公开，当前缓存也不足以还原“提交订单最终是否成交”的匹配链。应把它作为数据语义研究，而不是立即扩列。

JoinQuant 文本提供了有用的事件变量候选，但大部分收益来自涨停板资格、前日量价和经验阈值组合。正确做法是提取连续原子变量并在现有因子、技术因子和 orderbook 因子基线上做增量检验，不直接移植回测收益或成交假设。
