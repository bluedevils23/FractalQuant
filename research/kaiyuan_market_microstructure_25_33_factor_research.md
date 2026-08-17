# 开源证券市场微观结构研究系列（25-33）因子研究建议

## 范围与结论

本文逐篇阅读 `D:\研报\市场微观结构研究系列` 中第 25 至 33 篇，共 9 篇、160 页，覆盖 2024-06 至 2026-05。目标是识别能在 FractalQuant 现有股票/ETF 分钟和 L2 管线中复现的因子，而不是复述报告的历史回测收益。

本批最值得落地的是四组“条件化后的日频横截面因子”：

1. **成交量峰-岭-谷状态因子**：按同一交易时点的历史成交量异常和局部连续性切分分钟，再聚合量峰/量岭/量谷的数量、VWAP、成交额和间隔信息。
2. **价格跳跃峰-岭-谷状态因子**：以分钟振幅异常、邻近跳跃和价格缺口切分，再聚合跳跃状态的计数、收益、间隔和成交额跟随。
3. **高频振幅切割因子**：用价格或 1 分钟收益切分分钟振幅，分别保留跨日分钟池和先日内切割两种不相关构造。
4. **条件化分钟资金流残差**：将大小单净流按高振幅分钟、早盘和指数 5 分钟上涨情景分别筛选后，再做既有资金流残差处理。

订单方向长期记忆（报告 25）在理论上也有明确增量，但它要求逐笔**委托**数据，而不是仅有 L1/L2 快照或分钟成交；应在原始委托字段的方向、价格、数量、时钟顺序验证完备后列为第二阶段。报告 26、28、29、32 的主要贡献是失效诊断、监控和建模方法，不应直接增加为单一人工因子。

报告中的年化收益、IC 和 IR 均来自其历史样本、股票池、月末或双周调仓及双边千三等假设，不构成当前可复现的预期。所有日频候选因子必须用 d 日完整分钟/逐笔数据计算，并从 **d+1** 起可用；不可把完整日内统计回填到 d 日盘中决策。

## 研报覆盖

| 篇号 | 标题 | 主信息 | 结论 |
|---|---|---|---|
| 25 | 挂单方向长期记忆性的讨论与应用 | 委托方向 ACF、频谱、同类订单连续段 | L2 委托可用时值得实现 |
| 26 | 资金流与交易行为：因子失效的原因与讨论 | CNIR 的反转残差、分域与阈值漂移 | 作为资金流因子质量门控 |
| 27 | 高频成交量的峰、岭、谷信息 | 成交量状态切分和 20 个日频因子 | 最优先的分钟 OHLCV 扩展 |
| 28 | 因子切割论与深度学习的结合应用 | 切割三要素、DBD-GRU | 方法学，暂不直接落地模型 |
| 29 | 市场微观结构观察与高频因子回顾 | 早盘集中度、订单簿厚度、撤单率及既有因子跟踪 | 监控指标和回测筛选口径 |
| 30 | 高频振幅因子的内部切割 | 跨日分钟池及日内振幅切割 | 最优先的分钟 OHLCV 扩展 |
| 31 | 分钟资金流因子的构建方法 | 切割、时段、市场情景三类资金流残差 | 有逐笔成交时优先实现 |
| 32 | 深度学习赋能因子挖掘 2.0 | GRU+GAT、状态变量、特征集融合 | 后续模型研究，不作为一期因子 |
| 33 | 高频价格跳跃的峰、岭、谷信息 | 跳跃状态切分与 17 个日频因子 | 最优先的分钟 OHLCV 扩展 |

## 一、可直接实施的分钟 OHLCV 因子

### 1. 成交量峰-岭-谷（报告 27，第 4-21 页）

报告先按过去 20 个交易日的**同一交易分钟**标准化成交量，解决 A 股开盘和收盘的 U 型日内季节性。对股票 `i`、日期 `d`、分钟 `m`，建议冻结为下列仅用历史数据的定义：

```text
zv[i,d,m] = (volume[i,d,m] - mean(volume[i,d-20:d-1,m]))
            / std(volume[i,d-20:d-1,m])
eruption[i,d,m] = zv[i,d,m] > 1
valley[i,d,m]    = zv[i,d,m] <= 1
peak[i,d,m]      = eruption[i,d,m] and not eruption[i,d,m-1] and not eruption[i,d,m+1]
ridge[i,d,m]     = eruption[i,d,m] and (eruption[i,d,m-1] or eruption[i,d,m+1])
```

这里的 `zv > 1` 是把原文“高于 1 倍标准差”具体化的实现选择；原文未把均值中心化、等号归属写成代码，首次实验应将 `zv > 1`、`zv >= 1` 和仅正异常三种定义固定为小网格，而不能在全样本后选择最佳版本。跨午休不得将 11:30 与 13:00 视为相邻分钟。

优先保留信息较独立的六个候选，不要照搬报告全部 20 个高度相关因子：

```text
volume_peak_count_20d = sum(1{peak}) over last 20 sessions
volume_ridge_return_20d = sum(ret_1m * 1{ridge}) over last 20 sessions
volume_valley_vwap_rel_20d = mean(VWAP_valley / VWAP_day) over last 20 sessions
volume_peak_interval_kurt_20d = kurtosis(gaps between peak minutes) over last 20 sessions
volume_peak_ridge_amount_ratio_20d = sum(amount * 1{peak}) / sum(amount * 1{ridge})
volume_eruption_follow_ratio_20d = sum(amount[m+1] * 1{eruption[m]}) / sum(amount[m] * 1{eruption[m]})
```

报告方向可作为初始假设：量峰计数、量谷相对 VWAP、量峰间隔峰度、峰岭成交额比偏正；量岭收益及喷发成交额跟随比例偏负。方向必须由本地同口径回测重估。报告测试为月频、行业市值中性、双边千三；这是一组**日频截面因子**，并非用于同一分钟下单的特征。

与现有 `volume_spike_60s`、`volume_clustering_60s` 有底层数据重叠，但不重复：现有实现是固定滚动窗口的分钟级特征；本组因子按时点季节性校正，再利用孤立/连续形态进行日频聚合。

### 2. 高频振幅内部切割（报告 30，第 4-8 页）

报告给出两条不同的数据组织路径，两者约 30% 相关，可分别保留：

```text
amplitude[m] = high[m] / low[m] - 1

# 跨日分钟池，回看 N=10 个完整交易日，lambda=25%
V_high = mean(amplitude[m] for highest-lambda close[m])
V_low  = mean(amplitude[m] for lowest-lambda close[m])
minute_ideal_amplitude = V_high - V_low

# 先日内切割，再聚合
V_day[d] = mean(amplitude[d,m] for highest-lambda sentiment[d,m])
         - mean(amplitude[d,m] for lowest-lambda sentiment[d,m])
intraday_amplitude_cut = zscore_cs(mean(V_day, 10)) + zscore_cs(std(V_day, 10))
```

报告的优选参数为跨日分钟池 `N=10, lambda=25%`，日内版本 `N=10, lambda=20%`，情绪切割变量为 1 分钟收益。这里 `zscore_cs` 必须在可交易股票横截面、可得时点上计算。该家族与 `fz_methods.py` 的 `ZhenFuBoYi`、日内振幅相关方法有“振幅 + 切割”层面的重叠，但前者按收益排序进行累计博弈，本文是高/低状态的振幅均值差及其 10 日均值/标准差合成，建议作为新家族而非复用已有名称。

### 3. 价格跳跃峰-岭-谷（报告 33，第 4-16 页）

以分钟振幅作跳跃代理，先以同一交易分钟的历史振幅计算异常分数。跳跃后同时按局域情绪和跳跃结果分类：

```text
za[i,d,m] = zscore against i's prior-20-day same-minute amplitude
jump[m] = za[m] > 1

local_state[m] = (jump[m-1], jump[m+1])
gap[m] = max(low[m-1], low[m+1]) > min(high[m-1], high[m+1])

price_peak = jump and local_state is high-high and not gap
price_ridge = jump and local_state is low-low and gap
price_valley = not jump
```

`price_peak` 和 `price_ridge` 是对报告“非局域情绪高涨的无缺口跳跃”及“非局域情绪低迷的有缺口跳跃”的可运行表述；最终应把四类局域状态与 gap/no-gap 的八格完整保存，避免以过强的先验丢失信息。优先候选：

```text
price_peak_count_20d = sum(1{price_peak})
price_ridge_return_20d = sum(ret_1m * 1{price_ridge})
price_valley_vwap_rel_20d = mean(VWAP_valley / VWAP_day)
price_ridge_interval_skew_20d = skewness(gaps between price-ridge minutes)
price_jump_amount_leadlag_corr_20d = corr(amount[m], amount[m+1]) on jump minutes
```

报告称前四类以及跳跃成交额相关性在小市值域更强。现有 `TiaoYueDu`、`FanZhuanZhenFu` 是日线 Taylor 残差/振幅处理，`volume_spike` 是成交量异常；都不等同于本组“分钟跳跃形态 + 局域缺口”的状态因子。

## 二、逐笔成交和 L2 委托因子

### 4. 条件化分钟资金流残差（报告 31，第 3-9 页；报告 26，第 4-9 页）

报告 31 将逐笔成交合成为分钟大单、小单净流入，并在三种条件下筛选：高振幅分钟、前半小时、以及上证指数前一小时中 5 分钟收益最高的 50% 分钟。报告 26 给出了资金流剥离反转的核心形式：

```text
IMB = log((buy_notional + eps) / (sell_notional + eps))
IMB = alpha + beta * contemporaneous_return + residual
factor = mean(residual over past 20 sessions)
```

建议不要以固定金额定义大/小单。先按每个交易日、全市场或流动性分层的成交名义金额分位数分桶，并在数据字典中固定门槛。可构建：

```text
flow_large_high_amp_resid = residualized flow_large on the highest 50% own-amplitude minutes
flow_small_high_amp_resid = residualized flow_small on the highest 50% own-amplitude minutes
flow_large_open30_resid  = residualized flow_large during 09:30-10:00
flow_small_open30_resid  = residualized flow_small during 09:30-10:00
flow_large_index_up5_resid / flow_small_index_up5_resid
  = residualized flow on matching minutes where index 5m return is in its prior-sample upper half
```

报告 31 的经验方向是大单残差偏正、小单残差偏负；情景版本优于原始残差，但三种情景版本彼此相关较高。首轮每个方向只保留 `index_up5` 版本作为主候选，其他两种用于增量检验。现有 `trade_count_imbalance_60s`、`trade_qty_imbalance_60s`、`trade_notional_quantile_position_60s`、OFI/MLOFI 提供原始流和盘口压力，但没有“金额桶 + 情景筛选 + 日频反转残差”的完整构造，因此是增量。

报告 26 还给出重要的否定结论：加入市值、行业、流动性哑变量或使用单股时序回归，并未稳定优于当日截面回归；大小单最优阈值会漂移，资金流在非稳定市场环境和小市值域明显变弱。故这一家族必须输出 `bucket_threshold`、`coverage`、`residual R2`、按市值桶 IC 以及滚动 IC，而不是只输出因子值。

### 5. 挂单方向长期记忆与拆单痕迹（报告 25，第 4-22 页）

此组以**逐笔委托**而非成交方向为输入。将买入挂单编码为 `+1`，卖出挂单编码为 `-1`，对每只股票、每个交易日的序列 `X_n` 计算 1 至 100 阶自相关：

```text
gamma_k = Cov(X_n, X_(n-k))
rho_k = gamma_k / gamma_0
rho_k = a + b * log(k) + error_k,  k = 1,...,100
LMS = a
MEMO = 0.5 * zscore_cs(skew(rho_1:rho_100))
     + 0.5 * zscore_cs(kurtosis(rho_1:rho_100))
```

报告的 MEMO 还采用“仅最后半小时、价优且小额的委托”、20 日平滑。可进一步测试：

```text
OST = kurtosis(abs(FFT(X_small_order)))
order_island_mean = mean(lengths of consecutive same-side order runs)
order_island_std  = std(lengths of consecutive same-side order runs)
```

现有 `trade_direction_persistence_60s` 与这个逻辑仅部分相近：它基于成交方向持续性，且输出为 60 秒分钟值；LMS/MEMO 量化的是委托序列的多阶记忆形态，并形成日频横截面暴露。若原始委托 `side` 含义、撤单/修改事件、时间排序与交易所撮合顺序未验明，禁止用推断的成交方向替代并宣称复现。

## 三、可借鉴的方法，而非直接新增因子

### 因子切割和 DBD-GRU（报告 28）

报告将切割抽象为“对象、刀法、产出”：对象必须可加，刀法是可区分状态的指标，产出通常用高低状态的差或比。以理想反转为例：过去 20 日按日均单笔成交额分高/低各 10 日，`M = sum(ret_high) - sum(ret_low)`。这正是本批量峰岭谷、振幅、资金流因子的统一研发模板。

DBD-GRU 的可借鉴点是：以切割指标的时序中位数产生 high/low mask，两个 GRU 分支分别编码，最后用隐藏状态之差输出预测。报告训练采用滚动 5 年窗口、每年更新、未来 20 日收益、ICLoss。该模型需要完整股票截面、严格走样本外训练、模型注册与版本化；在手工状态因子完成基线验证前，不应直接进入生产因子目录。

### 市场微观结构监控（报告 29）

应新增为研究监控而不是 alpha：

```text
early_volume_share = volume[09:30-10:00] / volume[full day]
mean_order_notional = mean(order price * quantity)
book_depth = sum(bid_notional_l1:l5 + ask_notional_l1:l5)
hf_cancel_share = count(cancel lifetime < 1 second) / count(all cancels)
```

其中订单簿深度、冲击成本及部分撤单指标已有实现或原始数据支撑。报告 29 最有价值的结论是：应对因子做 20 日滚动跟踪并按 2023 年后、股票池和市场状态分段报告；拆单相关 MEMO/LMS、强反转和彩票委托不应因为旧报告表现好而跳过本地样本外验证。

### 深度学习 2.0（报告 32）

报告提出 GRU 抽取时序信息、GAT 抽取行业/财务/资金流关系，使用过去 20 日 Barra 因子收益经 MLP/softmax 做状态自适应（SA）加权；高频数据先降到日度。该架构适合作为后续“多因子合成模型”研究，前置条件是：可点时复现的财务数据、成分和行业映射、明确的训练/验证/测试时间切分、模型产物治理及与线性组合的增量比较。

本项目当前最稳妥的路径是先把本批人工状态因子作为 `HF` 特征组，与已有 OFI、MLOFI、成交冲击和技术因子一起做 walk-forward ridge/LightGBM 基线。只有基线在独立测试窗中有稳定增量，才评估 GRU/GAT；不能以报告的同一历史区间收益作为模型上线依据。

## 四、与 FractalQuant 现有实现的重合度

| 本批主题 | 现有实现 | 重合判断 | 推荐动作 |
|---|---|---|---|
| 原始订单簿失衡、深度、冲击 | `depth_imbalance_l5`、OFI/MLOFI、`orderbook_liquidity_l5`、`market_impact_60s` | 已覆盖核心原始信号 | 不重复新增 |
| 成交方向持续性 | `trade_direction_persistence_60s` | 与委托长期记忆的输入和统计量不同 | 有委托数据后新增 LMS/MEMO |
| 成交金额位置 | `trade_notional_quantile_position_60s` | 仅分钟成交位置，不含大/小单情景残差 | 新增条件化资金流日频因子 |
| 成交量异常/聚类 | `volume_spike_60s`、`volume_clustering_60s` | 没有同分钟季节性校正和峰岭谷形态 | 新增量峰岭谷日频家族 |
| 跳跃/振幅 | `TiaoYueDu`、`FanZhuanZhenFu`、`ZhenFuBoYi` | 日频 Taylor/博弈构造，不含分钟状态与缺口 | 新增跳跃峰岭谷、振幅切割 |
| 量价/VWAP | `volume_weighted_price`、`price_volume_decoupling` | 未按量谷、价谷等条件化 | 用作新因子的比较基线 |
| 撤单、彩票委托 | 撤单/订单簿因子已有部分基础 | 本批未提供完整可复现公式 | 仅纳入监控和后续 L2 研究 |

## 五、实施优先级和验收口径

| 优先级 | 因子组 | 输入 | 建议载体 | 验收标准 |
|---|---|---|---|---|
| P0 | 量峰岭谷 | 1min OHLCV/amount | 新建日频状态因子生成器 | 同时点历史窗口无泄漏；与 `volume_spike` 相关性、横截面 IC 和成本后表现均报告 |
| P0 | 价格跳跃峰岭谷 | 1min OHLCV/amount | 同一生成器 | 八类跳跃状态可审计；午休无跨段邻接；市值桶分域稳定性通过 |
| P0 | 高频振幅切割 | 1min OHLCV | 同一生成器 | 10 日/20-25% 参数预注册；跨日与日内版本有低相关且增量有效 |
| P1 | 条件化分钟资金流残差 | 逐笔成交、可靠方向 | 独立逐笔日频生成器 | 分桶阈值、方向来源、覆盖率、残差回归和市值分域均写入审计输出 |
| P1 | 委托长期记忆 | 逐笔委托 | `stock_orderbook.py` 的日频聚合旁路或独立生成器 | 只用明确委托事件；会话重置；ACF/FFT 样本长度和价优/小额过滤可复现 |
| P2 | 微观结构状态监控 | 分钟、订单簿、撤单 | 研究/质量仪表盘 | 20 日滚动、市场状态、股票池覆盖和原始数据异常共同输出 |
| P3 | DBD-GRU / GRU+GAT | 全量 PIT 面板 | 独立 research pipeline | 严格 walk-forward、特征消融、线性基线增量与模型版本复现 |

首轮不应使用报告中几十个同家族候选全量扫描。每个家族先冻结 3-6 个代表信号，做行业/市值中性、不同股票池、不同延迟和成本敏感性检验；只保留在未参与参数选择的时间窗中仍有增量的信号。对 ETF，量峰岭谷、跳跃和振幅可在基金自身 1 分钟数据上计算，但报告的股票横截面定价结论不能直接外推；ETF 应先做基金横截面、流动性分层和对应指数/资产类别分层验证。

## 六、来源定位

所有来源均位于 `D:\研报\市场微观结构研究系列`：

* 第 25 篇，第 4-22 页：订单方向编码、ACF 回归、MEMO、OST 与订单小岛；
* 第 26 篇，第 4-9 页：CNIR 反转残差、回归替代方案和大小单阈值漂移；
* 第 27 篇，第 4-23 页：成交量峰岭谷定义、因子构造与相关性；
* 第 28 篇，第 3-10 页：因子切割论和 DBD-GRU；
* 第 29 篇，第 4-10 页：微观监控维度和高频因子跟踪；
* 第 30 篇，第 4-8 页：分钟理想振幅和日内振幅切割；
* 第 31 篇，第 3-9 页：分钟资金流的切割、时段和情景残差；
* 第 32 篇，第 5-21 页：GRU+GAT、SA 加权、特征集和应用；
* 第 33 篇，第 4-18 页：价格跳跃峰岭谷定义、代表因子和组合。
