# 开源证券市场微观结构研究系列（21-24）因子研究建议

## 范围与结论

本文核阅 `D:\研报\市场微观结构研究系列` 第 21 至 24 篇，共 4 篇、77 页：

| 篇号 | 日期 | 标题 | 页数 | 可落地结论 |
|---|---:|---|---:|---|
| 21 | 2023-09-19 | 订单流系列：关于市场微观结构变迁的故事 | 23 | 主要是市场结构状态和因子失效监控，不应直接再注册为 alpha |
| 22 | 2024-01-24 | 订单流系列：撤单行为规律初探 | 15 | TRI 和 TOX 有明确增量，但需要保留订单 ID 的全生命周期逐笔委托数据 |
| 23 | 2024-01-28 | 大小单资金流为核心的综合行业轮动方案 | 23 | 行业级资金流事件因子与股票/ETF 分钟订单簿因子不同，具备低重合的上层增量 |
| 24 | 2024-05-24 | 深度学习赋能交易行为因子 | 16 | 是模型合成方案，不能作为一期手工因子直接加入 `factor` 目录 |

结论按实现优先级如下：

1. **P0 - 数据契约与监控**：将早盘成交集中度、订单执行时长、订单名义金额和高频撤单占比作为每日市场结构监控；它们用于识别样本漂移和因子失效，不作为盘中万能预测特征。
2. **P1 - 撤单生命周期因子**：待全量逐笔委托输入保留 `order_id`、事件类型和时间后，独立实现 `auction_sell_cancel_tri`、`cancel_tox_5s_30s` 与撤单类型审计字段。不能由当前的 1 分钟 K 线或无订单号的成交/委托切片伪造。
3. **P1 - 行业资金流事件**：在具备 point-in-time 行业归属、按方向和订单规模划分的日频资金流后，实现行业大单极端突破；ETF 可将行业映射替换为基金资产类别/跟踪指数主题，但必须单独验证。
4. **P2 - 研究模型**：先用严格 walk-forward Ridge/LightGBM 检验 21-23 篇新增字段的增量，再考虑报告 24 的 LSTM。

报告中的收益、Rank IC 和 Rank ICIR 来自其股票池、历史区间和调仓口径，不能当作当前项目预期。所有以完整日内数据汇总的日频信号均在 **d 日收盘后生成，最早 d+1 可交易**；不得将其回填为 d 日盘中可得。

## 现有实现和数据边界

当前股票/ETF L2 主链是 `FractalQuant/factor/stock_orderbook.py` 与 `scripts/generate_stock_orderbook_factors.py`：快照为主索引，逐笔委托和逐笔成交做 10/30/60/300 秒因果窗口并回填到 1 分钟。现有输出已经覆盖 OFI/MLOFI、五档深度/斜率/韧性、订单和成交方向失衡、VPIN、冲击、成交方向持续性、成交金额分位位置以及盘口条件化异常。

这条主链的原始委托适配器只读 `side`、`price`、`qty` 和时间，不保留订单 ID 或撤单事件类型。因此它可以产生短窗订单流失衡，**不能**重建一笔订单的下单-成交-撤单生命周期。

竞价脚本 `scripts/generate_auction_factors.py` 是例外：它在 09:15-09:20 重建带订单 ID 的新增/撤单事件，并已输出买卖撤单数量比、撤单金额比、撤单失衡、尾段撤单份额以及大单撤单比例。它没有输出全撤/部撤/废单分类，也不覆盖全天的 5 秒与 30 秒撤单寿命分布。

项目还存在 `factor/ml.py` 中的通用滚动机器学习因子和 `torch` 依赖，但没有报告 24 所需的面板 LSTM、PIT 财务拼接、时间切分、模型注册或样本外训练管线；其中逐条时间序列拟合的通用预测类不能视为报告 24 的复现。

## 第 21 篇：市场微观结构变迁

### 原文的核心观察

报告以逐笔委托和逐笔成交说明三类长期变化：

```text
订单执行时长 = 委托发生至最终成交或撤回的时间
平均挂单金额 = mean(order_price * order_qty)
早盘成交集中度 = volume[开盘早段] / volume[全天]
高频撤单占比 = count(cancel_lifetime <= 10s) / count(all_cancels)
```

它还指出早盘成交逐步前移，且 2018 年后“从日内反转中剔除开盘首小时”的旧优化不再稳定。这个结论是一个明确的反例：不能因为旧样本有效就硬编码早盘过滤。

### 与本地因子的重合度

| 报告主题 | 现有近似实现 | 判断 | 建议 |
|---|---|---|---|
| 订单执行速度 | 无生命周期时长 | 无公式重合；当前无订单 ID | 仅在委托生命周期数据合格后做日频监控 |
| 平均挂单金额 | `trade_notional_quantile_position_60s`、订单名义金额失衡 | 都使用金额，但对象分别为成交窗口/方向失衡与单笔委托均值 | 不新增同义短窗因子；作为市场结构监控 |
| 早盘成交集中度/成交重心 | 早盘区间、`early_range_*` 等盘中价格路径字段 | 无成交量重心日频统计 | 新增监控 `early_amount_share_30m`、`early_volume_centroid_60m`，d+1 使用 |
| 高频撤单结构 | 竞价取消字段，且仅竞价时段 | 部分底层数据重合；没有全天生命周期 | 与第 22 篇共用专用撤单生成器 |
| 日内反转的开盘过滤 | 现有日内策略与分钟量价因子 | 是策略条件问题而非新原始特征 | 在回测中用市场结构状态分组，禁止固定剔除首小时 |

建议冻结以下监控定义，先跨股票池、流动性桶和年份画 20 日滚动曲线：

```text
early_amount_share_30m[d] = sum(amount[09:30,10:00)) / sum(amount[all session])
early_volume_centroid_60m[d] = sum(minutes_from_open * volume) / sum(volume)
mean_order_notional[d] = mean(price * original_order_qty)
cancel_fast_share_10s[d] = N(cancel_lifetime <= 10s) / N(cancellations)
```

前两个只能用 1 分钟 OHLCV/成交额做成日后状态；后两个必须使用带订单 ID 的逐笔委托。它们首先是数据质量与模型失效检测指标：若分布突变，触发对相关 alpha 的重新分期和重新训练，而不是把监控值直接当成盘中下单信号。

## 第 22 篇：撤单行为规律

### 原文公式和实现口径

报告给出基础定义：

```text
cancel_rate  = cancelled_qty / free_float_shares
cancel_ratio = cancelled_qty / total_order_qty
```

简单的撤单比例与流通市值、换手和波动高度相关，报告本身也认为其选股稳定性较差。订单必须区分成交、全撤、部撤、废单；不能把所有撤单相加后宣称复现。

两个有明确操作定义的候选为：

```text
# 报告的绝对撤单率口径
sell_type_cancel_rate = sell_type_cancel_qty / free_float_shares

# TRI：09:15-09:20 的卖方撤单率等权合成
tri = mean(
    sell_full_cancel_qty / free_float_shares,
    sell_partial_cancel_qty / free_float_shares,
    sell_negative_cancel_qty / free_float_shares
)

# TOX：全时段撤单生命周期的短寿命占比
tox = N(cancel_lifetime <= 5s) / N(cancel_lifetime <= 30s)
```

这里 `negative_cancel`（废单）的交易所编码和判定口径须从原始字段字典验证；不能擅自把“未完全撤掉”或负向价格变化当作废单。`full` 与 `partial` 也必须以同一订单的原始数量、累计成交量和累计撤单量共同判定。

### 精确重合判断

| 候选 | 当前覆盖 | 重合度 | 结论 |
|---|---|---:|---|
| TRI 的早盘窗口 | `generate_auction_factors.py` 已重建 09:15-09:20 事件 | 中等 | 已有取消数量、买卖侧和原始金额，但当前比率分母是该侧新增委托量，不是报告的自由流通股本；仍缺撤单三分类和最终 TRI |
| TRI 的全撤/部撤/废单 | 当前竞价事件保存 `original_quantity`，但未形成三分类输出 | 低 | 不应以现有 `auction_*cancel*` 名称替代；专用生命周期聚合后新增 |
| TOX 5 秒/30 秒 | 无全时段订单生命期 | 很低 | 新增，但前提是订单 ID 与新增/撤单记录能跨事件关联 |
| 普通撤单率/比例 | 竞价取消数量比和金额比 | 部分 | 不单独加入；先做市值、换手、波动中性残差才有研究意义 |

推荐的新生成器应是日频旁路，例如 `scripts/generate_order_lifecycle_factors.py`，而不是向 60 秒快照因子中掺入依赖日终状态的字段。最小审计输出：

```text
trade_date, ts_code, lifecycle_reconstruction_ok,
orders_added, orders_with_cancel, orders_with_trade, unmatched_cancels,
sell_full_cancel_qty, sell_partial_cancel_qty, sell_negative_cancel_qty,
auction_sell_cancel_tri, cancel_tox_5s_30s, factor_date, available_time
```

`available_time` 应为 d+1 开盘前的日频可用时间。TRI 若专门做为 09:20 后的竞价信号，可以另设当日 09:20 的可用时点，但必须确认交易所 09:20-09:25 的撤单规则、数据到达延迟和实际下单约束；不能把这个例外与全天日频因子混用。

## 第 23 篇：大小单资金流的行业轮动

### 原文方法

报告的重点不是把个股资金流简单平均到行业。它明确比较过多种自下而上聚合，结论是直接行业层面的构造更可靠。三个关键构造是：

```text
# 主动超大单强度的高位规避
K = mean(rank(active_superlarge_strength), past 2 months)
    / mean(rank(active_superlarge_strength), past 12 months)
improved_strength = rank(current_active_superlarge_strength) / K

# 行业羊群效应
herding = RankCorr(delta(industry_return[t]),
                   delta(non_active_small_net_flow[t+1]))

# 大单净流入的行业事件
extreme[d] = +1 if net_flow[d] > mean_120d + 0.5 * std_120d
           = -1 if net_flow[d] < mean_120d - 0.5 * std_120d
           =  0 otherwise
large_flow_breakout_N = sum(extreme[d-N+1:d])
```

报告使用行业主动超大单和羊群效应的月频组合，也使用 `N=10` 的大单极端突破做周频行业轮动。上式中的 `t+1` 是因子构造的时间配对，绝不能理解为在 t 日已知道 t+1 的小单流；调仓时只能使用配对已完整发生的历史观察。

### 与本地项目的重合度

| 报告主题 | 本地近似实现 | 重合度 | 推荐动作 |
|---|---|---:|---|
| 分钟/短窗买卖失衡 | `trade_*_imbalance_60s`、`order_flow_imbalance_60s`、OFI/MLOFI | 原始流层面高 | 不重复注册同义分钟失衡 |
| 成交金额位置 | `trade_notional_quantile_position_60s` | 部分 | 它不是大小单日频资金流，更不是行业时间序列事件 |
| 市场/资产主动流上下文 | `market_active_notional_imbalance_1m`、`asset_minus_market_active_flow_1m` | 部分 | 可作为资金流方向与市场对照，但没有大小单桶、行业 PIT 聚合或事件标准化 |
| 大/小单残差和散户羊群 | 研究建议中列为 `moneyflow.parquet` 候选 | 概念相邻 | 未确认进入生产生成器；先验证输入数据、方向、规模阈值及 PIT 行业归属 |
| 行业大单极端突破 | 无 | 低 | 作为本批最值得新增的上层日频因子 |

建议先做最小、可审计的行业事件版本，而不照搬整套多因子行业轮动：

```text
industry_large_net_flow[d] =
  sum(active_buy_notional_large - active_sell_notional_large)
  / sum(active_buy_notional_large + active_sell_notional_large)

industry_large_flow_breakout_10[d] =
  sum(significant_extreme(industry_large_net_flow), previous 10 completed days)
```

规模分桶必须使用交易日内或预先固定的、可复现的名义金额分位阈值，输出阈值和覆盖率。行业归属和指数成分必须按 d 日可得版本处理；行业变更与停牌不可用最终版分类回填。

对 ETF，不应把股票行业因子的历史结论直接外推。可采用两条独立路线：

1. 先在 ETF 自身按跟踪资产类别、指数主题或基金类型建立横截面，计算基金自身的资金流事件。
2. 再以 ETF 当日可得的底层行业敞口映射行业事件，但必须使用 point-in-time 持仓/指数权重，并将披露滞后纳入可用时点。

两条路线都需分流动性分层报告；低成交 ETF 的方向与大单分类很容易被单笔交易主导。

## 第 24 篇：深度学习赋能交易行为因子

### 报告方案

报告的输入是日频量价、分钟频量价和大小单资金流。LSTM 对时序变量编码后，才把较慢变化的财务指标以分位形式拼接到输出层前的隐藏表征：

```text
sequence input = daily price/volume + minute price/volume + sized money flow
static input   = point-in-time financial percentile features
score          = MLP(LSTM(sequence input), static input)
label          = future 20-trading-day return
loss           = -cross_sectional_IC(score, label)
```

其月频示例使用 4 个月回看、6 年滚动训练窗、训练/验证 9:1、年末更新、两层 LSTM、自注意力和早停。多输出版本在损失中加入候选输出之间的相关性惩罚；报告自身指出该版本训练成本更高且不比单输出 `LSTM_pro` 明显更优。

### 本地判断

| 项目能力 | 是否已具备 | 判断 |
|---|---|---|
| 分钟与订单簿特征 | 是 | 可作为未来时序输入，但必须先统一成 d 日末可用面板 |
| `torch` 依赖 | 是 | 仅表示依赖可用，不代表已有训练系统 |
| 通用 `factor/ml.py` 模型 | 部分 | 是单序列滚动预测/异常类代码，缺横截面 IC 损失、面板标签和严格训练切分，不能复用为 LSTM_pro |
| PIT 财务拼接 | 未形成这条模型管线 | 财务字段必须以 `ann_date`/实际披露时点对齐，不能以报告期末 `end_date` 回填 |
| Walk-forward、产物版本、特征快照 | 未确认 | 是进入神经网络研发前的硬门槛 |

因此本篇不新增普通因子。应先完成以下基线，且在完全独立的测试窗证明显著增量，才启动 LSTM：

1. 冻结 21-23 篇及现有订单簿字段的版本、可用时间和缺失处理。
2. 以月度或周度截面为单位，使用 expanding/rolling Ridge 和 LightGBM，标签为未来 5/20 日收益。
3. 按训练、验证、最终测试做时间切分；调参窗口绝不接触最终测试期。
4. 报告相对线性基线的 Rank IC、ICIR、行业/市值中性 IC、换手、1/2/5 bps 单边成本表现、覆盖率和特征消融。
5. 仅在该基线稳定后，使用记录随机种子、输入 schema 哈希、训练区间、权重文件和预测日版本的 LSTM 实验。

ETF 不能直接拼接股票财务输入，所以报告 24 的 `LSTM_pro` 不适用于 ETF 的原样复现。ETF 版本若要研究，应以基金规模、跟踪误差、折溢价、流动性、指数/资产类别特征替代财务分支，并与仅分钟/订单簿的 Ridge 基线比较。

## 实施顺序与验收标准

| 优先级 | 工作项 | 数据依赖 | 交付物 | 必须通过的验收 |
|---|---|---|---|---|
| P0 | 市场结构监控 | 1min OHLCV/amount；可选订单生命周期 | 日频监控 parquet 和分布报告 | 分时段完整性、20 日滚动、按年份/流动性桶分组，日后可用 |
| P1 | TRI / TOX | 含 ID 的逐笔委托、成交、撤单编码 | 独立日频生命周期生成器和审计列 | 无未匹配撤单才产出；全撤/部撤/废单可回查；d+1 可用 |
| P1 | 行业大单极端突破 | 方向正确的大小单流、PIT 行业归属 | 行业日频事件面板 | 阈值、分桶、覆盖、行业映射版本都落盘；与简单聚合基准比较 |
| P2 | ETF 对应验证 | ETF 分钟/逐笔、PIT 分类或底层权重 | 独立 ETF 横截面研究 | 按流动性和资产类别分层，不套用股票方向 |
| P3 | LSTM 模型 | 完整 PIT 面板和训练治理 | 可复现实验包 | 相对 Ridge/LightGBM 的独立窗增量和成本后稳健性 |

任何新增日频因子至少报告：原值与行业/市值中性值、1/5/20 日前瞻收益、按流动性和市值桶的 IC、滚动 IC、缺失率、日间自相关、与已选特征相关性，以及 1/2/5 bps 单边成本下的非重叠组合路径。对于依赖 L2 的因子，还必须报告原始订单/成交行数、订单 ID 匹配率、方向字段验证结果和每日日志错误数。

## 来源与核验定位

来源均为 `D:\研报\市场微观结构研究系列`：

- 第 21 篇 pp. 9-18：订单执行时长、10 秒内撤单、平均挂单金额、早盘成交前移及日内反转结构变化。
- 第 22 篇 pp. 9-11：09:15-09:20 的三类卖方撤单率 TRI，以及 `TOX = 5 秒内撤单数 / 30 秒内撤单数`。
- 第 23 篇 pp. 5-12：2/12 月趋势比、120 日均值加减 0.5 倍标准差的极端流、行业羊群效应和 10 日大单极端突破。
- 第 24 篇 pp. 5-8、12-13：三类 LSTM 输入、6 年滚动训练、未来 20 日标签、低相关多输出损失和理想反转改进。

已对第 21 篇 p.9、第 22 篇 p.9、第 23 篇 p.6 和第 24 篇 p.5 渲染页做视觉核验；文本提取仅用于定位其余公式和上下文。
