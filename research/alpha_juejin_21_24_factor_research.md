# Alpha掘金系列第21-24篇：因子与研究框架评估

## 范围和结论

本次核阅 `D:\研报\alpha掘金系列` 的第 21 至 24 篇。报告中的历史回测口径、股票池、交易成本和样本期均不等同于本项目，不能作为当前股票或 ETF 的预期收益；以下建议只定义待验证的候选与研发门槛。

| 篇号 | 发布日期 | 标题 | 页数 | 与项目的主要关系 | 结论 |
|---|---:|---|---:|---|---|
| 21 | 2025-12-27 | 基于 LLM 的全天候财务逻辑因子挖掘框架 | 17 | 因子治理、表达式搜索 | 借鉴 PIT、相关性和表达式校验；不直接注册报告公式 |
| 22 | 2026-04-10 | 基于 GFlowNet 的低相关性量价因子挖掘策略 | 21 | 日频/分钟聚合特征搜索 | 小规模原型的 P1 研发项，不是单一新因子 |
| 23 | 2026-06-21 | 论坛散户观点有价值吗？--散户舆情选股投资手册 | 34 | 另类文本因子 | 无可审计 PIT 论坛输入，暂不进入生产 |
| 24 | 2026-07-08 | 基于 GFlowNet 和 AlphaEval 的分钟频因子挖掘筛选框架 | 23 | 1 分钟特征搜索、因子筛选与计算架构 | 先实现小候选集和 walk-forward 审计；MemMap/GFlowNet 为 P2 |

项目现有主链已经覆盖大量手工量价、分钟和 L2 订单簿特征：`FractalQuant/factor/microstructure.py`、`FractalQuant/factor/orderbook.py`、`FractalQuant/factor/stock_orderbook.py` 和 `scripts/generate_stock_orderbook_factors.py` 包含 OHLCV/VWAP、流动性、成交冲击、OFI/MLOFI、盘口深度/斜率、订单/成交失衡、VPIN、短窗冲击与异常片段等。因此第 21、22、24 篇的核心增量是**候选生成、去冗余和审计流程**，不是把既有字段换一个名称重复注册。

所有用完整 d 日日内数据聚合得到的日频因子，最早在 **d+1** 可交易；不能回填为 d 日盘中可得。第 24 篇的“分钟频”输入是 1 分钟 OHLCV/amount 及其因果日内计算，不能用来替代依赖逐笔或 Level2 的订单簿因子，也不能反过来将 L2 因子用 1 分钟 bar 伪造。

## 第 21 篇：LLM 全天候财务逻辑因子挖掘

### 原文方法

原文 pp. 5-10 将 LLM 的作用限定为受控表达式生成与改进，而非直接预测收益。其改进后的最大边际相关性（MMR）同时惩罚横截面、时序及风险因子相似性：

```text
MMR(fi) = lambda * IC(fi)
          - (1 - lambda) * max[fj in Ss union Sm](
              alpha * Rel_cs(fi, fj) + (1 - alpha) * Rel_ts(fi, fj)
            )
```

`Ss` 是已选因子集，`Sm` 包含 Barra 风险因子；`Rel_cs` 与 `Rel_ts` 分别是截面和时序相似度。原文还明确将挖掘/筛选限制在 2010-2019 年，将 2020-2025 年作为入库前的样本外验证。这是防止“全样本生成后再挑选”泄露的重要约束。

框架包括双层循环（槽位内生成-优化-初筛、槽位间收益和相关性控制）、成熟因子库 RAG 启发、反思 idea 池、截面 `rank`/`zscore`/归一化算子、量纲一致性约束，以及针对基本面字段的 AST/类型校验和可用日期对齐。pp. 11-13 展示的量价式只应视为候选示例，例如：

```text
EMA(Slope(close, 5) * Cov(close, volume, 5) / Var(close, 5)
    * Slope(volume, 5), 5)

(close - Max(high, 5)) / (Max(high, 5) - Min(low, 5))
    * EMA(volume, 5)
```

### 与本地能力的重合和缺口

| 原文模块 | 本地近似能力 | 重合度 | 判断 |
|---|---|---:|---|
| 常规量价算子 | `price.py`、`trend.py`、`volatility.py`、`microstructure.py` | 高 | 基础字段与多数经济直觉已覆盖，避免直接重复公式 |
| 因子选择与相关性 | `factor/selector.py`、`factor/analysis.py` | 部分 | 有选择/分析工具，未确认同时约束截面、时序、风险暴露的入库治理 |
| LLM/RAG/反思搜索 | 无专用生产管线 | 低 | 仅能作为离线研究工具；候选须由确定性表达式执行器复算 |
| 基本面 PIT 对齐 | 日频因子生成器可用，但无该表达式治理链 | 低 | 不得按报告期末 `end_date` 回填；必须以公告/实际可得时间对齐 |
| ETF 版本 | ETF 无个股财务报表横截面 | 很低 | 不原样迁移；仅可研究 ETF 自身规模、折溢价、跟踪误差、流动性和指数特征 |

### 建议

P0 先固化因子入库审计，而不是接入 LLM：候选表达式、字段 schema、窗口、可用时间、训练/验证/测试边界、方向、缺失率、复杂度和去重结果都落盘。对任何基本面候选，增加字段单位和最早 `available_time` 检查；`price - return` 这类量纲错误应拒绝执行。

P2 才考虑离线 LLM 生成。LLM 只能提出受限 grammar 内的公式，表达式须经过 AST 解析、类型检查、等价式归一化和确定性重算；RAG 不得携带最终测试期的因子表现或测试结论。

## 第 22 篇：GFlowNet 低相关量价因子挖掘

### 原文方法与可借鉴部分

原文 pp. 7-9 使用表达式树作为终止状态，以 Trajectory Balance（TB）训练 GFlowNet：

```text
L_TB = [log(
  Z * product_t P_F(s_t | s_(t-1))
  / (R(x) * product_t P_B(s_(t-1) | s_t))
)]^2
```

动作由算子、窗口、叶子特征组成；只有语法合法且未超过复杂度上限的动作可采样。原文的日频 grammar 覆盖一元、时序一元、二元、时序二元和截面算子，窗口为 5/10/20/40/60。奖励使用**市值中性化后的** `abs(IC)`，并在样本外再按与已选因子的 Spearman 相关性小于 0.4 筛选。其要点是 GFlowNet 生成一组按奖励加权、而非收敛到单一公式的候选；不应将报告中“相关性较低”的统计结果移植为本地事实。

分钟部分（pp. 12-14）先把 1 分钟数据压缩为约 40 个日频日内统计，再复用日频搜索。代表性输入包括：日内 Amihud、收盘区间位置、开盘/尾盘收益、日内趋势斜率与拟合度、收益自相关、实现偏度/峰度、成交量集中度、收盘相对 VWAP 偏离、日内回撤和上下行波动比。这些是基于完整日内路径的**日频候选特征**，不是盘中实时信号。

### 重合度与最小候选集

| 候选日内统计 | 现有近似项 | 重合度 | 推荐动作 |
|---|---|---:|---|
| 日内 Amihud / VWAP 偏离 | `amihud_illiquidity_5m`、价格/成交量因子 | 部分 | 对 1 分钟全日聚合另做日频字段，d+1 使用，不能与 5 分钟 L2 窗口同义替代 |
| 日内趋势 / R-square / 收益自相关 | `PriceVelocityFactor`、`MarketEfficiencyFactor` 等 | 部分 | 先按统一 1 分钟 session 定义做基线，评估正交增量 |
| 开盘/尾盘强度、时间位置、量能集中度 | 现有分钟与竞价/日内路径字段存在邻近项 | 部分 | 不直接注册数十个同类字段；以 6-10 个明确公式建立研究面板 |
| Kyle 冲击、买卖冲击不对称 | `market_impact_60s`、OFI/MLOFI、短窗失衡 | 部分到高 | 先做相关性和特征消融，避免与 L2 流因子混淆 |
| GFlowNet 搜索器 | 无 | 低 | 研究基础设施，P2；先用确定性小 grammar 做穷举/随机基线 |

建议的 P1 研究面板仅包含下列日后可用字段，先证明其相对现有 1 分钟技术因子的增量：

```text
intraday_trend_slope[d]       = slope(log(close_1m), completed session d)
intraday_trend_rsquare[d]     = R2(log(close_1m), completed session d)
close_position[d]             = (close[d] - low[d]) / (high[d] - low[d])
ret_first30[d], ret_last30[d] = completed interval returns
realized_skew[d], realized_kurt[d]
volume_concentration[d]       = sum((vol_1m / sum(vol_1m))^2)
vwap_close_dev[d]             = close[d] / VWAP[d] - 1
max_drawdown_intraday[d]
```

零分母、涨停停牌、半日交易和缺失分钟必须显式标记；不能用跨午休滚动填补。该面板对股票和 ETF 都可计算，但必须分别做横截面与流动性分层验证。

## 第 23 篇：论坛散户观点与主题因子

### 原文因子

原文 pp. 10-11 使用 FinBERT2 将主帖分为正/负面，并在 T 日仅汇总 `T-7` 至 `T-1` 的帖子，构造周频因子：

```text
weekly_positive      = N(positive posts, [T-7, T-1])
weekly_pos_ratio     = weekly_positive / weekly_total
pos_acceleration     = weekly_positive - lag_7d(weekly_positive)
pos_change_ratio     = weekly_positive / lag_7d(weekly_positive) - 1
pos_momentum_90      = mean_7d(weekly_positive) / mean_90d(weekly_positive) - 1
```

同样的数量、占比、差分/比率和 30/90/180 日动量适用于总帖、负帖和情感占比。原文在中证 1000 中发现高正面热度之后存在较强的周度反转，但沪深 300 有效因子更少，说明这一方向高度依赖股票池及论坛覆盖。

pp. 16-24 的第二条路径是 `Fin-Retriever embedding -> UMAP -> HDBSCAN -> c-TF-IDF -> LLM 命名` 的 BERTopic 主题流程。主题被归为基本面经营、技术分析、行业政策、情绪观点、非投资及未分类，并以主题周度占比和其 30/90/180 日动量构成因子。原文的关键反例是：对技术/基本面帖子再分类“看多/看空”后，在中证 500 中不显著；因此不能将方向标签误当成稳健 alpha。

### 本地适用性

| 条件 | 当前判断 | 影响 |
|---|---|---|
| 论坛原始文本、标的映射、发布时间、删除/转发状态 | 未确认有可重放的 PIT 生产数据 | 不能实现或回测该因子 |
| 金融情感/主题模型 | 有通用另类数据思路，未确认 FinBERT2 与主题模型生产链 | 不能把模型名称当作特征来源 |
| 论坛覆盖率和停牌/改名处理 | 未建立审计产物 | 高频发帖标的可能只是覆盖偏差 |
| ETF 映射 | 股票讨论不能无条件映射至 ETF | 需独立 ETF 论坛数据，或以披露滞后的 PIT 成分权重聚合 |

因此本篇为 P3 数据项目，而不是当前可新增因子。若未来具备合规、可回放的文本输入，最小输出应保存 `asof_time`、`source_id`、原帖/转发标记、`model_version`、`symbol_mapping_version`、类别概率和未分类比例；周频因子应在 T 日开盘前冻结，并以周初后未来一周收益评估。先按大/中/小市值、行业、覆盖率分层并与纯关注度比较，才判断情绪标签或主题是否有增量。

## 第 24 篇：分钟 GFlowNet 与 AlphaEval

### 原文的分钟文法和计算设计

原文 pp. 13-16 给出两阶段文法：分钟层先做分钟算子和日内 mask，再做日内聚合变成日频值，最后才允许日频算子。其结构为：

```text
MinExpr   = MinuteFeature | MinuteOp(MinExpr[, MinExpr or window])
MaskExpr  = MaskOp(MinExpr[, MinExpr])
BlockExpr = ReduceOp(MinExpr | MaskExpr[, MinExpr])
DailyExpr = BlockExpr | DailyOp(DailyExpr[, DailyExpr or window])
```

分钟叶子包括 `open/high/low/close/vol/amount`、`ret/vwap/hl_pct/bar_pos/amihud/rv/signed_vol/signed_amt` 和日内累计 `vwap_cum/twap/obv/pvt`。关键新增并非某一个数学算子，而是**掩码**：开盘/尾盘/中段，极值位置，上下分位/IQR 区域，或“仅当另一序列为正/极端时”的条件区间。日内归约包括均值、波动、偏度/峰度、斜率/R-square/极值时点、相关/协方差和加权均值。

pp. 7-8 的奖励函数同时考虑训练 IC、多头 IR 和 Barra 时序风险暴露：

```text
reward = abs(train_ic)
       * (1 + lambda * clip(train_long_ir, 0, cap))
       * (1 - mu * clip(barra_ts_corr, 0, 1))
```

候选还须通过训练/测试 IC 同号、训练与测试 `abs(IC)`、训练/测试多头 IR、Barra 时序相关性及与已选因子收益序列相关性的门槛。AlphaEval 再讨论预测能力（PPS）、排序稳定性（RRE）、扰动鲁棒性、金融逻辑和多样性；原文实验中 DPP/log-det 多样性筛选有助于降低共线性，RRE 主要降低换手，其他筛选未必提高收益。

### 与项目的重合和实施判断

| 模块 | 当前项目 | 结论 |
|---|---|---|
| 1 分钟量价特征 | `generate_etf_minute_factors.py` 等已有生成链 | 可作为候选输入，但需统一交易时段和因果聚合定义 |
| L2 短窗订单簿因子 | `stock_orderbook.py` 已具较完整覆盖 | 与 1 分钟 grammar 是并行特征组；不可混作同一数据等级 |
| 复杂分钟缓存 | 无按 `(year, channel) -> (days, 241, stocks)` 的 MemMap/block cache | 若研究规模扩大，再评估；不应先建重型基础设施 |
| DPP/低相关选择 | 有通用选择工具，未确认 DPP 和收益序列多样性流程 | 可先独立实现研究脚本中的贪心相关性基线 |
| AlphaEval/LLM 逻辑评分 | 无 | 逻辑评分只能做人工审查辅助，不能替代样本外证据 |

P1 的最小路线是：冻结第 22 篇的小面板，再只加入 4 类带经济解释的 mask-aggregation 候选，例如 `last30_return`、`last30_volume_share`、`return_when_volume_above_median_mean`、`amihud_at_extreme_return_mean`。表达式必须声明完整 session、掩码空集行为、最小分钟数和 d+1 可用时间。

P2 才做可复现的自动搜索，顺序应为：小 grammar 的随机/穷举基线 -> 严格 walk-forward 筛选 -> 低相关贪心选择 -> DPP 原型 -> GFlowNet。应限制表达式深度/节点数并对交换律、双重负号、恒等式做规范化，否则搜索结果会被等价公式填满。对 ETF，使用 ETF 自身 1 分钟数据建立独立候选池；不要把股票全 A 横截面训练出的公式当成 ETF 已验证结果。

## 实施优先级与验收

| 优先级 | 工作项 | 数据/因果要求 | 交付物 | 验收标准 |
|---|---|---|---|---|
| P0 | 因子候选治理与 PIT 审计 | 公式 AST、字段单位、`available_time`、时间切分 | 候选 manifest 和可复算表达式 | 测试期未参与生成/选型；所有日频字段标为 d+1 |
| P1 | 1 分钟日内统计小面板 | 完整 session 的 OHLCV/amount；午休、缺失、半日交易审计 | 股票与 ETF 分开的日频 parquet、覆盖/缺失报告 | 相对现有分钟技术因子在独立窗有正交增量 |
| P1 | 低相关筛选基线 | 训练期特征、未来收益标签、风险暴露 | walk-forward 研究脚本与入选清单 | 报告横截面/时序相关、秩稳定性、换手和特征消融 |
| P2 | DPP/GFlowNet 与 MemMap 原型 | 已稳定的小 grammar 和足够分钟面板 | 可复现实验包 | 相对随机/贪心基线在独立窗稳定增量，复杂度和资源受控 |
| P3 | 论坛情感与主题 | 合规、PIT、可回放的文本和标的映射 | 带版本/时间戳的另类数据面板 | 覆盖率、去重、延迟、分层及相对纯关注度的增量均通过 |

每个候选至少在最终未参与调参的区间报告：1/5/20 日前瞻 RankIC 与 ICIR、滚动 IC、缺失率、日间自相关、与已选因子的截面/时序相关、行业/市值中性结果、流动性分层，以及 1/2/5 bps 单边成本的非重叠组合路径。ETF 另报资产类别、规模和成交额分层；股票结果不能作为 ETF 的替代验证。

## 来源与核验定位

- 第 21 篇 pp. 5-10：MMR 的截面/时序/Barra 扩展、2010-2019 训练与样本外流程、量纲和表达式校验；pp. 11-13：候选量价/基本面表达式。
- 第 22 篇 pp. 7-9：TB 目标、表达式树和奖励；pp. 12-15：分钟聚合特征和日频搜索结果。
- 第 23 篇 pp. 10-11：周频情感统计与 `T-7` 至 `T-1` 口径；pp. 16-24：BERTopic+LLM 主题流程、主题统计和观点方向无效的反例。
- 第 24 篇 pp. 7-8：日频 grammar、奖励和样本筛选门槛；pp. 13-20：分钟 grammar、MemMap/block cache、DPP 与 AlphaEval。

第 21 篇 p. 5、第 22 篇 p. 8、第 23 篇 p. 11 和第 24 篇 p. 14 已进行 PDF 页面渲染视觉核验；其余内容以同一 PDF 的可提取文本用于公式定位和上下文核对。
