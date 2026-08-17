# Alpha掘金系列第18-24篇：ETF映射、自动因子挖掘和另类数据评估

## 1. 批次范围和总览

本报告覆盖 `D:\研报\alpha掘金系列` 第 18-24 篇，共 7 篇、147 页。全目录缺少第 1、2、17 篇 PDF；本批从第18篇开始，不补写缺失编号结论。报告的历史回测结果均只属于原文，不能作为本地股票或 ETF 的预期收益。

| 篇号 | 标题（简写） | 页数 | 核心 | 项目判断 |
|---|---|---:|---|---|
| 18 | TimeMixer 改进选股因子到 ETF 轮动 | 17 | TSGRU、个股-指数-ETF 两步映射 | P1 ETF 研究框架；先做 PIT 映射和线性基线 |
| 19 | Mamba2 端到端选股 | 19 | 多频时序模型与模型合成 | P3 模型研发，不是新因子 |
| 20 | 热门概念板块 AI 预测和概念龙头 | 16 | 个股聚合概念、FCF2EV 龙头 | P3 概念 PIT 数据和组合方法 |
| 21 | LLM 全天候财务逻辑因子挖掘 | 17 | MMR/RAG/表达式校验 | P0 因子治理，P2 离线搜索 |
| 22 | GFlowNet 低相关量价因子 | 21 | TB、grammar、低相关搜索 | P1 小规模研究面板，P2 GFlowNet |
| 23 | 论坛散户观点 | 34 | FinBERT2、BERTopic、周频统计 | P3，缺 PIT 论坛输入 |
| 24 | GFlowNet + AlphaEval 分钟框架 | 23 | 分钟文法、MemMap、DPP/RRE | P1 候选面板，P2 搜索/计算架构 |

当前项目的 ETF 分钟、跨市场和订单簿主链提供了量价、订单簿和部分参考指数特征，但没有报告 18 的个股面板模型/指数映射治理、报告 21-24 的自动公式搜索器或可回放论坛文本。完整 d 日日内聚合因子只能在 d+1 使用；不得用 1 分钟 bar 伪造 Level2 或逐笔因子。

### 与项目重合度总表

| 主题 | 重合度 | 当前处理 |
|---|---:|---|
| ETF/指数分钟量价与规模字段 | 中 | 可复用数据链，但必须独立做 PIT 映射和可投池 |
| 订单簿/L2 与分钟 grammar | 中 | 并行特征组，不能降级或伪造数据等级 |
| 自动公式搜索、DPP、AlphaEval | 低 | 仅有通用选择能力，先做小 grammar 基线 |
| 股票财务/概念/论坛另类数据 | 低 | 缺完整 PIT 生产输入，暂不直接加入 |

## 2. 第18篇：从个股选股到 ETF 轮动

### 原文流程

原文 pp. 6-15 将 TimeMixer 的多尺度、季节/趋势分解接入 GRU，得到 TSGRU；再把 TSGRU hidden state 与传统因子输入 LightGBM。ETF 轮动采用两步：

```text
stock scores at t
  -> aggregate by point-in-time index constituents/weights
  -> index scores at t
  -> choose ETF tracking the selected index
  -> among same-index ETFs choose rolling-20d largest AUM ETF
```

原文选择滚动 20 日规模最大的 ETF 并动态调整可投池。这是本地最直接可借鉴的 ETF 研究框架，但不是可直接复制的因子公式。

### 本地实施边界

| 环节 | 必须的 PIT 输入 | 不可接受的替代 |
|---|---|---|
| 个股到指数 | d 日已知的成分股、权重、停牌状态、个股信号 `available_time` | 用最终成分股/最终权重回填历史 |
| 指数到 ETF | 跟踪指数映射、生效区间、基金成立/清盘状态 | 仅按名称模糊匹配或今日基金列表 |
| 可投 ETF | 同日规模、成交额、折溢价、上市状态 | 用以后披露的规模或静态最大基金 |
| 交易 | ETF 的真实分钟价格、最早可用时点与成本 | 用收盘得分同时以同日收盘成交 |

P1 的最小版本不用 TSGRU：先将已有个股因子或可靠的指数成分因子，以 d 日 PIT 权重聚合成 `index_factor[d]`，次日再择 ETF；与 ETF 自身分钟/订单簿因子并列建模。每次聚合须落盘 `constituent_version`、权重覆盖、ETF 映射版本、AUM 日期和选择原因。股票方向、指数方向与 ETF 方向必须各自验证。

## 3. 第19篇：Mamba2 多频端到端模型

原文比较 10/30/60 分钟和日频 OHLCV 的 GRU/Mamba2；日频优势更明显，模型/频率间横截面信号相关性约 70%-80%，可作为合成与分散的依据。三大宽基指数点位作为全市场附加输入有增益，时间信息与 Barra 风格输入未展示稳定增量。

本地可借鉴的不是 Mamba2 类名，而是实验设计：同标签、同可用时间、同成本、同股票池下比较频率和模型，并报告预测间相关性和消融。现有 `factor/ml.py` 不是面板 Mamba2 管线；ETF 也不应把股票的 Barra/财务分支照搬。P3 前置条件是已有 walk-forward Ridge/LightGBM 基线、版本化训练数据、权重/随机种子登记和最终独立测试期。

## 4. 第20篇：概念轮动和 FCF2EV 龙头

原文 pp. 8-14 将个股 alpha 按热门概念指数成分聚合，再轮动概念；为了降低持仓数，在入选概念内按自由现金流率选择龙头。其候选为：

```text
concept_score[d] = weighted_mean(stock_alpha[d], PIT concept members/weights)
FCF2EV[d]        = free_cash_flow[d] / enterprise_value[d]
```

该过程最主要的风险是概念定义和成分变化的回填，且 FCF2EV 需要公告可得的现金流、债务、市值/企业价值对齐。当前项目并无确认的 PIT 热门概念成分生产链，ETF 的主题/概念暴露也有持仓披露滞后。因此 P3 前不实现；先使用明确的指数/资产类别映射替代概念映射，避免名称标签造成的未来函数。

## 5. 第21篇：LLM 因子治理

原文 pp. 5-10 的实质贡献是治理而非报告示例公式。改进 MMR 同时控制已有因子和 Barra 风险因子的截面/时序相似度：

```text
MMR(fi) = lambda * IC(fi)
  - (1-lambda) * max[fj in selected union risk](
      alpha * Rel_cs(fi,fj) + (1-alpha) * Rel_ts(fi,fj))
```

原文将挖掘限制在 2010-2019，2020-2025 才用于样本外入库；并要求量纲一致、截面标准化、表达式 AST/类型校验以及基本面截止日期对齐。这些可直接转化为 P0 因子 manifest：表达式、字段单位、窗口、复杂度、训练截止、`factor_date`、`available_time`、横截面/时序相关和风险暴露。

LLM/RAG 仅可在 P2 离线提出受限 grammar 内候选，必须由确定性执行器复算。特别是基本面不能以报告期末 `end_date` 回填，需要公告或实际披露时间。ETF 不具个股财务横截面，不能原样复现。

## 6. 第22篇：GFlowNet 低相关量价因子

原文 pp. 7-14 使用 op/window/feature 的表达式树和 Trajectory Balance：

```text
L_TB = [log(Z * product P_F / (R(x) * product P_B))]^2
reward = abs(market-cap-neutral IC)
```

日频 grammar 包含一元、时序一元、二元、时序二元和截面算子；分钟数据先降为约 40 个日内统计，再进行日频搜索。项目已有许多相邻字段，例如短窗 Amihud、VWAP、冲击、价格速度、量价背离和 L2 失衡，因此先建立 6-10 个明确的 1 分钟日内统计，而不是引入整套搜索器：

```text
intraday_trend_slope, intraday_trend_rsquare, close_position,
ret_first30, ret_last30, realized_skew, realized_kurt,
volume_concentration, vwap_close_dev, max_drawdown_intraday
```

这些均用完整日内 1 分钟路径计算并标为 d+1。对股票和 ETF 分别做横截面、流动性桶和已有因子正交性检验；日内停牌、涨停、半日和缺失分钟不能静默填补。

## 7. 第23篇：论坛情感和主题

原文 pp. 10-24 以 `T-7` 至 `T-1` 的帖子构造周频数量、正负占比、加速度和 30/90/180 日动量：

```text
weekly_pos_ratio = weekly_positive / weekly_total
pos_acceleration = weekly_positive[t] - weekly_positive[t-7]
pos_momentum_90  = mean_7d(weekly_positive) / mean_90d(weekly_positive) - 1
```

另一条路径是 `Fin-Retriever embedding -> UMAP -> HDBSCAN -> c-TF-IDF -> LLM topic name`，再统计基本面、技术、情绪等主题占比。原文显示观点方向（看多/看空）在中证500不显著，是不可忽略的反例；中小盘结果也不应外推到大盘或 ETF。

当前没有已确认的 PIT 论坛原文、发布时间、标的映射、去重/删帖版本和模型版本，故为 P3 数据项目。未来产物必须带 `source_id/asof_time/model_version/symbol_mapping_version`，并比较文本因子相对纯发帖关注度的增量；股票文本向 ETF 映射还需要披露滞后的 PIT 权重。

## 8. 第24篇：分钟 GFlowNet 与 AlphaEval

### 原文文法和计算架构

原文 pp. 13-20 在分钟层先运算和 mask，再日内 reduce 为日频，最后应用日频算子：

```text
MinExpr   -> MinuteOp -> optional MaskOp -> ReduceOp -> DailyExpr
```

分钟字段为 OHLCV/amount、`ret/vwap/hl_pct/bar_pos/amihud/rv/signed_vol/signed_amt` 和累计 `vwap_cum/twap/obv/pvt`。Mask 覆盖开盘/尾盘、极端分钟、分位/IQR 区域及条件区间；Reduce 覆盖统计量、偏度/峰度、斜率/R-square/极值时点、协方差/相关等。

其奖励组合训练 IC、多头 IR 与 Barra 时序相关性惩罚：

```text
reward = abs(train_ic)
 * (1 + lambda * clip(train_long_ir, 0, cap))
 * (1 - mu * clip(barra_ts_corr, 0, 1))
```

AlphaEval 以预测能力、排序稳定性 RRE、扰动鲁棒性、金融逻辑和多样性筛选。原文实验中 DPP/log-det 去冗余有价值，RRE 主要降换手，而其他评分未必提高收益。

### 本地路线

P1 先实现少量可解释 mask-aggregation：`last30_return`、`last30_volume_share`、`return_when_volume_above_median_mean`、`amihud_at_extreme_return_mean`。每个候选应声明 session、最小分钟数、空 mask 语义、树复杂度和 d+1 可用时间。

P2 顺序应是小 grammar 随机/穷举基线 -> walk-forward 单因子/模型增量 -> 相关性贪心筛选 -> DPP 原型 -> GFlowNet。原文的 MemMap `(year, channel) -> (days, 241, stocks)`、block cache 和多进程设计仅在候选规模被证明确有瓶颈后采用；不可在无基线时先引入重型计算设施。

## 9. 优先级和验收

| 优先级 | 工作项 | 必需数据/产物 | 验收 |
|---|---|---|---|
| P0 | 因子和映射 manifest | PIT 成分/权重/ETF 映射、`available_time`、表达式树 | 无最终版回填；每一行可追溯到版本 |
| P1 | ETF 两步映射线性基线 | 股票/指数/ETF 独立面板与规模日期 | d+1 交易、跟踪关系和规模筛选可复算 |
| P1 | 1 分钟日内统计与低相关筛选 | 完整 session OHLCV/amount | 相对已有技术/订单簿组在独立窗有增量 |
| P2 | DPP/GFlowNet/LLM 离线候选 | 受限 grammar、PIT 训练集 | 相对随机/贪心基线稳定；复杂度和资源可控 |
| P3 | Mamba2、概念文本、论坛舆情 | 版本化面板/文本/概念数据 | 覆盖、延迟、分层、成本后和最终测试均通过 |

所有候选必须报告 1/5/20 日收益的 IC/RankIC、滚动稳定性、缺失率、日间自相关、横截面与时序相关、行业/市值/资产类别中性结果、换手和 1/2/5 bps 单边成本非重叠组合。ETF 额外按 AUM、成交额、资产类别和跟踪指数分层；股票训练结果不构成 ETF 验收。

## 来源与核验

- 第18篇 pp. 6-15：TimeMixer/TSGRU、个股-指数-ETF 两步映射与滚动20日规模筛选。
- 第19篇 pp. 6-16：Mamba2 对 GRU 的多频比较、宽基指数信息和模型合成。
- 第20篇 pp. 8-14：概念聚合、轮动和 FCF2EV 龙头筛选。
- 第21篇 pp. 5-13：改进 MMR、训练/样本外边界、量纲/表达式校验。
- 第22篇 pp. 7-14：TB、表达式 grammar、相对量价与分钟聚合特征。
- 第23篇 pp. 10-24：周频情感、BERTopic+LLM、主题和观点方向反例。
- 第24篇 pp. 7-20：多目标奖励、分钟 grammar、MemMap/block cache、DPP 与 AlphaEval。

已视觉核验第18篇 p. 13、第21篇 p. 5、第22篇 p. 8、第23篇 p. 11 和第24篇 p. 14；其余内容以同一 PDF 的逐页可提取文本定位和复核。
