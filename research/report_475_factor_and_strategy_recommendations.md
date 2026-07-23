# 475 份量化资料的因子与策略建议

研究日期：2026-07-21

## 1. 研究口径与覆盖

原始目录：`D:\BaiduNetdiskDownload\量化因子挖掘思路475份`

- 475 个文件中有 473 份 PDF、1 份 Markdown 和 1 张 PNG。
- PDF 合计 45,828 页，抽取文本约 6,911 万字符。
- 433 份 PDF 可直接抽取正文；4 份文件为字节级完全重复。
- 39 份 PDF 是纯扫描版，1 份 PDF 为部分可读，1 份 PDF 为零页损坏文件。
- 按用户确认，未开始 OCR 的扫描版可跳过。本报告没有把扫描版标题当成公式证据。
- 逐文件状态、重复映射、可读性、筛选结论和候选映射见 `research/report_475_review_ledger.csv`。

本次“适合加入项目”采用以下门槛：

1. 可以定义成因果、point-in-time 的原子特征，或能明确放入策略、组合、执行层。
2. 与当前注册表不是公式等价的重复项。
3. 当前本地数据可以实现，或能清楚说明额外数据依赖。
4. 不把样本内阈值、模型输出或持仓规则冒充成通用原子因子。

## 2. 当前项目基线

| 因子层 | 当前规模 | 本报告如何处理重合项 |
|---|---:|---|
| ETF 分钟普通因子 | base 52，multi 156 | 普通动量、波动率、技术指标不重复增加 |
| 高级非线性因子 | 46 | Hurst、熵、递归、DTW 等不重复增加 |
| 股票订单簿因子 | base 63，multi 135 | OFI、MLOFI、深度、韧性、VPIN、冲击已有 |
| 集合竞价因子 | 51 | 两阶段收益、撤单、路径、基准超额已有 |
| 日内策略因子 | P0 21，P0+P1 34 | 开盘路径、GFTD、ORB、SMT、早盘区间已有 |
| CICC ETF 日级暴露 | 58 | 收益分布、量价相关、时段成交等优先扩展到股票，不重写 ETF 公式 |
| FZ ETF 日级暴露 | 34 | 日内形态、潮汐和成交博弈类先做语义去重 |

当前本地数据足以支持三条新路线：股票/ETF 分钟 OHLCV 与金额、带 `ann_date/end_date` 的财务表、大小单资金流和带 `report_date` 的分析师报告。L2 生成器虽然已存在，但原始逐笔路径默认是 `E:\逐笔数据`，所以新增 L2 因子必须视原始数据是否在线而定。

## 3. 第一优先级：当前数据可直接实现

### 3.1 分钟数据聚合为日级/短周期选股因子

| 建议字段 | 定义摘要 | 与当前因子的差异 | 来源 |
|---|---|---|---|
| `pv_corr_avg_20d` | 每日 `corr(close_1m, vol_1m)`，再取 20 日均值 | CICC ETF 有相近字段；股票侧缺少同口径日级暴露 | 东吴《高频价量相关性》pp.6-7 |
| `pv_corr_std_20d` | 上述每日相关系数的 20 日标准差 | 当前没有“价量关系稳定性”维度 | 同上 pp.7-8 |
| `pv_corr_trend_20d` | 对 20 个每日价量相关系数按时间回归，取斜率 | 当前相关性因子没有跨日趋势 | 同上 p.12 |
| `uid_information_uniformity_20d` | `std(daily_intraday_vol_20d) / mean(daily_intraday_vol_20d)` | 当前波动率因子不表达信息冲击在日期间的均匀度 | 东吴《信息分布均匀度》p.5 |
| `volume_volatility_cv_20d` | 先算每日分钟成交量标准差，再算 20 日标准差/均值 | 当前 `volume_clustering` 是滚动分钟特征，统计对象不同 | 长江《高频波动中的时间序列信息》p.5 |
| `volume_diff_abs_mean_20d` | 每日 `mean(abs(diff(volume_1m))/mean(volume_1m))`，再取 20 日均值 | 提取成交量局部跳变，当前没有等价日级字段 | 同上 pp.11-12 |
| `volume_weighted_close_ratio` | 成交量加权平均分钟收盘价 / 等权平均分钟收盘价 | 当前 VWAP 字段是滚动价格偏离，不是全天高位成交密度 | 长江《高位成交因子》pp.6-7 |
| `volume_weighted_price_skew` | 以分钟成交量为权重计算日内价格三阶中心矩/标准差三次方 | 当前 `volatility_skew` 统计收益，不统计成交量加权价格分布 | 同上 p.8 |
| `tail_amount_share_20m_ewma15d` | 尾盘 20 分钟成交额/全天成交额，再做 15 日 EWMA | CICC ETF 有收盘成交量类字段；股票与“成交额”口径仍缺 | 华安《成交额蕴藏的 Alpha》pp.6, 9-11 |
| `structured_reversal_volume_q10_10m` | 低成交量 10% 区间用成交量倒数加权动量，高成交量区间用成交量加权反转，两者作差 | 当前普通动量、量价确认不能表达按成交活跃度切分的双状态 | 长江《结构化反转因子》pp.8, 16-17 |
| `path_illiquidity_5m` | `sum(log(1+abs(r_5m))) / sum(amount_5m)` | 当前 `liquidity_ratio` 是均量/收益波动率，不是完整价格轨迹冲击 | 长江《高频因子和交易行为》pp.4-5 |
| `benford_volume_deviation_5000` | 最近 5,000 个非零分钟量首位数频率与 Benford 分布的平方距离和 | 当前没有数字分布/机构痕迹状态 | 方正《本福特的启示》p.4 |

实现建议：前三项及收益分布类因子已有 CICC ETF 参考实现，应抽出股票/ETF 共用的“分钟到日级暴露”计算层，避免同一公式维护两份。

### 3.2 收益分布因子的股票侧补齐

| 建议字段 | 公式 | 来源 |
|---|---|---|
| `realized_return_skew_20d_1m` | 每日分钟收益三阶矩除以已实现方差的 `3/2` 次方，再取 20 日均值 | 海通《股票收益分布特征》pp.5-6 |
| `realized_return_kurtosis_20d_1m` | 每日分钟收益四阶矩除以已实现方差平方，再取 20 日均值 | 同上 pp.5-6 |
| `upside_variance_share_20d_1m` | `sum(r^2 * I[r>0]) / sum(r^2)`，再取 20 日均值 | 海通《已实现波动分解》p.7 |

这些字段与当前滚动 20/40/60 根分钟 bar 的偏度、峰度不是同一信息集。ETF CICC 层已有接近字段，因此优先补股票侧，并统一使用复权价格、完整交易日和 `d+1` 可用时点。

### 3.3 本地日频、财务、分析师和资金流因子

| 建议字段 | 计算口径 | 可用本地表 | 来源 |
|---|---|---|---|
| `earnings_acceleration_eaa` | `growth_t-growth_t-1`，其中 `growth=(EPS_t-EPS_t-4)/abs(EPS_t-4)` | `fina_indicator.parquet` | 海通《盈利加速》p.4 |
| `earnings_acceleration_eap` | 同上，但增长分母使用上季末股价 | 财务表 + 分钟收盘聚合 | 同上 p.4 |
| `earnings_acceleration_eav` | 同上，但增长分母使用最近 8 季 EPS 标准差 | `fina_indicator.parquet` | 同上 p.4 |
| `asset_growth_volatility_8q` | 最近 8 季资产同比增速标准差；总资产、流动资产分别计算 | `balancesheet.parquet` | 海通《资产增长稳定性》pp.6-7 |
| `capital_structure_change` | 股东权益率、资产负债率、长期负债率的同比或环比变化 | `balancesheet.parquet` | 同上 pp.9-10 |
| `delevered_ep` | 息前经营利润 / `(股权市值 + 金融负债 - 金融资产)` | 资产负债表、利润表、股价 | 国盛《对价值因子的思考和改进》pp.7-9 |
| `delevered_sp` | 主营收入 / 经营性净资产市值 | 同上 | 同上 pp.7-9 |
| `delevered_cfp` | 经营现金流 / 经营性净资产市值 | 资产负债表、现金流量表、股价 | 同上 pp.7-9 |
| `intangible_adjusted_book_to_market` | 用永续盘存法资本化研发/销售管理支出，调整净资产并扣除商誉，再除以市值 | `income.parquet`、`balancesheet.parquet` | 国盛《无形资产估值因子》pp.9-11；海通《不可忽视的无形资产》 |
| `effective_tax_rate_volatility_8q` | 最近 8 季 `income_tax/total_profit` 的稳健波动率 | `income.parquet` | 国盛《刻画财报信息质量》p.14 |
| `nondepreciating_asset_share` | 在建工程、无形资产、研发支出、商誉等占总资产比例 | `balancesheet.parquet` | 同上 pp.12-13 |
| `analyst_eps_revision_20d` | 同一标的、同一预测年度的 EPS/净利润预测相对上次报告的修正幅度和上调占比 | `report_rc_daily.parquet` | 中信《分析师预期调整事件》；国盛《盈利修正后的股价漂移》 |
| `analyst_target_price_revision_20d` | 目标价相对前次报告的修正幅度、覆盖机构数和分歧度 | `report_rc_daily.parquet` | 中信《券商金股与分析师因子再增强》 |
| `earnings_surprise_percent` | `(单季度实际净利润-单季度预期净利润)/abs(预期净利润)` | 财务表 + `report_rc_daily.parquet` | 中信《分析师超预期因子》p.4 |
| `moneyflow_npr_5d_{sm,md,lg,elg}` | `sum(buy-sell)/sum(buy+sell)`，按四类订单分别计算 | `moneyflow.parquet` | 逐鹿《聪明的资金流向数据》p.12 |
| `large_order_flow_residual_20d` | 大/中单净流入强度对同期收益回归后的残差 | `moneyflow.parquet` | 开源《大小单资金流 Alpha 2.0》pp.4-7 |
| `retail_flow_herding_20d` | 对历史配对 `(return[q-1], small_flow[q])` 计算 20 日滚动秩相关，`q<=d` | `moneyflow.parquet` | 同上 p.4 |
| `semibeta_{N,P,MN,MP}_{20,60,120}` | 按个股/基准收益正负区间拆分的四类半贝塔，回看 20/60/120 日 | 分钟股票与指数日线可直接聚合；作为风险/截面因子验证 | 广发《SemiBeta 因子研究》p.9 |

财务和分析师因子必须按实际发布日期对齐：`ann_date`、`report_date` 当日收盘后才算可知，保守实现从下一交易日开盘可用。不能按 `end_date` 回填历史。

## 4. 第二优先级：有公式，但依赖原始 L2/逐笔数据

| 建议字段 | 公式或含义 | 当前重合与判断 | 来源 |
|---|---|---|---|
| `positive_buy_cautious_buy_ratio_20d` | 过去 20 日保守买入量之和/积极买入量之和；成交价相对上一快照买一/卖一判断 | 当前成交方向失衡没有“等待成交/跨价成交”两类买入行为拆分 | 长江《基于买入行为构建情绪因子》p.7 |
| `mci_bid`、`mci_ask` | 五档对手盘 VWAP 相对中价的成本，再除以五档报单金额 | 当前 `orderbook_liquidity_l5` 不等价；值得消融 | 中信《买卖报单流动性》pp.3-4 |
| `soir_l5_decay` | 各档 `(bid_qty-ask_qty)/(bid_qty+ask_qty)` 按近端衰减加权 | 与 `weighted_depth_imbalance_l5` 高相关但公式不同，先做增量检验 | 中信《高频订单失衡及价差》p.4 |
| `mpc_1m_mean_max_skew`、`mpc_5m_mean_max_skew` | 中间价 1/5 分钟变化率的均值、最大值、偏度 | 当前只有 mid price/velocity，缺日级分布统计 | 同上 pp.5-6 |
| `trade_notional_quantile_position_20d` | 去除每日最大 10 笔后，计算 `(A10%-Amin)/(Amax-Amin)`，再取 20 日均值 | 当前成交规模因子未输出该日级分位结构 | 开源《分钟单笔金额序列》pp.7-9 |
| `trade_notional_q90_q10_ratio_20d` | 单笔金额 90%/10% 分位比，是上述 QUA 因子的解释性拆分 | 与主 QUA 高相关，仅在残差增量通过时保留 | 同上 p.8 |
| `institutional_active_flow_absr` | 将主动方订单按单号汇总，筛选订单总额大于 100 万，算 `(buyex-sellex)/(buyex+sellex)` | 需要买卖订单编号；当前输入适配器只保留 side/price/qty | 国联《机构主动资金流》p.7 |
| `price_band_trade_count_share` | 高/低价格分位区间内成交笔数占全天比例 | 当前无日内价格分位条件统计 | 东兴《行为追踪因子》pp.7-8 |
| `price_band_mean_trade_size` | 高/低价格分位区间平均每笔成交量相对全天均值 | 同上 | 同上 p.8 |
| `net_turnover_rate` | `(主动买入量-主动卖出量)/流通股本` | 需要 point-in-time 流通股本 | 海通《净换手率》p.5 |
| `active_execution_degree_by_size` | 买/卖订单按大中小分层后的主动成交度 | 当前成交流因子未按订单原始委托量分层 | 海通《主动成交中的隐藏信息》 |

原始逐笔数据如果不在线，不应从已有分钟 K 线伪造这些字段。尤其 `institutional_active_flow_absr` 需要订单编号，不能用简单成交方向替代。

## 5. 不应重复注册，但值得用于策略层

| 已有或近似已有信息 | 建议策略用途 |
|---|---|
| `vpin_50bucket`、订单流异常、流动性冲击 | 作为下单门控：毒性或冲击过高时降低参与率、延迟追单或禁止市价单 |
| 竞价两阶段、撤单、虚假压力、基准超额 | 做开盘信号确认，不再增加同义竞价因子 |
| 开盘平稳度、GFTD、ORB、延迟极值、早盘区间 | 组成日内状态机或 T+0 触发器，不直接混进所有时刻的通用选股模型 |
| CICC 日内偏度、上下行波动、量价相关、头尾成交 | 先扩展股票侧，再比较新增字段的正交 IC；不要在 ETF 主生成器重复写一套 |
| OFI/MLOFI、深度失衡、盘口斜率 | 作为 L2 执行/短周期预测特征；SOIR/MCI 仅在增量检验通过后加入 |

## 6. 交易与组合策略建议

### 6.1 短周期股票选股

建立独立的“分钟聚合日级因子”面板，信号在 `d` 日收盘后生成，`d+1` 开盘或 VWAP 成交。第一批组合建议：

```text
score = mean_rank(
    -pv_corr_avg_20d,
    -pv_corr_trend_20d,
    -uid_information_uniformity_20d,
    -tail_amount_share_20m_ewma15d,
    -structured_reversal_volume_q10_10m,
    -path_illiquidity_5m
)
```

符号不能直接照搬旧研报；应在当前 2020-2026 数据上重新验证。先做 1/5/20 日前瞻收益、行业市值中性和 1/2/5 bps 单边成本，再决定周频还是日频换仓。

### 6.2 高频负面 Alpha 剔除

多篇海通和长江报告都显示高频因子空头端通常比多头端稳定。对 long-only 模型，优先采用：

1. 用高频合成分数识别最差 5%-10% 标的。
2. 在主模型完成全市场打分后，只对订单候选集做剔除或惩罚。
3. 不改变主模型的全市场评分口径，避免把“空头过滤”误做成新的训练标签。

来源包括海通《如何利用高频因子的空头效应》《剔除高频多因子空头组合》以及长江《负面 Alpha 研究（三）》。

### 6.3 因子半衰期与动态权重

因子权重不宜固定等权。建议用以下三类状态生成权重，但必须保留稳定的长期基准权重：

```text
decay_weight_f = exp(-age / half_life_f)
dispersion_f   = mean(top_quantile exposure) - mean(bottom_quantile exposure)
crowding_f     = composite(turnover, valuation_spread, short_term_reversal, ownership_concentration)
```

动态权重可采用 `base_weight * decay_weight * zscore(dispersion) * crowding_penalty`，并设置单因子权重上下限。来源：中信《因子衰减在多因子选股中的应用》、国盛《因子择时的三个标尺》。

### 6.4 基本面事件增强

把 `earnings_surprise_percent`、分析师修正幅度、评级上调事件作为事件门控，和低估值、盈利质量、反转状态交互。报告证据更支持“事件 + 因子”的条件策略，而不是孤立使用一个二元事件。

推荐顺序：

```text
财报/研报事件到达
  -> 使用 ann_date/report_date 建立可用时点
  -> 计算 surprise/revision 连续分数
  -> 质量、估值、过去收益做条件过滤
  -> 下一交易日执行，持有 5/20/60 日做事件衰减比较
```

### 6.5 微观结构执行门控

现有 `vpin_50bucket`、`adverse_selection_markout_30s`、流动性冲击、盘口韧性已经足以先做执行层试验：

- `VPIN` 或流动性冲击处于历史高分位时降低下单参与率。
- 买入时要求卖盘 MCI、价差和预期冲击低于上限。
- 盘口异常段不取消选股信号，只推迟或拆分订单。
- 回测同时记录信号收益、成交损失和收益保留比例，避免把执行改进误算成 Alpha。

来源：招商 VPIN/VWPIN 系列、中信买卖报单流动性、海通《高频策略交易成本的分析和预测》。

### 6.6 ETF 与 CTA 条件策略

- ETF 折溢价情绪：需要可靠的分钟 IOPV/净值或跟踪指数映射；当前只有 ETF OHLCV 时不应伪造折溢价。数据补齐后可测试开盘折溢价、收盘相对日内均值和折溢价上行成交额占比。
- `RSJ + ROC`：用收盘前 1 小时的好/坏已实现波动率差做情绪状态，再以 ROC 确认方向。适合指数/股指期货择时，不宜直接迁移为股票截面因子。
- 北向资金 CTA：当前本地目录未确认分钟北向净买入历史，列为数据补齐后的策略，不纳入当前可实现清单。

## 7. 推荐工程拆分

### P0：分钟聚合日级因子

新增独立模块和生成器，不修改现有 52/156 因子的输出契约：

```text
FractalQuant/factor/minute_daily_research.py
scripts/generate_stock_minute_daily_research_factors.py
scripts/generate_etf_minute_daily_research_factors.py
```

首批只实现 12 个低依赖字段：价量相关均值/波动/趋势、UID、成交量波动/差分、高位成交两项、尾盘成交额、结构化反转、路径非流动性、Benford 偏离。

### P1：point-in-time 基本面与分析师层

```text
FractalQuant/factor/stock_fundamental_research.py
scripts/generate_stock_fundamental_research_factors.py
```

首批实现盈利加速 3 项、资产增长稳定性、资本结构变化、去杠杆价值 3 项、无形资产调整估值、分析师修正/超预期和 4 类 NPR。所有字段必须携带 `available_date`。

### P2：策略与验证

1. 分别验证单因子原值、行业市值中性值和对当前因子正交后的残差值。
2. 统一检验 1/5/20 日收益、换手、容量、停牌/涨跌停可交易性和交易成本。
3. 只把通过样本外增量检验的字段加入生产注册表。
4. L2 因子先在原始逐笔数据在线时做消融，不把缺订单编号的数据强行近似成 ABSR。

## 8. 暂缓项

- 扫描版报告：按用户要求不继续 OCR，台账中标记 `skipped_unreadable_per_user`。
- 纯模型输出：AlphaNet、RNN、TFT、GAN、遗传规划和 AutoML 输出属于模型层，不登记为原子因子。
- 外部文本/持仓/北向数据：新闻情绪、完整研报正文、基金持仓和分钟北向资金目前没有确认 point-in-time 数据源。
- 过度复杂的 PIN/EKOP 极大似然：已有 VPIN 和成交流特征，先做增量比较；没有显著提升则不增加维护成本。
- 传统技术指标大全和 Alpha101 全量复制：当前多窗口技术因子覆盖较广，应按正交增量逐个引入，而不是整库复制。
