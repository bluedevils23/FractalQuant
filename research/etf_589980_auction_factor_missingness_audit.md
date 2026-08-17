# 589980.SH ETF auction 因子缺失审计

## 结论

样本为 `589980.SH`，输出文件为
`D:\workspace\stockdata\etf-factors\etf_auction_factors\589980.SH.parquet`。
截至 2026-08-10，文件有 284 个不重复交易日（2025-05-12 至 2026-07-27）和完整的
112 列 schema，其中 76 列为 auction 因子。

当前 ETF 回测在进入模型前要求 `auction_has_match=True`、
`auction_event_reconstruction_ok=True` 且 `available_time <= 09:25:59`。本样本只有
78/284 日（27.5%）通过；204 日因开盘撮合未识别而被排除，另有 5 日事件重构失败。
因此，首先应解决的是开盘撮合识别覆盖，而不是对路径因子做数值填补。

原始数据抽查证明两类缺失应区别处理：

| 交易日 | 原始路径 | 有效虚拟价快照 | 09:25 成交 | 判断 |
| --- | --- | ---: | ---: | --- |
| 2025-05-12 | `589980.SZ`（文件名后缀与输出代码不同） | 27 | 3 笔有效成交 | 行情快照未给出可识别的开盘成交，但成交文件可望用于重建；须先逐所验证订单/成交语义。 |
| 2025-05-14 | `589980.SZ` | 3（阶段一、二均为 0） | 0 | 没有足够竞价簿路径，不能补造路径因子。 |
| 2025-06-10 | `589980.SZ` | 17 | 2 笔有效成交 | 行情快照已识别到开盘撮合，是当前可交易样本。 |

逐笔目录后缀不可靠：虽然文件名为 `589980.SZ`，委托结构按上交所规则可成功重构，且当前
资产代码为 `589980.SH`。后续重建必须以股票池/分钟数据的交易所归属和原始字段结构判定，
不能仅按逐笔目录后缀选择沪深解析器。

## 因子逐项审计

缺失率均按 284 个输出日期统计；括号内的“可交易日缺失率”按 78 个已通过现有回测门槛的
日期统计。同行因子使用同一数据依赖、改善方法和回测处理，但每个字段均在表中列出。

| 因子 | 缺失率 | 直接原因 | 可改善性 | 回测处理 |
| --- | ---: | --- | --- | --- |
| `auction_overnight_return` | 71.8% (0.0%) | 无可识别 09:25 开盘撮合。 | 高：验证后使用 09:25 成交重建开盘价/量/额。 | 保持现有撮合门槛；不可用日不入选。 |
| `auction_return_stage1` | 77.1% (30.8%) | 阶段一少于两笔有效虚拟价。 | 仅可换更完整报价源；不能插值。 | NaN 保留；路径模型可另设“阶段一完整”样本。 |
| `auction_return_stage2` | 69.4% (3.8%) | 阶段二少于两笔有效虚拟价。 | 同上。 | NaN 保留。 |
| `auction_amount_ratio_5d` | 73.6% (6.4%) | 无撮合或不足 5 个有效历史撮合日。 | 前者可由成交重建改善；后者为正常 warm-up。 | 前 5 有效历史日不填；训练/验证统一剔除 warm-up。 |
| `auction_imbalance_change_stage1` | 77.1% (30.8%) | 阶段一端点不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_imbalance_change_stage2` | 69.4% (3.8%) | 阶段二端点不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_commitment_shift` | 76.8% (28.2%) | 阶段一或二委比端点不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_stage2_slope_bps_per_min` | 75.0% (23.1%) | 阶段二少于 3 个有限价格点。 | 仅提高快照频率。 | 不以两点斜率替代。 |
| `auction_stage2_range_bps` | 68.3% (0.0%) | 阶段二路径不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_stage2_efficiency_ratio` | 69.4% (3.8%) | 阶段二价格变化不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_matched_volume_ratio_5d` | 73.6% (6.4%) | 无撮合或 5 日 warm-up。 | 同 `auction_amount_ratio_5d`。 | 不向前填充。 |
| `auction_unmatched_imbalance` | 67.3% (0.0%) | 最后有效三档委比不存在。 | 更完整报价源可改善。 | NaN 保留。 |
| `auction_bid_cancel_qty_ratio_stage1` | 1.8% (0.0%) | 5 日事件重构失败。 | 中：检查异常订单号/撤单回连。 | 保持事件重构门槛；不以 0 填充。 |
| `auction_ask_cancel_qty_ratio_stage1` | 1.8% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_cancel_notional_ratio_stage1` | 1.8% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_cancel_imbalance_stage1` | 1.8% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_late_cancel_notional_share` | 1.8% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_large_order_cancel_ratio_stage1` | 8.8% (0.0%) | 事件失败或 20 个有效订单历史日不足。 | warm-up 不应修补。 | 仅在 20 日后使用；不要以总体阈值替代。 |
| `auction_large_cancel_imbalance_stage1` | 8.8% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_stage2_add_imbalance` | 1.8% (0.0%) | 5 日事件重构失败。 | 检查原始订单号。 | 保持 NaN/剔除该事件日。 |
| `auction_stage2_commitment_ratio` | 1.8% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_stage2_last60s_add_share` | 1.8% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_fake_pressure_proxy` | 1.8% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_stage_reversal_strength_bps` | 79.2% (34.6%) | 两阶段收益至少一个缺失。 | 依赖完整报价。 | NaN 保留；不要把“无反转”编码成 0。 |
| `auction_stage2_mid_mean_return` | 68.3% (0.0%) | 阶段二中点路径不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_stage2_mid_max_return` | 68.3% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_stage2_mid_min_return` | 68.3% (0.0%) | 同上。 | 同上。 | 同上。 |
| `auction_stage2_total_variation_bps` | 69.4% (3.8%) | 阶段二少于两点。 | 仅完整报价源。 | NaN 保留。 |
| `auction_stage2_up_step_ratio` | 69.4% (3.8%) | 阶段二少于两点。 | 仅完整报价源。 | NaN 保留。 |
| `auction_stage2_reversal_count` | 69.4% (3.8%) | 阶段二少于两点。 | 仅完整报价源。 | NaN 保留。 |
| `auction_imbalance_relative_change_stage1` | 77.1% (30.8%) | 阶段一端点不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_imbalance_relative_change_stage2` | 69.4% (3.8%) | 阶段二端点不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_imbalance_fisher_change_stage1` | 77.1% (30.8%) | 阶段一端点不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_imbalance_fisher_change_stage2` | 69.4% (3.8%) | 阶段二端点不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_amount_to_prev5d_adv_240` | 71.8% (0.0%) | 无撮合或 5 日分钟 ADV warm-up。 | 可由 09:25 成交重建撮合额改善。 | 不使用全日成交额替代。 |
| `auction_amount_to_prev20d_adv` | 71.8% (0.0%) | 无撮合或 20 日 ADV warm-up。 | 同上。 | 20 日前保持 NaN。 |
| `auction_amount_zscore_20d` | 78.9% (25.6%) | 无撮合或 20 日统计量不足。 | 同上。 | 不用全样本均值/标准差。 |
| `auction_matched_volume_to_submitted_ratio` | 72.5% (0.0%) | 无撮合或事件失败。 | 可由 09:25 成交重建分子。 | 仅在撮合及事件均有效时使用。 |
| `auction_final_vs_stage2_twap` | 88.7% (60.3%) | 阶段二快照覆盖率低于 80%。 | 仅更高频报价；降低门槛会改变定义。 | ETF 基线模型暂不使用，或限定高覆盖子样本。 |
| `auction_l3_imbalance_twap_stage2` | 87.3% (60.3%) | 同上。 | 同上。 | 同上。 |
| `auction_relative_spread_twap_stage2` | 87.3% (60.3%) | 同上。 | 同上。 | 同上。 |
| `prevday_intraday_drawdown_from_session_high` | 0.4% (0.0%) | 首个可用日没有前一日。 | 无需改善。 | 保持 NaN；样本起点自然 warm-up。 |
| `prevday_intraday_rebound_from_session_low` | 0.4% (0.0%) | 同上。 | 无需改善。 | 同上。 |
| `prevday_intraday_return_from_prev_close` | 0.4% (0.0%) | 同上。 | 无需改善。 | 同上。 |
| `prev_2d_return_rank_cs` | 1.1% (0.0%) | 2 日收益的样本起点。 | 无需改善。 | 统一从 warm-up 后训练。 |
| `prev_20d_return_rank_cs` | 7.4% (1.3%) | 连续 20 日历史不足。 | 无需改善。 | 前 20 日不填。 |
| `market_return_from_prev_close` | 18.7% (0.0%) | 基准 `510300.SH` 未识别开盘撮合（53 日）。 | 高：与本基金相同，验证后接入 09:25 成交。 | 现有撮合门槛下已不缺；不要用日线开盘替代。 |
| `market_above_ma20_prevclose` | 0.0% (0.0%) | 基准有长期日线历史。 | 无需改善。 | 可直接使用。 |
| `market_momentum_2d_prevclose` | 0.0% (0.0%) | 同上。 | 无需改善。 | 可直接使用。 |
| `auction_gap_excess_benchmark` | 72.9% (0.0%) | 本基金或基准无撮合。 | 可由双方 09:25 成交重建改善。 | 两者均有效才使用。 |
| `auction_stage2_excess_return_benchmark` | 72.9% (3.8%) | 本基金或基准阶段二路径不足。 | 仅完整报价源。 | NaN 保留。 |
| `auction_range_ratio` | 67.3% (0.0%) | 没有有效虚拟价。 | 更完整报价源。 | NaN 保留。 |
| `auction_stage1_range_ratio` | 75.7% (28.2%) | 阶段一没有有效虚拟价。 | 更完整报价源。 | NaN 保留。 |
| `auction_stage2_range_ratio` | 68.3% (0.0%) | 阶段二没有有效虚拟价。 | 更完整报价源。 | NaN 保留。 |
| `auction_stage1_end_return_from_prev_close` | 75.7% (28.2%) | 阶段一端点不存在。 | 更完整报价源。 | NaN 保留。 |
| `auction_stage2_end_return_from_stage1_end` | 76.8% (28.2%) | 任一阶段端点不存在。 | 更完整报价源。 | NaN 保留。 |
| `auction_up_step_ratio` | 67.3% (0.0%) | 没有有效虚拟价。 | 更完整报价源。 | NaN 保留。 |
| `auction_down_step_ratio` | 67.3% (0.0%) | 没有有效虚拟价。 | 更完整报价源。 | NaN 保留。 |
| `auction_snapshot_count_total` | 0.0% (0.0%) | 定义为计数，零快照记为 0 而非缺失。 | 无需改善。 | 可作为路径质量/缺失机制特征。 |
| `auction_l3_buy_share_final` | 67.3% (0.0%) | 最后有效三档委比不存在。 | 更完整报价源。 | NaN 保留。 |
| `auction_l3_buy_share_stage1_end` | 75.7% (28.2%) | 阶段一端点不存在。 | 更完整报价源。 | NaN 保留。 |
| `auction_l3_buy_share_change_stage2` | 76.8% (28.2%) | 两阶段端点不足。 | 更完整报价源。 | NaN 保留。 |
| `auction_stage1_max_return_from_prev_close` | 75.7% (28.2%) | 阶段一无有效价。 | 更完整报价源。 | NaN 保留。 |
| `auction_stage1_min_return_from_prev_close` | 75.7% (28.2%) | 阶段一无有效价。 | 更完整报价源。 | NaN 保留。 |
| `auction_open_pullback_from_stage1_max` | 79.9% (28.2%) | 无开盘撮合或阶段一无有效价。 | 撮合重建只能解决前者。 | 两个输入都有效才使用。 |
| `auction_open_rebound_from_stage1_min` | 79.9% (28.2%) | 同上。 | 同上。 | 同上。 |
| `auction_last60s_price_return` | 77.1% (32.1%) | 最后 60 秒少于两笔有效价。 | 仅更高频报价。 | NaN 保留。 |
| `auction_final_to_full_max` | 67.3% (0.0%) | 没有有效虚拟价。 | 更完整报价源。 | NaN 保留。 |
| `auction_volume_to_prevday_volume` | 71.8% (0.0%) | 无撮合；首日另缺昨量。 | 撮合重建可改善。 | 不以日内最终量替代。 |
| `auction_amount_to_float_mcap_prevclose` | 100.0% (100.0%) | `etf_daily.parquet` 没有 `circ_mv`。 | 可用权威 ETF 总份额 × NAV/AUM 建独立 ETF 规模口径；不能伪称流通市值。 | 当前 ETF 模型排除。 |
| `auction_open_to_prev_high` | 71.8% (0.0%) | 无撮合；首日另缺昨高。 | 撮合重建可改善。 | 两个输入都有效才使用。 |
| `auction_open_to_prev7d_close_max` | 71.8% (0.0%) | 无撮合或连续 7 日不足。 | 撮合重建只解决前者。 | 前 7 日保持 NaN。 |
| `auction_stage1_touched_limit_up` | 100.0% (100.0%) | ETF 日线没有官方 `up_limit`。 | 需供应商官方限价字段；不得按股票 ±10% 推算。 | 当前 ETF 模型排除。 |
| `auction_stage1_touched_limit_down` | 100.0% (100.0%) | ETF 日线没有官方 `down_limit`。 | 同上。 | 当前 ETF 模型排除。 |
| `auction_stage1_limit_up_distance_bps` | 100.0% (100.0%) | ETF 日线没有官方 `up_limit`。 | 同上。 | 当前 ETF 模型排除。 |
| `auction_stage1_limit_down_distance_bps` | 100.0% (100.0%) | ETF 日线没有官方 `down_limit`。 | 同上。 | 当前 ETF 模型排除。 |

## 回测处理

1. 现行开盘撮合/事件重构门槛是正确的，不能对 204 个未识别开盘撮合日填 0 或前向填充。它们没有可执行的 09:25 信号。LightGBM 训练、验证和测试必须使用同一门槛。
2. 在 78 个可交易日内，保留 LightGBM 原生 NaN；不得对路径、撤单或历史 warm-up 因子做横截面均值、零值或未来值填补。`auction_snapshot_count_total` 可作为缺失机制的显式质量特征。
3. 从 ETF 特征集永久排除 5 个恒缺字段：`auction_amount_to_float_mcap_prevclose`、`auction_stage1_touched_limit_up`、`auction_stage1_touched_limit_down`、`auction_stage1_limit_up_distance_bps`、`auction_stage1_limit_down_distance_bps`。只有取得定义一致的 ETF 规模/官方限价源后才重新纳入。
4. `auction_final_vs_stage2_twap`、`auction_l3_imbalance_twap_stage2`、`auction_relative_spread_twap_stage2` 在可交易日仍有 60.3% 缺失。基线 ETF LGBM 不应依赖它们；应以“完整阶段二路径”子样本单独做增量消融，并与同一交集上的基线比较。
5. 所有 5/20 日因子只在自然 warm-up 后评估；分割日期应位于各因子有效期之后。不要删除缺失日再跨日凑 5/20 个样本。
6. 回测入口当前会直接把 NaN 传给 LightGBM，且在特征读取前先执行撮合/事件门槛过滤。运行时必须显式传入当前目录 `D:\workspace\stockdata\etf-factors\etf_auction_factors`，因为该脚本的旧默认目录仍指向 `etf-data\etf_auction_factors`。

## 改善优先级

1. 验证沪深 09:25 成交的价格、数量、买卖委托号语义，并把开盘成交重建接入 opening-auction 管线；以 `auction_has_match` 覆盖率和与行情快照已识别日期的一致性验收。
2. 保持订单事件的严格重构规则，针对 5 个失败日记录具体订单号/撤单异常；不要以模糊去重修补。
3. 取得更完整的竞价报价快照后，再考虑路径/TWAP 因子；没有该源时不要降覆盖阈值或插值。
4. 为 ETF 单独引入有明确单位和时点的规模数据；官方涨跌停价若不存在，则这些限制价因子应永久不参与 ETF 模型。

## 跨基金 09:30 分钟线核验（2026-08-10）

为验证单基金结论是否具有普遍性，从 `etf_auction_factors` 的 1,422 个输出文件中按文件排序等距抽取 200 只 ETF；每只均存在对应的 `etf_1min` 文件。对每个 `auction_has_match=True` 的交易日，将逐笔报价已识别的开盘撮合字段与同日 `09:30:00` 分钟线比较。样本共包含 26,866 个已识别撮合日，其中 26,739 日两侧均有可比值。

| 比较字段 | 完全一致 | 可比日 | 一致率 | 结论 |
|---|---:|---:|---:|---|
| `auction_open_price` vs 09:30 `open` | 26,650 | 26,739 | 99.67% | 可作为开盘撮合价格的候选补充与交叉校验。|
| `auction_matched_volume` vs 09:30 `vol` | 14,907 | 26,739 | 55.75% | 不能作为纯集合竞价撮合量的替代。|
| `auction_amount` vs 09:30 `amount` | 13,879 | 26,739 | 51.91% | 不能作为纯集合竞价撮合额的替代。|

`09:30` 分钟线显然经常混入 09:30 连续竞价成交。例如 `589980.SH` 的 `2026-01-19`，逐笔识别的撮合量/额为 6,200 / 10,218，而分钟线为 7,800 / 12,840；两侧开盘价仍同为 1.648。价格也不是绝对可靠：本次抽样有 89 个可比日不一致，多个例子集中在 `2025-03-07`，因此接入前仍须按交易所和源数据版本核验。

这项验证只支持把 09:30 的 `open` 用作**经标记的价格补充候选**，且可用时间应保守设为 `09:31`；它不能补齐竞价路径、订单/撤单事件，亦不能填充开盘撮合量额。若策略维持当前 `09:25` 信号、`09:30` 后执行的因果定义，分钟线不能替代原始 09:25 成交重构。

同一批 200 只 ETF 的 57,559 个输出日中，当前 `auction_has_match=False` 的日期有 30,693 个；其中 30,638 个（**99.82%**）存在大于零的 09:30 分钟线 `open`，只有 55 个（0.18%）连这一价格候选也没有。按单基金计算，该覆盖率中位数为 100%。这说明分钟线几乎可以填补“开盘价格字段”的缺失，但不能把这 30,638 日改标为已识别撮合：其价格本身仍有约 0.33% 的已知不一致风险，且分钟线的可用时点、撮合量额、路径和订单事件均不满足现有 09:25 可交易门槛。

## QMT 09:30 分钟线的替代能力（2026-08-10）

对 `etf_1min_qmt` 与因子输出均存在的 200 只 ETF 做同样等距抽样。QMT 文件共覆盖 1,271 只 ETF；本次样本的主覆盖期为 2025-08-11 至 2026-08-10，中位每基金 186 个 09:30 bar。因此它不能补足更早的全量历史，但在其覆盖区间质量明显优于原分钟库。

| 比较字段（仅逐笔已识别撮合日） | 完全一致 | 可比日 | 一致率 |
|---|---:|---:|---:|
| `auction_open_price` vs QMT 09:30 `open` | 18,487 | 18,487 | 100.00% |
| `auction_matched_volume` vs QMT 09:30 `vol` | 18,089 | 18,487 | 97.85% |
| `auction_amount` vs QMT 09:30 `amount` | 14,672 | 18,487 | 79.36% |

本样本 59,509 个输出日中有 33,531 个 `auction_has_match=False`。QMT 有 09:30 bar 的交集为 32,761 日，其中 14,274 日当前无逐笔撮合识别，且**全部**具有正的 QMT 09:30 `open`。故 QMT 可以覆盖全部缺失撮合日的 **42.57%**（14,274 / 33,531），并在 QMT 覆盖期内覆盖 100% 的缺失开盘价格候选。若经逐基金验证后接受 QMT 09:30 bar 为开盘撮合，量的交叉验证支持度很高；成交额仍须以独立字段来源或 `price × volume` 的统一口径进一步验收。

即使接入 QMT，也应记录 `auction_match_source=qmt_0930`，并将其作为与逐笔报价/成交重构不同的来源。它能明显补足开盘价、在大多数日期补足撮合量，但不提供 09:15--09:25 路径与订单事件；若维持严格的 09:25 信号定义，仍需先确认 QMT 09:30 bar 的发布时间与撮合结束时点。
