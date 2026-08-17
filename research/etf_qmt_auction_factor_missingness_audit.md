# QMT ETF Auction 因子缺失审计

## 范围与结论

审计目录为 `D:\workspace\stockdata\etf-factors\etf_auction_factors_qmt_0930_match`。截至 2026-08-11，目录包含 729 个 ETF 文件、16,003 个 `ts_code/trade_date` 唯一行、114 列，日期范围为 2026-07-10 至 2026-08-10（22 个交易日）。全部文件 schema 一致，未发现标的内重复日期。

QMT tick 的竞价路径来自 `[09:15, 09:25)` 快照；QMT 分钟线的 `09:30:00 open/vol/amount` 按约定回填为 09:25 撮合。8,237 行（51.47%）有有效 QMT 撮合，均标记为 `auction_match_source=qmt_0930_minute`，其价格、量、额与对应 QMT 分钟线逐行一致。其余 7,766 行（48.53%）没有有效 QMT 09:30 bar，开盘撮合字段保持缺失。

最重要的限制不是数值缺失，而是数据能力边界：QMT tick 没有逐笔委托/撤单。故 `auction_event_reconstruction_ok=False` 覆盖全部 16,003 行，全部事件因子和提交量参与率均为 `NaN`。沿用要求该标记为真的既有 ETF auction 回测，会筛掉所有 QMT 行；QMT 版本必须使用明确的“快照/撮合子模型”而非假装事件数据已重构。

## 数据质量

| 检查项 | 结果 | 处理 |
|---|---:|---|
| 文件数 / 行数 | 729 / 16,003 | 无空文件。|
| 标的内重复 `trade_date` | 0 | 通过。|
| 有有效 QMT 09:30 撮合 | 8,237 (51.47%) | 开盘价、量、额可用，来源为 `qmt_0930_minute`。|
| 无有效 QMT 09:30 撮合 | 7,766 (48.53%) | 不填充，不标记为撮合。|
| 有效撮合行的 `available_time` | 8,237/8,237 为 09:25:00 | 符合本次约定的回填口径。|
| 事件重构成功 | 0/16,003 | QMT 不提供订单/撤单源，属于预期限制。|
| 事件因子非缺失 | 0 | 未生成伪造事件值。|

`available_time` 在没有撮合的行也为名义 09:25；因此任何下游筛选必须同时要求 `auction_has_match=True`，不能只以时间字段判断可交易性。

## 因子缺失

下表的“撮合行缺失率”只按 8,237 个 `auction_has_match=True` 行计算。它更能反映 QMT tick 的路径质量；总体缺失率还会混入 7,766 个无撮合日。

| 因子组 | 总体缺失率 | 撮合行缺失率 | 原因与处理 |
|---|---:|---:|---|
| `auction_overnight_return`、开盘价/量/额、`auction_gap_excess_benchmark`、`auction_open_to_prev_high` | 48.5%-48.9% | 0.0%（有足够日线历史时） | 仅由无 QMT 09:30 撮合造成；不得以日线全天量额补充。|
| 阶段二端点/范围/中点路径：`auction_stage2_range_bps`、`auction_return_stage2`、`auction_stage2_mid_*`、`auction_range_ratio` | 29.7%-31.1% | 0.4%-2.4% | QMT 有撮合时阶段二快照几乎完整；无撮合日约 60% 缺少有效路径。保持 `NaN`。|
| 阶段一端点：`auction_return_stage1`、`auction_imbalance_change_stage1`、`auction_stage1_range_ratio` | 35.0%-36.1% | 9.0%-10.1% | 阶段一快照不足；不插值。|
| `auction_stage2_slope_bps_per_min` | 32.1% | 3.8% | 少于三个有限阶段二价格点。|
| `auction_last60s_price_return` | 43.7% | 20.7% | 最后 60 秒少于两个有效快照，适合只在高覆盖路径子样本评估。|
| `auction_final_vs_stage2_twap`、阶段二 TWAP 因子 | 38.4%-55.0% | 12.7% | 时间覆盖率门槛未达到，不能降低定义门槛来填补。|
| 5 日撮合历史：`auction_amount_ratio_5d`、`auction_matched_volume_ratio_5d` | 68.9% | 39.5% | 22 日窗口内有效撮合日不足 5 天；属于自然 warm-up 与间断撮合。|
| 20 日撮合 z-score：`auction_amount_zscore_20d` | 100.0% | 100.0% | 当前 QMT tick 只有 22 个交易日，且每标的有效撮合不足 20 日；不能使用。|
| 分钟 ADV：`auction_amount_to_prev5d_adv_240`、`auction_amount_to_prev20d_adv` | 48.7% / 49.4% | 0.3% / 1.7% | QMT 分钟历史足够时可正常计算；总体缺失仅由无撮合造成。|
| 订单/撤单因子及 `auction_matched_volume_to_submitted_ratio` | 100.0% | 100.0% | 没有 QMT 委托、撤单和提交量，永久排除出 QMT 快照模型。|
| `auction_amount_to_float_mcap_prevclose` 与四个官方涨跌停字段 | 100.0% | 100.0% | ETF 日线无流通市值与官方涨跌停字段；不是 QMT tick 缺陷。|
| 前日/市场日线因子 | 0.0%-2.6% | 0.0%-2.1% | 仅样本起点或 20 日横截面历史不足；可保留。|

路径覆盖的质量指标也支持上述判断：`auction_snapshot_count_total` 从不缺失，均值为 28.48、中位数为 13；4,637 行（29.0%）为零快照，其中 4,604 行发生在无撮合日。有效撮合行仅 33 行（0.4%）为零快照。

## 回测与建模处理

1. QMT 模型应先要求 `auction_has_match=True`，不能把无撮合日的 09:25 名义时间当成可交易信号。
2. 不能复用要求 `auction_event_reconstruction_ok=True` 的现有回测资格门槛；应新增明确的 `qmt_snapshot_match` 模式，只使用非事件因子，并在训练、验证和测试中固定同一筛选规则。
3. 永久从该模式特征集移除全部事件因子、`auction_matched_volume_to_submitted_ratio`、`auction_amount_zscore_20d`、ETF 流通市值和官方涨跌停字段。
4. 将阶段一/阶段二完整度、`auction_snapshot_count_total` 与 `auction_stage2_twap_coverage_ratio` 保留为缺失机制或质量特征；LightGBM 可接收其余路径因子的自然 `NaN`，不得横截面填充或前向填充。
5. 当前 QMT tick 覆盖仅 22 个交易日，不能做独立的 20 日历史因子评估或稳健样本外回测；后续至少积累足够连续交易日后，再比较 QMT 快照模型与原逐笔事件模型。
