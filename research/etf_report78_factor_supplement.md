# 开源量化评论（78）ETF 因子补充

来源：2023-06-27《微观视角下的 ETF 轮动组合构建》。报告在行业主题 ETF 的跟踪指数层面筛选因子，先对个股因子做正交化，再按近 6 期最大 ICIR 给资金面、技术面、基本面三类加权；因子入选门槛为 RankICIR >= 0.4 或多头超额收益率（相对中证 800）>= 6.5%。报告给出的有效因子共 9 个：

| 类别 | 报告因子 | 本地 ETF 分钟数据状态 |
| --- | --- | --- |
| 基本面 | `Fundamental_TTM_GPOA`、`Fundamental_TTM_ROA` | 暂缓：需要带公告日的财务表，ETF 分钟目录没有该输入 |
| 技术面 | `Technical_IndayVolumeRatioLog`、`Technical_CVILLIQ`、`Technical_QUA`、`Technical_LongMom` | 已实现 ETF 级日频代理 |
| 资金面 | `Flow_Stock_InflowAmtRatio_FS`、`Flow_SynCorr`、`Flow_MTS` | 暂缓：需要北向/主力资金流，不能从 OHLCV 推断 |

## 已实现字段

入口：`scripts/generate_etf_report78_factors.py`；核心：`factor/etf_report78.py`。

- `technical_inday_volume_ratio_log`：`log1p(下行分钟收益平方和 / 全部分钟收益平方和)`。
- `technical_cvilliq`：分钟 Amihud 非流动性 `abs(return) / amount` 的变异系数。
- `technical_qua`：分钟成交额的中位数百分位排名。原报告的 QUA 是单笔成交金额分位数；本地 ETF 文件为分钟聚合，因此明确标记为代理，不冒充逐笔实现。
- `technical_long_mom`：日收盘价的 20 个已完成交易日动量。

每个 `trade_date=d` 的值只使用 d 日及之前的数据，输出 `available_date=d+1`、`available_time=d+1 09:30`。不存在下一交易日时 `available_date` 为空。金额字段缺失时，两个金额依赖字段保留为空，不用 `close * volume` 静默替代。

## 与现有分钟因子的关系

这些字段写入独立目录 `D:\workspace\stockdata\etf-data\etf_report78_factors`，不修改现有 `etf_1min_factors` 的 55 列 base 或 159 列 multi 输出契约。加入生产组合前仍需在当前 ETF/指数样本上做截面 RankIC、行业/市值中性、正交残差和样本外检验；报告中的 2014-2023 Wind 结果不能直接当作本地数据验证。

报告中的“优选 ETF 指数组合”还需要 ETF-跟踪指数映射、行业暴露和 20 日成交额筛选。当前补充只提供原子因子，不自动生成 Star-ETF 组合，避免把缺失的指数成分或资金流数据隐式补零。

