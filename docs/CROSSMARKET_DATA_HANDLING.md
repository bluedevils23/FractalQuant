# CrossMarket 数据处理约定

- 输入为 ETF 与参考指数分钟 K 线。计算保留缺失交易分钟（NaN），输出仅保留两者原始时间戳交集。
- 窗口按交易日重置；大陆市场午休 11:31–13:00 不计入交易分钟。输入已有的时间戳仍保留。
- 缺失分钟不前后填充；缺失价格两侧不能拼接成一分钟收益。
- 日内代理价格、同组累计收益在各自有效基准开盘价出现之前为 NaN。
- 因子使用当前分钟完整数据，只能在该分钟结束后发布使用。

## 前一交易日状态

`generate_etf_daily_states.py` 支持日线 MultiIndex 输入，使用 ETF 自身未筛选的分钟交易日历确定上一交易日；仅取该日的日线字段，不跨缺档回退。

状态文件包含 `ts_code`（六位基金代码）、`trade_date`、`prev_nav`、`prev_total_size`、`prev_total_share` 和 `source_trade_date`。加载时兼容带交易所后缀的代码，并用当前 ETF 分钟日历重新校验来源日期。

旧状态文件缺少 `source_trade_date` 时会明确报错，需要重新生成：

```powershell
uv run python scripts/generate_etf_daily_states.py --etf-root D:\workspace\stockdata\etf-data\etf_1min
```

若只生成部分基金状态，应将输出写入独立文件，并使用主生成器 `--etf-daily-state` 指向它。状态文件不存在时，主生成器继续从日线文件按 ETF 分钟日历对齐。

## 重生成

主生成器 `--overwrite` 更新指定日期范围内的基础列，保留范围外历史，以及共同输出行已有的补充因子列。指定范围内已无 ETF/指数交集的旧行会移除。未指定日期范围表示全量重建；全量无交集时移除旧文件。

基准残差补充脚本 `--overwrite` 只更新四个目标列，在指定日期范围内清空不再有共同输入分钟的旧值。默认模式仅填充目标列空值。

修改计算逻辑不会自动更新旧 parquet。受影响日期应先重生成基础因子，再用补充脚本 `--overwrite` 重算派生列；保留下来的补充列本身不会随基础因子更新而自动重算。OFI 脚本默认仍关闭。
