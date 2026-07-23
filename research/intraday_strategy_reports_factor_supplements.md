# 日内交易策略研报对当前因子的补充建议

研究日期：2026-07-19

## 1. 研究范围与方法

本报告只研究目录：

`D:\BaiduNetdiskDownload\日内交易&T0资料合集\日内交易策略`

目录中共有 16 篇 PDF，合计 317 页。16 篇均完成逐页文本提取；平稳度、资金流、低阶多项式、GFTD、开盘区间、厚尾峰度、遗传算法区间和日内延迟极值等关键公式页另行渲染为 PNG 复核。本文不与 `research/short_cycle_reports_factor_supplements.md` 所研究的另一目录混用。

完整源文件清单：

1. `另类交易策略之二十一：基于市场情绪平稳度的股指期货日内交易策略.pdf`（21 页）
2. `另类交易策略研究之十四：经验模态分解下的日内趋势交易策略.pdf`（21 页）
3. `另类交易策略系列之一：基于混沌理论的股指期货噪声趋势交易策略.pdf`（16 页）
4. `另类交易策略系列之七：在标度不变性破缺下洞察资金流向——MFT交易策略.pdf`（12 页）
5. `另类交易策略系列之三：多项式拟合的股指期货趋势交易(LPTT)策略.pdf`（14 页）
6. `另类交易策略系列之九：基于遗传规划的智能交易策略方法.pdf`（22 页）
7. `另类交易策略系列之二：一类波动收敛突变模式的趋势跟随策略.pdf`（20 页）
8. `另类交易策略系列之八：基于时域分形的相似性匹配日内低频交易策略（SMT）.pdf`（18 页）
9. `另类交易策略系列之六：基于GFTD的期指日内程序化交易策略.pdf`（15 页）
10. `另类交易策略系列之十一：日内突破模式及其资金管理的多重比较研究.pdf`（25 页）
11. `另类交易策略系列之十七：大数据深度学习系列之一：深度学习之股指期货日内交易策略.pdf`（24 页）
12. `另类交易策略系列之十九：带反转的加强版EMDT交易策略.pdf`（21 页）
13. `另类交易策略系列之十二：基于遗传规划多维变量的股指期货交易策略.pdf`（28 页）
14. `另类交易策略系列之十八：厚尾分布下的随机区间突破策略.pdf`（20 页）
15. `另类交易策略系列之十：基于遗传算法的期指日内交易系统.pdf`（19 页）
16. `另类交易策略系列之四：基于日内波动极值的股指期货趋势跟随系统.pdf`（21 页）

本次判断以当前工作树中的注册表为准：

| 生成器或因子族 | 当前规模 | 与本批研报最相关的已有信息 |
|---|---:|---|
| `scripts/generate_auction_factors.py` | 51 个竞价因子 | 两阶段竞价路径、参与度、历史状态、统一基准超额；另有 3 个分钟会话路径字段 |
| `scripts/generate_etf_minute_factors.py` | base 52，multi 156 | 动量、LSMA、ATR、布林带宽、波动率状态、分钟收益峰度、微观结构 |
| `scripts/generate_stock_orderbook_factors.py` | base 63，multi 135 | OFI/MLOFI、盘口深度、韧性、逐标的订单和成交流不平衡 |
| `factor.advanced_runtime` | 46 个默认可训练高级因子 | Hurst、熵、递归、DTW 等；默认排除未来收益标签 |
| CICC/FZ ETF 因子生成器 | 独立日级暴露族 | 日内收益分布、量价形态、尾部/头部成交等 |
| `scripts/generate_intraday_strategy_factors.py` | P0 21，P0+P1 34 | 本报告第一、第二优先级；`p0_p1` 为独立 opt-in profile |

“已有相似名字”不等于覆盖。本文按以下口径分类：

- 已覆盖：当前字段的公式、窗口和信息集能表达研报信息。
- 近似覆盖：当前字段相关，但窗口、横截面、事件状态或统计对象不同。
- 真正缺失：当前注册表无法从现有列直接等价派生。
- 不建议新增：属于模型结构、执行规则、过时样本内系数，或没有独立原子信息。

## 2. 总结

### 2.1 第一优先级：低依赖、高增量

建议先增加以下 6 组原子字段。它们都能用现有分钟 OHLCV 或日线数据实现，不需要 L2 权限或外部基本面。

| 因子组 | 建议字段 | 当前覆盖 | 最早可用时间 |
|---|---|---|---|
| 开盘路径平稳度 | `opening_path_mean_drawdown_50bar`、`opening_path_mean_reverse_drawdown_50bar`、`opening_path_smoothness_50bar` | 缺失 | 第 50 根开盘分钟 bar 完成后 |
| 低阶趋势形态 | `opening_poly_slope_60bar`、`opening_poly_curvature_60bar`、`opening_poly_trend_acceleration_60bar` | LSMA 仅近似斜率；曲率缺失 | 第 60 根开盘分钟 bar 完成后 |
| GFTD 状态机 | `gftd_setup_direction_5_3`、`gftd_setup_streak_5`、`gftd_buy_count_6`、`gftd_sell_count_6`、`gftd_signal_state_5_3_6` | 缺失 | 当前 bar 完成后 |
| 开盘突破距离 | `atr10_orb_width_to_open`、`dual_thrust_drange5_to_open`、两类上下边界距离 | ATR 已有；边界和距离缺失 | 当日开盘价确认后 |
| 日级厚尾状态 | `prev30d_open_close_return_pearson_kurtosis` | 分钟收益峰度仅近似 | 当日开盘前 |
| 延迟日内极值 | `distance_to_delayed_session_high_lag2`、`distance_to_delayed_session_low_lag2`、`delayed_session_extreme_breakout_state_lag2` | 当前会话回撤仅近似 | 当前 bar 完成后 |

#### 实现状态

第一优先级已经实现为独立因子层：

- 计算模块：`FractalQuant/factor/intraday_strategy.py`
- 股票/ETF 共用生成器：`scripts/generate_intraday_strategy_factors.py`
- 输出：4 个键、7 个源行情字段和 21 个因子字段，共 32 列
- 默认输出目录：`stock_1min_intraday_strategy_p0_factors` 和 `etf_1min_intraday_strategy_p0_factors`
- 本地分钟文件按已完成 bar 处理，`available_time=trade_time`
- 本地数据把 09:30 作为第一个存储 bar，因此固定 50/60-bar 字段在完整交易日分别从 10:19/10:29 行开始出现
- GFTD 的 5-bar 比较和状态在午休重置；研报明确定义为“开盘以来”的延迟日内极值是例外，午后继续保留上午极值
- ATR 开盘突破距离固定使用 `f=0.4`，Dual Thrust 距离固定使用 `f=0.2`；宽度归一化字段本身不乘 `f`
- 日线 ATR、DRange 和 30 日峰度全部通过 `< target_date` 的查找严格排除当日及未来数据，并在复权价格空间计算跨日边界
- P0 公式/因果/增量写入专项测试 7 项继续通过；加入 P1 后，补齐 `pytest-asyncio` 的当前全仓测试为 95 项通过
- 真实数据 smoke：`000001.SZ`、`510300.SH` 在 2026-07-17 均输出 241 行，21/21 因子至少有一个有效值且无无穷值

示例：

```powershell
uv run python scripts/generate_intraday_strategy_factors.py `
  --asset-type both `
  --symbols 000001.SZ 510300.SH `
  --date-from 20260717 --date-to 20260717 `
  --overwrite
```

### 2.2 第二优先级：需要跨日缓存、成分映射或额外工程

| 因子组 | 建议字段 | 额外依赖 |
|---|---|---|
| 市场主动资金流 | `market_active_notional_imbalance_1m`、`market_active_notional_imbalance_cum_session`、`asset_minus_market_active_flow_1m` | point-in-time 成分股、主动买卖方向、全市场分钟到齐时间 |
| 历史同刻路径匹配 | `path_knn_expected_next_return`、`path_knn_up_probability`、`path_knn_mean_distance`、`path_knn_direction_agreement` | 只含 `d-1` 及更早日期的同刻路径库和缓存 |
| 波动收敛后的突破 | `distance_to_prior26_upper_envelope`、`distance_to_prior26_lower_envelope`、`min_volatility_regime_7` | 研报未披露严格包络公式，必须采用透明适配口径 |
| 早盘区间反转位置 | `early_range_position_41m`、`distance_to_early_high_41m`、`distance_to_early_low_41m` | 固定早盘 41 分钟窗口和完成时点 |

#### 实现状态

第二优先级已经实现为 `p0_p1` profile：

- 默认 `p0` 仍为 21 个因子、32 列；`p0_p1` 为 34 个因子、45 列，并使用独立股票/ETF输出目录。
- 市场资金流要求两张显式 point-in-time 映射表：股票池成员区间和目标标的到股票池的区间映射；不使用静态板块快照回填历史。
- 逐笔成交只接受源数据的显式 `B/S` 方向。股票池内交易中成员缺逐笔、分钟时间轴不完整、目标缺映射或有效成交含未知方向时严格失败；停牌成员按零贡献处理。
- 逐笔事件按向上取整后的分钟结束标签对齐，避免把 09:30:xx 的成交提前写入 09:30 行；09:25 集合竞价不混入该连续成交资金流。
- 股票池资金流按 `pool_id + trade_date` 原子缓存，缓存同时记录成员数量和成员哈希，成员变化后自动重算。
- SMT 适配固定使用前 60 个交易日、K=10、至少 10 根当前路径；每个已完成前缀标准化后使用均方欧氏距离，历史日期严格满足 `h < d`。
- 按确认口径，P1 的路径、26-bar包络、7期最小波动率和累计资金流把午休视为全日连续；11:30 的历史下一根收益是 13:01。15:00 没有下一根，SMT字段为缺失。
- 早盘 41-bar 边界在本地完整日从 10:10 开始可用；26-bar 包络、7期最小波动率和 SMT 分别从 09:56、10:03、09:39 开始可用。
- 14 项专项测试和当前全仓 95 项测试通过。真实数据 smoke 中，`000001.SZ`、`159001.SZ` 在 2026-01-05 均输出 241×45，13/13 个 P1 因子有有效值且无无穷值。

示例：

```powershell
uv run python scripts/generate_intraday_strategy_factors.py `
  --asset-type both --priority-profile p0_p1 `
  --symbols 000001.SZ 159001.SZ `
  --date-from 20260105 --date-to 20260105 `
  --pool-membership-path pool_membership.csv `
  --target-pool-path target_pool.csv `
  --overwrite
```

### 2.3 第三优先级：研究型或低增量

- `emd_log_noise_trend_energy_ratio_41m`：公式有增量，但 EMD 端点效应强、计算重，必须逐时点只对前缀重算。
- `rough_entropy_noise_share`：原始粗糙熵和非线性拟合复杂，现有高级熵/Hurst 因子只能近似。
- `bars_since_20bar_high/low`、`mass_index`、`chaikin_money_flow`：易实现，但相对现有趋势、ATR、量价因子增量较低。
- `ga_breakout_width_2013`：可用于复现实验，不宜把 2010-2011 股指期货样本内系数直接作为股票/ETF生产因子。

不建议把遗传规划输出、深度学习预测值、随机突破宽度或 SAR 持仓规则直接登记为原子因子。它们分别属于模型、随机化执行或仓位管理层。

## 3. 第一优先级字段定义

### 3.1 开盘路径平稳度

来源：`基于市场情绪平稳度的股指期货日内交易策略`，p.6。

对开盘后的前 50 个已完成分钟 bar，取价格序列 `p[1:n]`。对每个 `i`：

```text
DD_i  = max(0, max_{j>i} ((p_i - p_j) / p_i))
RDD_i = max(0, max_{j>i} ((p_j - p_i) / p_i))
```

没有后继点时令该项为 0。原文公式未显式写出外层与 0 取最大值；实现时补上这个数值契约，避免严格单调路径产生“负回撤”：

```text
opening_path_mean_drawdown_50bar
    = mean(DD_i)

opening_path_mean_reverse_drawdown_50bar
    = mean(RDD_i)

opening_path_smoothness_50bar
    = min(mean(DD_i), mean(RDD_i))
```

值越小，路径越接近单边。方向应由独立的开盘累计收益表示，不要把方向乘入平稳度后丢失两个原子维度。

原文使用固定开盘观察窗。若另做滚动版本，应使用明确后缀，如 `_w50`，并在上午、下午分别滚动，不能让普通 50-bar 窗口跨午休。固定开盘版本则按原文保留上午全路径。

### 3.2 低阶多项式趋势形态

来源：`基于低阶多项式拟合的股指期货趋势交易(LPTT)策略`，pp.5-7。

为避免不同价格和窗口不可比，建议对时间归一化为 `x in [-1,1]`，价格使用 `y=log(p/p_0)`。分别拟合：

```text
y = a1*x + b1
y = a2*x^2 + b2*x + c2
```

输出：

```text
opening_poly_slope_60bar        = a1
opening_poly_curvature_60bar    = 2*a2
opening_poly_trend_acceleration_60bar
    = a1 * (2*a2)
```

乘积大于 0 表示当前趋势方向与曲率同向，即上涨加速或下跌加速；小于 0 表示趋势减速。当前 `lsma` 是固定滚动窗的原始价格线性斜率，不能替代二次曲率和二者交互。

原文使用更高频数据和固定开盘观察段；60 个 1 分钟 bar 是对当前数据的透明适配，不应宣称与原文回测完全相同。

### 3.3 GFTD 状态机

来源：`基于GFTD的期指日内程序化交易策略`，pp.3-5。原文优化参数为 `n1=5`、`n2=3`、`n3=6`。

第一层比较：

```text
ud_t = sign(close_t - close_{t-5})
```

连续同号累计；累计到 `-3` 形成买入启动，累计到 `+3` 形成卖出启动。买入计数要求同一 bar 同时满足：

```text
close_t >= high_{t-2}
high_t > high_{t-1}
close_t > previous_counted_close
```

卖出计数为对称条件。计数到 6 形成信号；新的同向启动会取消上一组未完成计数。

建议保留状态原子，而不是只输出最终 0/1 信号：

```text
gftd_setup_direction_5_3
gftd_setup_streak_5
gftd_buy_count_6
gftd_sell_count_6
gftd_signal_state_5_3_6
```

所有状态按交易日重置。普通滚动动量不能表达“启动、计数、被新启动取消”的事件记忆。

### 3.4 开盘区间突破距离

来源：`日内突破模式及其资金管理的多重比较研究`，pp.5-6。

ATR 型边界使用严格滞后的日线：

```text
TR_q = max(high[q], close[q-1]) - min(low[q], close[q-1])
avg_true_range_10[d] = mean(TR_q), q=d-10,...,d-1
```

Dual Thrust 的 5 日宽度为：

```text
HH = max(high[q]),  q=d-5,...,d-1
LL = min(low[q]),   q=d-5,...,d-1
HC = max(close[q]), q=d-5,...,d-1
LC = min(close[q]), q=d-5,...,d-1
DRange = max(HH - LC, HC - LL)
```

在当日开盘价 `O_d` 确认后，输出连续距离：

```text
atr10_orb_width_to_open = avg_true_range_10 / O_d
dual_thrust_drange5_to_open = DRange / O_d

distance_to_atr10_orb_upper = close_t / (O_d + f*avg_true_range_10) - 1
distance_to_atr10_orb_lower = close_t / (O_d - f*avg_true_range_10) - 1
distance_to_dual_thrust_upper = close_t / (O_d + f*DRange) - 1
distance_to_dual_thrust_lower = close_t / (O_d - f*DRange) - 1
```

先输出宽度和距离，再在模型或消融实验中研究 `f` 和突破标记；不要把全样本最优阈值固化为唯一因子。

### 3.5 严格滞后的日级厚尾状态

来源：`厚尾分布下的随机区间突破策略`，pp.15-17。

每日开盘到收盘收益：

```text
r_d = close_d / open_d - 1
```

因子使用 `d-30` 到 `d-1` 的 30 个完整交易日，计算经小样本修正的 Pearson kurtosis：

```text
prev30d_open_close_return_pearson_kurtosis[d]
    = pearson_kurtosis(r_q), q=d-30,...,d-1
```

原文用峰度大于 4 作为交易门控。建议先保留连续峰度，阈值标记作为后续交互。当前 `volatility_kurtosis` 统计分钟收益的滚动峰度，统计对象和可用时点均不同，不能等价替代。

### 3.6 延迟日内极值

来源：`基于日内波动极值的股指期货趋势跟随系统`，p.8。原文 `sL=2`、`cL=15`。

对交易日内第 `i` 根 bar：

```text
delayed_high_i = max(high_j), 1 <= j <= max(i-2, 1)
delayed_low_i  = min(low_j),  1 <= j <= max(i-2, 1)

distance_to_delayed_session_high_lag2 = close_i / delayed_high_i - 1
distance_to_delayed_session_low_lag2  = close_i / delayed_low_i - 1
```

第 16 根 bar 起：

```text
delayed_session_extreme_breakout_state_lag2 =
    +1, close_i >= delayed_high_i
    -1, close_i <= delayed_low_i
     0, otherwise
```

当前 `intraday_drawdown_from_session_high` 和 `intraday_rebound_from_session_low` 包含当前 bar，且上午、下午分别重置；原文是从当日开盘累计、排除最近两根 bar。因此只能算近似覆盖。新字段应按交易日累计，午后继续保留上午极值。

## 4. 第二优先级字段定义

### 4.1 市场主动资金流

来源：`在标度不变性破缺下洞察资金流向——MFT交易策略`，p.6。

对 point-in-time 指数成分或明确股票池：

```text
B_t = sum(active_buy_notional_{asset,t})
S_t = sum(active_sell_notional_{asset,t})

market_active_notional_imbalance_1m = (B_t-S_t)/(B_t+S_t)
market_active_buy_sell_ratio_1m = B_t/S_t
```

归一化不平衡比原始比值在 `S_t` 接近 0 时更稳定。另输出从开盘累计的同口径字段和标的减市场字段。

当前订单簿注册表只有逐标的 `trade_*_imbalance_*`、OFI/MLOFI，缺少成分股主动成交金额聚合。成交方向必须逐交易所验证；若只能使用 tick rule，字段必须带 `_proxy`。`available_time` 是全部入选成分在该分钟的最晚到达时间。

实现采用显式 `B/S`，分钟分母为零时不平衡记为 0。逐笔时间向上取整到已完成分钟标签，09:25 集合竞价不混入连续成交资金流。`market_active_notional_imbalance_cum_session` 从 09:30 累计到 15:00，午休不重置；离线输出只有在全部交易中成员数据完整时才令 `available_time=trade_time`。

### 4.2 历史同刻路径匹配

来源：`基于时域分形的相似性匹配日内低频交易策略（SMT）`，pp.12-16。

在交易日 `d` 的当前时段 `t`，用当前从开盘到 `t` 的收益路径与历史同刻路径比较。历史库必须满足 `h < d`：

```text
D_h(t) = distance(path_d[1:t], path_h[1:t])
N_K(d,t) = K nearest historical days by D_h(t)

path_knn_expected_next_return
    = mean(return_h[t+1], h in N_K)
path_knn_up_probability
    = mean(return_h[t+1] > 0, h in N_K)
path_knn_mean_distance
    = mean(D_h(t), h in N_K)

path_knn_direction_agreement
    = abs(mean(sign(return_h[t+1])), h in N_K)
```

实现固定 `K=10`、历史窗 60 个交易日、最短前缀 10 根。路径为从 09:30 起的累计对数收益，逐前缀标准化后使用均方欧氏距离；午休按全日连续处理。

当前 `dynamic_time_warping` 只比较同一滚动窗口内部的近期段、前一段和基准段，不使用“历史同刻 K 近邻的下一段收益”，故仅为近似。

### 4.3 波动收敛后的突破

来源：`一类波动收敛突变模式的趋势跟随策略`，pp.5-7。

报告没有披露包络线的完整数学定义，不能声称精确复现。当前已有布林带宽和短长波动率比，故不应重复新增一个同义“收敛因子”。可透明适配为：

```text
distance_to_prior26_upper_envelope
distance_to_prior26_lower_envelope
min_volatility_regime_7
```

其中边界只使用当前 bar 之前的 26 根数据；`min_volatility_regime_7` 只回看 7 个已完成值。突破交互在原子字段通过增量检验后再构造。

透明适配的具体口径为：上下边界分别取 `high.shift(1).rolling(26).max()` 和 `low.shift(1).rolling(26).min()`；`sigma26` 是 26 根一分钟对数收益的总体标准差，`min_volatility_regime_7 = sigma26.shift(1).rolling(7).min()`。三个窗口均跨午休连续，但不跨交易日。

### 4.4 早盘区间反转位置

来源：`带反转的加强版EMDT交易策略`，pp.9-11。

使用开盘后前 41 分钟已经完成的高低点：

```text
H41 = max(high[opening first 41 bars])
L41 = min(low[opening first 41 bars])
P41 = (H41+L41)/2

early_range_position_41m = 2*(close_t-L41)/(H41-L41)-1
distance_to_early_high_41m = close_t/H41-1
distance_to_early_low_41m  = close_t/L41-1
```

字段在第 41 根 bar 完成后才可用。若不实现 EMD 状态过滤，只能称为早盘区间位置适配，不能称为完整 EMDT 复现。

## 5. 16 篇研报逐篇审计

| # | 研报、日期、页数 | 核心可用信息 | 与当前注册表关系 | 结论 |
|---:|---|---|---|---|
| 1 | 市场情绪平稳度，2015-04-02，21 页 | 平均回撤和反向回撤的较小值（p.6） | 当前只有单点会话回撤 | P0，真正缺失 |
| 2 | 经验模态分解日内趋势，2014-03-31，21 页 | `log(std(noise)/std(trend))`（pp.10-14） | 高级熵/Hurst 仅近似 | P2，需前缀 EMD |
| 3 | 混沌理论 NTT，2011-05-11，16 页 | 粗糙熵估计噪声标准差占比（pp.5-10） | 递归、熵、混沌因子仅近似 | P2，复杂且重合度高 |
| 4 | MFT 资金流，2012-07-09，12 页 | 成分股主动买卖金额聚合（pp.5-9） | 只有逐标的订单和成交流 | P1，缺跨市场层 |
| 5 | 低阶多项式 LPTT，2011-10-11，14 页 | 线性斜率、二次曲率、加速/减速（pp.5-8） | LSMA 只覆盖线性斜率近似 | P0，曲率缺失 |
| 6 | 遗传规划方法，2012-09-16，22 页 | OHLCV 和日内序号驱动模型搜索（pp.10-16） | 输入族已覆盖 | 模型方法，不新增原子因子 |
| 7 | 波动收敛突变，2011-08-15，20 页 | 低波动后穿越包络（pp.5-7） | 波动率状态和布林宽已覆盖，交互缺失 | P1，透明适配 |
| 8 | SMT 相似性匹配，2012-09-03，18 页 | 历史同刻路径近邻的下一段方向（pp.12-16） | `dynamic_time_warping` 不是同一信息 | P1，真正缺失 |
| 9 | GFTD，2012-07-09，15 页 | 启动、计数和取消的状态机（pp.3-5） | 普通动量无法派生 | P0，真正缺失 |
| 10 | 日内突破比较，2013-06-17，25 页 | ATR/Dual Thrust 宽度和边界距离（pp.5-6） | ATR 已有，日级边界状态缺失 | P0，部分缺失 |
| 11 | 深度学习日内交易，2014-06-18，24 页 | 盘口、价差、中价、深度、成交量的 5 阶滞后（pp.16-19） | 股票/ETF 当前微观结构已大体覆盖；期货持仓量不可得 | 模型编码，不新增同义原子列 |
| 12 | 加强版 EMDT，2014-12-03，21 页 | EMD 趋势/反转状态和早盘枢轴区间（pp.5-11） | 两者均无精确字段 | 早盘区间 P1，EMD P2 |
| 13 | 多维变量遗传规划，2013-09-02，28 页 | ADX、MACD、CCI、ATR、CMF、Mass Index、距 20-bar 高点时间（p.14） | 大部分技术指标已覆盖 | 仅少数字段 P2 |
| 14 | 厚尾随机突破，2014-08-20，20 页 | 前 30 日开收收益 Pearson 峰度（pp.15-17） | 当前分钟峰度统计对象不同 | P0，真正缺失 |
| 15 | 遗传算法突破，2013-02-26，19 页 | 9 个前日 OHLC/真实区间变量的线性宽度（pp.7-9） | 原子日线可得，历史系数不可迁移 | P2，仅复现或滚动重估 |
| 16 | 日内波动极值，2011-12-09，21 页 | 排除最近 2 根的开盘以来高低点突破（p.8） | 当前高低点含当前 bar 且午休重置 | P0，真正缺失 |

## 6. 研究型候选和不直接落地项

### 6.1 EMD 能量比

研报 2 和 12 的共同定义是：

```text
s(t) = sum(IMF_i(t)) + r_4(t)
R(t) = log(std(s-r_4) / std(r_4))
```

`R` 低表示趋势成分相对强，`R` 高表示震荡/反转状态。EMD 的样条端点会随新增数据修订。若在完整交易日上分解后把过去时点值回填，会引入未来信息。合规实现只能在每个输出时点对 `<=t` 的前缀独立分解并只保留末端输出。

### 6.2 NTT 粗糙熵噪声占比

研报 3 定义：

```text
NTS = sigma_noisy / sigma_data
```

其中 `sigma_noisy` 来自相空间粗糙熵和非线性 LM 拟合。现有 Hurst、递归率、Kolmogorov entropy 等不等价。除非能复现完整估计过程并验证数值稳定性，否则应保留为研究候选，而不是用某个熵字段改名冒充。

### 6.3 遗传算法历史宽度

研报 15 的样本内线性宽度为：

```text
B = -0.34*prev_open
    +0.65*prev_high
    +0.31*prev_low
    -1.25*prev_close
    +0.81*prev2_close
    -0.99*true_high
    +0.79*true_low
    +0.52*true_range
    +1.21*avg_true_range_10
```

该系数由 2010-2011 股指期货优化得到。它可作为历史复现基线，但股票/ETF 实现应使用原子输入，随后在 purged walk-forward 内滚动估计，不应在当前生成器中把上述系数当成永久真值。

## 7. `available_time` 和防泄漏契约

1. 使用当前分钟 OHLC/close 的字段，最早在该分钟 bar 完成后可用；若源数据时间戳表示 bar 开始，则当前实现口径为 `bar_time + 1 minute`。交易和标签从下一可成交时点开始。
2. 固定开盘 41/50/60-bar 字段分别等对应窗口完整后才可用，不能回填到开盘。
3. GFTD、延迟极值和从开盘累计字段按交易日重置。延迟极值按原文跨午休保留上午信息；普通滚动窗不得把午休当成连续 1 分钟。
4. 日级 ATR、DRange、峰度和遗传算法输入严格只使用 `d-1` 及更早完整日。若字段还需要当日开盘价，则等待 09:25 开盘撮合结果。
5. 市场主动资金流和任何横截面字段的 `available_time` 是所需成分中最晚的事件/分钟到达时间。不得先算全市场最终结果再回填到较早标的。
6. 路径 KNN 的近邻库只能包含 `d-1` 及更早日期；标准化、距离阈值和 K 值也必须只在历史训练窗确定。
7. EMD 只能对 `<=t` 前缀运行；全日 EMD 回填过去时点属于未来泄漏。
8. 研报交易信号不能假定在同一根 bar 的收盘价、最高价或最低价成交。统一用下一根 bar 开盘或更保守的成交模型。
9. 15:00 bar 可能包含收盘集合竞价；其字段不能声称可用于 14:57 前执行。
10. `FutureReturnsFactor` 只允许作为标签，不能进入训练特征清单。

## 8. 验证方案

这些报告最早发表于 2011 年，最晚发表于 2015 年，且主要研究股指期货。历史收益不能证明对当前 A 股或 ETF 分钟模型仍有效。

建议按组实现并逐层消融：

1. 先实现 P0/P1 原子连续字段，不先固化报告阈值和交易规则。
2. 分股票、ETF、宽基指数单独验证，不把股指期货结论直接外推。
3. 使用 2021-2026 独立区间，采用 purged walk-forward；日级历史统计严格 `shift(1)`。
4. 标签建议为 `close[t+h] / open[t+1] - 1` 或同样可交易的下一时点口径，禁止同 bar 收盘建仓偏差。
5. 同时报告 Rank IC、ICIR、分组单调性、稳定年份比例、换手、容量、冲击和交易成本后收益。
6. 对当前 51 个竞价、156 个 ETF 多窗口、135 个订单簿和 46 个高级因子做增量消融，而不是只报告候选自身回测。
7. 开盘路径、LPTT、GFTD、突破、延迟极值、MFT、SMT、波动收敛和早盘区间分别作为独立组；只有原子组通过后再研究平稳度 x OFI、收敛 x 突破等交互。
8. 对窗口和阈值做敏感性平台检验，不在完整样本上挑单点最优参数。

## 9. 最终判断

16 篇研报中，6 篇提供了当前注册表真正缺失且可用现有数据直接构造的高优先级信息：

1. 研报 1 的整段路径平稳度；
2. 研报 5 的二阶趋势曲率；
3. 研报 9 的 GFTD 事件状态；
4. 研报 10 的严格滞后突破宽度和边界距离；
5. 研报 14 的前 30 日日内收益厚尾状态；
6. 研报 16 的排除最近两根 bar 的日内延迟极值。

研报 4、7、8、12 提供的第二优先级跨市场、历史路径和固定早盘区间信息已经实现为严格依赖检查的 `p0_p1` profile。它们仍需独立样本的增量验证，不能把实现完成等同于可部署有效。EMD、NTT、遗传规划和深度学习报告更适合作为模型研究或复现基线，不应简单改名扩列。
