# 5 分钟 BSP 收益模型与特征说明（代码审计版）

本文档依据当前仓库中的实际代码整理，核心来源为：

- `BuySellPoint/BSPointList.py`：各类 BSP 的原始结构特征生成；
- `BuySellPoint/BS_Point.py`：所有 BSP 的公共特征；
- `sliding_window_chan.py`：首次发现 BSP 时的快照、去重后的技术指标与市场状态；
- `pipelineCurrent.py`：标签生成、特征筛选、XGBoost/LSTM 训练、预测与交易执行。

本文严格区分：

1. BSP 对象生成过的原始特征；
2. 快照实际导出的字段；
3. `get_feature_columns()` 最终送入 5 分钟模型的字段；
4. 标签、标识符、文本字段以及未来信息等不应进入模型的字段。

## 1. 模型到底预测什么

当前 5 分钟模型不是分类模型，而是分方向训练的收益回归模型：

- Buy 模型：只使用 `direction == "buy"` 的 BSP；
- Sell 模型：只使用 `direction == "sell"` 的 BSP；
- 默认模型：`XGBRegressor`；也可以选择 LSTM；
- 目标列：`best_return_pct`。

`best_return_pct` 的计算方式：

- Buy BSP：从 BSP 所在 K 线之后开始，在 lookahead 窗口内寻找最高价；
  `100 × (未来最高价 - BSP收盘价) / BSP收盘价`；
- Sell BSP：从 BSP 所在 K 线之后开始，在 lookahead 窗口内寻找最低价；
  `100 × (BSP收盘价 - 未来最低价) / BSP收盘价`。

标签只有在完整未来窗口已经出现后才写入，模型训练时使用已完成标签的历史 BSP。实时预测本身不读取未来价格。

> 注意：`lookahead_days` 当前通过 `days × 24 × 60 / 5` 换算成 K 线数量。例如 2 天会得到 576 根 5 分钟 K 线。它代表 576 个数据 bar，不一定等于两个交易日，具体跨度取决于 CSV 是否包含盘前盘后以及交易时段。

## 2. 特征从哪里产生

一个 BSP 被 Chan 模型发现时，数据经过下面的路径：

```text
CBi / CSeg / ZS / KLine
    → BSPointList 为具体 BSP 类型生成 feature_dict
    → CBS_Point 添加公共特征 bsp_bi_amp
    → standardize_features_for_bsp 添加跨类型标准化特征
    → SlidingWindowChan 在 snapshot_first_seen 时导出快照
    → prepare_ml_dataset 添加类别编码
    → get_feature_columns 选择数值和布尔列
    → Buy/Sell 回归模型
```

快照只在 BSP 首次被发现时加入 `bsp_rows_5m`。因此快照中的结构和指标应解释为“首次发现时已知的状态”，而不是整段历史结束后回填的最终状态。

## 3. 当前实际进入模型的特征

按照当前导出结构和 `get_feature_columns()`，完整 schema 下共有 61 个候选输入列。某一种 BSP 不具备的类型专属特征会为空，并在进入矩阵时补为数值。

### 3.1 当前 K 线与信号基础字段

| 特征 | 来源/计算 | 含义与注意事项 |
|---|---|---|
| `klu_open` | BSP 结束 K 线的 open | 信号首次发现对应 K 线的开盘价。绝对价格会带来跨年份尺度漂移。 |
| `klu_high` | BSP 结束 K 线的 high | 当前 K 线最高价。 |
| `klu_low` | BSP 结束 K 线的 low | 当前 K 线最低价。 |
| `klu_close` | BSP 结束 K 线的 close | BSP 标签和多项价格特征的基准价。 |
| `klu_volume` | BSP 结束 K 线的 volume | 当前 K 线成交量；已删除重复的 `feat_volume`。 |
| `is_buy` | `int(bsp.is_buy)` | Buy=1、Sell=0。由于 Buy/Sell 模型分开训练，该列在单个模型内为常数，实际没有区分能力。 |
| `direction_encoded` | buy=1、sell=0、其他=-1 | 与 `is_buy` 重复；在分方向模型内同样为常数。 |
| `bsp_type_encoded` | `1→1, 2→2, 3a→3, 1p→4, 2s→5, 3b→6` | 文本 `bsp_type` 的模型编码。 |
| `is_segbsp` | BSP 对象属性 | 是否为线段级 BSP。当前代码中初始化为 False，未发现设置为 True 的路径，因此目前很可能是常数。 |

### 3.2 当前 K 线价格行为

| 特征 | 精确计算 | 含义 |
|---|---|---|
| `price_change_pct` | `100 × (close-open)/open` | 当前 5 分钟 K 线涨跌幅，单位为百分数。 |
| `high_low_spread_pct` | `100 × (high-low)/low` | 当前 K 线振幅，单位为百分数。 |
| `upper_shadow` | `high - max(open, close)` | 上影线的绝对价格长度，不是比例。 |
| `lower_shadow` | `min(open, close) - low` | 下影线的绝对价格长度，不是比例。 |
| `body_size` | `abs(close-open)` | 实体的绝对价格长度，不是比例。 |
| `is_bullish_candle` | `1 if close > open else 0` | 当前 K 线是否收阳。 |

### 3.3 技术指标（保留唯一规范列）

| 特征 | 来源/计算 | 说明 |
|---|---|---|
| `macd_value` | `2 × (DIF-DEA)` | MACD 柱值。 |
| `macd_diff` | 快 EMA - 慢 EMA | 代码中的 `DIF`；已统一名称并删除 `feat_macd_diff`。 |
| `macd_dea` | DIF 的 signal EMA | MACD 信号线；已删除 `feat_macd_dea`。 |
| `rsi` | 14 周期 Wilder 风格平滑涨跌 | 已启用真实计算；删除重复的 `feat_rsi`，不再用固定 0/50 冒充计算值。 |
| `kdj_k` | 9 周期 KDJ K | 已启用真实计算；删除 `feat_kdj_k`。 |
| `kdj_d` | 9 周期 KDJ D | 已启用真实计算；删除 `feat_kdj_d`。 |
| `kdj_j` | `3K-2D` | 已启用真实计算；删除 `feat_kdj_j`。 |
| `dmi_plus` | 14 周期 +DI | 已启用真实计算；不足周期时底层实现返回 0。 |
| `dmi_minus` | 14 周期 -DI | 已启用真实计算。 |
| `dmi_adx` | 由 DX 平滑得到 ADX | 已启用真实计算。 |
| `feat_ppo` | `(MACD fast EMA - slow EMA) / slow EMA` | 百分比价格振荡器，未乘 100。虽然前缀为 `feat_`，它是 BSP 创建时从同一结束 K 线取得。 |

### 3.4 所有 BSP 的公共/标准化 Chan 特征

| 特征 | 来源/计算 | 说明 |
|---|---|---|
| `feat_bsp_bi_amp` | `bsp.bi.amp()` | BSP 所挂接的 Bi 或 Seg 的绝对振幅。 |
| `feat_divergence_rate` | 随 BSP 类型标准化 | BSP1 为出段/入段力度比；BSP2/2s 使用 retrace rate；BSP3 使用 ZS height。跨类型数值含义并不完全相同。 |
| `feat_zs_cnt` | BSP1 取实际 `len(seg.zs_lst)`；其他类型使用规则默认值 | BSP2/2s 默认 1，BSP3 默认 2，其他默认 0。对非 BSP1 它不是重新统计得到的真实中枢数量。 |
| `feat_break_bi_amp` | 对应 break Bi 的绝对振幅 | BSP2/2s 使用真实 break Bi；BSP1/3 当前是从对应 BSP Bi 特征复制得到的标准化占位定义。 |
| `feat_break_bi_klu_cnt` | break Bi 覆盖的 KLU 数量 | BSP2/2s 为真实 break Bi 长度；BSP1/3 来自对应 Bi 长度。 |
| `feat_break_bi_amp_rate` | `break_bi_amp / break_bi_begin_value` | 比例值，未乘 100。 |
| `feat_bi_amp` | 当前类型对应 Bi 的绝对振幅 | 类型无专属值时回退到 `bsp_bi_amp`。 |
| `feat_bi_klu_cnt` | 当前类型对应 Bi 的 KLU 数量 | 某些回退路径会写 0，0 可能表示缺失而非真实零长度。 |
| `feat_bi_amp_rate` | `bi_amp / bi_begin_value` | 比例值，未乘 100；某些回退路径会写 0。 |
| `feat_level` | BSP1/1p=1，BSP2=2，BSP3=3；2s 使用 `bsp2s_lv` | BSP 层级；2s 的 level 带有真实层次数值。 |
| `feat_bsp_type` | `1/1p→1, 2→2, 2s→2.5, 3a→3, 3b→3.5` | BSP 创建阶段的另一套数值编码，与 `bsp_type_encoded` 重复表达类别，但映射不同。 |

### 3.5 BSP1 / BSP1P 专属特征

| 特征 | 精确来源 | 说明 |
|---|---|---|
| `feat_bsp1_bi_amp` | BSP1 对应最后一笔 `amp()` | 绝对价格振幅。 |
| `feat_bsp1_bi_klu_cnt` | 最后一笔 `get_klu_cnt()` | 笔覆盖的原始 K 线数量。 |
| `feat_bsp1_bi_amp_rate` | `amp / begin_value` | 相对振幅，未乘 100。 |

BSP1 的 `feat_divergence_rate` 原始公式为：

```text
out_metric / (in_metric + 1e-7)
```

其中 metric 由 BSP 配置的 `macd_algo` 决定；主 5 分钟流程当前使用 `macd_algo="peak"`。

### 3.6 BSP2 专属特征

| 特征 | 精确来源 | 说明 |
|---|---|---|
| `feat_bsp2_retrace_rate` | `bsp2_bi.amp() / break_bi.amp()` | 二类点回撤笔相对突破笔的幅度。 |
| `feat_bsp2_break_bi_amp` | `break_bi.amp()` | 突破笔绝对振幅。 |
| `feat_bsp2_break_bi_bi_klu_cnt` | `break_bi.get_klu_cnt()` | 字段名包含重复的 `bi`，但代码确实如此命名。 |
| `feat_bsp2_break_bi_amp_rate` | `break_bi.amp()/break_bi.get_begin_val()` | 突破笔相对振幅。 |
| `feat_bsp2_bi_amp` | `bsp2_bi.amp()` | BSP2 回撤笔绝对振幅。 |
| `feat_bsp2_bi_klu_cnt` | `bsp2_bi.get_klu_cnt()` | BSP2 回撤笔长度。 |
| `feat_bsp2_bi_amp_rate` | `bsp2_bi.amp()/bsp2_bi.get_begin_val()` | BSP2 回撤笔相对振幅。 |

### 3.7 BSP2S 专属特征

| 特征 | 精确来源 | 说明 |
|---|---|---|
| `feat_bsp2s_retrace_rate` | `abs(bsp2s_end-break_end)/break_bi.amp()` | 2s 点相对突破笔结束位置的回撤比例。 |
| `feat_bsp2s_break_bi_amp` | `break_bi.amp()` | 关联突破笔振幅。 |
| `feat_bsp2s_break_bi_klu_cnt` | `break_bi.get_klu_cnt()` | 关联突破笔长度。 |
| `feat_bsp2s_break_bi_amp_rate` | `break_bi.amp()/break_bi.get_begin_val()` | 关联突破笔相对振幅。 |
| `feat_bsp2s_bi_amp` | `bsp2s_bi.amp()` | 2s 当前笔绝对振幅。 |
| `feat_bsp2s_bi_klu_cnt` | `bsp2s_bi.get_klu_cnt()` | 2s 当前笔长度。 |
| `feat_bsp2s_bi_amp_rate` | `bsp2s_bi.amp()/bsp2s_bi.get_begin_val()` | 2s 当前笔相对振幅。 |
| `feat_bsp2s_lv` | `bias / 2` | 2s 在同一结构内的层次序号。 |

### 3.8 BSP3A / BSP3B 专属特征

| 特征 | 精确来源 | 说明 |
|---|---|---|
| `feat_bsp3_zs_height` | `(zs.high-zs.low)/zs.low` | 对比中枢的相对高度。 |
| `feat_bsp3_bi_amp` | `bsp3_bi.amp()` | 离开/回试相关笔的绝对振幅。 |
| `feat_bsp3_bi_klu_cnt` | `bsp3_bi.get_klu_cnt()` | BSP3 对应笔长度。 |
| `feat_bsp3_bi_amp_rate` | `bsp3_bi.amp()/bsp3_bi.get_begin_val()` | BSP3 对应笔相对振幅。 |

### 3.9 当前被 selector 误选的快照序号

| 特征 | 来源 | 风险 |
|---|---|---|
| `snapshot_first_seen` | SlidingWindowChan 处理到第几根 bar 时首次发现 BSP | 它是处理顺序/时间代理，不是市场结构特征；当前会进入模型。 |
| `snapshot_last_seen` | 快照最后一次出现序号 | 对 `new_rows` 通常等于 first_seen；当前也会进入模型，且与 first_seen 高度重复。 |

这两列应作为审计元数据保留在导出文件中，但建议从模型输入排除。

## 4. BSP 生成了、但当前模型没有使用的字段

| 字段 | 为什么未进入模型 | 是否应使用 |
|---|---|---|
| `timestamp` | selector 明确排除 | 作为事件时间保留，不直接作为数值输入；若需要时段特征，应显式生成 hour/minute/session。 |
| `klu_idx` | selector 明确排除 | 正确；它是行位置，不是市场特征。 |
| `direction` | 文本且明确排除 | Buy/Sell 已分模型，不需要再次输入。 |
| `bsp_type` | 文本且明确排除 | 使用 `bsp_type_encoded`。 |
| `bsp_types` | 文本列 | 可能包含同一 BSP 的多个类型；当前模型没有解析。 |
| `bi_direction` | 字符串，selector 只取 numeric/bool | **当前未使用**。如果模型需要，必须按 `up=1/down=-1/unknown=0` 显式编码。 |
| `segment_direction` | 字符串，selector 只取 numeric/bool | **当前未使用**。应保留 `unconfirmed=0`，不能用未来确认后的方向回填。 |
| `best_return_pct` | `LABEL_COLS` 排除 | 正确；它是训练目标。 |
| `next_bi_return` | BSP 创建时可能计算下一笔收益 | 快照导出已明确跳过；正确，因为它可能包含未来信息。 |
| `has_best_exit`、`best_exit_*` | selector 明确排除 | 正确；属于未来结果或交易评估。 |

## 5. 当前 61 个输入列的完整机器清单

```text
body_size
bsp_type_encoded
direction_encoded
dmi_adx
dmi_minus
dmi_plus
feat_bi_amp
feat_bi_amp_rate
feat_bi_klu_cnt
feat_break_bi_amp
feat_break_bi_amp_rate
feat_break_bi_klu_cnt
feat_bsp1_bi_amp
feat_bsp1_bi_amp_rate
feat_bsp1_bi_klu_cnt
feat_bsp2_bi_amp
feat_bsp2_bi_amp_rate
feat_bsp2_bi_klu_cnt
feat_bsp2_break_bi_amp
feat_bsp2_break_bi_amp_rate
feat_bsp2_break_bi_bi_klu_cnt
feat_bsp2_retrace_rate
feat_bsp2s_bi_amp
feat_bsp2s_bi_amp_rate
feat_bsp2s_bi_klu_cnt
feat_bsp2s_break_bi_amp
feat_bsp2s_break_bi_amp_rate
feat_bsp2s_break_bi_klu_cnt
feat_bsp2s_lv
feat_bsp2s_retrace_rate
feat_bsp3_bi_amp
feat_bsp3_bi_amp_rate
feat_bsp3_bi_klu_cnt
feat_bsp3_zs_height
feat_bsp_bi_amp
feat_bsp_type
feat_divergence_rate
feat_level
feat_ppo
feat_zs_cnt
high_low_spread_pct
is_bullish_candle
is_buy
is_segbsp
kdj_d
kdj_j
kdj_k
klu_close
klu_high
klu_low
klu_open
klu_volume
lower_shadow
macd_dea
macd_diff
macd_value
price_change_pct
rsi
snapshot_first_seen
snapshot_last_seen
upper_shadow
```

该清单来自当前代码生成的 BSP schema 再经过 `prepare_ml_dataset()` 和 `get_feature_columns()` 实际计算，而不是来自旧的 `features_used_in_training.csv`。旧 CSV 仍包含已经删除的 `feat_rsi`、`feat_volume`、`feat_macd_*` 等字段，不能再作为当前模型文档的依据。

## 6. 正确性审计结论

### 已正确处理

- RSI、KDJ、DMI 已在绘图/BSP 构建配置中启用真实计算；
- RSI、KDJ、MACD、volume 的重复 `feat_*` 列已删除，只保留规范列；
- 指标不存在时不再伪造 RSI=50、DMI=25 等值；
- `next_bi_return` 没有导出到训练快照；
- `best_return_pct` 和 best-exit 类未来结果已从 selector 排除；
- `klu_idx`、timestamp 等标识符已排除；
- 实时下单使用 next-bar open，而不是在信号已经完成的同一根 K 线上成交。

### 当前仍需修正或明确的地方

1. `bi_direction`、`segment_direction` 尚未编码，因此当前模型没有使用它们；
2. `snapshot_first_seen`、`snapshot_last_seen` 当前错误地进入模型，应排除；
3. `is_buy` 和 `direction_encoded` 在分方向模型中都是常数且互相重复；
4. `feat_bsp_type` 与 `bsp_type_encoded` 是两套类别数值编码，存在重复表达；
5. 训练时 `prepare_ml_dataset()` 用整批训练数据的列均值补缺失，但实时单行预测无法复用同一组训练均值，随后缺列会补 0，训练/预测缺失值处理并不一致；
6. 类型专属特征在其他 BSP 类型上为空。必须采用一致的缺失值策略，并最好增加 `is_bsp1/is_bsp2/is_bsp2s/is_bsp3` 等显式类型指示；
7. `feat_zs_cnt` 对 BSP2/2s/3 使用规则常数，不应解释成每个样本都重新计算出的真实中枢数量；
8. BSP1/3 的标准化 break-Bi 特征在代码注释中明确属于 placeholder 派生，不应解释成独立观测到的真实突破笔；
9. 绝对价格、绝对振幅、成交量跨年份和不同标的缺乏尺度稳定性，若跨标的训练应考虑相对化或标准化；
10. 当前特征集合是动态从 DataFrame 数值列获得的。不同训练窗口如果没有出现某一 BSP 类型，`feature_cols` 可能改变；模型包虽然保存列顺序，但重训之间 schema 可能不同。

## 7. 建议的“正确模型输入”版本

为了保持所有有效信息，同时减少重复和时序风险，建议：

- 保留所有唯一的价格行为、技术指标和 Chan 结构特征；
- 新增 `bi_direction_encoded` 与 `segment_direction_encoded`；
- `segment_direction` 未确认时编码为 0，禁止事后回填；
- 排除 `snapshot_first_seen`、`snapshot_last_seen`；
- 分 Buy/Sell 模型时排除 `is_buy`、`direction_encoded`；
- 只保留一套 BSP 类型编码，建议保留 `bsp_type_encoded`；
- 训练和实时预测统一把缺失值置 0，或在模型包中保存训练期 imputer 并在预测时复用；
- 固定一份版本化 `feature_cols`，不要让训练窗口决定 schema；
- 对 `feat_zs_cnt` 和 placeholder break-Bi 特征增加来源/有效性标志。

## 8. 模型训练与交易执行概要

1. SlidingWindowChan 每来一根 5 分钟 K 线重新计算当前窗口；
2. 只收集首次出现的 BSP 快照；
3. 等未来 lookahead 窗口完整后写入 `best_return_pct`；
4. 定期重训 Buy 和 Sell 两个回归模型；
5. 模型输出预测的最佳顺向收益百分比；
6. Buy/Sell 各自使用收益阈值过滤信号；
7. 满足阈值后在下一根 bar 的 open 执行；
8. 阈值通过近期已实现 replay 网格搜索更新，并计入 `fee_pct`。

这意味着模型学习的不是“价格下一根一定涨或跌”，而是：在某类 BSP 首次出现时，结合当时的 K 线、技术指标和 Chan 结构，未来窗口内顺着该信号方向最多可能出现多大的有利波动。
