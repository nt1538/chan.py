# Chan BSP 规则策略与交易架构

这部分把原来集中在 `pipelineCurrent.py` 的职责拆成三层，并保留旧的
`pipelineCurrent.ExecutionEngine` 名称供 notebook 和 checkpoint 使用。

完整 pipeline 的兼容入口现已迁移到 `Pipeline/DailyBandit5mPipeline.py`；
`pipelineCurrent.py` 只负责重新导出旧名称。数据、Chan、特征、Bandit、模型、
checkpoint 和报告分别通过 `DataPipeline/`、`ChanPipeline/`、`FeaturePipeline/`、
`Bandit/`、`ModelStrategy/`、`Checkpoint/` 与 `Reporting/` 访问。

## 目录职责

- `CustomBuySellPoint/`：把 BSP 行转换成首次发现快照信号；只负责规则和交易意图。
- `Trade/`：订单、下一根 K 线开盘成交、手续费、滑点、仓位和风控。
- `ModelStrategy/`：历史回放、绩效指标、跨周期测试和参数网格搜索。
- `Config/`：可选的 JSON/YAML 配置读取和示例参数。

策略不会重新计算历史上的 `bi_direction` 或 `segment_direction`。它直接读取
`bsp_df` 中 BSP 首次发现时保存的字段，因此不会用后来完成的线段修改过去决策。

## 最小回测示例

```python
from CustomBuySellPoint import SegBspStrategy, SegBspStrategyConfig
from ModelStrategy import BacktestChanConfig, run_bsp_backtest
from Trade import RiskConfig, RiskManager

strategy = SegBspStrategy(SegBspStrategyConfig(
    entry_segment_directions=frozenset({"up"}),
    required_sell_signals=3,
    sell_lookback_bars=8,
))

result = run_bsp_backtest(
    price_df,
    bsp_df,
    strategy,
    BacktestChanConfig(
        initial_capital=100_000,
        fee_pct=0.0005,
        slippage_pct=0.0002,
    ),
    RiskManager(RiskConfig(
        stop_loss_pct=0.03,
        take_profit_pct=0.08,
        trailing_stop_pct=0.025,
    )),
)

print(result.metrics)
print(result.trades)
result.trades.to_excel("outputs/strategy_trades.xlsx", index=False)
result.equity.to_excel("outputs/strategy_equity.xlsx", index=False)
```

## 参数搜索

```python
from ModelStrategy.parameterEvaluate.parameter_search import grid_search_seg_bsp

ranking = grid_search_seg_bsp(
    price_df,
    bsp_df,
    {
        "required_sell_signals": [1, 2, 3, 4],
        "sell_lookback_bars": [5, 8, 13],
        "exit_on_down_segment": [True, False],
    },
    score="sharpe",
)
```

参数搜索只能找出样本内表现最好的规则，不能证明未来最优。应使用不同月份或年份做
训练期、验证期和完全未参与选择的测试期，并检查交易次数、最大回撤和费用敏感性。
