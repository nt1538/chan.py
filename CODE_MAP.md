# Code Map

This project has several experiment notebooks, but the adaptive reward pipeline is mainly driven by a few Python modules.

## Main Adaptive Reward Files

### `adaptive_reward_checkpoint_fresh.py`

Main orchestration file for the adaptive reward strategy.

Responsibilities:

- Build adaptive reward snapshots with `build_adaptive_reward_snapshot`.
- Resume a saved snapshot with `run_adaptive_reward_from_snapshot`.
- Run/replay yfinance-based live or lookback simulations.
- Manage saved checkpoint bundles, warmup bars, model state, execution state, and output CSV/plots.
- Run the 5-minute execution phase through `_run_adaptive_reward_5m_phase`.
- Apply daily model gates (`FORCE_BUY`, `FREE`, `FORCE_SELL`) before accepting 5-minute buy/sell signals.

Important call parameters:

- `reset_execution_state=True`: resume flat from `initial_capital`.
- `threshold_ret_grid_override=...`: test a custom 5-minute return threshold grid without rebuilding the snapshot.
- `daily_threshold_lookback_days_override=...`: override the rolling lookback for daily threshold selection.

Important outputs:

- `daily_log.csv`: strategy equity, cash, position, thresholds, daily gate.
- `daily_reward_log.csv`: daily reward history used to fit daily thresholds.
- `trades.csv`: executed trades.
- `signal_decisions.csv`: every accepted/skipped 5-minute signal decision.
- `*_checkpoint.joblib`: persisted strategy state.

### `adaptive_trade_extensions.py`

Utility module for dynamic threshold selection and policy experiments.

Responsibilities:

- Define `RollingThresholdConfig`.
- Build threshold grids with `make_threshold_grid`.
- Build 5-minute return grids with `make_ret_grid`.
- Convert daily `p_day` into a gate with `gate_from_levels`.
- Score daily threshold pairs with `score_threshold_pair`.
- Select dynamic daily thresholds with `select_oracle_thresholds_from_daily_rewards`.
- Provide additional threshold/bandit helper classes for experiments.

Daily threshold meaning:

- `p_day <= buy_level` -> `FORCE_BUY`
- `p_day >= sell_level` -> `FORCE_SELL`
- otherwise -> `FREE`

### `adaptive_threshold_raw_walkforward.py`

Lower-level adaptive threshold and reward-evaluation helpers.

Responsibilities:

- Create Chan config for the adaptive pipeline.
- Evaluate per-day rewards for `FORCE_BUY`, `FREE`, and `FORCE_SELL`.
- Provide walk-forward pieces used by the adaptive reward pipeline.

### `pipelineCurrent.py`

Shared core pipeline and model utilities.

Responsibilities:

- Load OHLCV CSVs and macro features.
- Build Chan/K-line objects.
- Extract buy/sell point rows from Chan state.
- Build ML datasets and feature columns.
- Train/predict daily probability and 5-minute return models.
- Provide `ExecutionEngine` for simulated order placement and portfolio state.
- Provide plotting/model utility helpers used by the adaptive scripts.

This file is broad and acts like the shared library for the notebooks and adaptive scripts.

## Notebooks

### `PipelineAdaptiveRewardFreshStart.ipynb`

Notebook for building and resuming adaptive reward snapshots.

Typical use:

- Build a checkpoint from historical data.
- Resume from the checkpoint with a fresh trading state.
- Inspect `daily_reward_df`, `daily_log_df`, and `trades_df`.

### `PipelineAdaptiveActiveTQQQ.ipynb`

TQQQ-focused active/live experiment notebook.

Note: in the current workspace this file is empty, so use the Python module functions directly or restore notebook content from your older copy if needed.

### `PipelineAdaptiveActiveQQQ.ipynb` / `PipelineAdaptiveActiveSPY.ipynb`

Ticker-specific active experiment notebooks for QQQ and SPY.

### `PipelineAdaptiveActiveBacktest.ipynb`

Backtest/replay notebook for adaptive active runs.

### `PipelineAdaptiveThreshold*.ipynb`

Exploration notebooks for threshold logic and comparisons.

## Data Files

### `DataAPI/data/*.csv`

Local market and macro data used by the pipeline.

Examples:

- `TQQQ_DAY.csv`: daily TQQQ bars.
- `TQQQ_5M.csv`: 5-minute TQQQ bars.
- `QQQ_DAY.csv`, `QQQ_5M.csv`, `SPY_DAY.csv`, `SPY_5M.csv`: other ticker inputs.
- `VIX.csv`, `DXY.csv`, `US10Y.csv`, etc.: macro/risk features.

## Output Folders

### `output_adaptive_reward_snapshot_build_*`

Snapshot build outputs.

Contains:

- `snapshot_daily_log.csv`
- `snapshot_daily_reward_log.csv`
- `snapshot_trades.csv`
- `year_start_checkpoints/`

### `output_adaptive_reward_resumed_fresh_*`

Resume/backtest outputs from a saved snapshot.

Contains:

- `daily_log.csv`
- `daily_reward_log.csv`
- `trades.csv`
- `signal_decisions.csv`
- plots such as `equity_vs_buyhold.png`, `price_with_trades.png`, and `p_day.png`

### `output_adaptive_reward_live_*`

Live or paper-live outputs for a ticker/day.

Contains realtime-merged data, signal decisions, trades, logs, and live checkpoints.

### `output_adaptive_reward_range_backtest*`

Date-range replay/backtest outputs.

Useful for comparing a short period without reading a huge full-history output.

## Checkpoints

### `checkpoints/*.joblib`

Serialized strategy/model state.

Important checkpoint types:

- `*_fresh_start_252days_at_2020.joblib`: snapshot state built up to a chosen start point.
- `*__continued.joblib`: state after a resume/live run.

Be careful when reusing continued checkpoints: they may include an active execution state unless you pass `reset_execution_state=True`.

## Common Workflows

### Reproduce an old permissive 5-minute grid

```python
from adaptive_trade_extensions import make_ret_grid
from adaptive_reward_checkpoint_fresh import run_adaptive_reward_from_snapshot

res = run_adaptive_reward_from_snapshot(
    snapshot_path="checkpoints/TQQQ_adaptive_reward_fresh_start_252days_at_2020.joblib",
    end_time="2026-05-22",
    reset_execution_state=True,
    threshold_ret_grid_override=make_ret_grid(-0.5, 2.5, 0.05),
    output_dir="output_adaptive_reward_resumed_fresh_TQQQ_legacy_grid",
)
```

### Run with the newer percent-return grid

Omit `threshold_ret_grid_override`; the code will use the grid saved in the snapshot, or the default `0.0..25.0` grid when building a fresh snapshot.

### Change the daily dynamic threshold lookback

```python
res = run_adaptive_reward_from_snapshot(
    snapshot_path="checkpoints/TQQQ_adaptive_reward_fresh_start_252days_at_2020.joblib",
    end_time="2026-05-22",
    reset_execution_state=True,
    daily_threshold_lookback_days_override=30,
)
```

