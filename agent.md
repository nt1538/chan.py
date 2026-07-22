# Next-Day Gate Model Format

This repository now has a reusable starting format in `next_day_gate_model.py`.

## Purpose

Predict the correct next trading day's daily gate:

- `FORCE_BUY`
- `FREE`
- `FORCE_SELL`

Training labels are built from each day's candidate gate rewards. The highest reward wins. If `FORCE_BUY` and `FORCE_SELL` have the same reward, the label falls back to `day_close >= day_open` for `FORCE_BUY`, otherwise `FORCE_SELL`.

## No-Leak Rule

Each row uses information known after the current day's close to predict the next available trading day's gate.

Example:

```text
features from 2024-01-03 close -> predict gate for 2024-01-04
2024-01-04 reward columns -> score that prediction
```

During simulation, a prediction for a target day is trained only on rows with earlier target dates.

## Notebook Usage

```python
from next_day_gate_model import NextDayGateConfig, run_next_day_gate_experiment

cfg = NextDayGateConfig(
    rewards_csv="trades_with_gate_rewards.csv",
    market_csv="DataAPI/data/TQQQ_DAY.csv",
    start_year=2020,
    end_year=2026,
    initial_capital=100000,
    model_type="extra_trees",
    min_train_days=252,
    retrain_every_n_days=20,
    checkpoint_every_n_days=20,
    output_dir="output_next_day_gate_model_TQQQ",
    checkpoint_dir="checkpoints/next_day_gate_model_TQQQ",
)

result = run_next_day_gate_experiment(cfg)
result["summary"]
```

## Reuse Adaptive Reward Kwargs

If you already have a `common_kwargs` dict for `build_adaptive_reward_snapshot`, keep it. Use the adapter below for the direct next-day gate model:

```python
from next_day_gate_model import (
    P_DAY_LOGISTIC_KWARGS,
    next_day_gate_config_from_adaptive_kwargs,
    run_next_day_gate_experiment,
)

cfg = next_day_gate_config_from_adaptive_kwargs(
    common_kwargs,
    rewards_csv="output_adaptive_reward_resumed_fresh_TQQQ_252days_at_2020_W_Bonds/trades_with_gate_rewards.csv",
    start_year=2020,
    end_year=2026,
    output_dir="output_next_day_gate_model_TQQQ_W_Bonds",
    checkpoint_dir="checkpoints/next_day_gate_model_TQQQ_W_Bonds",
    model_type="extra_trees",
)

result = run_next_day_gate_experiment(cfg)
result["summary"]
```

The adapter reuses `daily_csv_path`, `code`, `initial_capital`, and `macro_files`. It intentionally does not use the previous p-day/logistic-regression settings:

- `N_confirm`
- `min_labeled_days_to_train`
- `retrain_every_new_labels`
- `dp_lookback`
- `static_buy_level`
- `static_sell_level`
- `daily_threshold_config`

Those settings belonged to the old flow:

```text
daily features -> logistic p_day -> p thresholds -> gate
```

This file uses the direct flow:

```text
current-day market/macro features -> next-day gate
```

## Temporal Fusion Transformer

Use `model_type="tft"` to train the PyTorch temporal-fusion-style gate classifier:

```python
cfg = next_day_gate_config_from_adaptive_kwargs(
    common_kwargs,
    trade_signals_csv="trade_signals_tqqq.csv",
    start_year=2020,
    end_year=2026,
    output_dir="output_next_day_gate_model_TQQQ_TFT",
    checkpoint_dir="checkpoints/next_day_gate_model_TQQQ_TFT",
    model_type="tft",
    feature_mode="rich_daily",
    tft_sequence_length=60,
    tft_hidden_size=64,
    tft_attention_heads=4,
    tft_epochs=30,
    tft_batch_size=64,
    retrain_every_n_days=20,
)

result = run_next_day_gate_experiment(cfg)
```

`feature_mode="rich_daily"` reuses the previous daily feature stack:

- daily kline/technical features from `compute_daily_kline_features`
- macro/index features from the existing macro loader
- Chan buy/sell point context
- Chan regime and base-direction flags

The old logistic p-value fields `p`, `dp_minK`, and `dp_maxK` are included only as zero placeholders, so the TFT does not depend on the previous logistic regression p-day model.

When `trade_signals_csv` is present, the direct model does not train a new 5-minute model. It derives gate rewards from the precomputed 5-minute signal file:

```text
trade_signals_tqqq.csv
trade_signals_spy.csv
trade_signals_qqq.csv
```

The adapter automatically looks for `trade_signals_{code.lower()}.csv`. You can override it:

```python
cfg = next_day_gate_config_from_adaptive_kwargs(
    common_kwargs,
    trade_signals_csv="trade_signals_tqqq.csv",
    start_year=2020,
    end_year=2026,
)
```

Signal-mode outputs include:

- `daily_predictions.csv`: model gate decision, probabilities, reward-label fields, and replay capital columns
- `daily_log.csv`: one row per target day with start capital, open equity, end capital, daily PnL, cash, position, and trade counts
- `executed_5m_trades.csv`: the actual 5-minute signal trades that executed after applying the predicted gate, plus any open-forced gate trades
- `yearly_summary.csv`: yearly summary using actual replay end capital

## Resume Usage

```python
cfg = NextDayGateConfig(
    resume_from_checkpoint="checkpoints/next_day_gate_model_TQQQ/latest.joblib",
    output_dir="output_next_day_gate_model_TQQQ",
    checkpoint_dir="checkpoints/next_day_gate_model_TQQQ",
)

result = run_next_day_gate_experiment(cfg)
```

## CLI Usage

```powershell
python next_day_gate_model.py --start-year 2020 --end-year 2026 --model-type extra_trees
```

## Outputs

The output directory contains:

- `daily_predictions.csv`: one no-leak decision row per target trading day
- `yearly_summary.csv`: year-by-year capital, reward, accuracy, and oracle comparison
- `config.json`: the exact configuration used

The checkpoint directory contains:

- `latest.joblib`: resumable checkpoint
- `year_start_YYYY.joblib`: optional start-of-year simulation checkpoint

## Future Model Contract

Future gate models should keep the same shape:

1. Build one daily row per date.
2. Label the best next-day gate from candidate rewards.
3. Use current-day features only.
4. Simulate year by year with `initial_capital=100000` unless overridden.
5. Save `daily_predictions.csv`, `yearly_summary.csv`, `config.json`, and resumable checkpoints.
6. Expose a dataclass config plus a notebook-callable `run_*_experiment(config)` function.
