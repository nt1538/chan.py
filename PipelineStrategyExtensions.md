# Pipeline Strategy Extensions

This note is based on the current `PipelineCode.ipynb` path, especially the latest SPY run block with:

- `p_buy_level=0.20`
- `p_sell_level=0.30`
- live 5m threshold optimization already enabled through `threshold_window_days`

## Current Situation

The latest `PipelineCode.ipynb` run uses:

- fixed daily probability cutoffs for the daily gate
- adaptive 5m return thresholds for the intraday XGBoost entries/exits

That means:

- the daily layer is still static
- the 5m layer is already dynamic

This is why the daily layer can underperform the more powerful oracle-like behavior you explored in `PipelineRL.ipynb`.

## Answers to Your Questions

### 1. Can the daily model use a moving threshold?

Yes.

The cleanest version is a rolling threshold optimizer over recent daily history:

- input:
  - recent `p_day`
  - realized reward for `FORCE_BUY`
  - realized reward for `FREE`
  - realized reward for `FORCE_SELL`
- search:
  - candidate `(p_buy_level, p_sell_level)` pairs
- objective:
  - maximize realized reward over the recent window
  - optionally penalize excessive gate switching

This is implemented as a starter in:

- `adaptive_trade_extensions.py`

Main entry point:

- `select_moving_daily_thresholds(...)`

### 2. Can RL optimize the `0.20 / 0.30` threshold selection?

Yes, and this is more aligned with your `PipelineRL` direction.

Instead of bandit actions being:

- `FORCE_BUY`
- `FREE`
- `FORCE_SELL`

the action can become:

- choose a threshold pair such as `(0.15, 0.30)` or `(0.25, 0.40)`

Then the chosen pair maps `p_day` into the gate for that day.

This is also implemented as a starter in:

- `adaptive_trade_extensions.py`

Main class:

- `ThresholdPairBandit`

This is likely the safest first RL extension because:

- it keeps your current daily probability model
- it only learns the threshold policy
- it is lower-risk than replacing the whole gate at once

## News-Shock Override Model

Yes, a separate model is the right shape for this.

It should not be folded into the normal daily/5m models because:

- breaking news is exogenous
- the feature timing is different
- the response logic is closer to risk override than prediction

Recommended design:

1. External news feed or event source
2. Event scorer:
   - `impact_score`
   - `direction_score`
   - `confidence`
3. Override layer:
   - `FORCE_EXIT_AND_HALT`
   - `FORCE_BUY_AND_HALT`
   - `FORCE_SELL_AND_HALT`
   - `NO_OVERRIDE`
4. Cooldown timer:
   - pause the normal trading loop for a fixed window

Starter shell:

- `NewsShockGuard` in `adaptive_trade_extensions.py`

This is intentionally a shell, not a live news scraper, because the repo does not currently have a realtime news source wired in.

## Interface

Starter interface file:

- `checkpoint_control_panel.py`

It provides:

- checkpoint browser
- resume-config builder
- fresh-run config builder
- news-override config tab

It currently exports configs and command previews rather than directly running your notebook functions.

That choice is intentional, because your current execution path still lives partly in notebooks, and I did not want to bind the UI to a function path that may shift while you are still iterating.

To launch it:

```bash
pip install streamlit
streamlit run checkpoint_control_panel.py
```

## Recommended Integration Order

### Phase 1

Add moving daily thresholds to the `PipelineCode.ipynb` path.

Reason:

- smallest change
- directly answers your current pain point
- easier to benchmark against the fixed `0.20 / 0.30` baseline

### Phase 2

Add threshold-pair bandit.

Reason:

- lets RL optimize threshold choice without replacing your daily probability model
- easier to debug than a fully unconstrained action policy

### Phase 3

Wire in a news-shock override feed.

Reason:

- this is a separate risk-control layer
- it should sit above the normal daily/5m pipeline

### Phase 4

Promote the interface from config-builder to live controller.

Reason:

- once the pipeline entrypoints are stable, the UI can directly call them
- at that point checkpoint resume and parameter editing become much easier to support cleanly

## Suggested Immediate Experiment

Run three versions on the same period:

1. Static daily thresholds:
   - `p_buy_level=0.20`
   - `p_sell_level=0.30`

2. Rolling daily thresholds:
   - use `select_moving_daily_thresholds(...)`

3. Threshold-pair bandit:
   - use `ThresholdPairBandit`

Compare:

- equity
- drawdown
- trade count
- percentage of days in `FORCE_BUY / FREE / FORCE_SELL`
- sensitivity during high-volatility periods

That should tell you whether the real bottleneck is:

- threshold placement
- reward design
- or the daily probability model itself
