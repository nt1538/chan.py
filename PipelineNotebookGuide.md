# Pipeline Notebook Guide

This file organizes the `Pipeline*.ipynb` notebooks by role, maturity, and likely output families.

## Recommended Working Structure

### 1. Canonical Current Path

These are the notebooks that best represent the current main trading pipeline:

- `PipelineCurrent.ipynb`
  - Best short-form snapshot of the current checkpoint/resume trading system.
  - Focus: daily probability gate + 5m Chan/XGB execution.
  - Includes macro/index features and checkpoint/resume runs.
  - Strongest match to recent output folders such as:
    - `output_dailyprob_gated_5m_xgb_QQQ_macro_us2`
    - `output_dailyprob_gated_5m_xgb_SPY_macro_us2`
    - `output_dailyprob_gated_5m_xgb_QQQ_noMacro_us2`
    - `output_dailyprob_gated_5m_xgb_QQQ_noMacro_2016-2024Cumulative`
    - `output_resumed_QQQ_macro`
    - `output_resumed_SPY_macro`

- `PipelineCode.ipynb`
  - Broad implementation notebook that appears to be the main engineering workbench.
  - Contains the reusable helper stack:
    - daily feature creation
    - macro feature loading
    - daily probability model
    - 5m XGB model
    - execution engine
    - checkpoint/resume support
  - Best notebook to mine when you want the full code path and the evolution from earlier versions.
  - Closely related to `pipelineCurrent.py`.

### 2. Daily-Only Research Branch

- `PipelineDaily.ipynb`
  - Separate research direction.
  - Focus: unified daily instability probability, pooled SPY/QQQ training, and LinUCB daily gate.
  - This is not the main intraday execution notebook.
  - It is the clearest expression of your higher-level idea for the daily regime layer.
  - Likely output family:
    - `output_daily_rl_decisions_QQQ`

### 3. Exact Real-Time / Leak-Free RL Branch

- `PipelineRL.ipynb`
  - Real-time exact walk-forward branch.
  - Adds stable checkpoint save/resume and explicit leakage-free processing.
  - Appears tuned around QQQ runs over 2019 onward.
  - Likely output family:
    - `output_QQQ_realtime_exact_2019_2021`
    - `output_QQQ_realtime_exact_2019_2026`

- `PipelineRL2.ipynb`
  - Same family as `PipelineRL.ipynb`, but expanded to older history.
  - Appears to test the same exact walk-forward architecture over earlier years.
  - Likely output family:
    - `output_QQQ_realtime_exact_2008_2017`
    - `output_QQQ_realtime_exact_2008_2026`

### 4. Model/Deployment Transition Notebook

- `PipelineModel.ipynb`
  - Transitional notebook between experimentation and deployable pipeline.
  - Adds stronger checkpoint/deploy/resume framing.
  - Includes:
    - train/deploy split thinking
    - bundle checkpoints
    - resumed runs
    - combined daily LR + bandit + 5m XGB experiments
  - This looks like a major milestone notebook rather than the cleanest current source.
  - Likely output family:
    - `output_dailyprob_gated_5m_xgb_QQQ_2025_new_with_checkpointing`
    - `output_dailyprob_gated_5m_xgb_QQQ_2026_new_with_checkpointing`
    - `output_resumed_QQQ_2026`
    - `output_daily_bandit_5m_xgb_adaptive*`

### 5. Historical Idea Dump / Archive Notebook

- `Pipeline.ipynb`
  - Earliest large research notebook.
  - Contains repeated function definitions, multiple generations of logic, and overlapping experiments.
  - Valuable as project memory, but not a good source of truth for current code.
  - Best treated as an archive of ideas and earlier architecture.
  - Likely output families from older experiments:
    - `output_chain_maturity_two_sided`
    - `output_regime_thresholds_no_lock*`
    - `output_no_maturity_daily_gate*`
    - older `output/chan_xgb_*`

## Recommended Source-of-Truth Hierarchy

If you want one mental model for where to look first:

1. `PipelineCurrent.ipynb`
   Use this for the latest practical end-to-end run flow.

2. `PipelineCode.ipynb`
   Use this for the fuller code base and helper definitions behind the current flow.

3. `pipelineCurrent.py`
   Use this as the exported script version of the current pipeline logic.

4. `PipelineDaily.ipynb`
   Use this when thinking about the daily gate as its own research problem.

5. `PipelineRL.ipynb` and `PipelineRL2.ipynb`
   Use these for exact walk-forward and bandit/realtime experiments.

6. `PipelineModel.ipynb`
   Use this to recover transition ideas around checkpointing and deployment.

7. `Pipeline.ipynb`
   Treat as archive/reference only.

## What Each Notebook Is Really Doing

### `PipelineCurrent.ipynb`

Current mainline:

- builds a daily model
- gates intraday trading
- uses macro/index features
- supports checkpoint/resume
- runs recent SPY/QQQ scenarios

This is the best notebook to keep iterating if the main goal is improving the active trading pipeline.

### `PipelineCode.ipynb`

Engineering notebook:

- contains the major helper functions
- includes older and newer versions in one place
- mixes implementation with run cells

This is the best notebook to refactor from if you want cleaner code extraction.

### `PipelineDaily.ipynb`

Daily-state notebook:

- focuses on the meaning of one daily probability
- treats the daily layer as a regime/risk decision engine
- uses a contextual bandit to choose `HOLD`, `FREE`, or `RISK_OFF`

This notebook contains some of the clearest strategic thinking in the repo.

### `PipelineModel.ipynb`

Checkpointing/deployment notebook:

- turns the system into something resumable
- experiments with train-period vs deploy-period separation
- mixes several related ideas in one notebook

Important for history, but less clean than `PipelineCurrent.ipynb` as a present-day control center.

### `PipelineRL.ipynb` and `PipelineRL2.ipynb`

Realtime/exact branch:

- stricter walk-forward framing
- more explicit leakage control
- own checkpoint system
- mostly separate from the current dailyprob-gated output family

These should be treated as a side branch, not mixed into the main notebook mentally.

### `Pipeline.ipynb`

Project-memory notebook:

- technical writeup
- multiple generations of code
- duplicated functions
- older threshold and regime experiments

Very useful for thought process, not good for active maintenance.

## Output Folder Map

### Main current pipeline family

Usually driven by:

- `PipelineCurrent.ipynb`
- `PipelineCode.ipynb`
- parts of `PipelineModel.ipynb`

Typical output prefixes:

- `output_dailyprob_gated_5m_xgb_*`
- `output_resumed_*`

### Daily-only bandit family

Usually driven by:

- `PipelineDaily.ipynb`

Typical output prefixes:

- `output_daily_rl_decisions_*`

### Exact realtime walk-forward family

Usually driven by:

- `PipelineRL.ipynb`
- `PipelineRL2.ipynb`

Typical output prefixes:

- `output_QQQ_realtime_exact_*`

### Older historical experiment family

Usually driven by:

- `Pipeline.ipynb`

Typical output prefixes:

- `output_chain_maturity_*`
- `output_regime_thresholds_*`
- `output_no_maturity_*`
- `output/chan_xgb_*`

## Practical Cleanup Recommendation

Without moving files yet, treat the notebooks like this:

- Active:
  - `PipelineCurrent.ipynb`
  - `PipelineCode.ipynb`

- Active research branch:
  - `PipelineDaily.ipynb`

- Side branch:
  - `PipelineRL.ipynb`
  - `PipelineRL2.ipynb`

- Historical milestone:
  - `PipelineModel.ipynb`

- Archive:
  - `Pipeline.ipynb`

## Suggested Next Refactor

If you want to simplify the project further, the cleanest next step is:

1. keep `PipelineCurrent.ipynb` as the run notebook
2. extract stable helpers from `PipelineCode.ipynb` into one or more `.py` modules
3. keep `PipelineDaily.ipynb` separate as the daily-regime research branch
4. leave `PipelineRL.ipynb` and `PipelineRL2.ipynb` as their own experimental lane
5. stop adding new logic to `Pipeline.ipynb`

## My Recommendation

For day-to-day work, think of the notebook stack like this:

- `PipelineCurrent.ipynb` = current driver
- `PipelineCode.ipynb` = code warehouse
- `PipelineDaily.ipynb` = future daily-regime idea branch
- `PipelineRL.ipynb` / `PipelineRL2.ipynb` = realtime exact branch
- `PipelineModel.ipynb` = checkpointing milestone
- `Pipeline.ipynb` = archive of project thinking
