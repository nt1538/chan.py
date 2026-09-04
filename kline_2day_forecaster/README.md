# 5-minute two-day extreme forecaster

This isolated package creates **one row for every 5-minute K-line**, retains all
source columns, adds causal technical and current Chan-state features, and trains
XGBoost baseline regressors plus a joint rolling-sequence LSTM for:

- `target_max_gain_2d`: signed return to the highest future high through two trading dates.
- `target_max_loss_2d`: signed return to the lowest future low through two trading dates.

The targets are not clipped at zero. If the whole future price range is below
the current close, maximum gain is negative; if it is entirely above the
current close, maximum loss is positive.

The current bar is excluded from both targets. With the default two-day horizon,
the window ends at the same clock time on the second subsequent trading date.
For example, Monday 10:00 runs through Wednesday 10:00. Weekends and holidays
are skipped. Splits are chronological and purged using each label's actual
horizon.

## Run

```powershell
python -m kline_2day_forecaster.cli --input-csv DataAPI/data/TQQQ_5M.csv --symbol TQQQ --output-dir outputs/tqqq_2day
```

Add `--save-enriched-csv` to persist the full row-level feature table. Use
`--no-chan` only for a fast technical-feature baseline. Outputs include the
trained `model.joblib`, exact config, feature manifest, metrics, and timestamped
test predictions.

## Jupyter notebook usage

Run the notebook from the repository root. Train the model:

```python
from kline_2day_forecaster import ForecastConfig, train_forecaster

config = ForecastConfig(
    input_csv="DataAPI/data/TQQQ_5M.csv",
    output_dir="outputs/tqqq_2day",
    symbol="TQQQ",
    train_start_date="2022-01-01",
    train_end_date="2025-12-31",
    test_start_date="2026-01-01",
    test_end_date="2026-12-31",
    enable_chan=True,
    n_estimators=300,
    lstm_sequence_length=78,
    lstm_epochs=10,
)
training_result = train_forecaster(config)
training_result["metrics"]
```

Plot the saved test results in Jupyter:

```python
from kline_2day_forecaster import plot_forecast_results

plot_paths = plot_forecast_results(
    "outputs/tqqq_2day",
    model="xgboost",
    rolling_bars=390,  # approximately five regular TQQQ sessions
    top_features=25,
    show=True,
)
```

This creates actual-versus-predicted, residual, rolling-error, and feature-
importance charts. Set ``show=False`` to save the PNG files without displaying
them in the notebook.

Create one review page per test day, with that day's five-minute candlesticks,
daily average predictions and targets, and the following day's realized range:

```python
from kline_2day_forecaster import plot_daily_forecast_review

daily_review = plot_daily_forecast_review(
    "outputs/tqqq_2day",
    model="xgboost",
    start="2021-01-01",
    end="2021-01-31",
    show=False,
    save_individual_pngs=True,
)
daily_review
```

All selected days are placed in one PDF. Use ``dates=["2021-01-04"]`` and
``show=True`` to display particular days directly in Jupyter.

Choose models directly in the function call:

```python
# XGBoost only
xgb_result = train_forecaster(config, model_types=("xgboost",), prediction_model="xgboost")

# LSTM only
lstm_result = train_forecaster(config, model_types=("lstm",), prediction_model="lstm")

# Train both and make the ensemble the default
both_result = train_forecaster(
    config,
    model_types=("xgboost", "lstm"),
    prediction_model="ensemble",
)
```

Feature families can be enabled independently in the function call:

```python
xgb_result = train_forecaster(
    config,
    model_types=("xgboost",),
    prediction_model="xgboost",
    use_standard_technical_features=False,
    use_chan_bsp_features=False,
)
```

Disabling ``use_chan_bsp_features`` excludes every ``chan_*`` input whose name
contains ``bsp`` while retaining Bi, segment, Zhongshu, time, and Chan technical
features.

Keep all three outputs while overriding each model's parameters:

```python
result = train_forecaster(
    config,
    model_types=("xgboost", "lstm"),
    prediction_model="ensemble",
    xgboost_params={
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "n_jobs": 1,
    },
    lstm_params={
        "sequence_length": 156,
        "hidden_size": 128,
        "layers": 2,
        "dropout": 0.20,
        "epochs": 20,
        "batch_size": 256,
        "learning_rate": 0.001,
        "train_stride": 1,
    },
    ensemble_xgboost_weight=0.60,
    # Limit the expensive rolling-Chan build directly in this call.
    train_start_date="2022-01-01",
    train_end_date="2025-12-31",
    test_start_date="2026-01-01",
    test_end_date="2026-12-31",
    chan_window_bars=500,
)
```

An ensemble weight of `0.60` means 60% XGBoost and 40% LSTM. All effective
parameters are stored in `config.json` and `model.joblib`.

With Chan enabled, every row rebuilds the Chan structure from at most the most
recent `chan_window_bars`. The loader automatically retains enough pre-training
rows to warm the window, while only the requested train/test dates enter model
evaluation.

Load it and predict the newest K-line:

```python
from kline_2day_forecaster import load_forecaster, predict_from_csv

model = load_forecaster("outputs/tqqq_2day/model.joblib")
latest_prediction = predict_from_csv(
    model_path="outputs/tqqq_2day/model.joblib",
    csv_path="DataAPI/data/TQQQ_5M.csv",
    latest_only=True,
    symbol="TQQQ",
    prediction_model="lstm",  # override saved default if desired
)
latest_prediction
```

Predict all supplied rows for analysis or backtesting:

```python
all_predictions = predict_from_csv(
    "outputs/tqqq_2day/model.joblib",
    "DataAPI/data/TQQQ_5M.csv",
    latest_only=False,
)
```

Provide at least 200 historical bars before the bar being predicted. A model
trained with Chan enabled must also use Chan during inference.

Prediction output contains `xgboost_*`, `lstm_*`, and equal-weight
`ensemble_*` columns. The generic `predicted_*` columns use the ensemble.

## Important assumptions

- Input bars must be ordered market bars without overnight placeholder rows.
- Prices should use one consistent adjustment convention.
- Results are predictive estimates, not guaranteed attainable trading returns;
  the labels ignore transaction costs, slippage, and execution latency.
