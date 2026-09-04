"""Configuration for dataset construction and model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ForecastConfig:
    """All paths and reproducibility controls for an experiment."""

    input_csv: str
    output_dir: str = "outputs/kline_2day_forecaster"
    symbol: str = "TQQQ"
    bars_per_day: int = 78
    horizon_days: int = 2
    # ``extremes`` preserves the original targets; ``exact_return`` predicts
    # close(t + horizon_days trading sessions) / close(t) - 1.
    target_mode: str = "extremes"
    direction_probability_threshold: float = 0.5
    # Restrict train/validation/test observations after features and labels are
    # built from the complete bar history.
    regular_session_only: bool = False
    regular_session_start: str = "09:30"
    regular_session_end: str = "15:55"
    # None keeps every bar. Typical de-overlapping values are 30 or 60.
    sample_every_minutes: int | None = None
    train_start_date: str | None = None
    train_end_date: str | None = None
    validation_start_date: str | None = None
    validation_end_date: str | None = None
    test_start_date: str | None = None
    test_end_date: str | None = None
    test_fraction: float = 0.20
    validation_fraction: float = 0.10
    random_seed: int = 42
    n_estimators: int = 300
    n_jobs: int = 1
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: float = 1.0
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    lstm_sequence_length: int = 78
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.10
    lstm_epochs: int = 10
    lstm_batch_size: int = 256
    lstm_learning_rate: float = 0.001
    lstm_train_stride: int = 1
    # Caps input values per recurrent batch; the effective batch is reduced
    # automatically for long sequences/wide feature sets to prevent OOM.
    lstm_max_batch_feature_values: int = 2_000_000
    model_types: tuple[str, ...] = ("xgboost", "lstm")
    prediction_model: str = "ensemble"
    ensemble_xgboost_weight: float = 0.5
    min_history_bars: int = 200
    # Five-minute horizons: 78 bars/session. These defaults span intraday,
    # 1/2/3/5-session and 10-session context without redefining legacy names.
    technical_windows: tuple[int, ...] = (
        2, 3, 5, 10, 12, 20, 26, 39, 50, 78, 100, 156, 200, 234, 390, 780,
    )
    technical_rsi_periods: tuple[int, ...] = (14, 39, 78, 156, 390)
    technical_atr_periods: tuple[int, ...] = (14, 39, 78, 156, 390)
    technical_macd_periods: tuple[tuple[int, int, int], ...] = (
        (12, 26, 9), (39, 78, 20), (78, 156, 39), (156, 390, 78),
    )
    technical_stochastic_period: int = 14
    technical_stochastic_d_period: int = 3
    technical_williams_period: int = 14
    technical_cci_period: int = 20
    technical_cci_constant: float = 0.015
    technical_mfi_period: int = 14
    technical_bollinger_period: int = 20
    technical_bollinger_std_multiplier: float = 2.0
    technical_dmi_period: int = 14
    technical_adx_period: int = 14
    technical_rsl_period: int = 14
    technical_ppo_periods: tuple[int, int] = (12, 26)
    technical_keltner_ema_period: int = 20
    technical_keltner_atr_period: int = 14
    technical_keltner_atr_multiplier: float = 2.0
    technical_starc_sma_period: int = 5
    technical_starc_atr_period: int = 10
    technical_starc_atr_multiplier: float = 2.0
    technical_tsi_periods: tuple[int, int] = (25, 13)
    technical_ultimate_periods: tuple[int, int, int] = (7, 14, 28)
    technical_ultimate_weights: tuple[float, float, float] = (4.0, 2.0, 1.0)
    technical_obv_z_period: int = 78
    technical_obv_z_min_periods: int = 20
    technical_ad_line_z_period: int = 78
    technical_ad_line_z_min_periods: int = 20
    technical_cmf_period: int = 20
    technical_doji_body_ratio: float = 0.10
    technical_pattern_shadow_body_ratio: float = 2.0
    technical_first_hour_minutes: int = 60
    technical_relative_volume_lookback_days: int = 20
    technical_relative_volume_min_days: int = 5
    # Controls the model inputs named ``tech_*``. Chan's own technical
    # indicators (``chan_tech_*``) are controlled separately by enable_chan.
    use_standard_technical_features: bool = True
    # Controls all Chan buy/sell-point inputs whose feature name contains
    # ``bsp``. Other Chan structure and Chan technical features remain enabled.
    use_chan_bsp_features: bool = True
    enable_chan: bool = True
    chan_window_bars: int = 500
    save_enriched_csv: bool = False
    verbose: bool = True
    progress_every_rows: int = 5_000

    @property
    def horizon_bars(self) -> int:
        """Legacy estimate retained for compatibility; labels use trading dates."""
        return int(self.bars_per_day * self.horizon_days)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
