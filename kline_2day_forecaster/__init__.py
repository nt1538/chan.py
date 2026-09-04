"""Point-in-time 5-minute feature engineering and return forecasting."""

from .config import ForecastConfig
from .comparison import (
    compare_bsp_type_vs_other_klines,
    compare_chan_bsp_feature_importance,
    create_paired_bsp_model_predictions,
)
from .evaluation import (
    plot_bsp_comparison_results,
    plot_daily_forecast_review,
    plot_direction_results,
    plot_direction_accuracy_vs_market_trend,
    plot_exact_return_results,
    plot_exact_return_rolling_quality,
    plot_forecast_results,
    plot_paired_bsp_model_results,
)
from .inference import load_forecaster, predict_from_csv
from .multilevel_chan import (
    ChanLevelSpec,
    MultiLevelChanConfig,
    attach_multilevel_chan_to_bsp,
    build_multilevel_chan_features,
    default_chan_levels,
    multilevel_chan_core_feature_columns,
    resample_chan_level,
)
from .multilevel_plotting import (
    plot_multilevel_bsp_confirmation,
    plot_selected_multilevel_bsp,
)
from .multilevel_reversal_plotting import plot_multilevel_bsp_reversal_confirmation
from .multilevel_entry_evaluation import plot_multilevel_entry_quality
from .pipeline import build_dataset, train_forecaster
from .three_class_forecaster import train_three_class_forecaster
from .three_class_evaluation import (
    plot_three_class_results,
    plot_three_class_tail_quality,
)
from .tail_feature_analysis import analyze_tail_feature_profiles

__all__ = [
    "ForecastConfig", "build_dataset", "train_forecaster",
    "train_three_class_forecaster", "load_forecaster",
    "plot_three_class_results", "plot_three_class_tail_quality",
    "analyze_tail_feature_profiles",
    "predict_from_csv", "plot_forecast_results", "plot_exact_return_results",
    "plot_exact_return_rolling_quality",
    "plot_direction_results",
    "plot_direction_accuracy_vs_market_trend",
    "plot_daily_forecast_review",
    "plot_bsp_comparison_results", "compare_chan_bsp_feature_importance",
    "compare_bsp_type_vs_other_klines", "create_paired_bsp_model_predictions",
    "plot_paired_bsp_model_results",
    "ChanLevelSpec", "MultiLevelChanConfig", "default_chan_levels",
    "resample_chan_level", "build_multilevel_chan_features",
    "attach_multilevel_chan_to_bsp", "multilevel_chan_core_feature_columns",
    "plot_multilevel_bsp_confirmation",
    "plot_selected_multilevel_bsp",
    "plot_multilevel_bsp_reversal_confirmation",
    "plot_multilevel_entry_quality",
]
