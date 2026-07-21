from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from next_day_gate_model import (
    GATES,
    P_DAY_LOGISTIC_KWARGS,
    REWARD_COLUMNS,
    build_supervised_frame,
    feature_columns,
    replay_predictions_with_trade_signals,
)


TARGET_REWARD_COLUMNS = {
    "FORCE_BUY": "target_reward_force_buy",
    "FREE": "target_reward_free",
    "FORCE_SELL": "target_reward_force_sell",
}


@dataclass
class DailyRewardGateConfig:
    """
    Notebook-friendly config for the daily reward-gate model.

    This uses the same return-model idea as the 5-minute model, but at daily
    frequency: fit one regressor per gate reward, then choose the gate with the
    highest predicted next-day reward.
    """

    rewards_csv: str = "trades_with_gate_rewards.csv"
    trade_signals_csv: Optional[str] = None
    market_csv: str = "DataAPI/data/TQQQ_DAY.csv"
    output_dir: str = "output_next_day_gate_reward_model"
    checkpoint_dir: str = "checkpoints/next_day_gate_reward_model"
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    initial_capital: float = 100000.0
    min_train_days: int = 252
    train_lookback_days: Optional[int] = None
    retrain_every_n_days: int = 20
    model_type: str = "xgboost"
    random_state: int = 42
    n_jobs: int = 1
    checkpoint_every_n_days: int = 20
    resume_from_checkpoint: Optional[str] = None
    save_year_start_checkpoints: bool = True
    feature_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 60)
    feature_mode: str = "rich_daily"
    daily_chan_start: Optional[str] = None
    daily_chan_max_klines: int = 500
    macro_files: dict[str, str] = field(default_factory=dict)
    macro_base_dir: str = "DataAPI/data"
    reward_tie_epsilon: float = 1e-9
    fee_per_day: float = 0.0
    fee_pct: float = 0.0
    verbose: bool = True


def _default_trade_signals_csv(code: str) -> Optional[str]:
    path = Path(f"trade_signals_{str(code).lower()}.csv")
    return str(path) if path.exists() else None


def daily_reward_gate_config_from_adaptive_kwargs(
    adaptive_kwargs: dict[str, Any],
    *,
    rewards_csv: str = "trades_with_gate_rewards.csv",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "xgboost",
    min_train_days: int = 252,
    retrain_every_n_days: int = 20,
    checkpoint_every_n_days: int = 20,
    **overrides: Any,
) -> DailyRewardGateConfig:
    """Build config from existing adaptive-reward kwargs."""

    adaptive_kwargs = dict(adaptive_kwargs or {})
    code = str(adaptive_kwargs.get("code", "MODEL"))
    cfg_values = {
        "rewards_csv": rewards_csv,
        "trade_signals_csv": overrides.pop("trade_signals_csv", _default_trade_signals_csv(code)),
        "market_csv": adaptive_kwargs.get("daily_csv_path", DailyRewardGateConfig.market_csv),
        "output_dir": output_dir or f"output_next_day_gate_reward_model_{code}",
        "checkpoint_dir": checkpoint_dir or f"checkpoints/next_day_gate_reward_model_{code}",
        "start_year": start_year,
        "end_year": end_year,
        "initial_capital": float(adaptive_kwargs.get("initial_capital", DailyRewardGateConfig.initial_capital)),
        "fee_pct": float(adaptive_kwargs.get("fee_pct", 0.0)),
        "min_train_days": int(min_train_days),
        "retrain_every_n_days": int(retrain_every_n_days),
        "model_type": model_type,
        "checkpoint_every_n_days": int(checkpoint_every_n_days),
        "macro_files": dict(adaptive_kwargs.get("macro_files") or {}),
        "daily_chan_start": adaptive_kwargs.get("daily_chan_start"),
        "daily_chan_max_klines": int(adaptive_kwargs.get("daily_chan_max_klines", 500)),
    }
    cfg_values.update(overrides)
    for old_key in P_DAY_LOGISTIC_KWARGS:
        cfg_values.pop(old_key, None)
    return DailyRewardGateConfig(**cfg_values)


def make_reward_regressor(config: DailyRewardGateConfig, seed_offset: int = 0) -> Pipeline:
    """Create one numeric reward regressor."""

    model_type = str(config.model_type).lower()
    seed = int(config.random_state) + int(seed_offset)
    if model_type in {"xgboost", "xgb", "xgb_regressor"}:
        reg = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=int(config.n_jobs),
            objective="reg:squarederror",
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)])
    if model_type in {"extra_trees", "extratrees", "et"}:
        reg = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=int(config.n_jobs),
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)])
    if model_type in {"random_forest", "rf"}:
        reg = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=int(config.n_jobs),
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)])
    raise ValueError("Unsupported model_type. Use xgboost, extra_trees, or random_forest.")


def _fit_reward_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    config: DailyRewardGateConfig,
) -> Optional[dict[str, Pipeline]]:
    """Fit one model per daily gate reward."""

    if len(train_df) < int(config.min_train_days):
        return None
    models: dict[str, Pipeline] = {}
    for offset, gate in enumerate(GATES):
        target_col = TARGET_REWARD_COLUMNS[gate]
        sub = train_df.dropna(subset=[target_col]).copy()
        if len(sub) < int(config.min_train_days):
            return None
        model = make_reward_regressor(config, seed_offset=offset)
        y = pd.to_numeric(sub[target_col], errors="coerce").astype(float)
        model.fit(sub[feature_cols], y)
        models[gate] = model
    return models


def _predict_gate_rewards(models: Optional[dict[str, Pipeline]], x: pd.DataFrame) -> dict[str, float]:
    """Predict next-day reward for every gate."""

    if not models:
        return {gate: 0.0 for gate in GATES}
    out = {}
    for gate in GATES:
        model = models.get(gate)
        out[gate] = float(model.predict(x)[0]) if model is not None else 0.0
    return out


def _choose_gate(predicted_rewards: dict[str, float]) -> str:
    """Choose the highest predicted reward, preferring FREE on exact ties."""

    order = {"FREE": 0, "FORCE_BUY": 1, "FORCE_SELL": 2}
    return max(GATES, key=lambda gate: (float(predicted_rewards.get(gate, 0.0)), -order[gate]))


def _reward_for_gate(row: pd.Series, gate: str) -> float:
    col = f"target_{REWARD_COLUMNS.get(str(gate), 'reward_free')}"
    val = pd.to_numeric(row.get(col, 0.0), errors="coerce")
    return 0.0 if pd.isna(val) else float(val)


def _checkpoint_payload(
    *,
    config: DailyRewardGateConfig,
    feature_cols: list[str],
    all_results: list[dict[str, Any]],
    completed_years: list[int],
    current_year: Optional[int],
    next_row_index: int,
    models: Optional[dict[str, Pipeline]],
) -> dict[str, Any]:
    return {
        "schema": "next_day_gate_reward_model_checkpoint_v1",
        "config": asdict(config),
        "feature_cols": feature_cols,
        "all_results": all_results,
        "completed_years": completed_years,
        "current_year": current_year,
        "next_row_index": int(next_row_index),
        "models": models,
        "saved_at": pd.Timestamp.now().isoformat(),
    }


def _save_checkpoint(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    joblib.dump(payload, tmp)
    try:
        os.replace(tmp, path)
        return str(path)
    except OSError:
        fallback = path.with_name(f"{path.stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}{path.suffix}")
        os.replace(tmp, fallback)
        return str(fallback)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if payload.get("schema") != "next_day_gate_reward_model_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint schema in {path}")
    return payload


def _write_outputs(
    output_dir: str | Path,
    results: list[dict[str, Any]],
    config: DailyRewardGateConfig,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(results)
    daily_log = pd.DataFrame()
    executed_trades = pd.DataFrame()

    if config.trade_signals_csv and not pred.empty:
        daily_log, executed_trades = replay_predictions_with_trade_signals(
            predictions=pred,
            trade_signals_csv=config.trade_signals_csv,
            fee_pct=float(config.fee_pct),
            initial_capital=float(config.initial_capital),
        )
        if not daily_log.empty:
            pred = pred.merge(
                daily_log[
                    [
                        "target_date",
                        "start_capital",
                        "open_equity",
                        "end_capital",
                        "daily_pnl",
                        "cash",
                        "position_qty",
                        "trade_count",
                        "five_min_signal_trade_count",
                    ]
                ],
                on="target_date",
                how="left",
            )

    pred_path = out_dir / "daily_predictions.csv"
    pred.to_csv(pred_path, index=False)
    daily_log_path = out_dir / "daily_log.csv"
    daily_log.to_csv(daily_log_path, index=False)
    executed_trades_path = out_dir / "executed_5m_trades.csv"
    executed_trades.to_csv(executed_trades_path, index=False)

    if pred.empty:
        summary = pd.DataFrame()
    elif config.trade_signals_csv and "end_capital" in pred.columns:
        summary = pred.groupby("year", as_index=False).agg(
            first_target_date=("target_date", "first"),
            last_target_date=("target_date", "last"),
            trading_days=("target_date", "count"),
            final_capital=("end_capital", "last"),
            total_reward=("daily_pnl", "sum"),
            accuracy=("is_correct", "mean"),
            oracle_reward=("oracle_reward", "sum"),
            executed_trades=("trade_count", "sum"),
            executed_5m_signal_trades=("five_min_signal_trade_count", "sum"),
        )
        summary["initial_capital"] = float(config.initial_capital)
        summary["return_pct"] = summary["final_capital"] / float(config.initial_capital) - 1.0
        summary["oracle_final_capital"] = float(config.initial_capital) + summary["oracle_reward"]
    else:
        summary = pred.groupby("year", as_index=False).agg(
            first_target_date=("target_date", "first"),
            last_target_date=("target_date", "last"),
            trading_days=("target_date", "count"),
            final_capital=("capital", "last"),
            total_reward=("realized_reward", "sum"),
            accuracy=("is_correct", "mean"),
            oracle_reward=("oracle_reward", "sum"),
        )
        summary["initial_capital"] = float(config.initial_capital)
        summary["return_pct"] = summary["final_capital"] / float(config.initial_capital) - 1.0
        summary["oracle_final_capital"] = float(config.initial_capital) + summary["oracle_reward"]

    summary_path = out_dir / "yearly_summary.csv"
    summary.to_csv(summary_path, index=False)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")
    return {
        "predictions": pred,
        "daily_log": daily_log,
        "executed_trades": executed_trades,
        "summary": summary,
        "predictions_path": str(pred_path),
        "daily_log_path": str(daily_log_path),
        "executed_trades_path": str(executed_trades_path),
        "summary_path": str(summary_path),
        "config_path": str(config_path),
    }


def run_daily_reward_gate_experiment(config: Optional[DailyRewardGateConfig] = None) -> dict[str, Any]:
    """
    Run a no-leak next-day simulation with three daily reward regressors.

    Row D uses features known after D close. The predicted gate is scored on the
    next available trading day. Training rows always have target dates earlier
    than the row being predicted.
    """

    config = config or DailyRewardGateConfig()
    verbose = bool(getattr(config, "verbose", True))
    if verbose:
        source = config.trade_signals_csv or config.rewards_csv
        print(
            f"[START] daily reward gate model source={source} "
            f"market={config.market_csv} years={config.start_year}-{config.end_year}"
        )
    df = build_supervised_frame(config)
    cols = feature_columns(df)
    if verbose:
        date_min = pd.to_datetime(df["target_date"]).min().date()
        date_max = pd.to_datetime(df["target_date"]).max().date()
        print(
            f"[DATA] supervised_rows={len(df):,} target_range={date_min}..{date_max} "
            f"features={len(cols):,}"
        )
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    main_checkpoint = checkpoint_dir / "latest.joblib"
    latest_checkpoint_path = str(main_checkpoint)

    all_results: list[dict[str, Any]] = []
    completed_years: list[int] = []
    resume_year = None
    resume_row_index = 0
    models: Optional[dict[str, Pipeline]] = None

    if config.resume_from_checkpoint:
        payload = _load_checkpoint(config.resume_from_checkpoint)
        all_results = list(payload.get("all_results", []))
        completed_years = [int(y) for y in payload.get("completed_years", [])]
        resume_year = payload.get("current_year")
        resume_row_index = int(payload.get("next_row_index", 0))
        models = payload.get("models")
        if verbose:
            print(
                f"[RESUME] checkpoint={config.resume_from_checkpoint} "
                f"current_year={resume_year} next_row_index={resume_row_index} "
                f"existing_results={len(all_results):,}"
            )

    years = sorted(df["year"].unique().tolist())
    if config.start_year is not None:
        years = [y for y in years if y >= int(config.start_year)]
    if resume_year is not None:
        years = [y for y in years if y >= int(resume_year)]
    if config.end_year is not None:
        years = [y for y in years if y <= int(config.end_year)]
    if not years:
        raise ValueError("No target years are available for the requested start/end range.")

    for year in years:
        if year in completed_years:
            if verbose:
                print(f"[YEAR {year}] already completed, skipping")
            continue

        year_df = df[df["year"] == year].copy().reset_index(drop=False).rename(columns={"index": "_global_index"})
        if year_df.empty:
            completed_years.append(year)
            continue
        if verbose:
            first_day = pd.Timestamp(year_df.iloc[0]["target_date"]).date()
            last_day = pd.Timestamp(year_df.iloc[-1]["target_date"]).date()
            print(f"[YEAR {year}] rows={len(year_df):,} target_range={first_day}..{last_day}")

        if config.save_year_start_checkpoints and resume_year is None:
            train_start_df = df[df["target_date"] < year_df.iloc[0]["target_date"]]
            if config.train_lookback_days is not None:
                train_start_df = train_start_df.tail(int(config.train_lookback_days))
            year_start_models = _fit_reward_models(train_start_df, cols, config)
            year_start_path = _save_checkpoint(
                checkpoint_dir / f"year_start_{year}.joblib",
                _checkpoint_payload(
                    config=config,
                    feature_cols=cols,
                    all_results=all_results,
                    completed_years=completed_years,
                    current_year=year,
                    next_row_index=0,
                    models=year_start_models,
                ),
            )
            if verbose:
                print(
                    f"[CHECKPOINT] year_start_{year} train_rows={len(train_start_df):,} "
                    f"model={'yes' if year_start_models is not None else 'no'} path={year_start_path}"
                )

        start_i = resume_row_index if resume_year == year else 0
        capital = float(config.initial_capital)
        if start_i > 0 and all_results:
            prior_for_year = [r for r in all_results if int(r["year"]) == int(year)]
            if prior_for_year:
                capital = float(prior_for_year[-1]["capital"])

        models = models if resume_year == year and models is not None else None
        days_since_fit = int(config.retrain_every_n_days)

        for local_i in range(start_i, len(year_df)):
            row = year_df.iloc[local_i]
            train_df = df[df["target_date"] < row["target_date"]]
            if config.train_lookback_days is not None:
                train_df = train_df.tail(int(config.train_lookback_days))
            if models is None or days_since_fit >= int(config.retrain_every_n_days):
                fitted = _fit_reward_models(train_df, cols, config)
                if fitted is not None:
                    models = fitted
                    days_since_fit = 0
                    if verbose:
                        print(
                            f"[TRAIN] year={year} row={local_i + 1}/{len(year_df)} "
                            f"target={pd.Timestamp(row['target_date']).date()} train_rows={len(train_df):,}"
                        )
                elif verbose and models is None and local_i == start_i:
                    print(
                        f"[TRAIN] year={year} waiting for min_train_days={config.min_train_days}; "
                        f"available={len(train_df):,}, using FREE fallback"
                    )

            x = pd.DataFrame([row[cols].to_dict()])
            pred_rewards = _predict_gate_rewards(models, x)
            predicted_gate = "FREE" if models is None else _choose_gate(pred_rewards)

            reward = _reward_for_gate(row, predicted_gate) - float(config.fee_per_day)
            oracle_gate = str(row["target_gate"])
            oracle_reward = _reward_for_gate(row, oracle_gate)
            capital += float(reward)
            result = {
                "decision_date": pd.Timestamp(row["date"]).date().isoformat(),
                "target_date": pd.Timestamp(row["target_date"]).date().isoformat(),
                "year": int(year),
                "predicted_gate": predicted_gate,
                "actual_best_gate": oracle_gate,
                "is_correct": bool(predicted_gate == oracle_gate),
                "realized_reward": float(reward),
                "oracle_reward": float(oracle_reward),
                "capital": float(capital),
                "pred_reward_force_buy": float(pred_rewards["FORCE_BUY"]),
                "pred_reward_free": float(pred_rewards["FREE"]),
                "pred_reward_force_sell": float(pred_rewards["FORCE_SELL"]),
                "target_reward_force_buy": float(row["target_reward_force_buy"]),
                "target_reward_free": float(row["target_reward_free"]),
                "target_reward_force_sell": float(row["target_reward_force_sell"]),
                "target_day_open": float(row["target_day_open"]),
                "target_day_close": float(row["target_day_close"]),
                "train_rows": int(len(train_df)),
                "model_available": bool(models is not None),
            }
            all_results.append(result)
            days_since_fit += 1

            if config.checkpoint_every_n_days > 0 and (local_i + 1) % int(config.checkpoint_every_n_days) == 0:
                latest_checkpoint_path = _save_checkpoint(
                    main_checkpoint,
                    _checkpoint_payload(
                        config=config,
                        feature_cols=cols,
                        all_results=all_results,
                        completed_years=completed_years,
                        current_year=year,
                        next_row_index=local_i + 1,
                        models=models,
                    ),
                )
                if verbose:
                    print(
                        f"[CHECKPOINT] year={year} next_row={local_i + 1}/{len(year_df)} "
                        f"path={latest_checkpoint_path}"
                    )

        completed_years.append(year)
        resume_year = None
        resume_row_index = 0
        models = None
        latest_checkpoint_path = _save_checkpoint(
            main_checkpoint,
            _checkpoint_payload(
                config=config,
                feature_cols=cols,
                all_results=all_results,
                completed_years=completed_years,
                current_year=None,
                next_row_index=0,
                models=None,
            ),
        )
        if verbose:
            year_results = [r for r in all_results if int(r["year"]) == int(year)]
            if year_results:
                print(
                    f"[YEAR {year}] done rows={len(year_results):,} "
                    f"final_capital={float(year_results[-1]['capital']):,.2f} "
                    f"checkpoint={latest_checkpoint_path}"
                )

    outputs = _write_outputs(config.output_dir, all_results, config)
    if verbose:
        print(f"[OUTPUT] predictions={outputs['predictions_path']}")
        print(f"[OUTPUT] summary={outputs['summary_path']}")
    outputs["checkpoint_path"] = latest_checkpoint_path
    outputs["feature_columns"] = cols
    return outputs


def run_daily_reward_gate_from_trade_signals(
    *,
    trade_signals_csv: str = "trade_signals_tqqq.csv",
    market_csv: str = "DataAPI/data/TQQQ_DAY.csv",
    start_year: Optional[int] = 2020,
    end_year: Optional[int] = 2026,
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    model_type: str = "xgboost",
    min_train_days: int = 252,
    train_lookback_days: Optional[int] = None,
    retrain_every_n_days: int = 20,
    checkpoint_every_n_days: int = 20,
    output_dir: str = "output_next_day_gate_reward_model_TQQQ_signals",
    checkpoint_dir: str = "checkpoints/next_day_gate_reward_model_TQQQ_signals",
    feature_mode: str = "rich_daily",
    daily_chan_start: Optional[str] = None,
    daily_chan_max_klines: int = 500,
    macro_files: Optional[dict[str, str]] = None,
    macro_base_dir: str = "DataAPI/data",
    n_jobs: int = 1,
    random_state: int = 42,
    resume_from_checkpoint: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the daily gate reward model from precomputed 5-minute trade signals.

    This is the fast path when `trade_signals_csv` already contains the FREE-gate
    5-minute model output. The function does not train or rerun the 5-minute
    model; it uses the saved trades to build daily gate rewards, then trains
    three daily reward regressors:

    - FORCE_BUY reward model
    - FREE reward model
    - FORCE_SELL reward model

    The selected daily gate is the gate with the highest predicted next-day
    reward.
    """

    cfg = DailyRewardGateConfig(
        trade_signals_csv=trade_signals_csv,
        market_csv=market_csv,
        start_year=start_year,
        end_year=end_year,
        initial_capital=float(initial_capital),
        fee_pct=float(fee_pct),
        model_type=model_type,
        min_train_days=int(min_train_days),
        train_lookback_days=train_lookback_days,
        retrain_every_n_days=int(retrain_every_n_days),
        checkpoint_every_n_days=int(checkpoint_every_n_days),
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        feature_mode=feature_mode,
        daily_chan_start=daily_chan_start,
        daily_chan_max_klines=int(daily_chan_max_klines),
        macro_files=dict(macro_files or {}),
        macro_base_dir=macro_base_dir,
        n_jobs=int(n_jobs),
        random_state=int(random_state),
        resume_from_checkpoint=resume_from_checkpoint,
        verbose=bool(verbose),
    )
    return run_daily_reward_gate_experiment(cfg)


def _combine_trade_signal_csvs(
    trade_signal_csvs: list[str],
    output_path: str | Path,
    verbose: bool = True,
) -> str:
    """Combine multiple precomputed 5-minute trade signal files into one CSV."""

    if not trade_signal_csvs:
        raise ValueError("trade_signal_csvs must contain at least one CSV path.")

    frames = []
    for path in trade_signal_csvs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing trade signal CSV: {p}")
        df = pd.read_csv(p)
        df["_source_file"] = str(p)
        frames.append(df)
        if verbose:
            print(f"[SIGNALS] loaded {p} rows={len(df):,}")

    out = pd.concat(frames, ignore_index=True, sort=False)
    for ts_col in ["exec_ts", "ts", "timestamp", "datetime"]:
        if ts_col in out.columns:
            out["_sort_ts"] = pd.to_datetime(out[ts_col], errors="coerce")
            out = out.sort_values(["_sort_ts", "_source_file"]).drop(columns=["_sort_ts"])
            break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    if verbose:
        print(f"[SIGNALS] combined rows={len(out):,} -> {output_path}")
    return str(output_path)


def run_daily_reward_gate_from_trade_signal_files(
    *,
    trade_signal_csvs: list[str] = None,
    market_csv: str = "DataAPI/data/TQQQ_DAY.csv",
    start_year: Optional[int] = 2020,
    end_year: Optional[int] = 2026,
    initial_capital: float = 100000.0,
    fee_pct: float = 0.0,
    model_type: str = "xgboost",
    min_train_days: int = 252,
    train_lookback_days: Optional[int] = None,
    retrain_every_n_days: int = 20,
    checkpoint_every_n_days: int = 20,
    output_dir: str = "output_next_day_gate_reward_model_TQQQ_combined_signals",
    checkpoint_dir: str = "checkpoints/next_day_gate_reward_model_TQQQ_combined_signals",
    combined_trade_signals_csv: Optional[str] = None,
    feature_mode: str = "rich_daily",
    daily_chan_start: Optional[str] = None,
    daily_chan_max_klines: int = 500,
    macro_files: Optional[dict[str, str]] = None,
    macro_base_dir: str = "DataAPI/data",
    n_jobs: int = 1,
    random_state: int = 42,
    resume_from_checkpoint: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run from multiple precomputed 5-minute signal files.

    Use this when one file contains build/warmup signals, for example 2012-2019,
    and another contains the simulation-era FREE-gate 5-minute model signals,
    for example 2020 onward.
    """

    trade_signal_csvs = trade_signal_csvs or ["trade_signals_build_tqqq.csv", "trade_signals_tqqq.csv"]
    combined_path = combined_trade_signals_csv or str(Path(output_dir) / "combined_trade_signals.csv")
    combined_path = _combine_trade_signal_csvs(list(trade_signal_csvs), combined_path, verbose=bool(verbose))
    return run_daily_reward_gate_from_trade_signals(
        trade_signals_csv=combined_path,
        market_csv=market_csv,
        start_year=start_year,
        end_year=end_year,
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        model_type=model_type,
        min_train_days=min_train_days,
        train_lookback_days=train_lookback_days,
        retrain_every_n_days=retrain_every_n_days,
        checkpoint_every_n_days=checkpoint_every_n_days,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        feature_mode=feature_mode,
        daily_chan_start=daily_chan_start,
        daily_chan_max_klines=daily_chan_max_klines,
        macro_files=macro_files,
        macro_base_dir=macro_base_dir,
        n_jobs=n_jobs,
        random_state=random_state,
        resume_from_checkpoint=resume_from_checkpoint,
        verbose=verbose,
    )


def parse_args() -> DailyRewardGateConfig:
    parser = argparse.ArgumentParser(description="Train daily gate reward regressors.")
    parser.add_argument("--rewards-csv", default=DailyRewardGateConfig.rewards_csv)
    parser.add_argument("--trade-signals-csv", default=None)
    parser.add_argument("--market-csv", default=DailyRewardGateConfig.market_csv)
    parser.add_argument("--output-dir", default=DailyRewardGateConfig.output_dir)
    parser.add_argument("--checkpoint-dir", default=DailyRewardGateConfig.checkpoint_dir)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--initial-capital", type=float, default=DailyRewardGateConfig.initial_capital)
    parser.add_argument("--min-train-days", type=int, default=DailyRewardGateConfig.min_train_days)
    parser.add_argument("--train-lookback-days", type=int, default=None)
    parser.add_argument("--retrain-every-n-days", type=int, default=DailyRewardGateConfig.retrain_every_n_days)
    parser.add_argument("--model-type", default=DailyRewardGateConfig.model_type)
    parser.add_argument("--random-state", type=int, default=DailyRewardGateConfig.random_state)
    parser.add_argument("--n-jobs", type=int, default=DailyRewardGateConfig.n_jobs)
    parser.add_argument("--checkpoint-every-n-days", type=int, default=DailyRewardGateConfig.checkpoint_every_n_days)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--feature-mode", default=DailyRewardGateConfig.feature_mode)
    parser.add_argument("--fee-pct", type=float, default=DailyRewardGateConfig.fee_pct)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["verbose"] = not bool(values.pop("quiet"))
    return DailyRewardGateConfig(**values)


if __name__ == "__main__":
    result = run_daily_reward_gate_experiment(parse_args())
    print("predictions:", result["predictions_path"])
    print("summary:", result["summary_path"])
    print("checkpoint:", result["checkpoint_path"])
