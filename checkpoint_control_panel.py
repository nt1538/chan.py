from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ResumeConfig:
    """User-selected settings for resuming a saved strategy checkpoint."""
    checkpoint_path: str
    daily_csv_path: str
    k5m_csv_path: str
    end_time: str
    output_dir: str
    dp_lookback_override: Optional[int] = None
    daily_threshold_lookback_days_override: Optional[int] = None
    verbose: bool = True
    plot_from_checkpoint: bool = True


@dataclass
class FreshRunConfig:
    """User-selected settings for starting a new daily/5m pipeline run."""
    daily_csv_path: str
    k5m_csv_path: str
    code: str
    daily_chan_start: str
    accumulation_start: str
    sim_start: str
    end_time: str
    p_buy_level: float
    p_sell_level: float
    threshold_window_days: float
    threshold_min_open_signals: int
    lookahead_days_5m: float
    retrain_every_days_5m: int
    save_checkpoint_path: str
    checkpoint_every_n_days: int
    output_dir: str


def list_checkpoints(base_dir: Path) -> list[Path]:
    """Return checkpoint files sorted by most recently modified first."""
    return sorted(base_dir.rglob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)


def default_output_dir(prefix: str = "output_control_panel") -> str:
    """Build a timestamped output directory for a Streamlit-triggered run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(prefix) / ts)


def export_run_config(config: dict, target_path: Path):
    """Write a selected run configuration to JSON for reproducibility."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def render_command_preview(resume: Optional[ResumeConfig], fresh: Optional[FreshRunConfig]) -> str:
    """Render a copyable Python call preview for the selected control-panel mode."""
    if resume is not None:
        data = asdict(resume)
        return (
            "resume_from_checkpoint(\n"
            f'    checkpoint_path="{data["checkpoint_path"]}",\n'
            f'    daily_csv_path="{data["daily_csv_path"]}",\n'
            f'    k5m_csv_path="{data["k5m_csv_path"]}",\n'
            f'    end_time="{data["end_time"]}",\n'
            f'    output_dir="{data["output_dir"]}",\n'
            f'    dp_lookback_override={data["dp_lookback_override"]},\n'
            f'    daily_threshold_lookback_days_override={data["daily_threshold_lookback_days_override"]},\n'
            f'    verbose={data["verbose"]},\n'
            f'    plot_from_checkpoint={data["plot_from_checkpoint"]},\n'
            ")"
        )

    if fresh is not None:
        data = asdict(fresh)
        return (
            "run_daily_prob_then_5m_xgb_gated(\n"
            f'    daily_csv_path="{data["daily_csv_path"]}",\n'
            f'    k5m_csv_path="{data["k5m_csv_path"]}",\n'
            f'    code="{data["code"]}",\n'
            f'    daily_chan_start="{data["daily_chan_start"]}",\n'
            f'    accumulation_start="{data["accumulation_start"]}",\n'
            f'    sim_start="{data["sim_start"]}",\n'
            f'    end_time="{data["end_time"]}",\n'
            f'    p_buy_level={data["p_buy_level"]},\n'
            f'    p_sell_level={data["p_sell_level"]},\n'
            f'    threshold_window_days={data["threshold_window_days"]},\n'
            f'    threshold_min_open_signals={data["threshold_min_open_signals"]},\n'
            f'    lookahead_days_5m={data["lookahead_days_5m"]},\n'
            f'    retrain_every_days_5m={data["retrain_every_days_5m"]},\n'
            f'    save_checkpoint_path="{data["save_checkpoint_path"]}",\n'
            f'    checkpoint_every_n_days={data["checkpoint_every_n_days"]},\n'
            f'    output_dir="{data["output_dir"]}",\n'
            ")"
        )

    return ""


def main():
    """Launch the Streamlit checkpoint browser and run-configuration UI."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "This control panel uses Streamlit.\n"
            "Install it with: pip install streamlit\n"
            "Then run: streamlit run checkpoint_control_panel.py"
        ) from exc

    root = Path(__file__).resolve().parent
    checkpoints_dir = root / "checkpoints"
    checkpoint_paths = list_checkpoints(checkpoints_dir)
    checkpoint_labels = [str(p.relative_to(root)) for p in checkpoint_paths]

    st.set_page_config(page_title="Trading Checkpoint Control Panel", layout="wide")
    st.title("Trading Checkpoint Control Panel")
    st.caption("Checkpoint browser and run-config builder for the Chan daily/5m pipeline.")

    mode = st.sidebar.radio(
        "Mode",
        ["Resume from checkpoint", "Start fresh run", "News override config"],
    )

    if mode == "Resume from checkpoint":
        st.subheader("Resume from Checkpoint")
        if not checkpoint_paths:
            st.warning("No `.joblib` checkpoints found under `checkpoints/`.")
            return

        selected_label = st.selectbox("Checkpoint", checkpoint_labels)
        selected_path = checkpoint_paths[checkpoint_labels.index(selected_label)]

        daily_csv_path = st.text_input("Daily CSV", "DataAPI/data/SPY_DAY.csv")
        k5m_csv_path = st.text_input("5m CSV", "DataAPI/data/SPY_5M.csv")
        end_time = st.text_input("End Time", "2026-12-31")
        output_dir = st.text_input("Output Dir", default_output_dir("output_control_panel_resume"))
        override_dp_lookback = st.checkbox("Override dp_lookback", value=False)
        dp_lookback_override = None
        if override_dp_lookback:
            dp_lookback_override = int(st.number_input("dp_lookback_override", min_value=1, value=5, step=1))

        override_daily_threshold_lookback = st.checkbox("Override daily threshold lookback", value=False)
        daily_threshold_lookback_days_override = None
        if override_daily_threshold_lookback:
            daily_threshold_lookback_days_override = int(
                st.number_input("daily_threshold_lookback_days_override", min_value=1, value=15, step=1)
            )
        verbose = st.checkbox("Verbose", value=True)
        plot_from_checkpoint = st.checkbox("Plot from checkpoint", value=True)

        resume = ResumeConfig(
            checkpoint_path=str(selected_path),
            daily_csv_path=daily_csv_path,
            k5m_csv_path=k5m_csv_path,
            end_time=end_time,
            output_dir=output_dir,
            dp_lookback_override=dp_lookback_override,
            daily_threshold_lookback_days_override=daily_threshold_lookback_days_override,
            verbose=verbose,
            plot_from_checkpoint=plot_from_checkpoint,
        )

        st.code(render_command_preview(resume=resume, fresh=None), language="python")

        if st.button("Export Resume Config"):
            target = root / "output" / "control_panel" / "resume_config.json"
            export_run_config(asdict(resume), target)
            st.success(f"Wrote {target}")

    elif mode == "Start fresh run":
        st.subheader("Start Fresh Run")
        col1, col2 = st.columns(2)

        with col1:
            code = st.text_input("Code", "SPY")
            daily_csv_path = st.text_input("Daily CSV", "DataAPI/data/SPY_DAY.csv")
            k5m_csv_path = st.text_input("5m CSV", "DataAPI/data/SPY_5M.csv")
            daily_chan_start = st.text_input("Daily Chan Start", "2016-06-01")
            accumulation_start = st.text_input("Accumulation Start", "2018-10-01")
            sim_start = st.text_input("Simulation Start", "2024-01-01")
            end_time = st.text_input("End Time", "2026-12-31")

        with col2:
            p_buy_level = st.number_input("p_buy_level", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
            p_sell_level = st.number_input("p_sell_level", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
            threshold_window_days = st.number_input("threshold_window_days", min_value=1.0, value=2.0, step=1.0)
            threshold_min_open_signals = st.number_input("threshold_min_open_signals", min_value=1, value=10, step=1)
            lookahead_days_5m = st.number_input("lookahead_days_5m", min_value=0.5, value=2.0, step=0.5)
            retrain_every_days_5m = st.number_input("retrain_every_days_5m", min_value=1, value=5, step=1)
            save_checkpoint_path = st.text_input("Save checkpoint", "checkpoints/control_panel_checkpoint.joblib")
            checkpoint_every_n_days = st.number_input("checkpoint_every_n_days", min_value=1, value=5, step=1)
            output_dir = st.text_input("Output Dir", default_output_dir("output_control_panel_fresh"))

        fresh = FreshRunConfig(
            daily_csv_path=daily_csv_path,
            k5m_csv_path=k5m_csv_path,
            code=code,
            daily_chan_start=daily_chan_start,
            accumulation_start=accumulation_start,
            sim_start=sim_start,
            end_time=end_time,
            p_buy_level=float(p_buy_level),
            p_sell_level=float(p_sell_level),
            threshold_window_days=float(threshold_window_days),
            threshold_min_open_signals=int(threshold_min_open_signals),
            lookahead_days_5m=float(lookahead_days_5m),
            retrain_every_days_5m=int(retrain_every_days_5m),
            save_checkpoint_path=save_checkpoint_path,
            checkpoint_every_n_days=int(checkpoint_every_n_days),
            output_dir=output_dir,
        )

        st.code(render_command_preview(resume=None, fresh=fresh), language="python")

        if st.button("Export Fresh-Run Config"):
            target = root / "output" / "control_panel" / "fresh_run_config.json"
            export_run_config(asdict(fresh), target)
            st.success(f"Wrote {target}")

    else:
        st.subheader("News Override Config")
        st.write("This tab is meant for wiring in a separate breaking-news model.")
        impact = st.slider("High-impact threshold", min_value=0.50, max_value=1.0, value=0.85, step=0.01)
        direction = st.slider("Directional threshold", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
        halt_minutes = st.number_input("Halt minutes", min_value=15, value=180, step=15)
        st.json(
            {
                "impact_score_threshold": impact,
                "direction_score_threshold": direction,
                "halt_minutes": int(halt_minutes),
                "override_actions": [
                    "FORCE_EXIT_AND_HALT",
                    "FORCE_BUY_AND_HALT",
                    "FORCE_SELL_AND_HALT",
                ],
            }
        )


if __name__ == "__main__":
    main()
