from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _safe_float(x: object, default: float = 0.0) -> float:
    """Return a finite float, or a fallback when notebook/log data is missing or dirty."""
    try:
        v = float(x)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return v


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    """Division helper that avoids zero-division in reward/state calculations."""
    return float(a) / (float(b) + eps)


@dataclass
class AdaptiveThresholdResult:
    """Result object returned by daily threshold optimizers."""
    buy_level: float
    sell_level: float
    gate: str
    score: float
    window_size: int


@dataclass
class RollingThresholdConfig:
    """Configuration for grid-searching daily buy/sell probability thresholds."""
    lookback_days: int = 60
    buy_grid: Sequence[float] = field(default_factory=lambda: np.round(np.arange(0.05, 0.46, 0.01), 4).tolist())
    sell_grid: Sequence[float] = field(default_factory=lambda: np.round(np.arange(0.15, 0.76, 0.01), 4).tolist())
    min_gap: float = 0.05
    max_gap: Optional[float] = None
    min_obs: int = 20
    switch_penalty: float = 0.0


def gate_from_levels(p_day: float, buy_level: float, sell_level: float) -> str:
    """Map the daily model probability into FORCE_BUY, FREE, FORCE_SELL, or NO_P."""
    if not np.isfinite(p_day):
        return "NO_P"
    if p_day >= sell_level:
        return "FORCE_SELL"
    if p_day <= buy_level:
        return "FORCE_BUY"
    return "FREE"


def reward_columns_available(df: pd.DataFrame) -> bool:
    """Check whether exact daily reward columns are present."""
    needed = {"reward_force_buy", "reward_free", "reward_force_sell"}
    return needed.issubset(df.columns)


def make_threshold_grid(start: float, end: float, step: float) -> list[float]:
    """Build a rounded inclusive grid for daily probability thresholds."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(np.floor((end - start) / step)) + 1
    return np.round(np.linspace(start, start + step * (n - 1), n), 6).tolist()


def make_ret_grid(start: float = -0.5, end: float = 2.5, step: float = 0.005) -> List[float]:
    """Build an inclusive grid for 5m return thresholds used by notebook experiments."""
    # Convenience helper for notebook callers that need to test legacy or custom 5m return thresholds.
    if step <= 0:
        raise ValueError("step must be positive")
    vals = np.arange(float(start), float(end) + 1e-12, float(step), dtype=float)
    return [float(x) for x in vals]


def build_proxy_reward_frame(df: pd.DataFrame, next_day_ret_col: str = "next_day_return") -> pd.DataFrame:
    """
    Build a simple reward frame when only daily next-day return is available.

    Interpretation:
    - FORCE_BUY: capture the next-day return
    - FREE: half exposure proxy
    - FORCE_SELL: flat proxy
    """
    if next_day_ret_col not in df.columns:
        raise ValueError(
            f"Expected either explicit reward columns or `{next_day_ret_col}` in the history DataFrame."
        )

    out = df.copy()
    ret = pd.to_numeric(out[next_day_ret_col], errors="coerce").fillna(0.0).astype(float)
    out["reward_force_buy"] = ret
    out["reward_free"] = 0.5 * ret
    out["reward_force_sell"] = 0.0
    return out


def build_equity_proxy_reward_frame(df: pd.DataFrame, equity_col: str = "equity") -> pd.DataFrame:
    """
    Build a proxy reward frame from the realized strategy equity curve.

    This is weaker than using oracle per-action rewards, but it works with the
    current `daily_log.csv` format produced by your pipeline notebooks.

    Interpretation:
    - `next_day_return` is inferred from the realized equity curve
    - FORCE_BUY uses the full next-day realized return
    - FREE uses half exposure as a conservative proxy
    - FORCE_SELL uses flat return
    """
    if equity_col not in df.columns:
        raise ValueError(
            f"Expected either explicit reward columns, `next_day_return`, or `{equity_col}` in the history DataFrame."
        )

    out = df.copy()
    eq = pd.to_numeric(out[equity_col], errors="coerce").astype(float)
    next_ret = (eq.shift(-1) / (eq + 1e-12) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["next_day_return"] = next_ret
    out["reward_force_buy"] = next_ret
    out["reward_free"] = 0.5 * next_ret
    out["reward_force_sell"] = 0.0
    return out


def _daily_history_rows(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep threshold fitting at daily granularity.

    Live polling can append more than one reward snapshot for the same trading
    date before the final checkpoint is saved. The threshold optimizer expects
    one row per day, so collapse duplicates before taking the rolling window.
    """
    if hist_df.empty or "date" not in hist_df.columns:
        return hist_df

    hist = hist_df.copy()
    hist["_threshold_date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    valid = hist["_threshold_date"].notna()
    if not valid.any():
        return hist_df

    no_date = hist[~valid].drop(columns=["_threshold_date"])
    dated = hist[valid].copy()
    dated["_source_order"] = np.arange(len(dated))
    dated = (
        dated.sort_values(["_threshold_date", "_source_order"])
        .drop_duplicates("_threshold_date", keep="last")
        .drop(columns=["_threshold_date", "_source_order"])
    )
    if no_date.empty:
        return dated.reset_index(drop=True)
    return pd.concat([no_date, dated], ignore_index=True)


def score_threshold_pair(
    hist_df: pd.DataFrame,
    buy_level: float,
    sell_level: float,
    switch_penalty: float = 0.0,
) -> float:
    """Score one daily threshold pair by replaying its historical gate decisions."""
    # Simulate what the daily gate would have chosen for each historical p_day,
    # then score that threshold pair by the realized reward of those choices.
    actions = hist_df["p_day"].apply(lambda p: gate_from_levels(_safe_float(p, np.nan), buy_level, sell_level))
    rewards = np.where(
        actions == "FORCE_BUY",
        hist_df["reward_force_buy"].to_numpy(dtype=float),
        np.where(
            actions == "FORCE_SELL",
            hist_df["reward_force_sell"].to_numpy(dtype=float),
            hist_df["reward_free"].to_numpy(dtype=float),
        ),
    )

    score = float(np.nansum(rewards))
    if switch_penalty > 0.0 and len(actions) > 1:
        switches = float((actions.shift(1) != actions).fillna(False).sum())
        score -= switch_penalty * switches
    return score


def classify_best_action_from_rewards(hist_df: pd.DataFrame) -> pd.Series:
    """Label each historical day with the best ex-post gate by realized reward."""
    rewards = hist_df[["reward_force_buy", "reward_free", "reward_force_sell"]].copy()
    col_to_action = {
        "reward_force_buy": "FORCE_BUY",
        "reward_free": "FREE",
        "reward_force_sell": "FORCE_SELL",
    }
    return rewards.idxmax(axis=1).map(col_to_action)


def threshold_confusion_score(
    hist_df: pd.DataFrame,
    buy_level: float,
    sell_level: float,
) -> dict:
    """Measure how often a threshold pair matches the ex-post best daily gate."""
    predicted = hist_df["p_day"].apply(lambda p: gate_from_levels(_safe_float(p, np.nan), buy_level, sell_level))
    best = classify_best_action_from_rewards(hist_df)
    correct = (predicted == best)
    return {
        "accuracy": float(correct.mean()) if len(correct) else 0.0,
        "n": int(len(correct)),
        "predicted": predicted,
        "best": best,
    }


def select_moving_daily_thresholds(
    history_df: pd.DataFrame,
    current_p_day: float,
    config: Optional[RollingThresholdConfig] = None,
    prev_buy_level: Optional[float] = None,
    prev_sell_level: Optional[float] = None,
) -> AdaptiveThresholdResult:
    """
    Optimize daily buy/sell probability thresholds on a rolling window.

    history_df should contain:
    - p_day
    - reward_force_buy / reward_free / reward_force_sell
      OR next_day_return
    """
    config = config or RollingThresholdConfig()
    hist = history_df.copy()
    if hist.empty:
        buy_level = _safe_float(prev_buy_level, 0.20)
        sell_level = _safe_float(prev_sell_level, 0.30)
        return AdaptiveThresholdResult(
            buy_level=buy_level,
            sell_level=sell_level,
            gate=gate_from_levels(current_p_day, buy_level, sell_level),
            score=0.0,
            window_size=0,
        )

    if not reward_columns_available(hist):
        if "next_day_return" in hist.columns:
            hist = build_proxy_reward_frame(hist)
        elif "equity" in hist.columns:
            hist = build_equity_proxy_reward_frame(hist)
        else:
            raise ValueError(
                "history_df needs either explicit reward columns, `next_day_return`, or `equity`."
            )

    if "date" in hist.columns:
        hist = hist.sort_values("date").reset_index(drop=True)
    elif "timestamp" in hist.columns:
        hist = hist.sort_values("timestamp").reset_index(drop=True)
    else:
        hist = hist.reset_index(drop=True)

    hist = _daily_history_rows(hist).tail(int(config.lookback_days)).copy()
    hist["p_day"] = pd.to_numeric(hist["p_day"], errors="coerce")
    hist = hist[np.isfinite(hist["p_day"])].copy()
    if len(hist) < int(config.min_obs):
        buy_level = _safe_float(prev_buy_level, 0.20)
        sell_level = _safe_float(prev_sell_level, 0.30)
        return AdaptiveThresholdResult(
            buy_level=buy_level,
            sell_level=sell_level,
            gate=gate_from_levels(current_p_day, buy_level, sell_level),
            score=0.0,
            window_size=len(hist),
        )

    best_score = -np.inf
    best_pair = (_safe_float(prev_buy_level, 0.20), _safe_float(prev_sell_level, 0.30))
    max_gap = getattr(config, "max_gap", None)
    for buy_level, sell_level in product(config.buy_grid, config.sell_grid):
        gap = float(sell_level) - float(buy_level)
        if gap < float(config.min_gap):
            continue
        if max_gap is not None and gap > float(max_gap):
            continue
        score = score_threshold_pair(
            hist_df=hist,
            buy_level=float(buy_level),
            sell_level=float(sell_level),
            switch_penalty=float(config.switch_penalty),
        )
        if score > best_score:
            best_score = score
            best_pair = (float(buy_level), float(sell_level))

    buy_level, sell_level = best_pair
    return AdaptiveThresholdResult(
        buy_level=buy_level,
        sell_level=sell_level,
        gate=gate_from_levels(current_p_day, buy_level, sell_level),
        score=float(best_score),
        window_size=len(hist),
    )


def select_oracle_thresholds_from_daily_rewards(
    history_df: pd.DataFrame,
    current_p_day: float,
    config: Optional[RollingThresholdConfig] = None,
    prev_buy_level: Optional[float] = None,
    prev_sell_level: Optional[float] = None,
    objective: str = "reward",
) -> AdaptiveThresholdResult:
    """
    Optimize threshold regions against per-day 5m-based rewards for:
    - FORCE_BUY
    - FREE
    - FORCE_SELL

    This matches your intended logic:
    - if FORCE_BUY is the best 5m-based action, p should be below buy threshold
    - if FORCE_SELL is the best 5m-based action, p should be above sell threshold
    - if FREE is best, p should lie between the thresholds

    objective:
    - "reward": maximize realized reward chosen by the threshold rule
    - "accuracy": maximize agreement with ex-post best action
    """
    config = config or RollingThresholdConfig()
    hist = history_df.copy()
    if hist.empty:
        buy_level = _safe_float(prev_buy_level, 0.20)
        sell_level = _safe_float(prev_sell_level, 0.30)
        return AdaptiveThresholdResult(
            buy_level=buy_level,
            sell_level=sell_level,
            gate=gate_from_levels(current_p_day, buy_level, sell_level),
            score=0.0,
            window_size=0,
        )

    if not reward_columns_available(hist):
        if "next_day_return" in hist.columns:
            hist = build_proxy_reward_frame(hist)
        elif "equity" in hist.columns:
            hist = build_equity_proxy_reward_frame(hist)
        else:
            raise ValueError(
                "history_df needs reward_force_buy/reward_free/reward_force_sell for exact threshold optimization."
            )

    if "date" in hist.columns:
        hist = hist.sort_values("date").reset_index(drop=True)
    elif "timestamp" in hist.columns:
        hist = hist.sort_values("timestamp").reset_index(drop=True)
    else:
        hist = hist.reset_index(drop=True)

    # Use one row per day and only the configured rolling lookback so live polling
    # does not let duplicate intraday snapshots overweight one trading day.
    hist = _daily_history_rows(hist).tail(int(config.lookback_days)).copy()
    hist["p_day"] = pd.to_numeric(hist["p_day"], errors="coerce")
    hist = hist[np.isfinite(hist["p_day"])].copy()

    if len(hist) < int(config.min_obs):
        buy_level = _safe_float(prev_buy_level, 0.20)
        sell_level = _safe_float(prev_sell_level, 0.30)
        return AdaptiveThresholdResult(
            buy_level=buy_level,
            sell_level=sell_level,
            gate=gate_from_levels(current_p_day, buy_level, sell_level),
            score=0.0,
            window_size=len(hist),
        )

    best_score = -np.inf
    best_pair = (_safe_float(prev_buy_level, 0.20), _safe_float(prev_sell_level, 0.30))
    # Grid search the daily probability thresholds. The selected pair maps the
    # latest p_day into FORCE_BUY, FREE, or FORCE_SELL.
    max_gap = getattr(config, "max_gap", None)
    for buy_level, sell_level in product(config.buy_grid, config.sell_grid):
        buy_level = float(buy_level)
        sell_level = float(sell_level)
        gap = sell_level - buy_level
        if gap < float(config.min_gap):
            continue
        if max_gap is not None and gap > float(max_gap):
            continue

        if objective == "accuracy":
            score = threshold_confusion_score(hist, buy_level, sell_level)["accuracy"]
        else:
            score = score_threshold_pair(
                hist_df=hist,
                buy_level=buy_level,
                sell_level=sell_level,
                switch_penalty=float(config.switch_penalty),
            )

        if score > best_score:
            best_score = float(score)
            best_pair = (buy_level, sell_level)

    buy_level, sell_level = best_pair
    return AdaptiveThresholdResult(
        buy_level=buy_level,
        sell_level=sell_level,
        gate=gate_from_levels(current_p_day, buy_level, sell_level),
        score=float(best_score),
        window_size=len(hist),
    )


@dataclass
class ThresholdPairBanditConfig:
    """Configuration for contextual bandit experiments over fixed threshold pairs."""
    threshold_pairs: Sequence[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.10, 0.25),
            (0.15, 0.30),
            (0.20, 0.30),
            (0.20, 0.35),
            (0.25, 0.40),
            (0.25, 0.50),
        ]
    )
    alpha: float = 0.50
    l2: float = 1.0


class ThresholdPairBandit:
    """
    A small contextual bandit over threshold pairs.

    The action is not BUY / SELL / FREE directly.
    The action is a candidate `(buy_level, sell_level)` pair.
    Once a pair is chosen, the daily gate is derived from `p_day`.
    """

    def __init__(self, n_features: int, config: Optional[ThresholdPairBanditConfig] = None):
        """Initialize one linear-UCB state per candidate threshold pair."""
        config = config or ThresholdPairBanditConfig()
        self.config = config
        self.threshold_pairs: List[tuple[float, float]] = [
            (float(b), float(s)) for (b, s) in config.threshold_pairs if float(s) > float(b)
        ]
        if not self.threshold_pairs:
            raise ValueError("threshold_pairs must contain at least one valid (buy, sell) pair.")
        self.n_features = int(n_features)
        self.alpha = float(config.alpha)
        self.l2 = float(config.l2)
        self.A = [np.eye(self.n_features, dtype=float) * self.l2 for _ in self.threshold_pairs]
        self.b = [np.zeros(self.n_features, dtype=float) for _ in self.threshold_pairs]

    def select_pair(self, x: np.ndarray) -> tuple[int, tuple[float, float]]:
        """Choose the threshold pair with the highest optimistic linear score."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"Expected feature vector length {self.n_features}, got {x.shape[0]}.")

        best_idx = 0
        best_score = -np.inf
        for idx, A in enumerate(self.A):
            A_inv = np.linalg.inv(A)
            theta = A_inv @ self.b[idx]
            mean = float(theta @ x)
            bonus = self.alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx, self.threshold_pairs[best_idx]

    def decide_gate(self, x: np.ndarray, p_day: float) -> dict:
        """Select a threshold pair and convert p_day into the corresponding daily gate."""
        action_idx, (buy_level, sell_level) = self.select_pair(x)
        gate = gate_from_levels(p_day, buy_level, sell_level)
        return {
            "action_idx": action_idx,
            "buy_level": buy_level,
            "sell_level": sell_level,
            "gate": gate,
            "p_day": float(p_day),
        }

    def update(self, action_idx: int, x: np.ndarray, reward: float):
        """Update the selected threshold pair with the realized reward."""
        x = np.asarray(x, dtype=float).reshape(-1)
        a = int(action_idx)
        self.A[a] += np.outer(x, x)
        self.b[a] += float(reward) * x


def realized_reward_for_gate(row: pd.Series, gate: str) -> float:
    """Read the realized reward for a chosen daily gate from one reward row."""
    if gate == "FORCE_BUY":
        return _safe_float(row.get("reward_force_buy", 0.0), 0.0)
    if gate == "FORCE_SELL":
        return _safe_float(row.get("reward_force_sell", 0.0), 0.0)
    return _safe_float(row.get("reward_free", 0.0), 0.0)


def run_walkforward_threshold_policy(
    daily_rewards_df: pd.DataFrame,
    config: Optional[RollingThresholdConfig] = None,
    initial_equity: float = 100000.0,
    objective: str = "reward",
    default_buy_level: float = 0.20,
    default_sell_level: float = 0.30,
) -> pd.DataFrame:
    """
    Online walk-forward simulation using adaptive daily thresholds.

    Required columns:
    - date
    - p_day
    - reward_force_buy
    - reward_free
    - reward_force_sell
    """
    config = config or RollingThresholdConfig()
    df = daily_rewards_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if not reward_columns_available(df):
        raise ValueError("daily_rewards_df must include reward_force_buy/reward_free/reward_force_sell")

    rows = []
    equity = float(initial_equity)
    buy_level = float(default_buy_level)
    sell_level = float(default_sell_level)

    for i in range(len(df)):
        row = df.iloc[i]
        hist = df.iloc[:i].copy()
        p_day = _safe_float(row.get("p_day", np.nan), np.nan)

        if len(hist) >= int(config.min_obs):
            out = select_oracle_thresholds_from_daily_rewards(
                history_df=hist,
                current_p_day=p_day,
                config=config,
                prev_buy_level=buy_level,
                prev_sell_level=sell_level,
                objective=objective,
            )
            buy_level = out.buy_level
            sell_level = out.sell_level
            gate = out.gate
            fit_score = out.score
            fit_window = out.window_size
        else:
            gate = gate_from_levels(p_day, buy_level, sell_level)
            fit_score = np.nan
            fit_window = len(hist)

        reward = realized_reward_for_gate(row, gate)
        equity *= (1.0 + reward)
        oracle_best = max(
            ["FORCE_BUY", "FREE", "FORCE_SELL"],
            key=lambda g: realized_reward_for_gate(row, g),
        )
        rows.append(
            {
                "date": row.get("date", i),
                "p_day": p_day,
                "buy_level": buy_level,
                "sell_level": sell_level,
                "gate": gate,
                "reward": reward,
                "equity": equity,
                "best_action_ex_post": oracle_best,
                "oracle_best_reward": realized_reward_for_gate(row, oracle_best),
                "fit_score": fit_score,
                "fit_window": fit_window,
            }
        )
    return pd.DataFrame(rows)


def run_walkforward_threshold_bandit_policy(
    daily_rewards_df: pd.DataFrame,
    bandit: ThresholdPairBandit,
    initial_equity: float = 100000.0,
) -> pd.DataFrame:
    """
    Online walk-forward simulation where the action is a threshold pair.
    """
    df = daily_rewards_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if not reward_columns_available(df):
        raise ValueError("daily_rewards_df must include reward_force_buy/reward_free/reward_force_sell")

    rows = []
    equity = float(initial_equity)
    equity_peak = float(initial_equity)
    current_pos = 0

    for _, row in df.iterrows():
        drawdown_rel = max(0.0, (equity_peak - equity) / max(equity_peak, 1e-12))
        x = make_daily_threshold_state(
            p_day=_safe_float(row.get("p_day", 0.5), 0.5),
            dp_min=0.0,
            dp_max=0.0,
            realized_vol_20=0.0,
            drawdown_rel=drawdown_rel,
            current_pos=current_pos,
        )
        decision = bandit.decide_gate(x=x, p_day=_safe_float(row.get("p_day", 0.5), 0.5))
        reward = realized_reward_for_gate(row, decision["gate"])
        bandit.update(decision["action_idx"], x, reward)

        equity *= (1.0 + reward)
        equity_peak = max(equity_peak, equity)
        if decision["gate"] == "FORCE_BUY":
            current_pos = 1
        elif decision["gate"] == "FORCE_SELL":
            current_pos = 0

        oracle_best = max(
            ["FORCE_BUY", "FREE", "FORCE_SELL"],
            key=lambda g: realized_reward_for_gate(row, g),
        )
        rows.append(
            {
                "date": row.get("date"),
                "p_day": _safe_float(row.get("p_day", np.nan), np.nan),
                "buy_level": decision["buy_level"],
                "sell_level": decision["sell_level"],
                "gate": decision["gate"],
                "reward": reward,
                "equity": equity,
                "best_action_ex_post": oracle_best,
                "oracle_best_reward": realized_reward_for_gate(row, oracle_best),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_bandit_reward(
    gate: str,
    force_buy_reward: float,
    free_reward: float,
    force_sell_reward: float,
) -> float:
    """Return the reward value corresponding to one selected gate."""
    if gate == "FORCE_BUY":
        return float(force_buy_reward)
    if gate == "FORCE_SELL":
        return float(force_sell_reward)
    return float(free_reward)


def make_daily_threshold_state(
    p_day: float,
    dp_min: float = 0.0,
    dp_max: float = 0.0,
    realized_vol_20: float = 0.0,
    drawdown_rel: float = 0.0,
    current_pos: int = 0,
) -> np.ndarray:
    """Pack daily context features into the vector expected by ThresholdPairBandit."""
    return np.array(
        [
            _safe_float(p_day, 0.5),
            _safe_float(dp_min, 0.0),
            _safe_float(dp_max, 0.0),
            _safe_float(realized_vol_20, 0.0),
            _safe_float(drawdown_rel, 0.0),
            float(int(current_pos)),
        ],
        dtype=float,
    )


@dataclass
class NewsShockDecision:
    """Output from the rule-based breaking-news guard."""
    action: str
    halt_trading: bool
    hold_minutes: int
    reason: str


@dataclass
class NewsShockConfig:
    """Trigger levels for the placeholder breaking-news risk guard."""
    high_impact_threshold: float = 0.85
    directional_threshold: float = 0.40
    halt_minutes: int = 180
    default_action_on_extreme: str = "FORCE_EXIT_AND_HALT"


class NewsShockGuard:
    """
    Rule-based shell for a future breaking-news model.

    Expected upstream event fields:
    - impact_score: 0..1
    - direction_score: -1..1
    - confidence: 0..1

    Suggested action semantics:
    - FORCE_EXIT_AND_HALT
    - FORCE_BUY_AND_HALT
    - FORCE_SELL_AND_HALT
    - NO_OVERRIDE
    """

    def __init__(self, config: Optional[NewsShockConfig] = None):
        """Store the news-shock trigger configuration."""
        self.config = config or NewsShockConfig()

    def decide(self, latest_event: Optional[dict]) -> NewsShockDecision:
        """Convert a parsed high-impact news event into a risk override decision."""
        if not latest_event:
            return NewsShockDecision("NO_OVERRIDE", False, 0, "no event")

        impact = _safe_float(latest_event.get("impact_score", 0.0), 0.0)
        direction = _safe_float(latest_event.get("direction_score", 0.0), 0.0)
        confidence = _safe_float(latest_event.get("confidence", 0.0), 0.0)
        headline = str(latest_event.get("headline", "")).strip()

        if impact < self.config.high_impact_threshold or confidence < 0.50:
            return NewsShockDecision("NO_OVERRIDE", False, 0, "event below trigger threshold")

        if abs(direction) < self.config.directional_threshold:
            return NewsShockDecision(
                self.config.default_action_on_extreme,
                True,
                int(self.config.halt_minutes),
                f"high-impact ambiguous news: {headline}",
            )

        if direction > 0:
            return NewsShockDecision(
                "FORCE_BUY_AND_HALT",
                True,
                int(self.config.halt_minutes),
                f"high-impact positive news: {headline}",
            )

        return NewsShockDecision(
            "FORCE_SELL_AND_HALT",
            True,
            int(self.config.halt_minutes),
            f"high-impact negative news: {headline}",
        )
