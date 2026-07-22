from __future__ import annotations

import numpy as np


class BanditAction:
    FORCE_BUY = 0
    FREE = 1
    FORCE_SELL = 2


ACTION_TO_NAME = {BanditAction.FORCE_BUY: "FORCE_BUY", BanditAction.FREE: "FREE", BanditAction.FORCE_SELL: "FORCE_SELL"}
NAME_TO_ACTION = {name: action for action, name in ACTION_TO_NAME.items()}


class LinUCBBandit:
    def __init__(self, n_actions: int, n_features: int, alpha: float = 0.75, l2: float = 1.0):
        self.n_actions, self.n_features = int(n_actions), int(n_features)
        self.alpha, self.l2 = float(alpha), float(l2)
        self.A = [np.eye(self.n_features, dtype=float) * self.l2 for _ in range(self.n_actions)]
        self.b = [np.zeros(self.n_features, dtype=float) for _ in range(self.n_actions)]

    def select_action(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        scores = []
        for action in range(self.n_actions):
            inverse = np.linalg.inv(self.A[action])
            theta = inverse @ self.b[action]
            bonus = self.alpha * float(np.sqrt(max(0.0, x @ inverse @ x)))
            scores.append(float(theta @ x) + bonus)
        return int(np.argmax(scores))

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        action = int(action)
        self.A[action] += np.outer(x, x)
        self.b[action] += float(reward) * x

    def state_dict(self) -> dict:
        return {"n_actions": self.n_actions, "n_features": self.n_features, "alpha": self.alpha, "l2": self.l2, "A": self.A, "b": self.b}

    @classmethod
    def from_state_dict(cls, state: dict) -> "LinUCBBandit":
        obj = cls(state["n_actions"], state["n_features"], state["alpha"], state["l2"])
        obj.A, obj.b = state["A"], state["b"]
        return obj
