from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinUCBConfig:
    n_arms: int = 4
    alpha: float = 0.1  # exploration strength (blueprint default)
    feature_dim: int = 5
    ridge_lambda: float = 1.0  # ridge prior (A starts as lambda * I)


class LinUCBAgent:
    """LinUCB with per-arm ridge regression (blueprint form).

    Per arm a:
      A_a = λI + Σ x xᵀ
      b_a = Σ r x
      θ_a = A_a^{-1} b_a
      p_a = θ_aᵀ x + α √(xᵀ A_a^{-1} x)
    """

    def __init__(self, config: LinUCBConfig):
        self.config = config
        d = config.feature_dim
        self.A: dict[int, np.ndarray] = {
            a: (config.ridge_lambda * np.eye(d, dtype=float)) for a in range(config.n_arms)
        }
        self.b: dict[int, np.ndarray] = {a: np.zeros((d,), dtype=float) for a in range(config.n_arms)}

    def select_action(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        assert x.shape[0] == self.config.feature_dim

        best_a = 0
        best_score = -float("inf")
        for a in range(self.config.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(x @ theta)
            bonus = self.config.alpha * float(np.sqrt(x @ (A_inv @ x)))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        assert x.shape[0] == self.config.feature_dim
        assert 0 <= action < self.config.n_arms

        # Ridge regression update rules:
        # A <- A + x x^T
        # b <- b + r x
        self.A[action] = self.A[action] + np.outer(x, x)
        self.b[action] = self.b[action] + float(reward) * x

