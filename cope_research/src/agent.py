from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinUCBConfig:
    n_arms: int = 4
    alpha: float = 1.5  # exploration strength
    feature_dim: int = 6  # includes bias term
    ridge_lambda: float = 1.0  # initial regularization


class LinUCBAgent:
    """LinUCB with per-arm linear models.

    Keeps A^{-1} and b for each arm so selection is:
      p_a = x^T theta_a + alpha * sqrt(x^T A^{-1}_a x)
      theta_a = A^{-1}_a b_a
    """

    def __init__(self, config: LinUCBConfig):
        self.config = config
        d = config.feature_dim
        self.A_inv: dict[int, np.ndarray] = {
            a: (1.0 / config.ridge_lambda) * np.eye(d, dtype=float)
            for a in range(config.n_arms)
        }
        self.b: dict[int, np.ndarray] = {a: np.zeros(d, dtype=float) for a in range(config.n_arms)}

    def select_action(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        assert x.shape[0] == self.config.feature_dim

        best_a = 0
        best_score = -float("inf")
        for a in range(self.config.n_arms):
            A_inv = self.A_inv[a]
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

        # b <- b + r x
        self.b[action] = self.b[action] + reward * x

        # Sherman-Morrison update for A_inv where A <- A + x x^T:
        # (A + u v^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
        # with u=v=x
        A_inv = self.A_inv[action]
        Ax = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv[action] = A_inv - np.outer(Ax, Ax) / denom

