from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_with_band(ax, y: np.ndarray, y_std: np.ndarray, label: str) -> None:
    x = np.arange(len(y))
    ax.plot(x, y, label=label)
    ax.fill_between(x, y - y_std, y + y_std, alpha=0.15)


def plot(result_json: str, out_dir: str | None = None) -> list[str]:
    data = _load(result_json)
    policies = data["policies"]

    out_dir = out_dir or os.path.join("cope_research", "results")
    _ensure_dir(out_dir)

    saved: list[str] = []
    base = os.path.splitext(os.path.basename(result_json))[0]

    # 1) Average cumulative reward
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in [("random", "Random"), ("static_best", "Static Best (Action 2)"), ("cope", "COPE (LinUCB)")]:
        y = np.array(policies[key]["cumulative_reward"], dtype=float)
        y_std = np.array(policies[key]["cumulative_reward_std"], dtype=float)
        _plot_with_band(ax, y, y_std, label=label)
    ax.set_title("Average Cumulative Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = os.path.join(out_dir, f"{base}_cumulative_reward.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    saved.append(p1)

    # 2) Schema success rate
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in [("random", "Random"), ("static_best", "Static Best (Action 2)"), ("cope", "COPE (LinUCB)")]:
        y = np.array(policies[key]["schema_success_rate"], dtype=float)
        y_std = np.array(policies[key]["schema_success_rate_std"], dtype=float)
        _plot_with_band(ax, y, y_std, label=label)
    ax.set_title("Schema Success Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    p2 = os.path.join(out_dir, f"{base}_success_rate.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    saved.append(p2)

    # 3) Average cumulative token cost
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in [("random", "Random"), ("static_best", "Static Best (Action 2)"), ("cope", "COPE (LinUCB)")]:
        y = np.array(policies[key]["cumulative_token_cost"], dtype=float)
        y_std = np.array(policies[key]["cumulative_token_cost_std"], dtype=float)
        _plot_with_band(ax, y, y_std, label=label)
    ax.set_title("Average Cumulative Token Cost (proxy)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Tokens")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p3 = os.path.join(out_dir, f"{base}_cumulative_token_cost.png")
    fig.tight_layout()
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    saved.append(p3)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=str)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    saved = plot(args.result_json, out_dir=args.out_dir)
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()

