from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import matplotlib.pyplot as plt

# Allow running as: `python cope_research/run_experiment.py ...`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cope_research.src.agent import LinUCBConfig, LinUCBAgent  # noqa: E402
from cope_research.src.data_gen import (  # noqa: E402
    MarkdownSample,
    generate_dataset,
    generate_drift_dataset,
)
from cope_research.src.environment import CopeEnvironment, Environment  # noqa: E402
from cope_research.src.features import extract_feature_dict, extract_features  # noqa: E402


@dataclass(frozen=True)
class EpisodeLog:
    reward: float
    token_cost: int
    action: int
    parse_ok: bool
    schema_ok: bool
    kind: str


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_policy(
    *,
    samples: list[MarkdownSample],
    policy_name: str,
    env: Environment,
    seed: int,
    alpha: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    cfg = LinUCBConfig(alpha=alpha)
    agent = LinUCBAgent(cfg)

    logs: list[EpisodeLog] = []
    for s in samples:
        x = extract_features(s.md)
        if policy_name == "random":
            action = int(rng.integers(0, cfg.n_arms))
        elif policy_name == "static_best":
            action = 2
        elif policy_name == "cope":
            action = agent.select_action(x)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        step = env.step(s.md, action, expected_keys=s.expected_keys, sample_kind=s.kind)
        if policy_name == "cope":
            agent.update(x, action, step.reward)

        logs.append(
            EpisodeLog(
                reward=float(step.reward),
                token_cost=int(step.token_cost),
                action=int(action),
                parse_ok=bool(step.parse_ok),
                schema_ok=bool(step.schema_ok),
                kind=str(s.kind),
            )
        )

    rewards = np.array([l.reward for l in logs], dtype=float)
    costs = np.array([l.token_cost for l in logs], dtype=float)
    schema_ok = np.array([l.schema_ok for l in logs], dtype=float)

    return {
        "policy": policy_name,
        "episodes": len(samples),
        "reward": rewards.tolist(),
        "cumulative_reward": np.cumsum(rewards).tolist(),
        "avg_reward": (np.cumsum(rewards) / (np.arange(len(rewards)) + 1)).tolist(),
        "token_cost": costs.tolist(),
        "cumulative_token_cost": np.cumsum(costs).tolist(),
        "schema_success": schema_ok.tolist(),
        "schema_success_rate": (np.cumsum(schema_ok) / (np.arange(len(schema_ok)) + 1)).tolist(),
        "actions": [l.action for l in logs],
    }


def _aggregate_runs(run_payloads: list[dict[str, object]]) -> dict[str, object]:
    # Stack metrics across runs and average.
    keys_to_avg = [
        "reward",
        "cumulative_reward",
        "avg_reward",
        "token_cost",
        "cumulative_token_cost",
        "schema_success",
        "schema_success_rate",
    ]
    out: dict[str, object] = {"runs": len(run_payloads)}
    for k in keys_to_avg:
        arr = np.array([p[k] for p in run_payloads], dtype=float)
        out[k] = arr.mean(axis=0).tolist()
        out[k + "_std"] = arr.std(axis=0).tolist()
    return out


def run_blueprint_mock(*, episodes: int, seed: int, out_path: str | None) -> str:
    """Blueprint experiment: Random vs Greedy(ZERO_SHOT) vs COPE(LinUCB) on a mock simulator."""

    rng = np.random.default_rng(seed)
    env = CopeEnvironment(seed=seed)

    # Synthetic trials: mix simple and deeply nested.
    # Simple: short text; Deep: lots of indentation.
    samples: list[str] = []
    for i in range(episodes):
        if rng.random() < 0.5:
            md = "# Simple\n\nShort line.\nAnother line.\n"
        else:
            # Ensure depth > 0.5 under blueprint feature scaling (indent_level/10).
            md = "# Deep\n\n- a\n  - b\n    - c\n      - d\n        - e\n          - f\n            - g\n"
        samples.append(md)

    cfg = LinUCBConfig(alpha=0.1, feature_dim=5)
    cope = LinUCBAgent(cfg)

    actions_random: list[int] = []
    actions_greedy: list[int] = []
    actions_cope: list[int] = []

    rewards_random: list[float] = []
    rewards_greedy: list[float] = []
    rewards_cope: list[float] = []

    for md in samples:
        f_dict = extract_feature_dict(md)
        x = extract_features(md)

        a_r = int(rng.integers(0, 4))
        a_g = 0  # always ZERO_SHOT
        a_c = cope.select_action(x)

        r_r, _ = env.simulate_step(f_dict, a_r)
        r_g, _ = env.simulate_step(f_dict, a_g)
        r_c, _ = env.simulate_step(f_dict, a_c)

        cope.update(x, a_c, r_c)

        actions_random.append(a_r)
        actions_greedy.append(a_g)
        actions_cope.append(a_c)
        rewards_random.append(float(r_r))
        rewards_greedy.append(float(r_g))
        rewards_cope.append(float(r_c))

    def cumavg(rs: list[float]) -> list[float]:
        arr = np.array(rs, dtype=float)
        return (np.cumsum(arr) / (np.arange(len(arr)) + 1)).tolist()

    payload: dict[str, object] = {
        "experiment": "blueprint-mock",
        "episodes": episodes,
        "seed": seed,
        "policies": {
            "random": {"reward": rewards_random, "cumavg_reward": cumavg(rewards_random), "actions": actions_random},
            "greedy_zero_shot": {
                "reward": rewards_greedy,
                "cumavg_reward": cumavg(rewards_greedy),
                "actions": actions_greedy,
            },
            "cope_linucb": {"reward": rewards_cope, "cumavg_reward": cumavg(rewards_cope), "actions": actions_cope},
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    _ensure_dir(os.path.join("cope_research", "results"))
    if out_path is None:
        out_path = os.path.join("cope_research", "results", f"blueprint_mock_{_now_tag()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    # Plot: cumulative average reward over time.
    fig, ax = plt.subplots(figsize=(9, 5))
    x_axis = np.arange(episodes)
    ax.plot(x_axis, payload["policies"]["random"]["cumavg_reward"], label="Random")  # type: ignore[index]
    ax.plot(x_axis, payload["policies"]["greedy_zero_shot"]["cumavg_reward"], label="Greedy (ZERO_SHOT)")  # type: ignore[index]
    ax.plot(x_axis, payload["policies"]["cope_linucb"]["cumavg_reward"], label="COPE (LinUCB)")  # type: ignore[index]
    ax.set_title("COPE Blueprint Mock: Cumulative Average Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Average Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_path = out_path.replace(".json", ".png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return out_path


def run_experiment(
    *,
    experiment: str,
    episodes: int,
    runs: int,
    backend: str,
    seed: int,
    alpha: float,
    out_path: str | None,
) -> str:
    if experiment == "learning-curve":
        samples = generate_dataset(
            n_a=episodes // 3,
            n_b=episodes // 3,
            n_c=episodes - 2 * (episodes // 3),
            seed=seed,
            shuffle=True,
        )
    elif experiment == "drift":
        samples = generate_drift_dataset(episodes=episodes, split_at=episodes // 2, seed=seed)
    else:
        raise ValueError("experiment must be one of: learning-curve, drift")

    env = Environment(backend=backend, seed=seed)

    policies = ["random", "static_best", "cope"]
    results: dict[str, object] = {
        "experiment": experiment,
        "backend": backend,
        "episodes": episodes,
        "runs": runs,
        "seed": seed,
        "alpha": alpha,
        "llama_repo_id": getattr(env, "llama_repo_id", None),
        "llama_filename": getattr(env, "llama_filename", None),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policies": {},
    }

    for p in policies:
        run_payloads: list[dict[str, object]] = []
        for r in range(runs):
            run_seed = seed + 1000 * r + (0 if p == "random" else 17)
            run_payloads.append(
                _run_policy(samples=samples, policy_name=p, env=env, seed=run_seed, alpha=alpha)
            )
        results["policies"][p] = _aggregate_runs(run_payloads)

    _ensure_dir(os.path.join("cope_research", "results"))
    if out_path is None:
        out_path = os.path.join("cope_research", "results", f"{experiment}_{backend}_{_now_tag()}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--episodes", type=int, default=100)
    common.add_argument("--runs", type=int, default=5)
    common.add_argument("--backend", type=str, default="llama_cpp", choices=["llama_cpp", "groq", "mock"])
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--alpha", type=float, default=1.5)
    common.add_argument("--out", type=str, default=None, help="Output JSON path")

    sub.add_parser("learning-curve", parents=[common])
    sub.add_parser("drift", parents=[common])
    bp = sub.add_parser("blueprint-mock")
    bp.add_argument("--episodes", type=int, default=100)
    bp.add_argument("--seed", type=int, default=0)
    bp.add_argument("--out", type=str, default=None)

    args = parser.parse_args()
    if args.cmd == "blueprint-mock":
        out_path = run_blueprint_mock(episodes=args.episodes, seed=args.seed, out_path=args.out)
        print(out_path)
        return
    out_path = run_experiment(
        experiment=args.cmd,
        episodes=args.episodes,
        runs=args.runs,
        backend=args.backend,
        seed=args.seed,
        alpha=args.alpha,
        out_path=args.out,
    )
    print(out_path)


if __name__ == "__main__":
    main()

