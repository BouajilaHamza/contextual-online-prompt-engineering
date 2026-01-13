from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

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
from cope_research.src.environment import Environment  # noqa: E402
from cope_research.src.features import extract_features  # noqa: E402


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

    args = parser.parse_args()
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

