from __future__ import annotations

"""
Run COPE experiments on Modal.

Usage (after you configure Modal auth on your machine):

  modal run cope_research/modal_app.py --experiment learning-curve --episodes 100 --runs 5

This will execute remotely and download the resulting JSON to local
`cope_research/results/` (plots are generated locally afterward).
"""

import os
import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/cope"


def _image() -> modal.Image:
    # NOTE: llama-cpp-python may compile on first build; Modal caches images.
    # If build time is high, consider switching to a prebuilt image later.
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install_from_requirements("requirements.txt")
    )


app = modal.App("cope-research")
image = _image()

hf_cache = modal.Volume.from_name("cope-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    timeout=60 * 60,  # 1h
    cpu=4,
    volumes={"/root/.cache/huggingface": hf_cache},
    mounts=[modal.Mount.from_local_dir(str(REPO_ROOT), remote_path=REMOTE_ROOT)],
)
def run_remote(experiment: str, episodes: int, runs: int, backend: str, seed: int, alpha: float) -> str:
    # Execute the existing runner in the mounted repo and return the output path.
    cmd = [
        "python3",
        f"{REMOTE_ROOT}/cope_research/run_experiment.py",
        experiment,
        "--backend",
        backend,
        "--episodes",
        str(episodes),
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--alpha",
        str(alpha),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    # Ensure HF cache is persisted
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main(
    experiment: str = "learning-curve",
    episodes: int = 100,
    runs: int = 5,
    backend: str = "llama_cpp",
    seed: int = 0,
    alpha: float = 1.5,
) -> None:
    out_path = run_remote.remote(experiment, episodes, runs, backend, seed, alpha)
    print(out_path)

    # The remote path is inside the mounted repo path. We mirror that locally.
    # Example remote output: cope_research/results/<file>.json
    local_out = REPO_ROOT / out_path
    if not local_out.exists():
        # In some Modal setups, stdout prints a relative path; fall back to local join.
        local_out = (REPO_ROOT / "cope_research" / "results" / os.path.basename(out_path)).resolve()

    # Generate plots locally for convenience.
    if local_out.exists() and local_out.suffix == ".json":
        subprocess.check_call(
            ["python3", str(REPO_ROOT / "cope_research" / "plot_results.py"), str(local_out)]
        )

