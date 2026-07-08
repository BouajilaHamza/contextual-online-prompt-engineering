from __future__ import annotations

"""
Run COPE experiments on Modal with vLLM (Fast GPU Inference).
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/cope"


def _image() -> modal.Image:
    return (
        modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
        .pip_install(
            "vllm==0.5.4",
            "outlines==0.0.46",
            "pyairports",
            "matplotlib",
            "requests",
            "huggingface-hub",
            "tqdm",
            "numpy<2.0.0"
        )
        .add_local_dir(str(REPO_ROOT), remote_path=REMOTE_ROOT)
    )


app = modal.App("cope-research")
image = _image()

hf_cache = modal.Volume.from_name("cope-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("cope-results", create_if_missing=True)


@app.function(
    image=image,
    timeout=10 * 60 * 60,
    gpu="A10G",
    cpu=4,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/results": results_vol
    },
)
def run_remote(experiment: str, episodes: int, runs: int, backend: str, seed: int, alpha: float) -> dict:
    out_dir = "/root/results"
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{experiment}_{backend}_{timestamp}.json")
    
    cmd = [
        "python3",
        "-u",
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
        "--out",
        out_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Switch back to the streaming loop to catch every single line of output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        cwd="/root",
        bufsize=1
    )
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        sys.stdout.write(line)
        sys.stdout.flush()

    return_code = process.wait()
    process.stdout.close()
    
    if return_code != 0:
        raise RuntimeError(f"Experiment failed with code {return_code}")
        
    with open(out_path, "r") as f:
        results = json.load(f)
    
    hf_cache.commit()
    results_vol.commit()
    return results


@app.local_entrypoint()
def main(
    experiment: str = "learning-curve",
    episodes: int = 100,
    runs: int = 5,
    backend: str = "vllm", 
    seed: int = 0,
    alpha: float = 1.5,
) -> None:
    print(f"🚀 Launching vLLM Experiment: {experiment}...")
    results = run_remote.remote(experiment, episodes, runs, backend, seed, alpha)
    
    results_dir = REPO_ROOT / "cope_research" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    local_out = results_dir / f"{experiment}_{backend}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(local_out, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {local_out}")
    plot_script = REPO_ROOT / "cope_research" / "plot_results.py"
    if plot_script.exists():
        subprocess.check_call(["python3", str(plot_script), str(local_out)])
