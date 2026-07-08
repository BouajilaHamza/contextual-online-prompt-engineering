from __future__ import annotations

"""
Deep Inspect vLLM on Modal.
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
    timeout=600,
    gpu="A10G",
    volumes={
        "/root/.cache/huggingface": hf_cache,
    },
)
def inspect_vllm():
    print("--- Environment Check ---")
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.get_device_name(0)}")
    
    print("\n--- vLLM Import ---")
    try:
        from vllm import LLM, SamplingParams
        print("vLLM imported successfully.")
    except Exception as e:
        print(f"vLLM import failed: {e}")
        return

    print("\n--- Model Loading (Qwen 0.5B) ---")
    # Use extremely low memory just to see it pass
    try:
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            max_model_len=1024
        )
        print("Model loaded successfully!")
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
        outputs = llm.generate(["The capital of France is"], sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            
    except Exception as e:
        print(f"Model loading or generation failed: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    inspect_vllm.remote()
