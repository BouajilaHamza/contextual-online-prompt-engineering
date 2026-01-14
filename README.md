# COPE: Contextual Online Prompt Engineering (Research Framework)

This repository contains a lightweight Python research scaffold for the COPE paper idea:
learning an **online contextual bandit policy** (LinUCB) that maps **Markdown file context**
to an optimal **prompting strategy** (arms 0–3).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run experiments (mock environment)

Learning curve (COPE vs baselines):

```bash
python cope_research/run_experiment.py learning-curve --episodes 100 --runs 20
python cope_research/plot_results.py cope_research/results/learning_curve.json
```

Blueprint mock (fast RL sanity check):

```bash
python cope_research/run_experiment.py blueprint-mock --episodes 100
```

Adaptation / drift:

```bash
python cope_research/run_experiment.py drift --episodes 100 --runs 20
python cope_research/plot_results.py cope_research/results/drift.json
```

## Real local open-source LLM (default)

By default, the experiment runner uses a **local open-source GGUF model** via `llama.cpp`
(`backend=llama_cpp`). On first run, it will auto-download a small model from Hugging Face:

- Repo: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- File: `qwen2.5-0.5b-instruct-q4_k_m.gguf`

You can override:

- `COPE_GGUF_REPO`
- `COPE_GGUF_FILE`
- `COPE_GGUF_PATH` (to point at a local GGUF file)

Example:

```bash
python cope_research/run_experiment.py learning-curve --backend llama_cpp --episodes 60 --runs 3
```

Results are saved to `cope_research/results/*.json` and plots to `*.png`.

## Run on Modal (recommended for longer runs)

If local inference is slow, you can run the same experiment remotely on Modal:

```bash
pip install -r requirements.txt
modal run cope_research/modal_app.py --experiment learning-curve --episodes 100 --runs 5
```

This uses the same `llama_cpp` backend and will populate `cope_research/results/` locally
with the JSON + plots (HF model cache is persisted in a Modal volume).

## Hosted open-source model via Groq (optional)

If you prefer a hosted open-weights model, the framework also supports **Groq** via an
OpenAI-compatible API (`backend=groq`).

Set:

- `GROQ_API_KEY`: your Groq API key
- `GROQ_MODEL`: optional (default: `llama-3.1-8b-instant`)

Then run:

```bash
python cope_research/run_experiment.py learning-curve --backend groq
```

> Note: LLM-based rewards depend heavily on the model; small/open models often benefit more
> from schema-enforced prompting and chunking strategies.

