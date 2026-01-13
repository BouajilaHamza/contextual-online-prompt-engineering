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

Adaptation / drift:

```bash
python cope_research/run_experiment.py drift --episodes 100 --runs 20
python cope_research/plot_results.py cope_research/results/drift.json
```

## Real LLM environment (optional)

By default, experiments use a **mock LLM** so results are reproducible without API keys.
The framework also includes an optional **Groq (OpenAI-compatible) backend** in
`cope_research/src/environment.py`.

Set:

- `GROQ_API_KEY`: your Groq API key
- `GROQ_MODEL`: optional (default: `llama-3.1-8b-instant`)

Then run:

```bash
python cope_research/run_experiment.py learning-curve --backend groq
```

> Note: LLM-based rewards depend heavily on the model; small/open models often benefit more
> from schema-enforced prompting and chunking strategies.

