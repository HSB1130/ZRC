# Zero-Resource Curriculum (ZRC)

This repository contains a reference Python implementation of the Zero-Resource Curriculum (ZRC) framework described in the paper "Zero-Resource Curriculum Reinforcement Learning for LLMs".

## Project layout

```
zrc/
  config.py              # Dataclasses collecting hyperparameters
  difficulty/            # Intrinsic difficulty estimation
  clustering/            # DP-means clustering implementation
  changepoint/           # Bayesian online change point detection
  sampling/              # Bandit-based curriculum sampling
  curriculum/            # Orchestration of the full curriculum loop
  rl/                    # Hooks for policy optimisation
scripts/
  run_training.py        # Toy driver showcasing curriculum updates
```

## Getting started

Create a virtual environment with Python 3.9+ and install the dependencies required for transformers-based models, TRL fine-tuning, and vLLM inference.

```
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118  # Choose the wheel that matches your CUDA/CPU setup
pip install numpy transformers \"trl[vllm]\" vllm datasets
```

Run the toy example to observe curriculum dynamics on synthetic tasks:

```
python scripts/run_training.py --split train[:200] --steps 20
```

The script now loads the **agentica-org/DeepScaleR-Preview-Dataset**, fine-tunes **Qwen/Qwen2.5-Math-7B** with TRL's **GRPOTrainer**, and performs curriculum-guided updates while using vLLM for all rollouts.

## Integrating with RL training

`ZeroResourceCurriculum` exposes methods to

1. `select_cluster()` – choose which cluster to draw tasks from.
2. `sample_tasks_from_cluster(cluster_id, num_tasks)` – list task IDs for the next batch.
3. `record_outcomes(task_ids, successes, verification_failures)` – update internal statistics after an RL step.

You can connect the curriculum to your RL pipeline by wiring these hooks around your policy optimisation loop. The `zrc.rl` module provides the `RLUpdater` helper, which accepts a user-defined update function for custom experiments. The demo script shows how to pair the curriculum with TRL's GRPOTrainer (configured with `use_vllm=True`) while the trainer handles vLLM-backed rollouts. `zrc.models.load_causal_lm` bootstraps transformer models, and the demo constructs semantic embeddings from the policy token embeddings—feel free to swap in a richer encoder if desired.

## Notes

- Modules rely on PyTorch, transformers, TRL, and vLLM; ensure GPU drivers and CUDA/cuDNN match your environment.
- Hugging Face `datasets` is required for loading `agentica-org/DeepScaleR-Preview-Dataset`.
- The DP-means and BOCPD implementations are concise reference versions intended for research prototyping. For large-scale training you may want to port performance-critical parts to JAX/PyTorch kernels.
- The stabilisation and bandit mechanisms follow the high-level description in the ZRC paper; tune the hyperparameters in `zrc/config.py` to match your setup.
