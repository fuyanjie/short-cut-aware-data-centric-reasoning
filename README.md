# Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models

Code for the NeurIPS 2026 submission: *"Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models"*.

## Overview

Large language models often exploit **spurious shortcuts** in training data rather than learning genuine reasoning. This project proposes a **gradient-based framework** to detect and correct shortcut learning, consisting of three components:

1. **ShortcutScore** — A per-sample metric that quantifies shortcut reliance using gradient alignment analysis:
   `S(s) = α · B(s) + β · C(s)`, where `B(s)` measures gradient misalignment with the validation set and `C(s)` captures answer-reasoning inconsistency.

2. **Shortcut-aware Reweighting** — Down-weights high-shortcut samples during training:
   `w(s) = exp(-λ · S(s))`

3. **Gradient Surgery** — Projects gradients to remove shortcut-correlated components via alignment projection and answer-gradient suppression.

## Project Structure

```
├── run_all.py              # Main experiment runner
├── hp_search.py            # Hyperparameter search (multi-GPU parallel grid search)
├── src/
│   ├── config.py           # Configuration (dual-profile: local/server)
│   ├── data.py             # Synthetic datasets (Math, Financial, Causal)
│   ├── data_realworld.py   # Real-world datasets (GSM8K, MATH)
│   ├── model.py            # SmallGPT transformer model
│   ├── methods.py          # Core: ShortcutScore, Reweighting, Gradient Surgery
│   ├── trainer.py          # Training functions (baselines + our method)
│   ├── evaluate.py         # Evaluation metrics (accuracy, robustness, F1)
│   └── visualize.py        # Result table generation
├── latex/                  # Paper LaTeX source
├── results/                # Experiment output tables
├── requirements.txt
└── README.md
```

## Datasets

### Synthetic (3 datasets)
| Dataset | True Rule | Shortcut Rule |
|---------|-----------|---------------|
| Math-Arithmetic | `a + b ≥ 10` → SAT | `a ≥ 5` → SAT |
| Financial-Analysis | `margin ≥ 5 AND debt < 5` → SAT | `revenue ≥ 5` → SAT |
| Causal-Reasoning | `x ≥ 5 AND z < 3` → CAUS | `corr_xy ≥ 5` → CAUS |

Training data uses 70% shortcut labels / 30% true labels. Validation and test always use true labels.

### Real-World (2 datasets)
| Dataset | Source | Shortcut Rule |
|---------|--------|---------------|
| GSM8K | Grade school math (OpenAI) | Sum of all numbers in question |
| MATH | Competition math (Hendrycks et al.) | Largest number in problem |

## Methods Compared (13 total)

| Method | Type | Description |
|--------|------|-------------|
| Standard Fine-Tuning | Baseline | Vanilla cross-entropy training |
| Self-Consistency | Inference | Sample multiple, majority vote |
| Data Filtering | Data-centric | Remove high-confidence shortcut samples |
| JTT | Data-centric | Just Train Twice (Liu et al., 2021) |
| Focal Loss | Loss-based | Down-weight easy examples (Lin et al., 2017) |
| Group DRO | Distributionally Robust | Minimize worst-group loss (Sagawa et al., 2020) |
| IRM | Invariant Learning | Invariant Risk Minimization (Arjovsky et al., 2019) |
| V-REx | Invariant Learning | Variance Risk Extrapolation (Krueger et al., 2021) |
| Fishr | Invariant Learning | Match gradient variance (Rame et al., 2022) |
| LfF | Debiasing | Learning from Failure (Nam et al., 2020) |
| Influence Filtering | Influence-based | Remove harmful training samples |
| Meta-Reweighting | Meta-learning | Learn sample weights on validation set |
| **Ours (Full)** | **Gradient-aware** | **ShortcutScore + Reweighting + Gradient Surgery** |

## Quick Start

### Installation

```bash
pip install torch numpy matplotlib datasets transformers
```

### Running Experiments

```bash
# Synthetic datasets only (default)
python3 run_all.py

# Real-world datasets only (GSM8K + MATH)
DATASET_TYPE=realworld python3 run_all.py

# All 5 datasets
DATASET_TYPE=all python3 run_all.py
```

### Scale Profiles

The codebase supports two scale profiles, auto-detected based on hardware:

| Profile | Model Size | Training Data | Use Case |
|---------|-----------|---------------|----------|
| `local` | 277K params | 500 samples | Quick local iteration |
| `server` | 19M params (synthetic) / 163M params (NL) | 10K+ samples | GPU servers |

Override with environment variable:
```bash
EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py
```

## Running on a Server

### Basic Experiments (SSH-safe)

```bash
# Option 1: nohup (simplest, survives SSH disconnection)
nohup env EXPERIMENT_SCALE=server python3 run_all.py > run.log 2>&1 &

# Option 2: tmux (recommended — can reconnect to see live output)
tmux new -s experiment
EXPERIMENT_SCALE=server python3 run_all.py
# Ctrl+B, D to detach; tmux attach -t experiment to reconnect

# Synthetic only
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=synthetic python3 run_all.py > run_synthetic.log 2>&1 &

# Real-world only (GSM8K + MATH)
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=realworld python3 run_all.py > run_realworld.log 2>&1 &

# All datasets
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py > run_all.log 2>&1 &
```

> **Note:** Use `nohup env VAR=val python3 ...` (not `nohup VAR=val python3 ...`). The `env` command is required for `nohup` to recognize environment variables.

### Hyperparameter Search

The `hp_search.py` script performs a parallel grid search over SART hyperparameters across all available GPUs.

```bash
# Phase 1: Primary search over (lambda, gamma, rho)
# 60 configs, ~2.25 hours on 4× H100
nohup env EXPERIMENT_SCALE=server python3 hp_search.py --output-dir hp_results > hp_search.log 2>&1 &

# Phase 2: Fine-tune (tau_A, tau_R, phase3_lr_factor) around best from Phase 1
# 36 configs, ~1.5 hours on 4× H100
nohup env EXPERIMENT_SCALE=server python3 hp_search.py --phase 2 --output-dir hp_results > hp_search_p2.log 2>&1 &

# Quick smoke test (8 configs, local profile, ~2 min)
python3 hp_search.py --smoke-test --output-dir hp_test
```

**Search space (Phase 1):**

| Parameter | Current Default | Search Values | Role |
|-----------|----------------|---------------|------|
| `lambda_` | 2.0 | 0.2, 0.5, 0.8, 1.0, 1.5 | Reweighting strength |
| `gamma` | 0.8 | 0.1, 0.2, 0.3, 0.5 | Gradient projection strength |
| `rho` | 0.7 | 0.1, 0.3, 0.5 | Answer-gradient suppression |

**Search space (Phase 2 — around best from Phase 1):**

| Parameter | Search Values | Role |
|-----------|---------------|------|
| `tau_A` | 0.1, 0.2, 0.3, 0.4 | Alignment threshold |
| `tau_R` | 0.3, 0.5, 0.7 | Concentration threshold |
| `phase3_lr_factor` | 0.3, 0.5, 0.7 | Phase 3 learning rate multiplier |

**Output files:**
- `hp_results/results_final.json` — All configs ranked by combined score
- `hp_results/best_config.json` — Best hyperparameter configuration
- `hp_results/results_partial.json` — Incremental results (check progress)

**Monitoring progress:**
```bash
# Check how many configs have completed
cat hp_results/results_partial.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} configs done')"

# View current top results
tail -20 hp_search.log
```

### After Hyperparameter Search

Once you find the best hyperparameters, update `src/config.py` and re-run `run_all.py` for final results:
```bash
# Re-run full experiments with optimized hyperparameters
nohup env EXPERIMENT_SCALE=server python3 run_all.py > run_final.log 2>&1 &
```

### Git on Server

```bash
# Save and push results
git add -A
git commit -m "Add experiment results"
git push origin main

# Pull latest changes
git config pull.rebase true   # one-time setup
git pull
```

## Results (Server-Scale, Synthetic Datasets)

### Main Results (averaged across 3 datasets)

| Method | Accuracy | Robustness | Reasoning | SC Det. F1 |
|--------|----------|------------|-----------|------------|
| Standard Fine-Tuning | 59.6% | 1.9% | 43.8% | — |
| Self-Consistency | 64.1% | 12.4% | 43.8% | — |
| Data Filtering | 68.6% | 20.2% | 57.7% | 0.82 |
| JTT | 59.0% | 1.0% | 43.1% | — |
| Focal Loss | 60.2% | 3.8% | 44.8% | — |
| Group DRO | **78.2%** | **48.2%** | **69.9%** | — |
| **SART (Ours)** | 75.8% | 39.9% | 66.0% | 0.66 |

SART achieves +16.2pp accuracy and +38.0pp robustness over SFT, and the **best robustness on Financial-Analysis** (58.1%, surpassing Group DRO's 53.9%).

### Ablation — Component Contributions

| Configuration | Accuracy | Robustness | Grad. Align. |
|---------------|----------|------------|--------------|
| Standard FT | 59.6% | 1.9% | −0.07 |
| Reweighting Only | 64.7% | 13.1% | — |
| Gradient Surgery Only | 81.0% | 51.3% | — |
| **Full Method (Both)** | 75.8% | 39.9% | **+0.10** |

Gradient Surgery is the primary mechanism. The full method achieves the best gradient alignment (+0.10 vs −0.07 for SFT).

## Model Architecture

**SmallGPT**: Minimal causal transformer (embedding + positional encoding + Transformer encoder + output head).

| Config | Vocab | d_model | Layers | Heads | d_ff | Max Seq Len |
|--------|-------|---------|--------|-------|------|-------------|
| Synthetic | 35 | 512 | 6 | 8 | 2048 | 24 |
| NL (GPT-2) | 50,257 | 768 | 12 | 12 | 3,072 | 512 |

## Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Alignment weight | α | 1.0 | Weight for alignment component in ShortcutScore |
| Concentration weight | β | 1.0 | Weight for concentration component |
| Alignment threshold | τ_A | 0.3 | Threshold: shortcut if alignment < τ_A |
| Concentration threshold | τ_R | 0.5 | Threshold: shortcut if concentration > τ_R |
| Reweighting strength | λ | 2.0 | Exponential decay rate for sample weights |
| Projection strength | γ | 0.8 | Gradient alignment projection intensity |
| Suppression strength | ρ | 0.7 | Answer-gradient suppression intensity |

All SART hyperparameters can be overridden via the `cfg` dict passed to `train_our_method()`, or searched automatically via `hp_search.py`.

## Citation

```bibtex
@inproceedings{fu2026gradient,
  title={Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models},
  author={Fu, Yanjie},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is for academic research purposes.
