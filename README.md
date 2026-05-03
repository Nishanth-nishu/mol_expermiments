# mol_experiments — 3D Molecular Conformer Generation via Equivariant Diffusion

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.3-green.svg)](https://www.rdkit.org/)

A research codebase for generating physically valid 3D molecular conformers using **E(3)-equivariant denoising diffusion probabilistic models (DDPM)** on the QM9 dataset, designed as a NeurIPS AI4Science submission pipeline.

---

## Overview

Given a molecular graph (atoms + bonds from SMILES), this system generates the physically correct 3D arrangement of those atoms — the **conformer**. The model is trained on DFT-optimized QM9 geometries and evaluated on standard MAT-R / COV-R conformer generation benchmarks.

**Four experiments** ablate architectural improvements over the fixed EGNN-DDPM baseline:

| Exp | Key Change | Reference |
|-----|-----------|-----------|
| A | Fixed baseline (DDPM + AdamW) | EDM — Hoogeboom et al. ICML 2022 |
| B | Multi-head attention over EGNN messages | EQGAT-diff — Le et al. ICLR 2024 |
| C | Conditional Flow Matching (Euler ODE, 20 steps) | Lipman et al. ICLR 2023 |
| D | Torsion-angle auxiliary loss (OPLS-AA) | TorDiff — Jing et al. NeurIPS 2022 |

---

## Repository Structure

```
mol_next_gen/
├── models/                          # Model architectures
│   ├── conformer_diffusion.py       # Base: E(3)-equivariant EGNN + DDPM (Exp A)
│   ├── attn_conformer_diffusion.py  # EQGAT-diff attention EGNN (Exp B)
│   ├── flow_matching.py             # Conditional Flow Matching (Exp C)
│   ├── geometry_constraints.py      # MMFF94/OPLS-AA geometry losses
│   └── __init__.py
│
├── autoresearch/                    # Training scripts & evaluation harness
│   ├── mol_prepare.py               # Fixed dataset loader + evaluation harness (READ-ONLY)
│   ├── mol_train.py                 # Exp A: fixed baseline trainer
│   ├── mol_train_expB.py            # Exp B: attention EGNN trainer
│   ├── mol_train_expC.py            # Exp C: flow matching trainer
│   ├── mol_train_expD.py            # Exp D: torsion auxiliary loss trainer
│   └── results.tsv                  # Aggregated experiment results
│
├── data/
│   └── prepare_qm9.py               # Download QM9 + convert to JSONL
│
├── scripts/                         # SLURM job scripts
│   ├── exp_A_baseline.sh
│   ├── exp_B_attention_egnn.sh
│   ├── exp_C_flow_matching.sh
│   ├── exp_D_torsion_aux.sh
│   └── submit_all_experiments.sh    # Auto-submit all 4 after data prep
│
├── experiments/                     # Per-experiment results (auto-generated)
│   ├── exp_A_baseline/train.log
│   ├── exp_B_attention_egnn/train.log
│   ├── exp_C_flow_matching/train.log
│   └── exp_D_torsion_aux/train.log
│
├── docs/                            # In-depth documentation
│   ├── README.md                    # Full pipeline: math, architecture, citations
│   ├── README_base_model.md         # ConformerDiffusion deep-dive
│   ├── README_exp_A.md              # Exp A: bugs fixed + baseline setup
│   ├── README_exp_B.md              # Exp B: EQGAT-diff attention math
│   ├── README_exp_C.md              # Exp C: flow matching derivation
│   └── README_exp_D.md              # Exp D: torsion angle loss
│
├── visualization/
│   └── plot_results.py              # Plot training curves + metrics table
│
├── PROMPT.txt                       # Full research audit record
└── .gitignore
```

---

## Quickstart

### 1. Setup Environment

```bash
cd mol_next_gen
python -m venv venv && source venv/bin/activate
pip install torch==2.5.1 torch-geometric rdkit-pypi numpy
```

### 2. Prepare QM9 Dataset

Downloads the QM9 SDF from public DeepChem S3 (~250 MB) and converts to JSONL:

```bash
python data/prepare_qm9.py --output data/qm9_selfies.jsonl
# ~5 minutes; writes 131,970 molecules
```

### 3. Verify Installation

```bash
PYTHONPATH=. python -c "
from models.conformer_diffusion import ConformerDiffusion
from models.attn_conformer_diffusion import AttnConformerDiffusion
from models.flow_matching import FlowMatchingConformer
from autoresearch.mol_prepare import make_dataloaders
print('All imports OK')
"
```

### 4. Run Experiments

**SLURM (recommended — runs all 4 in parallel):**
```bash
sbatch scripts/exp_A_baseline.sh
sbatch scripts/exp_B_attention_egnn.sh
sbatch scripts/exp_C_flow_matching.sh
sbatch scripts/exp_D_torsion_aux.sh
```

**Or auto-submit after data prep completes:**
```bash
bash scripts/submit_all_experiments.sh &
```

**Single GPU (sequential):**
```bash
PYTHONPATH=. python autoresearch/mol_train.py          # Exp A
PYTHONPATH=. python autoresearch/mol_train_expB.py     # Exp B
PYTHONPATH=. python autoresearch/mol_train_expC.py     # Exp C
PYTHONPATH=. python autoresearch/mol_train_expD.py     # Exp D
```

### 5. Compare Results

```bash
# Live training progress
tail -f logs/expA_*.log

# Final metrics across all experiments
grep "fully_valid:\|mat_r:" experiments/exp_*/train.log

# TSV results table
cat autoresearch/results.tsv
```

---

## Dataset: QM9

| Property | Value |
|----------|-------|
| Source | Ramakrishnan et al. Scientific Data 2014 |
| Size | 133,885 molecules |
| Atoms | H, C, N, O, F |
| Coordinates | DFT-optimized (B3LYP/6-31G(2df,p)) |
| Train / Val | 118,773 / 13,197 (90/10 split, seed=42) |

QM9 is the standard benchmark for 3D conformer generation. DFT coordinates are electronically-minimized energy geometries — physically stable and high-quality training signal.

---

## Model Architecture

The backbone is an **E(3)-equivariant EGNN** (Satorras et al., ICML 2021) trained with DDPM (Ho et al., NeurIPS 2020):

- **Input:** atom types, noisy 3D coords, bond graph, timestep
- **6 equivariant layers:** message passing with RBF distance features + bond embeddings
- **Output:** predicted clean 3D coordinates $\hat{x}_0$ (x₀ parameterization)
- **Geometry constraints:** MMFF94 bond/angle losses + OPLS-AA torsion (Exp D)
- **Inference:** DDIM (50 steps) or Euler ODE (20 steps for Exp C)

See [`docs/README.md`](docs/README.md) for the complete mathematical derivation with all equations.

---

## Evaluation Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **fully_valid** | % passing RDKit SanitizeMol | ↑ Higher |
| **MAT-R** | Mean min-RMSD to reference (Å) | ↓ Lower |
| **COV-R** | % references covered at 0.5 Å | ↑ Higher |
| **validity** | % with bond lengths within ±0.20 Å | ↑ Higher |
| **strain** | MMFF94 energy (kcal/mol) | ↓ Lower |

**MAT-R is the primary metric.** Published baselines: GeoMol=0.225 Å, EDM=0.44 Å, EQGAT-diff=0.17 Å.

---

## Documentation

| File | Contents |
|------|----------|
| [`docs/README.md`](docs/README.md) | Complete pipeline math — dataset to generation |
| [`docs/README_base_model.md`](docs/README_base_model.md) | ConformerDiffusion architecture deep-dive |
| [`docs/README_exp_A.md`](docs/README_exp_A.md) | Experiment A: 9 bugs fixed, baseline setup |
| [`docs/README_exp_B.md`](docs/README_exp_B.md) | Experiment B: attention EGNN math + ablation |
| [`docs/README_exp_C.md`](docs/README_exp_C.md) | Experiment C: flow matching derivation |
| [`docs/README_exp_D.md`](docs/README_exp_D.md) | Experiment D: torsion angle loss |

---

## Key References

1. **EDM** — Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D." ICML 2022. arXiv:2203.17003
2. **EGNN** — Satorras et al. "E(n) Equivariant Graph Neural Networks." ICML 2021. arXiv:2102.09844
3. **EQGAT-diff** — Le et al. "EQGAT-diff." ICLR 2024. arXiv:2306.01916
4. **TorDiff** — Jing et al. "Torsional Diffusion for Molecular Conformer Generation." NeurIPS 2022. arXiv:2206.01729
5. **Flow Matching** — Lipman et al. "Flow Matching for Generative Modeling." ICLR 2023. arXiv:2210.02747
6. **GeoMol** — Ganea et al. "GeoMol." NeurIPS 2021. arXiv:2106.07802
7. **DDPM** — Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020. arXiv:2006.11239
8. **DDIM** — Song et al. "Denoising Diffusion Implicit Models." ICLR 2021. arXiv:2010.02502
9. **QM9** — Ramakrishnan et al. Scientific Data 2014.

---

## License

Research code — for academic and non-commercial use.
