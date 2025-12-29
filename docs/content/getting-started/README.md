# Getting Started

> **Get up and running with Ternary VAE in minutes.**

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Training Run

```bash
# Run training with default config
python scripts/train/train.py --config configs/ternary.yaml

# Quick test run (fewer epochs)
python scripts/train/train.py --config configs/ternary_fast_test.yaml
```

### Analyze HIV Data

```bash
# Run comprehensive HIV analysis
python scripts/analyze_all_datasets.py

# Generate clinical decision support
python scripts/clinical_applications.py
```

---

## What Is This Project?

Ternary VAE is a novel machine learning architecture that uses **hyperbolic geometry** and **3-adic number theory** to encode biological sequences in ways that naturally capture evolutionary relationships.

### Key Insight

Protein evolution has a tree-like structure. Hyperbolic space is uniquely suited to represent trees because distance grows exponentially (unlike flat Euclidean space).

### What It Does

| Capability | Description |
|:-----------|:------------|
| Drug resistance prediction | Predicts which mutations confer resistance |
| Immune evasion mapping | Maps how viruses hide from antibodies |
| Vulnerability hotspots | Identifies regions that cannot mutate |
| Therapeutic targets | Suggests intervention points |

---

## Key Achievements

| Achievement | Result |
|:------------|:-------|
| HIV drug resistance | +0.89 Spearman correlation across 23 drugs |
| Glycan shield mapping | 7 sentinel positions identified |
| Multi-disease platform | 11 diseases with unified framework |
| Vaccine targets | 387 candidates ranked by stability |

---

## Next Steps

| Topic | Link |
|:------|:-----|
| Architecture overview | [architecture/README.md](../architecture/README.md) |
| Theory deep dive | [theory/README.md](../theory/README.md) |
| HIV research findings | [research/hiv/README.md](../research/hiv/README.md) |
| Multi-disease platform | [research/multi-disease/README.md](../research/multi-disease/README.md) |
| Contributing | [development/contributing.md](../development/contributing.md) |

---

## Tutorials

| Tutorial | Description |
|:---------|:------------|
| [Basic Training](tutorials/basic-training.md) | Train your first model |
| [HIV Analysis](tutorials/hiv-analysis.md) | Analyze HIV sequences |
| [Uncertainty Quantification](tutorials/uncertainty.md) | Add confidence intervals |
| [Transfer Learning](tutorials/transfer-learning.md) | Cross-disease transfer |

---

_Last updated: 2025-12-28_
