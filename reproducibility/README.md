# Reproducibility Package

This directory contains everything needed to reproduce the results from our paper:

**"P-adic Geometry Encodes Protein Thermodynamics: A Universal Framework for Drug Resistance Prediction"**

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate ternary-vae

# 2. Run all experiments
python run_all_experiments.py

# 3. Generate figures
python generate_figures.py
```

## Directory Structure

```
reproducibility/
├── README.md                    # This file
├── environment.yml              # Conda environment specification
├── run_all_experiments.py       # Master script to run all experiments
├── generate_figures.py          # Generate publication figures
├── expected_results.json        # Expected results for validation
└── notebooks/
    ├── Figure1_CrossDisease.ipynb    # Cross-disease performance
    ├── Figure2_Physics.ipynb         # Physics validation
    ├── Figure3_Architecture.ipynb    # Architecture diagram
    └── Figure4_Clinical.ipynb        # Clinical decision support
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

## Experiments

### 1. Cross-Disease Benchmark

Validates the p-adic VAE across 11 disease domains:

```bash
python ../scripts/experiments/run_cross_disease.py --output-dir ../results/benchmarks
```

Expected results:
- HIV: Spearman ≥ 0.89
- SARS-CoV-2: Spearman ≥ 0.85
- Tuberculosis: Spearman ≥ 0.85
- Overall: Spearman ≥ 0.85

### 2. Physics Validation

Confirms p-adic thermodynamics universality:

```bash
python ../scripts/experiments/run_physics_validation.py --output-dir ../results/physics
```

Expected results:
- ΔΔG correlation > 0.7
- Mass invariant confirmed
- 6/6 hierarchy levels passing
- Thermodynamics/kinetics separation confirmed

### 3. Individual Disease Analysis

```bash
# HIV drug resistance (23 drugs)
python ../scripts/experiments/run_full_validation.py

# SARS-CoV-2 variant analysis
python ../scripts/research/sars_cov2_analysis.py

# Tuberculosis MDR prediction
python ../scripts/research/tb_mdr_prediction.py
```

## Figures

| Figure | Description | Notebook |
|--------|-------------|----------|
| 1 | Cross-disease performance comparison | Figure1_CrossDisease.ipynb |
| 2 | P-adic physics hierarchy | Figure2_Physics.ipynb |
| 3 | Architecture diagram | Figure3_Architecture.ipynb |
| 4 | Clinical decision support | Figure4_Clinical.ipynb |
| S1 | Per-drug HIV results | Supplementary/FigureS1_HIV.ipynb |
| S2 | Transfer learning curves | Supplementary/FigureS2_Transfer.ipynb |

## Validation

After running experiments, validate against expected results:

```python
import json

with open("expected_results.json") as f:
    expected = json.load(f)

with open("../results/benchmarks/latest.json") as f:
    actual = json.load(f)

# Check key metrics
assert actual["overall_spearman_mean"] >= expected["overall_spearman_mean"] - 0.05
print("Validation passed!")
```

## Hardware Requirements

| Configuration | Time | GPU Memory |
|--------------|------|------------|
| CPU only | ~4 hours | N/A |
| Single GPU (RTX 3090) | ~30 min | 8 GB |
| Multi-GPU (4x A100) | ~10 min | 4x 20 GB |

## Citation

```bibtex
@article{ternary_vae_2025,
  title={P-adic Geometry Encodes Protein Thermodynamics},
  author={AI Whisperers},
  journal={Nature Methods},
  year={2025},
  doi={10.1038/s41592-025-XXXXX}
}
```

## Contact

For questions about reproducibility, please open an issue on GitHub or contact:
- Email: support@aiwhisperers.com

## License

- Code: PolyForm Non-Commercial 1.0.0
- Results: CC-BY 4.0
