# Research Partner Packages

This directory contains specialized bioinformatics tools developed for research partners.

## Partner Overview

| Partner | Directory | Focus Area | Status |
|---------|-----------|------------|--------|
| Jose Colbes | `jose_colbes/` | Protein stability (DDG) prediction | 95% Ready |
| Alejandra Rojas | `alejandra_rojas/` | Arbovirus primers & surveillance | 85% Ready |
| Carlos Brizuela | `carlos_brizuela/` | Antimicrobial peptide optimization | 70% Ready |
| HIV Research | `hiv_research_package/` | HIV clinical decision support | Complete |

## Package Status Summary

### Jose Colbes - Protein Stability (95%)
- LOO CV validated: Spearman rho = 0.585, p < 0.001
- 95% CI: [0.341, 0.770] via bootstrap
- Rosetta-blind instability detection working
- **Gap:** Documentation polish only

### Alejandra Rojas - Arbovirus (85%)
- Pan-arbovirus primers: DENV-1/2/3/4, ZIKV, CHIKV, MAYV
- Clade-specific DENV-4 primers designed
- In-silico PCR validation complete
- **Gap:** Extend to additional viruses (scripts ready)

### Carlos Brizuela - AMP Design (70%)
- MIC prediction: PeptideVAE (r=0.74) working
- NSGA-II sequence-space optimization fixed
- Pathogen-specific design (B1) generating candidates
- **Gap:** Toxicity/stability are heuristics, not ML

### HIV Research - Clinical Tools (Complete)
- TDR screening (H6) validated
- LA selection (H7) integrated with Stanford HIVdb

## Quick Links

### Notebooks
- [AMP Navigator](carlos_brizuela/notebooks/brizuela_amp_navigator.ipynb) - NSGA-II peptide optimization
- [Rotamer Scoring](jose_colbes/notebooks/colbes_scoring_function.ipynb) - Protein stability prediction
- [Serotype Forecaster](alejandra_rojas/notebooks/rojas_serotype_forecast.ipynb) - Arbovirus surveillance

### Key Scripts
| Script | Partner | Description |
|--------|---------|-------------|
| `A2_pan_arbovirus_primers.py` | Rojas | Pan-arbovirus primer design |
| `B1_pathogen_specific_design.py` | Brizuela | WHO priority pathogen targeting |
| `B8_microbiome_safe_amps.py` | Brizuela | Microbiome-compatible peptides |
| `C1_rosetta_blind_detection.py` | Colbes | Rosetta-missed instability detection |
| `H6_tdr_screening.py` | HIV | Transmitted drug resistance |

## Usage

Each partner package is self-contained with:
- `README.md` - Package documentation
- `scripts/` - Command-line tools
- `notebooks/` - Interactive Jupyter notebooks
- `data/` - Reference datasets
- `docs/` - User guides
- `results/` - Output directories

See individual package READMEs for detailed usage instructions.

---
*Last updated: 2026-01-08*
