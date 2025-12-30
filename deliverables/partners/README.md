# Research Partner Packages

This directory contains specialized bioinformatics tools developed for research partners.

## Partner Overview

| Partner | Directory | Focus Area |
|---------|-----------|------------|
| Alejandra Rojas | `alejandra_rojas/` | Arbovirus surveillance & RT-PCR primer design |
| Carlos Brizuela | `carlos_brizuela/` | Antimicrobial peptide multi-objective optimization |
| Jose Colbes | `jose_colbes/` | Protein stability & rotamer scoring |
| HIV Research | `hiv_research_package/` | HIV clinical decision support tools |

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
