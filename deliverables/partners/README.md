# Bioinformatics Research Packages

**Production-ready tools for drug discovery and pathogen surveillance**

[![Packages](https://img.shields.io/badge/packages-4%20validated-blue.svg)]()

This directory contains specialized bioinformatics tools built on p-adic hyperbolic geometry from the [Ultrametric Antigen AI](../../README.md) project.

---

## Package Overview

| Package | Directory | Focus Area | Status | Key Metric |
|---------|-----------|------------|:------:|------------|
| **Protein Stability** | `protein_stability_ddg/` | DDG prediction | 95% Ready | LOO ρ=0.521 |
| **Arbovirus Surveillance** | `arbovirus_surveillance/` | Primers & forecasting | 90% Ready | 7 viruses covered |
| **Antimicrobial Peptides** | `antimicrobial_peptides/` | AMP optimization | 90% Ready | Mean r=0.656 |
| **HIV Research** | `hiv_research_package/` | Clinical decision support | Complete | Stanford HIVdb |

---

## Package Status Summary

### Protein Stability Prediction (95%)
- **LOO CV validated:** Spearman ρ = 0.521, p < 0.001
- **95% CI:** [0.21, 0.80] via bootstrap (N=52)
- **Speed advantage:** <0.1 sec/mutation vs 30 sec - 30 min (FoldX/Rosetta)
- **Unique capability:** Rosetta-blind instability detection (23.6% of cases)
- **Caveat:** N=52 result NOT comparable to N=669 literature benchmarks

### Arbovirus Surveillance (90%)
- **Pan-arbovirus primers:** DENV-1/2/3/4, ZIKV, CHIKV, MAYV
- **CDC primer validation:** 60% recovery rate
- **Key discovery:** DENV-4 cryptic diversity (97.4% no conserved windows)
- **P-adic integration:** Hyperbolic variance identifies orthogonal signals

### Antimicrobial Peptide Design (90%)
- **5 pathogen models:** All statistically significant (p<0.001)
- **Mean Spearman:** r=0.656 (outperforms sklearn baseline 0.56)
- **Per-pathogen:** General r=0.608, P. aeruginosa r=0.506, E. coli r=0.492
- **Limitation:** S. aureus has moderate confidence (r=0.35)

### HIV Clinical Tools (Complete)
- **TDR screening (H6):** Validated against Stanford HIVdb
- **LA selection (H7):** Eligibility assessment for long-acting injectables
- **200K sequences analyzed** with 7 conjectures validated

---

## Quick Links

### Notebooks
- [AMP Navigator](antimicrobial_peptides/notebooks/brizuela_amp_navigator.ipynb) - NSGA-II peptide optimization
- [Scoring Function](protein_stability_ddg/notebooks/colbes_scoring_function.ipynb) - Protein stability prediction
- [Serotype Forecaster](arbovirus_surveillance/notebooks/rojas_serotype_forecast.ipynb) - Arbovirus surveillance

### Key Scripts
| Script | Package | Description |
|--------|---------|-------------|
| `A2_pan_arbovirus_primers.py` | Arbovirus | Pan-arbovirus primer design |
| `B1_pathogen_specific_design.py` | AMP | WHO priority pathogen targeting |
| `B8_microbiome_safe_amps.py` | AMP | Microbiome-compatible peptides |
| `C1_rosetta_blind_detection.py` | DDG | Rosetta-missed instability detection |
| `C4_mutation_effect_predictor.py` | DDG | DDG prediction CLI |
| `H6_tdr_screening.py` | HIV | Transmitted drug resistance |
| `H7_la_injectable_selection.py` | HIV | LA injectable eligibility |

---

## Usage

Each package is self-contained with:
- `README.md` - Package documentation
- `scripts/` - Command-line tools
- `notebooks/` - Interactive Jupyter notebooks
- `data/` - Reference datasets (where applicable)
- `docs/` - User guides
- `results/` - Output directories
- `validation/` - Validation reports

See individual package READMEs for detailed usage instructions.

---

## Outreach Resources

| Resource | Description |
|----------|-------------|
| [CONTACT_LIST_200_RESEARCHERS.md](CONTACT_LIST_200_RESEARCHERS.md) | 200 researcher contacts across 8 categories |
| [CONTACTS_VERIFIED_EMAILS.csv](CONTACTS_VERIFIED_EMAILS.csv) | CSV with verified emails for easy import |
| [OUTREACH_PITCH_DECKS.md](OUTREACH_PITCH_DECKS.md) | Audience-specific pitch templates |

---

## Technical Documentation

| Document | Description |
|----------|-------------|
| [DELIVERABLES_INDEX.md](DELIVERABLES_INDEX.md) | Consolidated verification index |
| [CLAUDE.md](CLAUDE.md) | AI assistant context for this directory |

---

## Citation

If you use any of these packages, please cite:

```bibtex
@software{ternary_vae_bioinformatics,
  author = {{AI Whisperers}},
  title = {Ternary VAE Bioinformatics: P-adic Geometric Tools for Molecular Biology},
  year = {2026},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics}
}
```

---

## Contact

- **Repository:** [github.com/Ai-Whisperers/ternary-vaes-bioinformatics](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics)
- **Issues:** GitHub Issues
- **Email:** ai.whisperer.wvdp@gmail.com

---

*Version 2.1 · Updated 2026-02-04*
