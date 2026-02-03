# Bioinformatics Research Packages

**Doc-Type:** Package Index · Version 2.0 · Updated 2026-01-26 · AI Whisperers

This directory contains specialized bioinformatics tools built on p-adic hyperbolic geometry from the Ternary VAE project.

---

## Package Overview

| Package | Directory | Focus Area | Status | Key Metric |
|---------|-----------|------------|:------:|------------|
| **Protein Stability** | `protein_stability_ddg/` | DDG prediction | 95% Ready | LOO rho=0.585 |
| **Arbovirus Surveillance** | `arbovirus_surveillance/` | Primers & forecasting | 90% Ready | 7 viruses covered |
| **Antimicrobial Peptides** | `antimicrobial_peptides/` | AMP optimization | 70% Ready | Spearman=0.656 |
| **HIV Research** | `hiv_research_package/` | Clinical decision support | Complete | Stanford HIVdb |

---

## Package Status Summary

### Protein Stability Prediction (95%)
- **LOO CV validated:** Spearman rho = 0.585, p < 0.001
- **95% CI:** [0.341, 0.770] via bootstrap
- **Features:** Rosetta-blind instability detection, sequence-only predictions
- **Caveat:** N=52 result NOT comparable to N=669 literature benchmarks

### Arbovirus Surveillance (90%)
- **Pan-arbovirus primers:** DENV-1/2/3/4, ZIKV, CHIKV, MAYV
- **Key discovery:** DENV-4 cryptic diversity (97.4% no conserved windows)
- **P-adic integration:** TrainableCodonEncoder producing meaningful results
- **Features:** Hyperbolic trajectory forecasting, primer stability scanning

### Antimicrobial Peptide Design (70%)
- **MIC prediction:** PeptideVAE (Spearman=0.656)
- **NSGA-II optimization:** Working in latent space
- **Limitations:** Pseudomonas/Staphylococcus models non-significant
- **Features:** Pathogen-specific (B1), microbiome-safe (B8), synthesis-optimized (B10)

### HIV Clinical Tools (Complete)
- **TDR screening (H6):** Validated against Stanford HIVdb
- **LA selection (H7):** Eligibility assessment for long-acting injectables
- **Features:** Clinical decision support workflows

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
- **Email:** support@aiwhisperers.com

---

*Last updated: 2026-01-26*
