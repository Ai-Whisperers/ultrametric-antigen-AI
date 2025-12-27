# Ternary VAE Bioinformatics

[![License: PolyForm Non‑Commercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Non‑Commercial%201.0.0-lightgrey.svg)](LEGAL_AND_IP/LICENSE)
[![License: CC‑BY‑4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LEGAL_AND_IP/RESULTS_LICENSE.md)
[![Open Medicine Policy](https://img.shields.io/badge/Open%20Medicine-Policy-blue.svg)](LEGAL_AND_IP/OPEN_MEDICINE_POLICY.md)
[![GitHub stars](https://img.shields.io/github/stars/Ai-Whisperers/ternary-vaes-bioinformatics?style=social)](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics)

---

## 📖 Overview

**Ternary VAE** is a cutting‑edge variational auto‑encoder that learns representations in **hyperbolic geometry** and **3‑adic number spaces**. It is designed for bioinformatics applications such as:

- **Geometric vaccine design** for HIV and emerging pathogens.
- **Drug‑interaction modeling** using manifold‑centric embeddings.
- **Codon‑space exploration** for synthetic biology.
- **Agricultural drug discovery** (e.g., Pasteur Molecule‑Binding project).

The project follows an **Open‑Medicine** philosophy: all scientific outputs (data, figures, model weights) are released under **CC‑BY‑4.0**, while the source code remains under the **PolyForm Non‑Commercial 1.0.0** license to prevent exclusive commercial exploitation.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the core training script (example)
python scripts/train/train.py --config configs/ternary.yaml
```

> **Tip**: The repository includes a `Dockerfile` for reproducible container builds.

---

## 📚 Documentation

- **Theory & Foundations** – detailed mathematical background, biological context, and validation strategy: `DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/`
- **Research Proposals** – organized proposals for future work: `DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/09_BIBLIOGRAPHY_AND_RESOURCES/RESEARCH_PROPOSALS/`
- **Project Management** – roadmaps, risk register, and code‑health metrics: `DOCUMENTATION/02_PROJECT_MANAGEMENT/`
- **API Reference** – generated automatically from the `src/` package (see `docs/` after running `scripts/doc_builder.py`).

### 🧬 HIV Analysis Documentation

Comprehensive analysis of 200,000+ HIV records using p‑adic hyperbolic geometry:

| Document | Description |
|----------|-------------|
| [Executive Summary](research/bioinformatics/codon_encoder_research/hiv/documentation/EXECUTIVE_SUMMARY.md) | Key findings and methodology overview |
| [Quick Start](research/bioinformatics/codon_encoder_research/hiv/documentation/quickstart/QUICK_START.md) | Run the analysis in 10 minutes |
| [Literature Review](research/bioinformatics/codon_encoder_research/hiv/documentation/LITERATURE_REVIEW.md) | 150 papers for further reading |
| [Full Documentation](research/bioinformatics/codon_encoder_research/hiv/README.md) | Complete HIV module README |

**Key Results:**
- **387 vaccine targets** ranked by evolutionary stability
- **85% tropism prediction accuracy** using geometric features
- **Position 22** discovered as top tropism determinant (novel finding)
- **r = 0.41 correlation** between hyperbolic distance and drug resistance
- **P-adic vs Hamming**: Spearman r = 0.8339 (strong geometry validation)

### Clinical Decision Support (NEW - 2025-12-26)

| Finding | Value |
|---------|-------|
| Top Vaccine Candidate | **TPQDLNTML** (Gag, priority: 0.970) |
| MDR High-Risk Sequences | 2,489 (34.8% of screened) |
| Druggable Tat Targets | 247 kinases/receptors |
| MDR Mutations Identified | 1,032 enriched signatures |
| Host-Directed Targets | 19 HIV proteins → 3+ druggable hosts |

See [`results/clinical_applications/CLINICAL_REPORT.md`](results/clinical_applications/CLINICAL_REPORT.md) for full details.

### P-adic VAE Drug Resistance Prediction (NEW - 2025-12-27)

**Comprehensive validation across all 23 HIV drugs with 4 drug classes:**

| Drug Class | Avg Correlation | Best Drug | Architecture |
|------------|-----------------|-----------|--------------|
| PI (8 drugs) | **+0.928** | LPV (+0.956) | All perform well |
| NRTI (6 drugs) | **+0.887** | 3TC (+0.981) | Attention VAE |
| NNRTI (5 drugs) | **+0.853** | NVP (+0.959) | Transformer VAE |
| INI (4 drugs) | **+0.863** | EVG (+0.963) | Transformer VAE |
| **Overall** | **+0.890** | 3TC (+0.981) | - |

**Key Innovations:**
- **P-adic ranking loss**: +0.6 correlation improvement over standard MSE
- **Attention analysis**: 65-70% F1 match with known resistance mutations
- **Cross-resistance modeling**: Captures TAM cross-resistance (AZT-D4T: +0.97) and M184V resensitization

**Quick Start:**
```bash
# Run full validation across 23 drugs
python scripts/experiments/run_full_validation.py

# Run interpretability analysis
python scripts/experiments/run_attention_analysis.py

# Test cross-resistance patterns
python scripts/experiments/run_cross_resistance_test.py
```

See [`UNDERSTANDING/34_FINAL_SUMMARY_AND_RECOMMENDATIONS.md`](UNDERSTANDING/34_FINAL_SUMMARY_AND_RECOMMENDATIONS.md) for full documentation.

---

## 🛠️ Installation & Development

The project uses a standard Python stack. For development, we recommend the following tools:

- **ruff** – fast Python linter/formatter (`ruff.toml` is already configured).
- **pytest** – test suite (`tests/`).
- **pre‑commit** – hooks for linting and SPDX header checks.
- **GitHub Actions** – CI pipelines for linting, testing, and code‑health dashboards.

To set up pre‑commit:

```bash
pip install pre-commit
pre-commit install
```

---

## 📦 License & Legal

### Software (Code)

- **License**: PolyForm Non‑Commercial 1.0.0
- **Permitted**: Academic, educational, and non‑profit use.
- **Commercial Use**: Requires a separate commercial license – contact `support@aiwhisperers.com`.

### Research Outputs (Data, Figures, Models)

- **License**: CC‑BY‑4.0 – free for any reuse with attribution.
- **Open‑Medicine Policy**: See [`LEGAL_AND_IP/OPEN_MEDICINE_POLICY.md`](LEGAL_AND_IP/OPEN_MEDICINE_POLICY.md) for detailed terms.

All legal documents are collected in the [`LEGAL_AND_IP/`](LEGAL_AND_IP/) directory:

- [`LICENSE`](LEGAL_AND_IP/LICENSE)
- [`NOTICE`](LEGAL_AND_IP/NOTICE)
- [`OPEN_MEDICINE_POLICY.md`](LEGAL_AND_IP/OPEN_MEDICINE_POLICY.md)
- [`RESULTS_LICENSE.md`](LEGAL_AND_IP/RESULTS_LICENSE.md)
- [`CLA.md`](LEGAL_AND_IP/CLA.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- [`SECURITY.md`](SECURITY.md)
- [`CITATION.cff`](CITATION.cff)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Read the Contributor License Agreement** ([`CLA.md`](LEGAL_AND_IP/CLA.md)) and sign it.
2. **Review the Code of Conduct** ([`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)).
3. **Check the contribution guidelines** ([`CONTRIBUTING.md`](CONTRIBUTING.md)) for branch naming, commit style, and testing requirements.
4. **Open a Pull Request** – link it to an existing issue or create a new one.
5. **Ensure all CI checks pass** (ruff, pytest, SPDX header verification).

---

## 🔒 Security

Report any vulnerabilities via the [`SECURITY.md`](SECURITY.md) policy. We use a coordinated disclosure process and will acknowledge contributors.

---

## 📑 Citation

Please cite the project using the provided `CITATION.cff`. Example BibTeX entry:

```bibtex
@software{ternary_vae,
  author = {{AI Whisperers}},
  title = {Ternary VAE Bioinformatics},
  year = {2025},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics}
}
```

> **Note**: DOI pending Zenodo registration. See [CITATION.cff](CITATION.cff) for full citation details.

---

## 👥 Authors & Acknowledgments

- **Primary Authors** – see [`AUTHORS.md`](LEGAL_AND_IP/AUTHORS.md).
- **Contributors** – see [`CONTRIBUTORS.md`](LEGAL_AND_IP/CONTRIBUTORS.md).
- **Funding** – this work is supported by open‑science grants and institutional collaborations.

---

## 📞 Contact

For general questions, open an issue. For commercial licensing inquiries, email `support@aiwhisperers.com`.

---

_Last updated: 2025-12-27_
