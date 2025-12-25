# External Documentation Index

> **Registry of documentation that exists outside the DOCUMENTATION/ folder.**

This index links to all project documentation that, for practical reasons, lives alongside the code or data it describes rather than in the centralized DOCUMENTATION folder.

---

## Root Level

Standard project governance files at repository root:

| File | Purpose |
|:-----|:--------|
| [README.md](../README.md) | Main project overview and entry point |
| [AUTHORS.md](../LEGAL_AND_IP/AUTHORS.md) | Project authors |
| [CONTRIBUTORS.md](../LEGAL_AND_IP/CONTRIBUTORS.md) | Contributors list |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guidelines |
| [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | Community code of conduct |
| [CLA.md](../LEGAL_AND_IP/CLA.md) | Contributor License Agreement |
| [SECURITY.md](../SECURITY.md) | Security policy and vulnerability reporting |
| [LICENSE](../LICENSE) | PolyForm Noncommercial 1.0.0 |
| [NOTICE](../NOTICE) | Third-party attributions |

---

## Source Code (`src/`)

Architecture documentation that lives with the code:

| File | Purpose | Canonical Doc |
|:-----|:--------|:--------------|
| [src/README.md](../src/README.md) | Source architecture overview | - |
| [src/models/README.md](../src/models/README.md) | Model components guide | [Architecture Specs](03_PRESENTATION_TIERS/03_TIER_3_TECHNICAL/05_SPECS_AND_GUIDES/ARCHITECTURE.md) |
| [src/losses/README.md](../src/losses/README.md) | Loss functions documentation | - |
| [src/training/README.md](../src/training/README.md) | Training modules guide | - |

---

## Research (`research/`)

Research-specific documentation:

| File | Purpose |
|:-----|:--------|
| [research/alphafold3/README.md](../research/alphafold3/README.md) | AlphaFold3 integration setup |
| [research/alphafold3/HYBRID_APPROACH.md](../research/alphafold3/HYBRID_APPROACH.md) | Lightweight AF3 strategy (6300x storage reduction) |
| [research/LICENSE.md](../research/LICENSE.md) | Research data licensing terms |

---

## Results (`results/`)

Output and analysis documentation:

| File | Purpose |
|:-----|:--------|
| [results/discoveries/hiv/DISCOVERY_HIV_PADIC_RESISTANCE.md](../results/discoveries/hiv/DISCOVERY_HIV_PADIC_RESISTANCE.md) | HIV resistance findings |
| [results/alphafold_predictions/VALIDATION_RESULTS.md](../results/alphafold_predictions/VALIDATION_RESULTS.md) | AF3 structural validation |
| [results/benchmarks/RESOLUTION_COMPARISON.md](../results/benchmarks/RESOLUTION_COMPARISON.md) | Manifold resolution analysis |

**Session logs** (in `results/run_history/`):
- SESSION.md, ANALYSIS_AND_IMPROVEMENTS.md, etc.
- Historical session documentation

---

## Configurations (`configs/`)

Configuration documentation:

| File | Purpose |
|:-----|:--------|
| [configs/archive/README.md](../configs/archive/README.md) | Legacy config documentation (v5.6-v5.9) |

---

## Conductor (`conductor/`)

**Note**: Kept separate as project management tooling.

| File | Purpose |
|:-----|:--------|
| [conductor/product.md](../conductor/product.md) | Product guide |
| [conductor/product-guidelines.md](../conductor/product-guidelines.md) | Product guidelines |
| [conductor/tech-stack.md](../conductor/tech-stack.md) | Technology stack |
| [conductor/code_styleguides/](../conductor/code_styleguides/) | Code style guides |

---

## GitHub Templates (`.github/`)

| File | Purpose |
|:-----|:--------|
| [.github/ISSUE_TEMPLATE/bug_report.md](../.github/ISSUE_TEMPLATE/bug_report.md) | Bug report template |
| [.github/ISSUE_TEMPLATE/feature_request.md](../.github/ISSUE_TEMPLATE/feature_request.md) | Feature request template |
| [.github/PULL_REQUEST_TEMPLATE.md](../.github/PULL_REQUEST_TEMPLATE.md) | PR template |

---

## AI Assistant Configuration

| File | Purpose |
|:-----|:--------|
| [.claude/commands/](../.claude/commands/) | Claude Code command definitions |
| [.claude/settings.json](../.claude/settings.json) | Claude Code project settings |

---

## Why These Are External

| Category | Reason |
|:---------|:-------|
| **Root governance** | Standard convention; discoverable by tools |
| **src/ READMEs** | Navigate to module and find its docs immediately |
| **research/ docs** | Research-specific, lives with data |
| **results/ docs** | Output documentation, lives with outputs |
| **conductor/** | Separate project management system |
| **GitHub templates** | Required location for GitHub integration |

---

## Adding External Docs

When creating documentation outside DOCUMENTATION/:

1. Add entry to this index
2. Consider adding banner pointing to canonical DOCUMENTATION location
3. Update NAVIGATION_GUIDE.md if relevant to stakeholders

---

*Last updated: 2025-12-25*
