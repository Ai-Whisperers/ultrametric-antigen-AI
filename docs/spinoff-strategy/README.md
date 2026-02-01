# Production Spin-off Strategy

> **Philosophy:** Users should not be persuaded—they should have reproducible metrics they measure themselves.

---

## Overview

As components in [ultrametric-antigen-AI](https://github.com/Ai-Whisperers/ultrametric-antigen-AI) reach production quality, they are open-sourced in **separate repositories** with independent quality gates.

This approach:
- Prevents "AI hallucination" perception
- Avoids "messy mixture of fields" dismissal
- Enables user approval through **self-measured metrics**
- Accelerates iteration while improving credibility

---

## Documents

| Document | Purpose |
|----------|---------|
| [SPINOFF_PLAN.md](SPINOFF_PLAN.md) | Strategy, candidates, timeline |
| [SPINOFF_README_TEMPLATE.md](SPINOFF_README_TEMPLATE.md) | README template for spin-offs |
| [REPRODUCIBILITY_TEMPLATE.md](REPRODUCIBILITY_TEMPLATE.md) | Reproducibility guide template |
| [CI_TEMPLATE.yml](CI_TEMPLATE.yml) | GitHub Actions quality gates |

---

## Repository Hierarchy

```
Ai-Whisperers/
├── 3-adic-ml                    # FOUNDATION
├── ultrametric-antigen-AI       # APPLICATION HUB
└── [Production Spin-offs]       # PRODUCTION
    ├── padic-ddg-predictor
    ├── hyperbolic-codon-encoder
    └── ...
```

---

## Quality Gates (Required for All Spin-offs)

| Gate | Criteria | Automated |
|------|----------|-----------|
| Unit Tests | 100% pass | Yes |
| Reproducibility | Metrics within tolerance | Yes |
| Documentation | All APIs documented | Yes |
| Security | No vulnerabilities | Yes |
| License | Compatible deps | Yes |

---

## Current Status

| Component | Validation | Spin-off Status |
|-----------|------------|-----------------|
| DDG Predictor | LOO ρ=0.52 | **Ready** |
| Codon Encoder | LOO ρ=0.61 | **Ready** |
| AMP Fitness | r=0.63 | Near-ready |
| Contact Prediction | AUC=0.67 | Research stage |

---

*Last updated: 2026-01-30*
