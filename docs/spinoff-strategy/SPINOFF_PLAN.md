# Production Spin-off Strategy

**Doc-Type:** Strategy Document · Version 1.0 · 2026-01-30

---

## Philosophy

> **Users should not be persuaded—they should have reproducible metrics they measure themselves.**

Each production-quality component is open-sourced in a **separate repository** with:
- Independent quality gates
- Reproducibility checks
- Clear documentation that enables skeptical verification

This approach:
1. Avoids "AI hallucination" perception
2. Prevents "messy mixture of fields" dismissal
3. Enables user approval through **self-measured metrics**
4. Accelerates iteration while improving credibility

---

## Repository Hierarchy

```
Ai-Whisperers/
├── 3-adic-ml                    # FOUNDATION: Mathematical framework
│   └── (3-adic math, hyperbolic geometry, VAE theory)
│
├── ultrametric-antigen-AI       # APPLICATION HUB: Bioinformatics R&D
│   └── (Research, partner packages, evolutionary history)
│
└── [Production Spin-offs]       # PRODUCTION: Individual tools
    ├── padic-ddg-predictor      # Protein stability prediction
    ├── hyperbolic-codon-encoder # Codon embeddings
    ├── amp-fitness-predictor    # Antimicrobial peptides
    └── ...
```

---

## Spin-off Candidates

### Tier 1: Ready for Spin-off

| Component | Validation | Metrics | Priority |
|-----------|------------|---------|----------|
| **DDG Predictor** | LOO ρ=0.52 (shipped), 0.58 (retrained) | Rosetta-blind 23.6% | **Highest** |
| **TrainableCodonEncoder** | LOO ρ=0.61 on S669 | +105% over baseline | **High** |

### Tier 2: Near-Ready

| Component | Validation | Remaining Work |
|-----------|------------|----------------|
| **AMP Fitness Predictor** | Pearson r=0.63 | Documentation, tests |
| **Contact Prediction** | AUC=0.67 | More protein validation |

### Tier 3: Research Stage

| Component | Status |
|-----------|--------|
| Arbovirus Primers | Dual-layer (production + research) |
| HIV Resistance | Stanford HIVdb integration |

---

## DDG Predictor Spin-off Plan

### Proposed Repository Name
`padic-ddg-predictor` or `hyperbolic-ddg-predictor`

### Structure

```
padic-ddg-predictor/
├── README.md                    # Clear, skeptic-friendly documentation
├── REPRODUCIBILITY.md           # Step-by-step reproduction guide
├── VALIDATION_REPORT.md         # Pre-computed validation results
│
├── src/
│   └── ddg_predictor/
│       ├── __init__.py
│       ├── predictor.py         # ValidatedDDGPredictor
│       ├── features.py          # Feature extraction
│       └── embeddings.py        # Codon embedding interface
│
├── models/
│   └── trained_predictor.pt     # Shipped model weights
│
├── data/
│   ├── s669_subset.json         # Validation dataset (N=52)
│   └── s669_full.json           # Full dataset for comparison
│
├── scripts/
│   ├── validate.py              # Run full validation
│   ├── reproduce_loo.py         # Reproduce LOO results
│   └── compare_rosetta.py       # Rosetta-blind detection
│
├── tests/
│   ├── test_predictor.py
│   ├── test_reproducibility.py  # Automated metric verification
│   └── conftest.py
│
├── benchmarks/
│   ├── BENCHMARK_PROTOCOL.md
│   └── results/
│       └── validation_metrics.json
│
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

### Quality Gates (Must Pass Before Release)

| Gate | Criteria | Automated? |
|------|----------|------------|
| **Unit Tests** | 100% pass | Yes |
| **Reproducibility Test** | LOO ρ within 0.02 of claimed | Yes |
| **Documentation Check** | All public APIs documented | Yes |
| **Metric Verification** | Output matches validation report | Yes |
| **Dependency Audit** | No security vulnerabilities | Yes |
| **License Compliance** | All deps compatible | Yes |

### README Template for DDG Predictor

See [SPINOFF_README_TEMPLATE.md](SPINOFF_README_TEMPLATE.md)

---

## Codon Encoder Spin-off Plan

### Proposed Repository Name
`hyperbolic-codon-encoder`

### Key Differentiator
Provides pre-trained codon embeddings on the Poincaré ball with 3-adic valuation hierarchy.

### Structure

```
hyperbolic-codon-encoder/
├── README.md
├── REPRODUCIBILITY.md
│
├── src/
│   └── codon_encoder/
│       ├── encoder.py           # TrainableCodonEncoder
│       ├── embeddings.py        # Pre-extracted embeddings
│       └── distances.py         # Hyperbolic distance utilities
│
├── models/
│   └── trained_encoder.pt
│
├── embeddings/
│   └── codon_embeddings.pt      # (64, 16) pre-computed
│
└── ...
```

---

## Spin-off Workflow

### Step 1: Identify Ready Component
- Validation metrics stable
- Code isolated from main repo
- Documentation exists

### Step 2: Extract and Clean
```bash
# Create new repo structure
mkdir -p new-repo/src new-repo/tests new-repo/data

# Copy relevant code
cp -r source/predictor.py new-repo/src/

# Remove internal dependencies
# Update imports to use standalone packages
```

### Step 3: Add Quality Infrastructure
- GitHub Actions for CI/CD
- Automated reproducibility tests
- Documentation generation

### Step 4: Validation Report
Generate `VALIDATION_REPORT.md` with:
- Exact commands to reproduce metrics
- Expected outputs
- Environment specifications

### Step 5: Release Checklist
- [ ] All quality gates pass
- [ ] README is skeptic-friendly
- [ ] Reproducibility verified on fresh machine
- [ ] License files present
- [ ] Citation information included

---

## Maintenance Strategy

### Updates to Foundation (3-adic-ml)
- Spin-offs pin to specific versions
- Breaking changes trigger spin-off updates
- Changelog tracks compatibility

### Updates to Spin-offs
- Semantic versioning
- Backward compatibility for public APIs
- Deprecation warnings before removal

---

## Timeline

| Phase | Target | Components |
|-------|--------|------------|
| Q1 2026 | DDG Predictor | `padic-ddg-predictor` |
| Q1 2026 | Codon Encoder | `hyperbolic-codon-encoder` |
| Q2 2026 | AMP Fitness | `amp-fitness-predictor` |
| Q2 2026 | Contact Prediction | (if validation sufficient) |

---

## Success Metrics

A spin-off is successful when:
1. **External users reproduce claimed metrics** without our help
2. **Citation/usage** by independent researchers
3. **Issues** are about extension, not basic functionality
4. **No "this doesn't work" without user error**

---

*Last updated: 2026-01-30*
