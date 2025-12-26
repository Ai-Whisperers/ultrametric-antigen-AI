# Supplementary Materials

> **Getting started guides, FAQ, reproducibility, and limitations**

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [QUICK_START.md](QUICK_START.md) | 10-minute setup and run guide |
| [FAQ.md](FAQ.md) | Common questions and answers |
| [REPRODUCIBILITY_GUIDE.md](REPRODUCIBILITY_GUIDE.md) | Step-by-step reproduction |
| [LIMITATIONS_AND_CAVEATS.md](LIMITATIONS_AND_CAVEATS.md) | Known limitations |

---

## Quick Start (5 Minutes)

```bash
# 1. Navigate to project
cd ternary-vaes-bioinformatics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run clinical applications pipeline
python scripts/clinical_applications.py

# 4. Run research discoveries pipeline
python scripts/research_discoveries.py

# 5. View results
cat results/clinical_applications/CLINICAL_REPORT.md
cat results/research_discoveries/RESEARCH_FINDINGS.md
```

---

## Common Questions

| Question | Answer |
|----------|--------|
| What is p-adic geometry? | A mathematical framework based on prime number divisibility |
| Why hyperbolic space? | Natural representation of hierarchical evolutionary relationships |
| How accurate is tropism prediction? | 85% with geometric features |
| What data is required? | Stanford HIVDB, LANL, CATNAP (all public) |

---

## Known Limitations

1. **Computational**: Poincaré ball operations are slower than Euclidean
2. **Data**: Limited to HIV-1 (not HIV-2 or SIV)
3. **Validation**: Structural predictions need AlphaFold3 confirmation
4. **Clinical**: Not FDA-approved; research use only

---

_See [FAQ.md](FAQ.md) for complete Q&A._
