# Methodology & Technical Documentation

> **Complete technical specifications for the HIV p-adic hyperbolic analysis**

---

## Documents in This Section

| Document | Description |
|----------|-------------|
| [METHODOLOGY.md](METHODOLOGY.md) | Complete technical methods and algorithms |
| [API_REFERENCE.md](API_REFERENCE.md) | Python function documentation |
| [STATISTICAL_GUIDE.md](STATISTICAL_GUIDE.md) | Statistics interpretation guide |
| [BENCHMARKING.md](BENCHMARKING.md) | Comparison with existing methods |
| [SOFTWARE_ARCHITECTURE.md](SOFTWARE_ARCHITECTURE.md) | System design and architecture |

---

## Core Concepts

### P-adic Valuation
```python
def compute_padic_valuation(n, p=3):
    """v_p(n) = highest power of p dividing n"""
    if n == 0:
        return float('inf')
    valuation = 0
    while n % p == 0:
        valuation += 1
        n //= p
    return valuation
```

### Hyperbolic Distance (Poincaré Ball)
```
d(x, y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
```

### Codon Encoding
- 64 codons indexed 0-63
- Index = B1×16 + B2×4 + B3 (where B ∈ {T=0, C=1, A=2, G=3})
- P-adic distance respects wobble hypothesis

---

## Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `src/biology/codons.py` | Genetics | Codon constants, conversions |
| `src/analysis/immunology/` | Immunology | Epitope encoding, HLA risk |
| `src/training/hyperbolic_trainer.py` | Training | Pure hyperbolic geometry |
| `scripts/research_discoveries.py` | Analysis | 5 research directions |
| `scripts/clinical_applications.py` | Clinical | Decision support |

---

_See [API_REFERENCE.md](API_REFERENCE.md) for complete function documentation._
