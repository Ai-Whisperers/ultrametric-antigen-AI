# Project Roadmap

> **Current status and future plans.**

---

## Current Status (2025-12-28)

### Completed

| Feature | Status | Details |
|:--------|:-------|:--------|
| BaseVAE abstraction | Done | Unified base for 19+ VAE variants |
| Uncertainty quantification | Done | MC Dropout, Evidential, Ensemble |
| Epistasis modeling | Done | Mutation interaction network |
| Transfer learning | Done | 5 strategies, cross-disease |
| Structure-aware VAE | Done | AlphaFold2, SE(3) encoders |
| Multi-disease platform | Done | 11 diseases supported |
| Test suite | Done | 231 tests, 97.4% pass rate |
| Documentation consolidation | Done | Unified docs/content/ |

### In Progress

| Feature | Status | ETA |
|:--------|:-------|:----|
| Legacy docs cleanup | 80% | 2025-12-29 |
| Integration tests | 90% | 2025-12-29 |

---

## Next Steps

### Short-term (Q1 2025)

1. **Clinical validation**
   - Partner with HIV clinic for prospective validation
   - FDA pre-submission meeting

2. **TB module completion**
   - MDR/XDR classification validation
   - Integration with TB diagnostics workflows

3. **API development**
   - REST API for predictions
   - Batch processing support

### Medium-term (Q2-Q3 2025)

1. **Regulatory pathway**
   - FDA 510(k) preparation
   - CE marking (EU)

2. **Commercial partnerships**
   - Pharma API integrations
   - Diagnostic lab partnerships

3. **Platform expansion**
   - Additional disease modules
   - Real-time surveillance integration

### Long-term (2026+)

1. **Novel target discovery**
   - Drug-resistant mutation prediction
   - Vaccine antigen design

2. **Point-of-care integration**
   - Rapid testing integration
   - Resource-limited settings

---

## Architecture Evolution

```
2024 Q4: Single VAE → Dual VAE (VAE-A + VAE-B)
2025 Q1: Dual VAE → Multi-disease platform
2025 Q4: Platform → Clinical-grade system
2026+:   Clinical → Regulatory-approved diagnostic
```

---

## Contributing

See [contributing.md](contributing.md) for how to contribute to these goals.

---

_Last updated: 2025-12-28_
