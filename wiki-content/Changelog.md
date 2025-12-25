# Changelog

All notable changes to Ternary VAE Bioinformatics are documented here.

This project follows [Semantic Versioning](https://semver.org/).

---

## [5.11] - 2025-12-25

### Highlights
- Stable release with homeostasis controller
- Full documentation wiki
- Comprehensive loss registry system

### Added
- **HomeostasisController**: Adaptive KL weighting for training stability
- **SwarmVAE**: Multi-agent collaborative learning architecture
- **HolographicPoincareManifold**: AdS/CFT-inspired boundary behavior
- **ViralEvolutionPredictor**: Predict viral mutations
- **mRNAStabilityPredictor**: Predict mRNA stability from sequence
- Complete wiki documentation (19 pages)
- WCAG AA compliant Mermaid diagrams

### Changed
- Improved numerical stability in all geometric operations
- Loss registry now supports dynamic composition
- Better default hyperparameters

### Fixed
- Edge cases in `exp_map_zero` near boundary
- Gradient clipping in Riemannian optimizer
- Memory leaks in large batch training

---

## [5.11.9] - 2025-12-20

### Added
- Zero structure loss for balanced training
- HolographicEncoder with attention mechanism

### Fixed
- NaN gradients with very small curvature values

---

## [5.11.8] - 2025-12-15

### Added
- Enhanced numerical stability guards
- Automatic gradient scaling

### Changed
- Default EPSILON from 1e-7 to 1e-8
- Improved projection clamping

---

## [5.11.7] - 2025-12-10

### Added
- HomeostasisController for adaptive beta scheduling
- CurriculumScheduler for progressive training

### Changed
- Refactored loss components for better modularity

---

## [5.10] - 2025-11-15

### Added
- SwarmVAE multi-agent architecture
- PheromoneField for agent communication
- AgentRole enum (EXPLORER, EXPLOITER, SCOUT)

### Changed
- Upgraded to PyTorch 2.0 compatibility
- Improved memory efficiency

---

## [5.9] - 2025-10-01

### Added
- p-adic ranking loss components
- RadialStratificationLoss
- PAdicRankingLoss

### Changed
- Restructured losses module with registry pattern

---

## [5.8] - 2025-09-01

### Added
- Full Riemannian optimization support
- RiemannianAdam optimizer
- ManifoldParameter for learnable manifold points

### Fixed
- Parallel transport implementation
- Geodesic interpolation accuracy

---

## [5.7] - 2025-08-01

### Added
- AutoimmuneCodonRegularizer
- GlycanShieldAnalyzer
- CD4CD8AwareRegularizer

### Changed
- Moved biological losses to specialized module

---

## [5.6] - 2025-07-01

### Added
- Configuration system with YAML support
- Environment variable overrides
- Config validation

---

## [5.5] - 2025-06-01

### Added
- Callback system for training
- EarlyStoppingCallback
- CheckpointCallback
- MetricsCallback

---

## [5.0] - 2025-05-01

### Added
- Initial public release
- TernaryVAE core architecture
- Poincare ball geometry
- Basic loss functions

---

## Version Numbering

- **Major**: Breaking API changes
- **Minor**: New features, backwards compatible
- **Patch**: Bug fixes, documentation

---

## Upgrade Guide

### From 5.10 to 5.11

```python
# Old
from src.models import VAE
model = VAE(...)

# New
from src.models import TernaryVAE
model = TernaryVAE(...)
```

### From 5.9 to 5.10

No breaking changes. SwarmVAE is optional.

---

*See also: [[Home]], [[Installation]], [[Contributing-Guide]]*
