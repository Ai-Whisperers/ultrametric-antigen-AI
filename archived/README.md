# Archived Code

This directory contains deprecated code that has been superseded by newer implementations.
These files are kept for historical reference only and are **NOT** part of the active codebase.

## Contents

### models/

| File | Version | Superseded By | Notes |
|------|---------|---------------|-------|
| `ternary_vae_v5_6.py` | v5.6 | `src/models/ternary_vae.py` (v5.11) | Original StateNet v2 |
| `ternary_vae_v5_7.py` | v5.7 | `src/models/ternary_vae.py` (v5.11) | Added metric attention |
| `ternary_vae_v5_10.py` | v5.10 | `src/models/ternary_vae.py` (v5.11) | Pure hyperbolic geometry |
| `appetitive_vae.py` | experimental | N/A | Experimental architecture (abandoned) |

### losses/

| File | Description | Superseded By |
|------|-------------|---------------|
| `appetitive_losses.py` | Experimental appetitive loss functions | N/A (abandoned) |

## Why Keep These?

1. **Historical Reference**: Understanding the evolution of the architecture
2. **Reproducibility**: Running experiments from earlier papers/versions
3. **Rollback**: Emergency fallback if issues found in new code

## Usage

These files should **NOT** be imported from active code. If you need functionality
from these files, it should be properly migrated to the canonical implementation
in `src/`.

## Deletion Policy

These files can be safely deleted after:
1. All active experiments use v5.11+
2. No publications reference these versions
3. At least 3 months since last use

---
*Archived on: 2024-12-25*
*Canonical version: v5.11 (TernaryVAEV5_11)*
