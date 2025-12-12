# Archived Configs

These config files are archived as part of the v5.10.1 unification.

## Status

| File | Reason | Notes |
|------|--------|-------|
| ternary_v5_6.yaml | Legacy | Base config, kept for reference |
| ternary_v5_7.yaml | Legacy | Added StateNet v3, kept for reference |
| ternary_v5_8.yaml | Orphaned | Model file never existed, config was misleading |
| ternary_v5_9.yaml | Orphaned | Model file never existed, config was misleading |
| ternary_v5_9_2.yaml | Orphaned | Model file never existed, config was misleading |

## v5.8/v5.9 Issue

These configs defined features (two-phase training, continuous feedback) that were never implemented because the corresponding model files (`ternary_vae_v5_8.py`, `ternary_vae_v5_9.py`) were never created. The training scripts silently imported the v5.6 model instead.

## Current Primary Config

Use `configs/ternary_v5_10.yaml` - the unified config that includes all features from previous versions.

## Checkpoint Compatibility

- v5.6 checkpoints: Load with `src/models/ternary_vae_v5_6.py`
- v5.7 checkpoints: Load with `src/models/ternary_vae_v5_7.py`
- v5.8/v5.9 checkpoints: Actually contain v5.6 model weights (mislabeled)

---

Archived: 2025-12-12 as part of v5.10.1 cleanup
