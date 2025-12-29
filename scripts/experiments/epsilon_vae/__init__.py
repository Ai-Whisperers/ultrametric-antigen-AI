# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Epsilon-VAE research experiments and training scripts.

This package contains experimental training variants and analysis tools
for the Epsilon-VAE architecture research.

Training Scripts:
    train_epsilon_vae.py        Base Epsilon-VAE training
    train_epsilon_vae_enhanced.py   Enhanced variant
    train_epsilon_vae_hybrid.py     Hybrid architecture
    train_epsilon_coupled.py        Coupled VAE training
    train_fractional_padic.py       Fractional p-adic training
    train_radial_collapse.py        Radial collapse training
    train_radial_target.py          Radial target training
    train_hierarchy_focused.py      Hierarchy-focused training
    train_balanced_radial.py        Balanced radial training
    train_homeostatic_rich.py       Homeostatic training
    train_soft_radial.py            Soft radial training

Analysis Scripts:
    analyze_run.py              Analyze training run results
    analyze_padic_structure.py  Analyze p-adic structure
    analyze_weight_structure.py Analyze weight structure
    analyze_progressive_checkpoints.py  Analyze checkpoint progression

Utility Scripts:
    extract_embeddings.py       Extract embeddings from checkpoints
    collect_checkpoints.py      Collect and organize checkpoints
    collect_checkpoints_enhanced.py  Enhanced checkpoint collection
    compare_frozen_vs_unfrozen.py    Compare frozen vs unfrozen encoders
    apply_radial_snap.py        Apply radial snapping to embeddings
    optimize_p3_baseline.py     Optimize P3 baseline

Usage:
    python scripts/epsilon_vae/train_epsilon_vae.py --config configs/epsilon.yaml
    python scripts/epsilon_vae/analyze_run.py --run outputs/runs/latest
"""
