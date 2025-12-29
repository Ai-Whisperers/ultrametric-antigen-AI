# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""HIV-specific analysis entry points.

CLI entry points for HIV research. Library code is in src/analysis/hiv/,
src/clinical/hiv/, and src/research/hiv/.

Scripts:
    run_hiv_analysis.py         Unified entry point for HIV analyses
    validate_hiv_setup.py       Validate HIV analysis environment
    download_hiv_datasets.py    Download HIV datasets from LANL, Stanford, etc.
    train_codon_vae_hiv.py      Train codon VAE on HIV sequences

Usage:
    python scripts/hiv/run_hiv_analysis.py --analysis all
    python scripts/hiv/validate_hiv_setup.py
    python scripts/hiv/download_hiv_datasets.py
"""
