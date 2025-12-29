# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hybrid Structure Predictor: Ternary VAE + PDB Structures.

This module combines our ternary VAE model (v5_11_11_production/best.pt)
with PDB structural data for HIV integrase reveal mutation validation.

The hybrid approach eliminates the need for AlphaFold3's massive requirements:
- No A100 GPU needed
- No 630GB databases needed
- Only ~100MB of HIV integrase structures required
"""

from .pdb_analyzer import PDBAnalyzer
from .structure_predictor import HybridStructurePredictor

__all__ = ["HybridStructurePredictor", "PDBAnalyzer"]
