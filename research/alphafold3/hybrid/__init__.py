"""Hybrid Structure Predictor: Ternary VAE + PDB Structures.

This module combines our ternary VAE model (v5_11_11_production/best.pt)
with PDB structural data for HIV integrase reveal mutation validation.

The hybrid approach eliminates the need for AlphaFold3's massive requirements:
- No A100 GPU needed
- No 630GB databases needed
- Only ~100MB of HIV integrase structures required
"""

from .structure_predictor import HybridStructurePredictor
from .pdb_analyzer import PDBAnalyzer

__all__ = ["HybridStructurePredictor", "PDBAnalyzer"]
