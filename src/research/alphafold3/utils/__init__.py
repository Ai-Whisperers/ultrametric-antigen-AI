# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

# AlphaFold3 Extracted Utilities
# Pure Python constants extracted from AF3 for hybrid approach.
# License: CC-BY-NC-SA 4.0 (AlphaFold3 source code license)

"""
Extracted pure Python utilities from AlphaFold3.

These modules contain no C++ dependencies and can be used directly.
Original source: https://github.com/google-deepmind/alphafold3
"""

from . import atom_types, residue_names

__all__ = ["residue_names", "atom_types"]
