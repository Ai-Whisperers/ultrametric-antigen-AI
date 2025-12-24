# AlphaFold3 Extracted Utilities
# Pure Python constants extracted from AF3 for hybrid approach.
# License: CC-BY-NC-SA 4.0 (AlphaFold3 source code license)

"""
Extracted pure Python utilities from AlphaFold3.

These modules contain no C++ dependencies and can be used directly.
Original source: https://github.com/google-deepmind/alphafold3
"""

from . import residue_names
from . import atom_types

__all__ = ["residue_names", "atom_types"]
