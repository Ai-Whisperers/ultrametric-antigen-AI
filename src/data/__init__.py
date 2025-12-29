# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Backward compatibility shim - use src.dataio instead.

This module re-exports from src.dataio for backward compatibility.
New code should import from src.dataio directly.
"""

import warnings

warnings.warn(
    "src.data is deprecated, use src.dataio instead",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from dataio
from src.dataio import *
from src.dataio import __all__
