# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 AI Whisperers - Ivan Googhardson (ivan@aiwhisperers.com)
"""
Experimental modules for research and development.

These modules are not production-ready and may change or be removed.
"""

# Submodule imports for lazy loading
from . import categorical
from . import category
from . import contrastive
from . import diffusion
from . import equivariant
from . import graphs
from . import implementations
from . import information
from . import linguistics
from . import meta
from . import physics
from . import quantum
from . import topology
from . import tropical

__all__ = [
    "categorical",
    "category",
    "contrastive",
    "diffusion",
    "equivariant",
    "graphs",
    "implementations",
    "information",
    "linguistics",
    "meta",
    "physics",
    "quantum",
    "topology",
    "tropical",
]
