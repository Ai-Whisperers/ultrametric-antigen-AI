# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Experimental modules - now in production.

These modules were previously experimental but are now production-ready
in their respective packages.

Available modules (now in production):
- meta_learning: MAML and few-shot adaptation -> src.meta
- All other _future modules -> src.<module_name>

Usage:
    # Preferred: import from production modules directly
    from src.meta import MAML, Task

    # Legacy: import from experimental (deprecated)
    from src.experimental import MAML, Task
"""

# Import from production modules for backwards compatibility
from src.meta import (
    MAML,
    Task,
)

__all__ = [
    "MAML",
    "Task",
]
