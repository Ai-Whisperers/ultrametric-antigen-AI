# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DEPRECATED: Use src.config.paths instead.

This module has been consolidated into src/config/paths.py for better organization.
Imports are redirected for backward compatibility.

Usage (new):
    from src.config.paths import (
        PROJECT_ROOT,
        RESEARCH_DIR,
        get_research_path,
        get_data_path,
    )
"""

import warnings
from pathlib import Path

warnings.warn(
    "src.research is deprecated. Use src.config.paths instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility redirects
from src.config.paths import (
    DATA_DIR,
    PROJECT_ROOT,
    RESEARCH_DIR,
    RESULTS_DIR,
    get_config_path,
    get_research_path,
    list_datasets,
    list_research_experiments,
)


def get_project_root() -> Path:
    """DEPRECATED: Use src.config.paths.PROJECT_ROOT instead."""
    return PROJECT_ROOT


def get_data_path(subpath: str = "") -> Path:
    """Get path to data directory or subdirectory.

    DEPRECATED: Use src.config.paths.DATA_DIR instead.
    """
    if subpath:
        return DATA_DIR / subpath
    return DATA_DIR


def get_results_path(subpath: str = "") -> Path:
    """Get path to results directory or subdirectory.

    DEPRECATED: Use src.config.paths.RESULTS_DIR instead.
    """
    if subpath:
        return RESULTS_DIR / subpath
    return RESULTS_DIR


__all__ = [
    "get_project_root",
    "get_research_path",
    "get_data_path",
    "get_results_path",
    "get_config_path",
    "list_research_experiments",
    "list_datasets",
]
