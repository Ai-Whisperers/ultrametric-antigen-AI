# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Research path utilities bridge.

NOTE: This is NOT a research module - it provides PATH UTILITIES for accessing
the research/ directory from src/ code. The actual research code is in research/.

This module provides a bridge between the core library (src/) and research
experiments (research/). It enables:

1. Easy import of research utilities from core code
2. Consistent path handling for research data
3. Standardized experiment configuration

Usage:
    from src.research import get_research_path, get_data_path

    # Get path to research experiment
    hiv_path = get_research_path("bioinformatics/codon_encoder_research/hiv")

    # Get path to external data
    data_path = get_data_path("external/github/HIV-data")
"""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root (parent of src/)
    """
    return Path(__file__).resolve().parents[2]


def get_research_path(subpath: str = "") -> Path:
    """Get path to research directory or subdirectory.

    Args:
        subpath: Optional subdirectory path (e.g., "bioinformatics/hiv")

    Returns:
        Path to research directory or subdirectory
    """
    research_dir = get_project_root() / "research"
    if subpath:
        return research_dir / subpath
    return research_dir


def get_data_path(subpath: str = "") -> Path:
    """Get path to data directory or subdirectory.

    Args:
        subpath: Optional subdirectory path (e.g., "external/github")

    Returns:
        Path to data directory or subdirectory
    """
    data_dir = get_project_root() / "data"
    if subpath:
        return data_dir / subpath
    return data_dir


def get_results_path(subpath: str = "") -> Path:
    """Get path to results directory or subdirectory.

    Args:
        subpath: Optional subdirectory path

    Returns:
        Path to results directory or subdirectory
    """
    results_dir = get_project_root() / "results"
    if subpath:
        return results_dir / subpath
    return results_dir


def get_config_path(config_name: str) -> Optional[Path]:
    """Get path to a configuration file.

    Args:
        config_name: Name of config file (e.g., "ternary.yaml")

    Returns:
        Path to config file if exists, None otherwise
    """
    config_path = get_project_root() / "configs" / config_name
    return config_path if config_path.exists() else None


def list_research_experiments() -> list[str]:
    """List available research experiments.

    Returns:
        List of experiment directory names
    """
    research_dir = get_research_path()
    if not research_dir.exists():
        return []

    experiments = []
    for item in research_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            experiments.append(item.name)
    return sorted(experiments)


def list_datasets() -> list[str]:
    """List available datasets in data/ directory.

    Returns:
        List of dataset directory names
    """
    data_dir = get_data_path()
    if not data_dir.exists():
        return []

    datasets = []
    for item in data_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            datasets.append(item.name)
    return sorted(datasets)


__all__ = [
    "get_project_root",
    "get_research_path",
    "get_data_path",
    "get_results_path",
    "get_config_path",
    "list_research_experiments",
    "list_datasets",
]
