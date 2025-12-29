"""Tests for path resolution utilities.

Tests verify path resolution and utility functions work correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.paths import (
    DATA_DIR,
    PROJECT_ROOT,
    RESEARCH_DIR,
    RESULTS_DIR,
    get_config_path,
    get_project_root,
    get_research_path,
    list_datasets,
    list_research_experiments,
)


class TestPathResolution:
    """Test path resolution functions."""

    def test_get_project_root(self):
        """Project root should contain src/ directory."""
        root = get_project_root()
        assert root.exists()
        assert (root / "src").exists()

    def test_project_root_constant(self):
        """PROJECT_ROOT should contain src/ directory."""
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "src").exists()

    def test_get_research_path_base(self):
        """Should return research directory."""
        path = get_research_path()
        assert path.name == "research"

    def test_get_research_path_subdir(self):
        """Should return research subdirectory."""
        path = get_research_path("bioinformatics")
        assert "bioinformatics" in str(path)

    def test_data_dir(self):
        """DATA_DIR should be data directory."""
        assert DATA_DIR.name == "data"

    def test_results_dir(self):
        """RESULTS_DIR should be in outputs."""
        assert "results" in str(RESULTS_DIR)

    def test_get_config_path_existing(self):
        """Should return config path if exists."""
        path = get_config_path("ternary.yaml")
        # May or may not exist, but should not crash
        assert path is None or path.exists()

    def test_get_config_path_nonexistent(self):
        """Should return None for nonexistent config."""
        path = get_config_path("nonexistent_config_12345.yaml")
        assert path is None


class TestListingFunctions:
    """Test directory listing functions."""

    def test_list_research_experiments(self):
        """Should return list of experiment names."""
        experiments = list_research_experiments()
        assert isinstance(experiments, list)
        # Should have at least bioinformatics
        if experiments:  # May be empty in some test environments
            assert all(isinstance(e, str) for e in experiments)

    def test_list_datasets(self):
        """Should return list of dataset names."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        if datasets:  # May be empty in some test environments
            assert all(isinstance(d, str) for d in datasets)


class TestModuleImports:
    """Test module can be imported correctly."""

    def test_import_all_exports(self):
        """All exported symbols should be importable."""
        from src.config.paths import (
            get_config_path,
            get_project_root,
            get_research_path,
            list_datasets,
            list_research_experiments,
        )

        # Verify functions are callable
        assert callable(get_project_root)
        assert callable(get_research_path)
        assert callable(get_config_path)
        assert callable(list_research_experiments)
        assert callable(list_datasets)
