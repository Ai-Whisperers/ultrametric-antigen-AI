# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src.config.paths module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest


class TestProjectRoot:
    """Tests for PROJECT_ROOT detection."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should point to an existing directory."""
        from src.config.paths import PROJECT_ROOT

        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_has_markers(self):
        """PROJECT_ROOT should contain project marker files."""
        from src.config.paths import PROJECT_ROOT

        # At least one of these should exist
        markers = ["pyproject.toml", "setup.py", ".git", "LICENSE"]
        found = [m for m in markers if (PROJECT_ROOT / m).exists()]
        assert len(found) > 0, f"No project markers found in {PROJECT_ROOT}"

    def test_project_root_contains_src(self):
        """PROJECT_ROOT should contain src directory."""
        from src.config.paths import PROJECT_ROOT

        assert (PROJECT_ROOT / "src").exists()
        assert (PROJECT_ROOT / "src").is_dir()


class TestDataDirectories:
    """Tests for data directory paths."""

    def test_data_dir_path(self):
        """DATA_DIR should be under PROJECT_ROOT."""
        from src.config.paths import DATA_DIR, PROJECT_ROOT

        assert DATA_DIR == PROJECT_ROOT / "data"

    def test_subdirectories_are_children(self):
        """Data subdirectories should be children of DATA_DIR."""
        from src.config.paths import (
            CACHE_DIR,
            DATA_DIR,
            EXTERNAL_DATA_DIR,
            PROCESSED_DATA_DIR,
            RAW_DATA_DIR,
        )

        assert RAW_DATA_DIR == DATA_DIR / "raw"
        assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
        assert EXTERNAL_DATA_DIR == DATA_DIR / "external"
        assert CACHE_DIR == DATA_DIR / "cache"


class TestOutputDirectories:
    """Tests for output directory paths."""

    def test_output_dir_path(self):
        """OUTPUT_DIR should be under PROJECT_ROOT."""
        from src.config.paths import OUTPUT_DIR, PROJECT_ROOT

        assert OUTPUT_DIR == PROJECT_ROOT / "outputs"

    def test_subdirectories_are_children(self):
        """Output subdirectories should be children of OUTPUT_DIR."""
        from src.config.paths import (
            CHECKPOINTS_DIR,
            LOGS_DIR,
            OUTPUT_DIR,
            REPORTS_DIR,
            RESULTS_DIR,
            RUNS_DIR,
            VIZ_DIR,
        )

        assert RESULTS_DIR == OUTPUT_DIR / "results"
        assert CHECKPOINTS_DIR == OUTPUT_DIR / "models"
        assert RUNS_DIR == OUTPUT_DIR / "runs"
        assert REPORTS_DIR == OUTPUT_DIR / "reports"
        assert VIZ_DIR == OUTPUT_DIR / "visualizations"
        assert LOGS_DIR == OUTPUT_DIR / "logs"


class TestHelperFunctions:
    """Tests for path helper functions."""

    def test_ensure_dirs_creates_directories(self, tmp_path):
        """ensure_dirs should create directories."""
        from src.config.paths import ensure_dirs

        dir1 = tmp_path / "test1"
        dir2 = tmp_path / "test2" / "nested"

        assert not dir1.exists()
        assert not dir2.exists()

        ensure_dirs(dir1, dir2)

        assert dir1.exists()
        assert dir2.exists()

    def test_ensure_dirs_idempotent(self, tmp_path):
        """ensure_dirs should not fail if directories exist."""
        from src.config.paths import ensure_dirs

        dir1 = tmp_path / "existing"
        dir1.mkdir()

        # Should not raise
        ensure_dirs(dir1)
        assert dir1.exists()

    def test_get_checkpoint_path_without_version(self):
        """get_checkpoint_path should return correct path without version."""
        from src.config.paths import CHECKPOINTS_DIR, get_checkpoint_path

        path = get_checkpoint_path("my_model")
        assert path == CHECKPOINTS_DIR / "my_model.pt"

    def test_get_checkpoint_path_with_version(self):
        """get_checkpoint_path should return correct path with version."""
        from src.config.paths import CHECKPOINTS_DIR, get_checkpoint_path

        path = get_checkpoint_path("my_model", version="v5_11")
        assert path == CHECKPOINTS_DIR / "v5_11" / "my_model.pt"

    def test_get_results_path_default_ext(self):
        """get_results_path should use json extension by default."""
        from src.config.paths import RESULTS_DIR, get_results_path

        path = get_results_path("analysis")
        assert path == RESULTS_DIR / "analysis.json"

    def test_get_results_path_custom_ext(self):
        """get_results_path should support custom extensions."""
        from src.config.paths import RESULTS_DIR, get_results_path

        path = get_results_path("data", ext="csv")
        assert path == RESULTS_DIR / "data.csv"

    def test_get_data_path_processed(self):
        """get_data_path should return processed path by default."""
        from src.config.paths import PROCESSED_DATA_DIR, get_data_path

        path = get_data_path("features.pt")
        assert path == PROCESSED_DATA_DIR / "features.pt"

    def test_get_data_path_raw(self):
        """get_data_path should return raw path when specified."""
        from src.config.paths import RAW_DATA_DIR, get_data_path

        path = get_data_path("sequences.fasta", processed=False)
        assert path == RAW_DATA_DIR / "sequences.fasta"


class TestLegacyPathResolution:
    """Tests for legacy path resolution."""

    def test_resolve_results_path(self):
        """resolve_legacy_path should map results/ to new location."""
        from src.config.paths import RESULTS_DIR, resolve_legacy_path

        path = resolve_legacy_path("results/analysis.json")
        assert path == RESULTS_DIR / "analysis.json"

    def test_resolve_sandbox_checkpoints(self):
        """resolve_legacy_path should map sandbox-training/checkpoints/."""
        from src.config.paths import CHECKPOINTS_DIR, resolve_legacy_path

        path = resolve_legacy_path("sandbox-training/checkpoints/v5_11/best.pt")
        assert path == CHECKPOINTS_DIR / "v5_11/best.pt"

    def test_resolve_data_raw_path(self):
        """resolve_legacy_path should map data/raw/."""
        from src.config.paths import RAW_DATA_DIR, resolve_legacy_path

        path = resolve_legacy_path("data/raw/sequences.fasta")
        assert path == RAW_DATA_DIR / "sequences.fasta"

    def test_resolve_data_processed_path(self):
        """resolve_legacy_path should map data/processed/."""
        from src.config.paths import PROCESSED_DATA_DIR, resolve_legacy_path

        path = resolve_legacy_path("data/processed/features.pt")
        assert path == PROCESSED_DATA_DIR / "features.pt"

    def test_resolve_outputs_viz_path(self):
        """resolve_legacy_path should map outputs/viz/."""
        from src.config.paths import VIZ_DIR, resolve_legacy_path

        path = resolve_legacy_path("outputs/viz/plot.png")
        assert path == VIZ_DIR / "plot.png"

    def test_resolve_unknown_path(self):
        """resolve_legacy_path should resolve unknown paths against PROJECT_ROOT."""
        from src.config.paths import PROJECT_ROOT, resolve_legacy_path

        path = resolve_legacy_path("some/unknown/path.txt")
        assert path == PROJECT_ROOT / "some/unknown/path.txt"


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_data_dir_env_override(self, tmp_path, monkeypatch):
        """DATA_DIR should respect TERNARY_DATA_DIR environment variable."""
        custom_data = tmp_path / "custom_data"
        custom_data.mkdir()

        monkeypatch.setenv("TERNARY_DATA_DIR", str(custom_data))

        # Need to reload the module to pick up env var
        import importlib

        import src.config.paths

        importlib.reload(src.config.paths)

        assert src.config.paths.DATA_DIR == custom_data

        # Cleanup: reload again without env var
        monkeypatch.delenv("TERNARY_DATA_DIR", raising=False)
        importlib.reload(src.config.paths)

    def test_output_dir_env_override(self, tmp_path, monkeypatch):
        """OUTPUT_DIR should respect TERNARY_OUTPUT_DIR environment variable."""
        custom_output = tmp_path / "custom_output"
        custom_output.mkdir()

        monkeypatch.setenv("TERNARY_OUTPUT_DIR", str(custom_output))

        import importlib

        import src.config.paths

        importlib.reload(src.config.paths)

        assert src.config.paths.OUTPUT_DIR == custom_output

        # Cleanup
        monkeypatch.delenv("TERNARY_OUTPUT_DIR", raising=False)
        importlib.reload(src.config.paths)


class TestInitProjectDirs:
    """Tests for init_project_dirs function."""

    def test_init_creates_directories(self, tmp_path, monkeypatch):
        """init_project_dirs should create all standard directories."""
        # Set up temporary directories
        monkeypatch.setenv("TERNARY_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("TERNARY_OUTPUT_DIR", str(tmp_path / "outputs"))

        import importlib

        import src.config.paths

        importlib.reload(src.config.paths)

        # Call init
        src.config.paths.init_project_dirs()

        # Check directories were created
        assert (tmp_path / "data" / "raw").exists()
        assert (tmp_path / "data" / "processed").exists()
        assert (tmp_path / "outputs" / "results").exists()
        assert (tmp_path / "outputs" / "models").exists()

        # Cleanup
        monkeypatch.delenv("TERNARY_DATA_DIR", raising=False)
        monkeypatch.delenv("TERNARY_OUTPUT_DIR", raising=False)
        importlib.reload(src.config.paths)


class TestImportFromConfig:
    """Tests for importing paths from src.config."""

    def test_import_from_config_module(self):
        """Paths should be importable from src.config."""
        from src.config import (
            CHECKPOINTS_DIR,
            DATA_DIR,
            OUTPUT_DIR,
            PROJECT_ROOT,
            RESULTS_DIR,
        )

        assert PROJECT_ROOT is not None
        assert DATA_DIR is not None
        assert OUTPUT_DIR is not None
        assert RESULTS_DIR is not None
        assert CHECKPOINTS_DIR is not None

    def test_import_helper_functions(self):
        """Helper functions should be importable from src.config."""
        from src.config import (
            ensure_dirs,
            get_checkpoint_path,
            get_data_path,
            get_results_path,
            init_project_dirs,
            resolve_legacy_path,
        )

        assert callable(ensure_dirs)
        assert callable(get_checkpoint_path)
        assert callable(get_results_path)
        assert callable(get_data_path)
        assert callable(resolve_legacy_path)
        assert callable(init_project_dirs)
