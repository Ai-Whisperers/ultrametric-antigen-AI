"""Tests for visualization core functionality.

These tests verify the visualization module imports correctly and
basic functionality works without requiring actual plot rendering.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestVisualizationImports:
    """Test that visualization modules import correctly."""

    def test_import_styles_palettes(self):
        """Test palette module imports."""
        from src.visualization.styles.palettes import TOLVIBRANT, SEMANTIC

        assert TOLVIBRANT is not None
        assert len(TOLVIBRANT) > 0
        assert SEMANTIC is not None

    def test_import_styles_themes(self):
        """Test theme module imports."""
        from src.visualization.styles.themes import ThemeConfig, Context

        assert ThemeConfig is not None
        assert Context is not None

    def test_import_core_base(self):
        """Test core base module imports."""
        from src.visualization.core.base import create_figure, despine

        assert callable(create_figure)
        assert callable(despine)


class TestPaletteColors:
    """Test palette color definitions."""

    def test_tolvibrant_length(self):
        """TolVibrant palette should have expected colors."""
        from src.visualization.styles.palettes import TOLVIBRANT

        assert len(TOLVIBRANT) >= 7  # Standard TolVibrant has 7+ colors

    def test_semantic_colors(self):
        """Semantic colors should have required keys."""
        from src.visualization.styles.palettes import SEMANTIC

        assert hasattr(SEMANTIC, 'primary') or 'primary' in dir(SEMANTIC)

    def test_get_categorical_cmap(self):
        """Test categorical colormap generation."""
        from src.visualization.styles.palettes import get_categorical_cmap

        cmap = get_categorical_cmap("tolvibrant", n_colors=5)
        assert cmap is not None
        # Should be callable to get colors
        assert callable(cmap)


class TestThemeConfig:
    """Test theme configuration."""

    def test_theme_config_creation(self):
        """Test creating a ThemeConfig."""
        from src.visualization.styles.themes import ThemeConfig

        config = ThemeConfig(name="test")
        assert config.name == "test"

    def test_context_enum(self):
        """Test Context enum values."""
        from src.visualization.styles.themes import Context

        assert Context.PAPER is not None
        assert Context.TALK is not None
        assert Context.POSTER is not None


class TestFigureCreation:
    """Test figure creation utilities."""

    @patch('matplotlib.pyplot.figure')
    def test_create_figure_mocked(self, mock_figure):
        """Test create_figure with mocked matplotlib."""
        from src.visualization.core.base import create_figure

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_figure.return_value = mock_fig

        fig, ax = create_figure()
        assert mock_figure.called

    def test_despine_mocked(self):
        """Test despine with mocked matplotlib."""
        from src.visualization.core.base import despine

        mock_ax = MagicMock()

        # Should not raise
        despine(mock_ax)


class TestPlotHelpers:
    """Test plot helper functions."""

    def test_manifold_plot_imports(self):
        """Test manifold plot module imports."""
        try:
            from src.visualization.plots.manifold import plot_manifold
            assert callable(plot_manifold)
        except ImportError:
            pytest.skip("Manifold plot module not available")

    def test_training_plot_imports(self):
        """Test training plot module imports."""
        try:
            from src.visualization.plots.training import plot_training_curves
            assert callable(plot_training_curves)
        except ImportError:
            pytest.skip("Training plot module not available")


class TestProjections:
    """Test projection utilities."""

    def test_poincare_projection_import(self):
        """Test Poincare projection imports."""
        try:
            from src.visualization.projections.poincare import project_to_poincare_disk
            assert callable(project_to_poincare_disk)
        except ImportError:
            pytest.skip("Poincare projection not available")


class TestDataPreparation:
    """Test data preparation for visualization."""

    def test_embedding_to_2d(self):
        """Test embedding dimension reduction."""
        from sklearn.decomposition import PCA

        # Create fake embeddings
        embeddings = np.random.randn(100, 10)

        # Should reduce to 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        assert reduced.shape == (100, 2)

    def test_color_by_label(self):
        """Test color assignment by label."""
        from src.visualization.styles.palettes import get_categorical_cmap

        labels = np.array([0, 1, 2, 0, 1, 2])
        cmap = get_categorical_cmap("tolvibrant", n_colors=3)

        colors = [cmap(int(label)) for label in labels]
        assert len(colors) == len(labels)
