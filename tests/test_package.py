"""Tests for package structure and imports."""

import pytest
from importlib import import_module


class TestPackageStructure:
    """Test package structure and imports."""

    def test_main_package_import(self):
        """Test that main package imports correctly."""
        import hiv_analysis
        
        assert hasattr(hiv_analysis, '__version__')
        assert hasattr(hiv_analysis, '__author__')
        assert hasattr(hiv_analysis, '__email__')
        
        assert hiv_analysis.__author__ == "AI Whisperers"
        assert hiv_analysis.__email__ == "research@ai-whisperers.com"

    def test_submodule_imports(self):
        """Test that submodules can be imported."""
        # These should not raise ImportError
        import hiv_analysis.cli
        import hiv_analysis.core
        import hiv_analysis.utils
        import hiv_analysis.data
        import hiv_analysis.scripts

    def test_script_imports(self):
        """Test that individual scripts can be imported."""
        from hiv_analysis.scripts import conservation_scorer
        from hiv_analysis.scripts import alignment_viewer
        from hiv_analysis.scripts import format_exporter
        from hiv_analysis.scripts import mafft_wrapper
        from hiv_analysis.scripts import setup_hiv_data
        from hiv_analysis.scripts import los_alamos_downloader

        # Test that key classes/functions exist
        assert hasattr(conservation_scorer, 'ConservationScorer')
        assert hasattr(conservation_scorer, 'ConservationConfig')

    def test_cli_module_structure(self):
        """Test CLI module structure."""
        from hiv_analysis import cli
        
        assert hasattr(cli, 'main')
        assert callable(cli.main)

    def test_version_format(self):
        """Test that version follows expected format."""
        import hiv_analysis
        
        version = hiv_analysis.__version__
        assert isinstance(version, str)
        
        # Should follow semantic versioning (x.y.z)
        parts = version.split('.')
        assert len(parts) >= 2
        assert all(part.isdigit() for part in parts[:2])

    def test_package_exports(self):
        """Test that __all__ exports work correctly."""
        import hiv_analysis
        
        # Should have defined exports
        assert hasattr(hiv_analysis, '__all__')
        assert isinstance(hiv_analysis.__all__, list)
        
        # All exported names should be importable
        for name in hiv_analysis.__all__:
            assert hasattr(hiv_analysis, name)

    @pytest.mark.integration
    def test_dynamic_imports(self):
        """Test dynamic importing of modules."""
        module_paths = [
            'hiv_analysis.cli',
            'hiv_analysis.scripts.conservation_scorer',
            'hiv_analysis.scripts.alignment_viewer'
        ]
        
        for module_path in module_paths:
            try:
                module = import_module(module_path)
                assert module is not None
            except ImportError as e:
                pytest.fail(f"Failed to import {module_path}: {e}")

    def test_module_docstrings(self):
        """Test that key modules have docstrings."""
        import hiv_analysis
        import hiv_analysis.cli
        from hiv_analysis.scripts import conservation_scorer
        
        assert hiv_analysis.__doc__ is not None
        assert len(hiv_analysis.__doc__) > 0
        assert hiv_analysis.cli.__doc__ is not None
        assert conservation_scorer.__doc__ is not None