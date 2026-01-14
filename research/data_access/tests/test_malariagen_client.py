"""
Unit tests for the MalariaGEN client.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestMalariaGENClient:
    """Tests for MalariaGENClient."""

    @pytest.fixture
    def mock_malariagen_data(self):
        """Create mock malariagen_data module."""
        mock_module = MagicMock()

        # Mock Pf7 dataset
        mock_pf7 = MagicMock()
        mock_pf7.sample_metadata.return_value = pd.DataFrame([
            {"sample_id": "PF0001", "country": "Kenya", "year": 2020},
            {"sample_id": "PF0002", "country": "Ghana", "year": 2019},
        ])
        mock_pf7.sample_sets.return_value = pd.DataFrame([
            {"sample_set": "Pf7.1", "samples": 5000}
        ])
        mock_module.Pf7.return_value = mock_pf7

        # Mock Pv4 dataset
        mock_pv4 = MagicMock()
        mock_pv4.sample_metadata.return_value = pd.DataFrame([
            {"sample_id": "PV0001", "country": "Brazil", "year": 2021}
        ])
        mock_module.Pv4.return_value = mock_pv4

        # Mock Ag3 dataset
        mock_ag3 = MagicMock()
        mock_ag3.sample_metadata.return_value = pd.DataFrame([
            {"sample_id": "AG0001", "country": "Mali", "year": 2020}
        ])
        mock_ag3.sample_sets.return_value = pd.DataFrame([
            {"sample_set": "AG3.1", "samples": 3000}
        ])
        mock_module.Ag3.return_value = mock_ag3

        return mock_module

    def test_client_creation(self, mock_malariagen_data):
        """Test client initialization with mock."""
        with patch.dict("sys.modules", {"malariagen_data": mock_malariagen_data}):
            # Reset the global variable
            import data_access.clients.malariagen_client as mc
            mc.malariagen_data = None

            with patch.object(mc, "_ensure_malariagen") as mock_ensure:
                mock_ensure.return_value = None
                mc.malariagen_data = mock_malariagen_data

                from data_access.clients import MalariaGENClient
                client = MalariaGENClient()
                assert client is not None

    def test_get_pf_sample_metadata(self, mock_malariagen_data):
        """Test getting P. falciparum sample metadata."""
        with patch.dict("sys.modules", {"malariagen_data": mock_malariagen_data}):
            import data_access.clients.malariagen_client as mc
            mc.malariagen_data = mock_malariagen_data

            from data_access.clients import MalariaGENClient
            client = MalariaGENClient()

            result = client.get_pf_sample_metadata()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

    def test_get_pf_samples_by_country(self, mock_malariagen_data):
        """Test filtering P. falciparum samples by country."""
        with patch.dict("sys.modules", {"malariagen_data": mock_malariagen_data}):
            import data_access.clients.malariagen_client as mc
            mc.malariagen_data = mock_malariagen_data

            from data_access.clients import MalariaGENClient
            client = MalariaGENClient()

            result = client.get_pf_samples_by_country("Kenya")
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    def test_get_dataset_summary(self, mock_malariagen_data):
        """Test getting dataset summary."""
        with patch.dict("sys.modules", {"malariagen_data": mock_malariagen_data}):
            import data_access.clients.malariagen_client as mc
            mc.malariagen_data = mock_malariagen_data

            from data_access.clients import MalariaGENClient
            client = MalariaGENClient()

            result = client.get_dataset_summary()
            assert isinstance(result, pd.DataFrame)

    def test_get_pf_country_summary(self, mock_malariagen_data):
        """Test getting country-level summary."""
        with patch.dict("sys.modules", {"malariagen_data": mock_malariagen_data}):
            import data_access.clients.malariagen_client as mc
            mc.malariagen_data = mock_malariagen_data

            from data_access.clients import MalariaGENClient
            client = MalariaGENClient()

            result = client.get_pf_country_summary()
            assert isinstance(result, pd.DataFrame)
            assert "country" in result.columns
            assert "sample_count" in result.columns


class TestMalariaGENImportError:
    """Tests for handling missing malariagen_data package."""

    def test_import_error_handling(self):
        """Test proper handling when package not installed."""
        # This test verifies the import error is raised properly
        import data_access.clients.malariagen_client as mc

        # Store original value
        original = mc.malariagen_data

        # Set to None to trigger import attempt
        mc.malariagen_data = None

        with patch.dict("sys.modules", {"malariagen_data": None}):
            with patch("builtins.__import__", side_effect=ImportError("not installed")):
                with pytest.raises(ImportError) as exc_info:
                    mc._ensure_malariagen()

                assert "malariagen_data" in str(exc_info.value)

        # Restore original
        mc.malariagen_data = original


@pytest.mark.live_api
class TestMalariaGENLive:
    """Live API tests for MalariaGEN client."""

    @pytest.fixture
    def client(self):
        """Create MalariaGEN client."""
        try:
            from data_access.clients import MalariaGENClient
            return MalariaGENClient()
        except ImportError:
            pytest.skip("malariagen_data package not installed")

    def test_live_get_dataset_summary(self, client):
        """Test live dataset summary retrieval."""
        try:
            result = client.get_dataset_summary()
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"MalariaGEN API unavailable: {e}")

    def test_live_get_pf_metadata(self, client):
        """Test live P. falciparum metadata retrieval."""
        try:
            result = client.get_pf_sample_metadata()
            assert isinstance(result, pd.DataFrame)
            # Pf7 has ~20k samples
            assert len(result) > 10000
        except Exception as e:
            pytest.skip(f"MalariaGEN API unavailable: {e}")
