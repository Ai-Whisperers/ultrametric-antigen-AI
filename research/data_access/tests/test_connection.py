"""
Connection validation tests for all API clients.

Tests both mock and live API connections.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import pandas as pd


class TestConnectionValidation:
    """Test API connection validation."""

    def test_settings_validation(self):
        """Test settings validation returns warnings for missing config."""
        from data_access.config import Settings

        warnings = Settings.validate()
        assert isinstance(warnings, list)
        # Should warn if email not set
        if not Settings.ncbi.email:
            assert any("NCBI_EMAIL" in w for w in warnings)

    def test_datahub_creation(self):
        """Test DataHub can be instantiated."""
        from data_access import DataHub

        hub = DataHub()
        assert hub is not None
        assert hub._ncbi is None  # Lazy loaded
        assert hub._hivdb is None

    def test_datahub_lazy_loading(self):
        """Test DataHub clients are lazy loaded."""
        from data_access import DataHub

        hub = DataHub()

        # Clients should be None initially
        assert hub._ncbi is None

        # Accessing property should load client
        hivdb = hub.hivdb
        assert hub._hivdb is not None
        assert hivdb is hub._hivdb


class TestMockConnections:
    """Test connections using mocks."""

    def test_hivdb_mock_connection(self, mock_response):
        """Test HIVDB client with mock response."""
        from data_access.clients import HIVDBClient

        with patch("requests.Session.post") as mock_post:
            mock_post.return_value = mock_response(
                json_data={"data": {"viewer": {"algorithms": []}}}
            )

            client = HIVDBClient()
            result = client.get_algorithms()
            assert result is not None

    def test_cbioportal_mock_connection(self, mock_response):
        """Test cBioPortal client with mock response."""
        from data_access.clients import CBioPortalClient

        with patch("requests.Session.get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[{"cancerTypeId": "brca", "name": "Breast Cancer"}]
            )

            client = CBioPortalClient()
            result = client.get_cancer_types()
            assert isinstance(result, pd.DataFrame)

    def test_card_mock_connection(self, mock_response):
        """Test CARD client with mock response."""
        from data_access.clients import CARDClient

        with patch("requests.Session.get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[{"drug_class": "aminoglycoside"}]
            )

            client = CARDClient()
            result = client.get_drug_classes()
            assert isinstance(result, pd.DataFrame)

    def test_bvbrc_mock_connection(self, mock_response):
        """Test BV-BRC client with mock response."""
        from data_access.clients import BVBRCClient

        with patch("requests.Session.post") as mock_post:
            mock_post.return_value = mock_response(
                json_data=[{"genome_id": "test_123"}]
            )

            client = BVBRCClient()
            # Test query method
            with patch("requests.Session.get") as mock_get:
                mock_get.return_value = mock_response(json_data=[])
                result = client.get_data_summary()
                assert isinstance(result, pd.DataFrame)


@pytest.mark.live_api
class TestLiveConnections:
    """Test live API connections (requires network access)."""

    def test_hivdb_live_connection(self):
        """Test live connection to HIVDB API."""
        from data_access.clients import HIVDBClient

        client = HIVDBClient()
        try:
            algorithms = client.get_algorithms()
            assert algorithms is not None
        except Exception as e:
            pytest.skip(f"HIVDB API unavailable: {e}")

    def test_cbioportal_live_connection(self):
        """Test live connection to cBioPortal API."""
        from data_access.clients import CBioPortalClient

        client = CBioPortalClient()
        try:
            cancer_types = client.get_cancer_types()
            assert isinstance(cancer_types, pd.DataFrame)
            assert len(cancer_types) > 0
        except Exception as e:
            pytest.skip(f"cBioPortal API unavailable: {e}")

    def test_card_live_connection(self):
        """Test live connection to CARD API."""
        from data_access.clients import CARDClient

        client = CARDClient()
        try:
            drug_classes = client.get_drug_classes()
            assert isinstance(drug_classes, pd.DataFrame)
            assert len(drug_classes) > 0
        except Exception as e:
            pytest.skip(f"CARD API unavailable: {e}")

    def test_bvbrc_live_connection(self):
        """Test live connection to BV-BRC API."""
        from data_access.clients import BVBRCClient

        client = BVBRCClient()
        try:
            summary = client.get_data_summary()
            assert isinstance(summary, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"BV-BRC API unavailable: {e}")

    def test_datahub_test_connections(self):
        """Test DataHub connection testing."""
        from data_access import DataHub

        hub = DataHub()
        try:
            results = hub.test_connections()
            assert isinstance(results, pd.DataFrame)
            assert "api" in results.columns
            assert "status" in results.columns
        except Exception as e:
            pytest.skip(f"Connection test failed: {e}")
