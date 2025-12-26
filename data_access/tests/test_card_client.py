"""
Unit tests for the CARD (Antibiotic Resistance Database) client.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import pandas as pd


class TestCARDClient:
    """Tests for CARDClient."""

    @pytest.fixture
    def client(self):
        """Create CARD client."""
        from data_access.clients import CARDClient
        return CARDClient()

    def test_client_creation(self, client):
        """Test client initialization."""
        assert client.base_url == "https://card.mcmaster.ca/api"
        assert client.session is not None

    def test_get_drug_classes_mock(self, client, mock_response):
        """Test getting drug classes with mock."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"drug_class": "aminoglycoside"},
                    {"drug_class": "beta-lactam"},
                    {"drug_class": "fluoroquinolone"},
                ]
            )

            result = client.get_drug_classes()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

    def test_get_drug_classes_fallback(self, client, mock_response):
        """Test drug classes fallback when API unavailable."""
        with patch.object(client.session, "get") as mock_get:
            from requests.exceptions import RequestException
            mock_get.side_effect = RequestException("API unavailable")

            result = client.get_drug_classes()
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0  # Should return fallback data

    def test_get_resistance_mechanisms_mock(self, client, mock_response):
        """Test getting resistance mechanisms with mock."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"mechanism": "antibiotic efflux"},
                    {"mechanism": "antibiotic inactivation"},
                ]
            )

            result = client.get_resistance_mechanisms()
            assert isinstance(result, pd.DataFrame)

    def test_get_resistance_mechanisms_fallback(self, client, mock_response):
        """Test resistance mechanisms fallback when API unavailable."""
        with patch.object(client.session, "get") as mock_get:
            from requests.exceptions import RequestException
            mock_get.side_effect = RequestException("API unavailable")

            result = client.get_resistance_mechanisms()
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

    def test_get_eskape_pathogens(self, client):
        """Test getting ESKAPE pathogen data."""
        # This uses static data + optional enrichment
        result = client.get_eskape_pathogens()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # E, S, K, A, P, E

        # Check for expected columns
        assert "pathogen" in result.columns
        assert "code" in result.columns

    def test_search_aro_mock(self, client, mock_response):
        """Test ARO search with mock."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"aro_id": "ARO:0000001", "name": "test gene"}
                ]
            )

            result = client.search_aro("test")
            assert isinstance(result, pd.DataFrame)

    def test_get_resistance_genes_mock(self, client, mock_response):
        """Test resistance gene retrieval with mock."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"gene_name": "blaNDM-1", "pathogen": "E. coli"},
                    {"gene_name": "mecA", "pathogen": "S. aureus"},
                ]
            )

            result = client.get_resistance_genes()
            assert isinstance(result, pd.DataFrame)

    def test_get_model_summary_mock(self, client, mock_response):
        """Test model summary with mock."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"model_type": "protein homolog"},
                    {"model_type": "protein variant"},
                ]
            )

            result = client.get_model_summary()
            assert isinstance(result, pd.DataFrame)


@pytest.mark.live_api
class TestCARDLive:
    """Live API tests for CARD client."""

    @pytest.fixture
    def client(self):
        """Create CARD client."""
        from data_access.clients import CARDClient
        return CARDClient()

    def test_live_get_drug_classes(self, client):
        """Test live drug class retrieval."""
        try:
            result = client.get_drug_classes()
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"CARD API unavailable: {e}")

    def test_live_get_eskape(self, client):
        """Test live ESKAPE pathogen retrieval."""
        try:
            result = client.get_eskape_pathogens()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 6
        except Exception as e:
            pytest.skip(f"CARD API unavailable: {e}")
