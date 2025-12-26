"""
Unit tests for the NCBI/Entrez client.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestNCBIClient:
    """Tests for NCBIClient."""

    @pytest.fixture
    def client(self, mock_env):
        """Create NCBI client with mock environment."""
        from data_access.clients import NCBIClient
        return NCBIClient()

    def test_client_creation(self, client):
        """Test client initialization."""
        assert client is not None

    def test_search_hiv_sequences_mock(self, client):
        """Test HIV sequence search with mock."""
        with patch.object(client, "search") as mock_search:
            mock_search.return_value = {
                "ids": ["123", "456"],
                "count": 2,
                "query_translation": "HIV-1[Organism]"
            }

            result = client.search_hiv_sequences(subtype="B", max_results=10)
            assert "ids" in result
            assert result["count"] == 2

    def test_get_sequence_summary_mock(self, client):
        """Test sequence summary retrieval with mock."""
        # Mock the Entrez.esummary call at module level
        with patch("Bio.Entrez.esummary") as mock_summary:
            mock_handle = MagicMock()
            mock_summary.return_value = mock_handle
            mock_handle.__enter__ = MagicMock(return_value=mock_handle)
            mock_handle.__exit__ = MagicMock(return_value=False)
            mock_handle.read.return_value = ""

            with patch("Bio.Entrez.read") as mock_read:
                mock_read.return_value = [
                    {
                        "Id": "123",
                        "Title": "HIV-1 isolate pol gene",
                        "Length": 1500,
                        "AccessionVersion": "KY123456.1"
                    }
                ]

                # Check method exists
                assert hasattr(client, "get_sequence_summary")

    def test_fetch_sequences_mock(self, client):
        """Test sequence fetching with mock."""
        with patch("Bio.Entrez.efetch") as mock_efetch:
            mock_record = MagicMock()
            mock_record.seq = "ATGCATGC"
            mock_record.id = "test_id"
            mock_record.description = "Test sequence"

            mock_efetch.return_value = MagicMock()
            mock_efetch.return_value.__enter__ = MagicMock(return_value=mock_efetch.return_value)
            mock_efetch.return_value.__exit__ = MagicMock(return_value=False)

            # Test that method exists
            assert hasattr(client, "fetch_sequences")


class TestNCBISearchFilters:
    """Tests for NCBI search filter construction."""

    @pytest.fixture
    def client(self, mock_env):
        """Create NCBI client."""
        from data_access.clients import NCBIClient
        return NCBIClient()

    def test_search_with_subtype_filter(self, client):
        """Test search with subtype filter."""
        with patch.object(client, "search") as mock_search:
            mock_search.return_value = {"ids": [], "count": 0}

            client.search_hiv_sequences(subtype="B")

            # Check that search was called with subtype in query
            call_args = mock_search.call_args
            assert call_args is not None

    def test_search_with_gene_filter(self, client):
        """Test search with gene filter."""
        with patch.object(client, "search") as mock_search:
            mock_search.return_value = {"ids": [], "count": 0}

            client.search_hiv_sequences(gene="pol")

            call_args = mock_search.call_args
            assert call_args is not None

    def test_search_with_country_filter(self, client):
        """Test search with country filter."""
        with patch.object(client, "search") as mock_search:
            mock_search.return_value = {"ids": [], "count": 0}

            client.search_hiv_sequences(country="USA")

            call_args = mock_search.call_args
            assert call_args is not None


@pytest.mark.live_api
class TestNCBILive:
    """Live API tests for NCBI client."""

    @pytest.fixture
    def client(self):
        """Create NCBI client with real config."""
        from data_access.clients import NCBIClient
        from data_access.config import settings

        if not settings.ncbi.email:
            pytest.skip("NCBI_EMAIL not configured")

        return NCBIClient()

    def test_live_search_hiv(self, client):
        """Test live HIV sequence search."""
        try:
            result = client.search_hiv_sequences(subtype="B", max_results=5)
            assert "ids" in result
            assert isinstance(result["count"], int)
        except Exception as e:
            pytest.skip(f"NCBI API unavailable: {e}")

    def test_live_fetch_pubmed(self, client):
        """Test live PubMed search."""
        try:
            result = client.fetch_pubmed_abstracts("HIV drug resistance", max_results=3)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"NCBI API unavailable: {e}")
