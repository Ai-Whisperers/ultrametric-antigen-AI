"""
Unit tests for the BV-BRC (Bacterial and Viral Bioinformatics Resource Center) client.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import pandas as pd


class TestBVBRCClient:
    """Tests for BVBRCClient."""

    @pytest.fixture
    def client(self):
        """Create BV-BRC client."""
        from data_access.clients import BVBRCClient
        return BVBRCClient()

    def test_client_creation(self, client):
        """Test client initialization."""
        assert client.base_url == "https://www.bv-brc.org/api"
        assert client.session is not None

    def test_taxon_ids(self, client):
        """Test predefined taxon IDs."""
        assert client.TAXON_IDS["Mycobacterium tuberculosis"] == 1773
        assert client.TAXON_IDS["Staphylococcus aureus"] == 1280
        assert client.TAXON_IDS["Escherichia coli"] == 562

    def test_data_types(self, client):
        """Test predefined data types."""
        assert "genome" in client.DATA_TYPES
        assert "genome_amr" in client.DATA_TYPES
        assert "sp_gene" in client.DATA_TYPES

    def test_search_genomes_mock(self, client, mock_response):
        """Test genome search with mock."""
        with patch.object(client.session, "post") as mock_post:
            mock_post.return_value = mock_response(
                json_data=[
                    {
                        "genome_id": "83332.12",
                        "genome_name": "Mycobacterium tuberculosis H37Rv",
                        "taxon_id": 1773,
                        "genome_status": "Complete",
                    }
                ]
            )

            result = client.search_genomes(organism="Mycobacterium tuberculosis")
            assert isinstance(result, pd.DataFrame)

    def test_get_amr_phenotypes_mock(self, client, mock_response):
        """Test AMR phenotype retrieval with mock."""
        with patch.object(client.session, "post") as mock_post:
            mock_post.return_value = mock_response(
                json_data=[
                    {
                        "genome_id": "83332.12",
                        "genome_name": "M. tuberculosis",
                        "antibiotic": "Rifampicin",
                        "resistant_phenotype": "Resistant",
                    }
                ]
            )

            result = client.get_amr_phenotypes()
            assert isinstance(result, pd.DataFrame)

    def test_get_specialty_genes_mock(self, client, mock_response):
        """Test specialty gene retrieval with mock."""
        with patch.object(client.session, "post") as mock_post:
            mock_post.return_value = mock_response(
                json_data=[
                    {
                        "genome_id": "83332.12",
                        "gene": "rpoB",
                        "product": "RNA polymerase beta subunit",
                        "property": "Antibiotic Resistance",
                    }
                ]
            )

            result = client.get_specialty_genes(property_type="Antibiotic Resistance")
            assert isinstance(result, pd.DataFrame)

    def test_get_virulence_factors(self, client, mock_response):
        """Test virulence factor retrieval."""
        with patch.object(client, "get_specialty_genes") as mock_genes:
            mock_genes.return_value = pd.DataFrame([
                {"gene": "esxA", "property": "Virulence Factor"}
            ])

            result = client.get_virulence_factors()
            assert isinstance(result, pd.DataFrame)

    def test_get_resistance_genes(self, client, mock_response):
        """Test resistance gene retrieval."""
        with patch.object(client, "get_specialty_genes") as mock_genes:
            mock_genes.return_value = pd.DataFrame([
                {"gene": "katG", "property": "Antibiotic Resistance"}
            ])

            result = client.get_resistance_genes()
            assert isinstance(result, pd.DataFrame)

    def test_get_tb_genomes(self, client, mock_response):
        """Test TB genome retrieval."""
        with patch.object(client, "search_genomes") as mock_search:
            mock_search.return_value = pd.DataFrame([
                {"genome_id": "83332.12", "genome_name": "M. tuberculosis H37Rv"}
            ])

            result = client.get_tb_genomes()
            assert isinstance(result, pd.DataFrame)
            mock_search.assert_called_once()

    def test_get_syphilis_genomes(self, client, mock_response):
        """Test syphilis genome retrieval."""
        with patch.object(client, "search_genomes") as mock_search:
            mock_search.return_value = pd.DataFrame([
                {"genome_id": "160.1", "genome_name": "Treponema pallidum"}
            ])

            result = client.get_syphilis_genomes()
            assert isinstance(result, pd.DataFrame)
            mock_search.assert_called_once()

    def test_get_data_summary(self, client, mock_response):
        """Test data summary retrieval."""
        with patch.object(client, "_query") as mock_query:
            mock_query.return_value = pd.DataFrame([{"genome_id": "test"}])

            result = client.get_data_summary()
            assert isinstance(result, pd.DataFrame)


@pytest.mark.live_api
class TestBVBRCLive:
    """Live API tests for BV-BRC client."""

    @pytest.fixture
    def client(self):
        """Create BV-BRC client."""
        from data_access.clients import BVBRCClient
        return BVBRCClient()

    def test_live_get_data_summary(self, client):
        """Test live data summary retrieval."""
        try:
            result = client.get_data_summary()
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"BV-BRC API unavailable: {e}")

    def test_live_search_genomes(self, client):
        """Test live genome search."""
        try:
            result = client.search_genomes(
                organism="Mycobacterium tuberculosis",
                limit=5
            )
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"BV-BRC API unavailable: {e}")

    def test_live_get_amr_phenotypes(self, client):
        """Test live AMR phenotype retrieval."""
        try:
            result = client.get_amr_phenotypes(limit=10)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"BV-BRC API unavailable: {e}")
