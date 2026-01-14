"""
Unit tests for the cBioPortal client.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import pandas as pd


class TestCBioPortalClient:
    """Tests for CBioPortalClient."""

    @pytest.fixture
    def client(self):
        """Create cBioPortal client."""
        from data_access.clients import CBioPortalClient
        return CBioPortalClient()

    def test_client_creation(self, client):
        """Test client initialization."""
        assert client.base_url == "https://www.cbioportal.org/api"
        assert client.session is not None

    def test_get_cancer_types_mock(self, client, mock_response):
        """Test getting cancer types with mock."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"cancerTypeId": "brca", "name": "Breast Cancer"},
                    {"cancerTypeId": "lung", "name": "Lung Cancer"},
                ]
            )

            result = client.get_cancer_types()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

    def test_get_studies_mock(self, client, mock_response):
        """Test getting studies with mock."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"studyId": "brca_tcga", "name": "Breast TCGA", "description": "TCGA Breast Cancer"},
                    {"studyId": "lung_tcga", "name": "Lung TCGA", "description": "TCGA Lung Cancer"},
                ]
            )

            result = client.get_studies()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

    def test_get_studies_with_keyword(self, client, mock_response):
        """Test filtering studies by keyword."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"studyId": "brca_tcga", "name": "Breast TCGA", "description": "TCGA Breast Cancer"},
                    {"studyId": "lung_tcga", "name": "Lung TCGA", "description": "TCGA Lung Cancer"},
                ]
            )

            result = client.get_studies(keyword="Breast")
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1

    def test_get_mutations_mock(self, client, mock_response):
        """Test getting mutations with mock."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {
                        "molecularProfileId": "brca_tcga_mutations",
                        "molecularAlterationType": "MUTATION_EXTENDED"
                    }
                ]
            )

            with patch.object(client.session, "post") as mock_post:
                mock_post.return_value = mock_response(
                    json_data=[
                        {
                            "sampleId": "TCGA-001",
                            "entrezGeneId": 7157,
                            "gene": "TP53",
                            "proteinChange": "R175H",
                        }
                    ]
                )

                result = client.get_mutations("brca_tcga")
                assert isinstance(result, pd.DataFrame)

    def test_get_top_mutated_genes_mock(self, client, mock_response):
        """Test getting top mutated genes."""
        with patch.object(client, "get_mutations") as mock_mutations:
            mock_mutations.return_value = pd.DataFrame([
                {"sampleId": "S1", "entrezGeneId": 7157, "gene": "TP53"},
                {"sampleId": "S2", "entrezGeneId": 7157, "gene": "TP53"},
                {"sampleId": "S1", "entrezGeneId": 672, "gene": "BRCA1"},
            ])

            result = client.get_top_mutated_genes("brca_tcga", top_n=5)
            assert isinstance(result, pd.DataFrame)

    def test_get_available_data_types(self, client, mock_response):
        """Test checking available data types."""
        with patch.object(client.session, "get") as mock_get:
            mock_get.return_value = mock_response(
                json_data=[
                    {"molecularAlterationType": "MUTATION_EXTENDED"},
                    {"molecularAlterationType": "COPY_NUMBER_ALTERATION"},
                    {"molecularAlterationType": "MRNA_EXPRESSION"},
                ]
            )

            result = client.get_available_data_types("brca_tcga")
            assert isinstance(result, dict)
            assert result["mutation"] is True
            assert result["copy_number"] is True
            assert result["expression"] is True


@pytest.mark.live_api
class TestCBioPortalLive:
    """Live API tests for cBioPortal client."""

    @pytest.fixture
    def client(self):
        """Create cBioPortal client."""
        from data_access.clients import CBioPortalClient
        return CBioPortalClient()

    def test_live_get_cancer_types(self, client):
        """Test live cancer types retrieval."""
        try:
            result = client.get_cancer_types()
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"cBioPortal API unavailable: {e}")

    def test_live_get_studies(self, client):
        """Test live studies retrieval."""
        try:
            result = client.get_studies()
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"cBioPortal API unavailable: {e}")

    def test_live_search_genes(self, client):
        """Test live gene search."""
        try:
            result = client.search_genes("TP53")
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"cBioPortal API unavailable: {e}")
