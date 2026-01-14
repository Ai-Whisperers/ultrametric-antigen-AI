"""
Integration tests for the DataHub class.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import pandas as pd


class TestDataHub:
    """Tests for DataHub integration class."""

    @pytest.fixture
    def hub(self):
        """Create DataHub instance."""
        from data_access import DataHub
        return DataHub()

    def test_hub_creation(self, hub):
        """Test DataHub initialization."""
        assert hub is not None
        assert hub._ncbi is None
        assert hub._hivdb is None
        assert hub._cbioportal is None

    def test_lazy_loading_hivdb(self, hub):
        """Test HIVDB client lazy loading."""
        assert hub._hivdb is None
        client = hub.hivdb
        assert hub._hivdb is not None
        assert client is hub._hivdb

    def test_lazy_loading_cbioportal(self, hub):
        """Test cBioPortal client lazy loading."""
        assert hub._cbioportal is None
        client = hub.cbioportal
        assert hub._cbioportal is not None
        assert client is hub._cbioportal

    def test_lazy_loading_card(self, hub):
        """Test CARD client lazy loading."""
        assert hub._card is None
        client = hub.card
        assert hub._card is not None
        assert client is hub._card

    def test_lazy_loading_bvbrc(self, hub):
        """Test BV-BRC client lazy loading."""
        assert hub._bvbrc is None
        client = hub.bvbrc
        assert hub._bvbrc is not None
        assert client is hub._bvbrc

    def test_validate(self, hub):
        """Test configuration validation."""
        warnings = hub.validate()
        assert isinstance(warnings, list)


class TestDataHubConvenienceMethods:
    """Tests for DataHub convenience methods."""

    @pytest.fixture
    def hub(self):
        """Create DataHub instance."""
        from data_access import DataHub
        return DataHub()

    def test_search_hiv_resistance(self, hub, sample_hiv_sequence):
        """Test HIV resistance analysis convenience method."""
        with patch.object(hub.hivdb, "analyze_sequence") as mock_analyze:
            mock_analyze.return_value = {"data": {"viewer": {}}}

            result = hub.search_hiv_resistance(sample_hiv_sequence)
            assert result is not None
            mock_analyze.assert_called_once_with(sample_hiv_sequence)

    def test_get_hiv_sequences(self, hub):
        """Test HIV sequence retrieval convenience method."""
        with patch.object(hub.ncbi, "search_hiv_sequences") as mock_search:
            mock_search.return_value = {"ids": ["123", "456"], "count": 2}

            with patch.object(hub.ncbi, "get_sequence_summary") as mock_summary:
                mock_summary.return_value = pd.DataFrame([
                    {"id": "123", "title": "HIV-1 sequence"}
                ])

                result = hub.get_hiv_sequences(subtype="B", max_results=10)
                assert isinstance(result, pd.DataFrame)

    def test_get_amr_data(self, hub):
        """Test AMR data retrieval convenience method."""
        with patch.object(hub.bvbrc, "get_amr_phenotypes") as mock_amr:
            mock_amr.return_value = pd.DataFrame([
                {"antibiotic": "Rifampicin", "phenotype": "Resistant"}
            ])

            result = hub.get_amr_data()
            assert isinstance(result, pd.DataFrame)

    def test_get_cancer_mutations(self, hub):
        """Test cancer mutation retrieval convenience method."""
        with patch.object(hub.cbioportal, "get_studies") as mock_studies:
            mock_studies.return_value = pd.DataFrame([
                {"studyId": "brca_tcga"},
                {"studyId": "lung_tcga"}
            ])

            with patch.object(hub.cbioportal, "get_mutations_by_gene") as mock_mutations:
                mock_mutations.return_value = pd.DataFrame([
                    {"gene": "TP53", "mutation": "R175H"}
                ])

                result = hub.get_cancer_mutations("TP53", study_limit=2)
                assert isinstance(result, pd.DataFrame)

    def test_get_tb_genomes(self, hub):
        """Test TB genome retrieval convenience method."""
        with patch.object(hub.bvbrc, "get_tb_genomes") as mock_tb:
            mock_tb.return_value = pd.DataFrame([
                {"genome_id": "83332.12", "genome_name": "M. tuberculosis H37Rv"}
            ])

            result = hub.get_tb_genomes(limit=10)
            assert isinstance(result, pd.DataFrame)
            mock_tb.assert_called_once_with(limit=10)

    def test_get_syphilis_genomes(self, hub):
        """Test syphilis genome retrieval convenience method."""
        with patch.object(hub.bvbrc, "get_syphilis_genomes") as mock_syphilis:
            mock_syphilis.return_value = pd.DataFrame([
                {"genome_id": "160.1", "genome_name": "Treponema pallidum"}
            ])

            result = hub.get_syphilis_genomes(limit=10)
            assert isinstance(result, pd.DataFrame)

    def test_get_eskape_summary(self, hub):
        """Test ESKAPE pathogen summary convenience method."""
        with patch.object(hub.card, "get_eskape_pathogens") as mock_eskape:
            mock_eskape.return_value = pd.DataFrame([
                {"code": "S", "pathogen": "Staphylococcus aureus"}
            ])

            result = hub.get_eskape_summary()
            assert isinstance(result, pd.DataFrame)


class TestDataHubTestConnections:
    """Tests for DataHub connection testing."""

    @pytest.fixture
    def hub(self):
        """Create DataHub instance."""
        from data_access import DataHub
        return DataHub()

    def test_test_connections_mock(self, hub):
        """Test connection testing with mocks."""
        with patch.object(hub.hivdb, "get_algorithms") as mock_hivdb:
            mock_hivdb.return_value = []

            with patch.object(hub.cbioportal, "get_cancer_types") as mock_cbio:
                mock_cbio.return_value = pd.DataFrame()

                with patch.object(hub.card, "get_drug_classes") as mock_card:
                    mock_card.return_value = pd.DataFrame()

                    with patch.object(hub.bvbrc, "get_data_summary") as mock_bvbrc:
                        mock_bvbrc.return_value = pd.DataFrame()

                        result = hub.test_connections()
                        assert isinstance(result, pd.DataFrame)
                        assert "api" in result.columns
                        assert "status" in result.columns

    def test_test_connections_handles_errors(self, hub):
        """Test connection testing handles API errors gracefully."""
        with patch.object(hub.hivdb, "get_algorithms") as mock_hivdb:
            mock_hivdb.side_effect = Exception("Connection failed")

            with patch.object(hub.cbioportal, "get_cancer_types") as mock_cbio:
                mock_cbio.return_value = pd.DataFrame()

                with patch.object(hub.card, "get_drug_classes") as mock_card:
                    mock_card.return_value = pd.DataFrame()

                    with patch.object(hub.bvbrc, "get_data_summary") as mock_bvbrc:
                        mock_bvbrc.return_value = pd.DataFrame()

                        result = hub.test_connections()
                        # Should still return a DataFrame
                        assert isinstance(result, pd.DataFrame)
                        # HIVDB should show error
                        hivdb_row = result[result["api"] == "HIVDB"]
                        assert hivdb_row["status"].values[0] == "ERROR"


@pytest.mark.live_api
class TestDataHubLive:
    """Live API tests for DataHub."""

    @pytest.fixture
    def hub(self):
        """Create DataHub instance."""
        from data_access import DataHub
        return DataHub()

    def test_live_test_connections(self, hub):
        """Test live connection testing."""
        try:
            result = hub.test_connections()
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 4  # At least HIVDB, cBioPortal, CARD, BV-BRC
        except Exception as e:
            pytest.skip(f"Connection test failed: {e}")
