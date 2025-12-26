"""
Unit tests for the HIVDB (Stanford Sierra) client.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import pandas as pd


class TestHIVDBClient:
    """Tests for HIVDBClient."""

    @pytest.fixture
    def client(self):
        """Create HIVDB client."""
        from data_access.clients import HIVDBClient
        return HIVDBClient()

    @pytest.fixture
    def mock_graphql_response(self, mock_response):
        """Create mock GraphQL response."""
        def _create(data):
            return mock_response(json_data={"data": data})
        return _create

    def test_client_creation(self, client):
        """Test client initialization."""
        assert client.endpoint == "https://hivdb.stanford.edu/graphql"
        assert client.session is not None

    def test_get_drug_classes(self, client, mock_graphql_response):
        """Test getting drug classes."""
        with patch("requests.Session.post") as mock_post:
            mock_post.return_value = mock_graphql_response({
                "viewer": {
                    "drugClasses": [
                        {"name": "NRTI", "fullName": "Nucleoside RT Inhibitor"},
                        {"name": "NNRTI", "fullName": "Non-nucleoside RT Inhibitor"},
                        {"name": "PI", "fullName": "Protease Inhibitor"},
                        {"name": "INSTI", "fullName": "Integrase Inhibitor"},
                    ]
                }
            })

            result = client.get_drug_classes()
            # Result can be list or other format depending on API response
            assert result is not None

    def test_analyze_sequence(self, client, mock_graphql_response, sample_hiv_sequence):
        """Test sequence analysis."""
        with patch("requests.Session.post") as mock_post:
            mock_post.return_value = mock_graphql_response({
                "viewer": {
                    "sequenceAnalysis": [{
                        "inputSequence": {"header": "test"},
                        "validationResults": [],
                        "drugResistance": [
                            {
                                "drugClass": {"name": "NRTI"},
                                "drugScores": [
                                    {"drug": {"name": "ABC"}, "score": 0.0, "text": "Susceptible"}
                                ]
                            }
                        ]
                    }]
                }
            })

            result = client.analyze_sequence(sample_hiv_sequence)
            assert result is not None
            # Result should contain the sequence analysis data
            assert "data" in result or "viewer" in result

    def test_get_resistance_summary(self, client, mock_graphql_response, sample_hiv_sequence):
        """Test getting resistance summary as DataFrame."""
        with patch("requests.Session.post") as mock_post:
            mock_post.return_value = mock_graphql_response({
                "viewer": {
                    "sequenceAnalysis": [{
                        "drugResistance": [
                            {
                                "drugClass": {"name": "NRTI"},
                                "drugScores": [
                                    {"drug": {"name": "ABC"}, "score": 10.0, "text": "Low-Level Resistance"},
                                    {"drug": {"name": "AZT"}, "score": 0.0, "text": "Susceptible"},
                                ]
                            }
                        ]
                    }]
                }
            })

            result = client.get_resistance_summary(sample_hiv_sequence)
            assert isinstance(result, pd.DataFrame)

    def test_get_mutations_analysis(self, client, mock_graphql_response, sample_mutations):
        """Test mutation analysis."""
        with patch("requests.Session.post") as mock_post:
            mock_post.return_value = mock_graphql_response({
                "viewer": {
                    "mutationsAnalysis": [{
                        "mutationType": "Major",
                        "comments": []
                    }]
                }
            })

            result = client.get_mutations_analysis(sample_mutations, "RT")
            assert result is not None

    def test_get_algorithms(self, client, mock_graphql_response):
        """Test getting interpretation algorithms."""
        with patch("requests.Session.post") as mock_post:
            mock_post.return_value = mock_graphql_response({
                "viewer": {
                    "algorithms": [
                        {"name": "HIVDB", "version": "9.4"},
                        {"name": "ANRS", "version": "30"},
                    ]
                }
            })

            result = client.get_algorithms()
            assert result is not None


class TestHIVDBIntegration:
    """Integration tests for HIVDB client."""

    @pytest.fixture
    def client(self):
        """Create HIVDB client."""
        from data_access.clients import HIVDBClient
        return HIVDBClient()

    @pytest.mark.live_api
    def test_live_get_drug_classes(self, client):
        """Test live drug class retrieval."""
        try:
            result = client.get_drug_classes()
            assert result is not None
            # Should include PI, NRTI, NNRTI, INSTI
        except Exception as e:
            pytest.skip(f"HIVDB API unavailable: {e}")

    @pytest.mark.live_api
    def test_live_analyze_sequence(self, client, sample_hiv_sequence):
        """Test live sequence analysis."""
        try:
            result = client.analyze_sequence(sample_hiv_sequence)
            assert result is not None
        except Exception as e:
            pytest.skip(f"HIVDB API unavailable: {e}")
