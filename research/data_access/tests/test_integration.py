"""
Unit tests for the integration module.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
import pandas as pd


class TestHIVAnalysisIntegration:
    """Tests for HIV analysis integration."""

    @pytest.fixture
    def integration(self):
        """Create integration instance with mocked client."""
        from data_access.integration import HIVAnalysisIntegration
        from data_access.clients import HIVDBClient

        mock_client = MagicMock(spec=HIVDBClient)
        return HIVAnalysisIntegration(hivdb_client=mock_client)

    def test_parse_mutation_simple(self, integration):
        """Test parsing simple mutation notation."""
        mut = integration.parse_mutation("M184V")
        assert mut.position == 184
        assert mut.wild_type == "M"
        assert mut.mutant == "V"
        assert mut.notation == "M184V"

    def test_parse_mutation_with_gene(self, integration):
        """Test parsing mutation with gene prefix."""
        mut = integration.parse_mutation("RT:K103N")
        assert mut.position == 103
        assert mut.wild_type == "K"
        assert mut.mutant == "N"
        assert mut.gene == "RT"

    def test_parse_mutation_invalid(self, integration):
        """Test parsing invalid mutation raises error."""
        with pytest.raises(ValueError):
            integration.parse_mutation("invalid")

    def test_analyze_sequence(self, integration, sample_hiv_sequence):
        """Test sequence analysis with mocked API."""
        integration.hivdb.analyze_sequence.return_value = {
            "data": {
                "viewer": {
                    "sequenceAnalysis": [{
                        "validationResults": [],
                        "drugResistance": [{
                            "drugClass": {"name": "NRTI"},
                            "gene": {"name": "RT"},
                            "drugScores": [
                                {
                                    "drug": {"name": "ABC"},
                                    "score": 10.0,
                                    "text": "Low-Level Resistance",
                                    "partialScores": [{
                                        "mutations": [{"text": "M184V"}]
                                    }]
                                }
                            ]
                        }],
                        "alignedGeneSequences": [{
                            "gene": {"name": "RT"},
                            "mutations": [{"text": "M184V"}]
                        }]
                    }]
                }
            }
        }

        result = integration.analyze_sequence(sample_hiv_sequence, "test_seq")

        assert result.sequence_id == "test_seq"
        assert result.gene == "RT"
        assert len(result.resistance_results) == 1
        assert result.resistance_results[0].drug == "ABC"

    def test_get_resistance_profile(self, integration, sample_hiv_sequence):
        """Test getting resistance profile as DataFrame."""
        integration.hivdb.analyze_sequence.return_value = {
            "data": {
                "viewer": {
                    "sequenceAnalysis": [{
                        "drugResistance": [{
                            "drugClass": {"name": "NRTI"},
                            "gene": {"name": "RT"},
                            "drugScores": [
                                {"drug": {"name": "ABC"}, "score": 0.0, "text": "Susceptible"},
                                {"drug": {"name": "AZT"}, "score": 15.0, "text": "Low-Level"},
                            ]
                        }]
                    }]
                }
            }
        }

        df = integration.get_resistance_profile(sample_hiv_sequence)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "drug" in df.columns
        assert "score" in df.columns


class TestSequenceProcessor:
    """Tests for sequence processor."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        from data_access.integration import SequenceProcessor
        return SequenceProcessor()

    def test_clean_sequence(self, processor):
        """Test sequence cleaning."""
        dirty = "atg TTt\nAGA  aca"
        cleaned = processor.clean_sequence(dirty)

        assert cleaned == "ATGTTTAGAACA"  # All characters preserved
        assert " " not in cleaned
        assert "\n" not in cleaned

    def test_clean_sequence_rna(self, processor):
        """Test RNA to DNA conversion."""
        rna = "AUGUUU"
        dna = processor.clean_sequence(rna)

        assert dna == "ATGTTT"

    def test_validate_sequence_valid(self, processor):
        """Test validation of valid sequence."""
        is_valid, messages = processor.validate_sequence("ATGTTTAGA")
        assert is_valid

    def test_validate_sequence_invalid_chars(self, processor):
        """Test validation catches invalid characters."""
        is_valid, messages = processor.validate_sequence("ATGXYZ")
        assert not is_valid
        assert any("Invalid" in m for m in messages)

    def test_extract_codons(self, processor):
        """Test codon extraction."""
        codons = list(processor.extract_codons("ATGTTTAGA"))

        assert len(codons) == 3
        assert codons[0].codon == "ATG"
        assert codons[0].amino_acid == "M"
        assert codons[1].codon == "TTT"
        assert codons[1].amino_acid == "F"

    def test_process_sequence(self, processor):
        """Test full sequence processing."""
        result = processor.process_sequence("ATGTTTAGA", "test", "pol")

        assert result.sequence_id == "test"
        assert result.gene == "pol"
        assert result.codon_count == 3
        assert result.translated_sequence == "MFR"

    def test_sequence_to_dataframe(self, processor):
        """Test conversion to DataFrame."""
        result = processor.process_sequence("ATGTTTAGA", "test")
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "codon" in df.columns
        assert "amino_acid" in df.columns

    def test_get_synonymous_codons(self, processor):
        """Test getting synonymous codons."""
        synonyms = processor.get_synonymous_codons("TTT")

        assert "TTT" in synonyms
        assert "TTC" in synonyms  # Both encode Phe

    def test_codon_statistics(self, processor):
        """Test codon statistics calculation."""
        # Sequence with repeated codons
        seq = "ATGATGATG"  # 3x Met
        stats = processor.calculate_codon_statistics(seq)

        assert isinstance(stats, pd.DataFrame)
        assert "codon" in stats.columns
        assert "count" in stats.columns


class TestResultsExtractor:
    """Tests for results extractor."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        from data_access.integration import ResultsExtractor
        return ResultsExtractor()

    def test_extract_hivdb_resistance(self, extractor):
        """Test HIVDB results extraction."""
        mock_result = {
            "data": {
                "viewer": {
                    "sequenceAnalysis": [{
                        "drugResistance": [{
                            "drugClass": {"name": "NRTI"},
                            "gene": {"name": "RT"},
                            "drugScores": [
                                {"drug": {"name": "ABC"}, "score": 10, "text": "Low"}
                            ]
                        }]
                    }]
                }
            }
        }

        result = extractor.extract_hivdb_resistance(mock_result, "test")

        assert result.source == "HIVDB"
        assert result.data_type == "drug_resistance"
        assert not result.is_empty
        assert result.record_count == 1

    def test_extract_ncbi_sequences(self, extractor):
        """Test NCBI sequence extraction."""
        sequences = [
            {"AccessionVersion": "KY123.1", "Title": "HIV-1", "Length": 1000},
            {"AccessionVersion": "KY456.1", "Title": "HIV-2", "Length": 500},
        ]

        result = extractor.extract_ncbi_sequences(sequences)

        assert result.source == "NCBI"
        assert result.data_type == "sequences"
        assert result.record_count == 2

    def test_comprehensive_results(self, extractor):
        """Test comprehensive results aggregation."""
        from data_access.integration import ComprehensiveResults

        results = ComprehensiveResults()

        # Add multiple results
        result1 = extractor.extract_ncbi_sequences([{"Id": "1"}])
        result2 = extractor.extract_card_resistance(pd.DataFrame([{"gene": "test"}]))

        results.add_result(result1)
        results.add_result(result2)

        assert results.summary["total_sources"] == 2
        assert results.summary["total_records"] == 2

    def test_combined_dataframe(self, extractor):
        """Test getting combined DataFrame."""
        from data_access.integration import ComprehensiveResults

        results = ComprehensiveResults()

        result1 = extractor.extract_ncbi_sequences([{"Id": "1", "Title": "Test"}])
        results.add_result(result1)

        combined = results.get_combined_dataframe()

        assert isinstance(combined, pd.DataFrame)
        assert "_source" in combined.columns

    def test_analysis_report(self, extractor):
        """Test report generation."""
        from data_access.integration import ComprehensiveResults

        results = ComprehensiveResults()
        result1 = extractor.extract_ncbi_sequences([{"Id": "1"}])
        results.add_result(result1)

        report = extractor.create_analysis_report(results)

        assert isinstance(report, str)
        assert "NCBI" in report
        assert "SUMMARY" in report
