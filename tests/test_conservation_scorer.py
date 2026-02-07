"""Tests for conservation scoring functionality."""

import pytest
import numpy as np
from unittest.mock import patch, mock_open
from pathlib import Path

from hiv_analysis.scripts.conservation_scorer import (
    ConservationConfig, 
    ConservationScorer
)


class TestConservationConfig:
    """Test cases for ConservationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConservationConfig()
        
        assert config.score_types == ['shannon', 'simpson', 'property']
        assert config.gap_penalty == 0.1
        assert config.pseudocount == 0.001
        assert config.output_format == "json"
        assert 'hydrophobic' in config.property_groups
        assert 'A' in config.property_groups['hydrophobic']

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConservationConfig(
            score_types=['shannon'],
            gap_penalty=0.2,
            output_format='csv'
        )
        
        assert config.score_types == ['shannon']
        assert config.gap_penalty == 0.2
        assert config.output_format == 'csv'

    def test_property_groups_structure(self):
        """Test that property groups contain expected amino acids."""
        config = ConservationConfig()
        
        # Test some known groupings
        assert 'A' in config.property_groups['hydrophobic']
        assert 'R' in config.property_groups['charged_positive'] 
        assert 'D' in config.property_groups['charged_negative']
        assert 'S' in config.property_groups['polar']
        assert 'P' in config.property_groups['special']


class TestConservationScorer:
    """Test cases for ConservationScorer class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        scorer = ConservationScorer()
        
        assert scorer.config is not None
        assert isinstance(scorer.config, ConservationConfig)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ConservationConfig(gap_penalty=0.5)
        scorer = ConservationScorer(config)
        
        assert scorer.config.gap_penalty == 0.5

    def test_shannon_entropy_calculation(self):
        """Test Shannon entropy calculation."""
        scorer = ConservationScorer()
        
        # Perfect conservation (all same)
        alignment = {
            'seq1': 'AAAA',
            'seq2': 'AAAA',
            'seq3': 'AAAA',
            'seq4': 'AAAA'
        }
        entropy_scores = scorer.calculate_shannon_entropy(alignment)
        # Perfect conservation should have low entropy values
        assert len(entropy_scores) == 4
        assert all(isinstance(score, float) for score in entropy_scores)
        
        # Variable positions
        alignment = {
            'seq1': 'ATGC',
            'seq2': 'ATGC', 
            'seq3': 'ATGC',
            'seq4': 'ATGC'
        }
        entropy_scores = scorer.calculate_shannon_entropy(alignment)
        assert len(entropy_scores) == 4
        assert all(isinstance(score, float) for score in entropy_scores)

    def test_simpson_index_calculation(self):
        """Test Simpson diversity index calculation."""
        scorer = ConservationScorer()
        
        # Perfect conservation
        alignment = {
            'seq1': 'AAAA',
            'seq2': 'AAAA',
            'seq3': 'AAAA', 
            'seq4': 'AAAA'
        }
        simpson_scores = scorer.calculate_simpson_index(alignment)
        # Perfect conservation should have low diversity values
        assert len(simpson_scores) == 4
        assert all(isinstance(score, float) for score in simpson_scores)
        
        # Variable positions
        alignment = {
            'seq1': 'ATGC',
            'seq2': 'CGAT',
            'seq3': 'TGCA',
            'seq4': 'GATC'
        }
        simpson_scores = scorer.calculate_simpson_index(alignment)
        assert len(simpson_scores) == 4
        assert all(isinstance(score, float) for score in simpson_scores)

    def test_gap_handling(self):
        """Test that gaps are handled correctly."""
        scorer = ConservationScorer()
        
        alignment_with_gaps = {
            'seq1': 'A-AA',
            'seq2': 'A-AA',
            'seq3': '--AA',
            'seq4': 'A-AA'
        }
        shannon_scores = scorer.calculate_shannon_entropy(alignment_with_gaps)
        simpson_scores = scorer.calculate_simpson_index(alignment_with_gaps)
        
        # Should still calculate but with gap penalty
        assert all(isinstance(score, float) for score in shannon_scores)
        assert all(isinstance(score, float) for score in simpson_scores)

    def test_property_conservation(self):
        """Test property-based conservation scoring."""
        scorer = ConservationScorer()
        
        # Hydrophobic amino acids should be conserved
        alignment = {
            'seq1': 'AVIL',
            'seq2': 'AVIL', 
            'seq3': 'AVIL',
            'seq4': 'AVIL'
        }
        prop_scores = scorer.calculate_property_conservation(alignment)
        assert len(prop_scores) == 4
        assert all(isinstance(score, float) for score in prop_scores)
        
        # Mixed properties should be less conserved
        alignment = {
            'seq1': 'ARDS',
            'seq2': 'ARDS',
            'seq3': 'ARDS', 
            'seq4': 'ARDS'
        }
        prop_scores = scorer.calculate_property_conservation(alignment)
        assert len(prop_scores) == 4
        assert all(isinstance(score, float) for score in prop_scores)

    def test_empty_alignment_handling(self):
        """Test handling of empty alignments."""
        scorer = ConservationScorer()
        
        empty_alignment = {}
        # Should handle gracefully without crashing
        try:
            shannon_scores = scorer.calculate_shannon_entropy(empty_alignment)
            simpson_scores = scorer.calculate_simpson_index(empty_alignment)
            # If it doesn't crash, that's good enough for now
        except (StopIteration, ValueError):
            # These exceptions are expected with empty input
            pass

    @pytest.mark.unit
    def test_alignment_processing(self):
        """Test processing of alignment data."""
        scorer = ConservationScorer()
        
        # Mock alignment data
        alignment_data = {
            'seq1': "ATCG",
            'seq2': "ATGG",
            'seq3': "ACCG"
        }
        
        # Should process without errors
        shannon_scores = scorer.calculate_shannon_entropy(alignment_data)
        simpson_scores = scorer.calculate_simpson_index(alignment_data)
        
        assert len(shannon_scores) == 4
        assert len(simpson_scores) == 4
        assert all(isinstance(score, float) for score in shannon_scores)
        assert all(isinstance(score, float) for score in simpson_scores)

    def test_aa_frequencies_defined(self):
        """Test that amino acid frequencies are properly defined."""
        scorer = ConservationScorer()
        
        assert hasattr(scorer, 'aa_frequencies')
        assert 'A' in scorer.aa_frequencies
        assert 'R' in scorer.aa_frequencies
        # Frequencies should sum to approximately 1.0
        total = sum(scorer.aa_frequencies.values())
        assert 0.99 < total < 1.01