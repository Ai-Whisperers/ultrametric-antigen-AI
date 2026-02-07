"""Tests for sequence utility functions."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch


class TestSequenceUtils:
    """Test utility functions for sequence handling."""

    def test_fasta_content_parsing(self, sample_fasta_content):
        """Test parsing of FASTA format content."""
        lines = sample_fasta_content.strip().split('\n')
        sequences = {}
        current_id = None
        
        for line in lines:
            if line.startswith('>'):
                current_id = line[1:]
                sequences[current_id] = ''
            else:
                sequences[current_id] += line
                
        assert len(sequences) == 3
        assert 'seq1' in sequences
        assert sequences['seq1'] == 'ATCGATCGATCG'

    def test_sequence_validation(self):
        """Test basic sequence validation."""
        valid_dna = "ATCGATCGATCG"
        valid_protein = "ARNDCQEGHILKMFPSTWYV"
        invalid_sequence = "ATCGXYZ"
        
        # DNA validation
        assert all(c in 'ATCG' for c in valid_dna)
        
        # Protein validation
        standard_aa = set('ARNDCQEGHILKMFPSTWYV')
        assert all(c in standard_aa for c in valid_protein)
        
        # Invalid sequence
        assert not all(c in 'ATCG' for c in invalid_sequence)

    def test_sequence_length_consistency(self, sample_sequence_data):
        """Test that aligned sequences have consistent lengths."""
        sequences = list(sample_sequence_data.values())
        
        # All sequences should have the same length in an alignment
        lengths = [len(seq) for seq in sequences]
        assert all(length == lengths[0] for length in lengths)

    @pytest.mark.unit
    def test_gap_handling_in_sequences(self):
        """Test proper handling of gaps in sequence alignment."""
        aligned_sequences = {
            'seq1': 'ATC-GATC',
            'seq2': 'AT-GGATC', 
            'seq3': 'ATCGGA-C'
        }
        
        # Count gaps per position
        alignment_length = len(list(aligned_sequences.values())[0])
        gap_counts = []
        
        for pos in range(alignment_length):
            gaps = sum(1 for seq in aligned_sequences.values() if seq[pos] == '-')
            gap_counts.append(gaps)
        
        assert len(gap_counts) == alignment_length
        assert all(count >= 0 for count in gap_counts)

    def test_file_format_detection(self):
        """Test detection of sequence file formats."""
        fasta_content = ">seq1\nATCG\n>seq2\nGCTA\n"
        clustal_content = "CLUSTAL W (1.81) multiple sequence alignment\n\nseq1    ATCG\nseq2    GCTA\n"
        
        # Simple format detection
        assert fasta_content.startswith('>')
        assert 'CLUSTAL' in clustal_content.upper()

    @pytest.mark.integration
    def test_temporary_file_handling(self, temp_dir, sample_fasta_content):
        """Test working with temporary files."""
        # Write test data to temporary file
        temp_file = temp_dir / "test_sequences.fasta"
        temp_file.write_text(sample_fasta_content)
        
        assert temp_file.exists()
        assert temp_file.read_text() == sample_fasta_content
        
        # Test reading back
        content = temp_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) == 6  # 3 headers + 3 sequences

    def test_sequence_statistics(self, sample_sequence_data):
        """Test basic sequence statistics calculation."""
        sequences = list(sample_sequence_data.values())
        
        stats = {}
        for seq_id, sequence in sample_sequence_data.items():
            stats[seq_id] = {
                'length': len(sequence),
                'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence),
                'at_content': (sequence.count('A') + sequence.count('T')) / len(sequence)
            }
        
        # Verify statistics make sense
        for seq_stats in stats.values():
            assert seq_stats['length'] > 0
            assert 0 <= seq_stats['gc_content'] <= 1
            assert 0 <= seq_stats['at_content'] <= 1
            # GC + AT should sum to 1 for DNA sequences
            assert abs((seq_stats['gc_content'] + seq_stats['at_content']) - 1.0) < 0.01

    @pytest.mark.slow
    def test_large_sequence_handling(self):
        """Test handling of larger sequence datasets."""
        # Create a moderately large synthetic dataset
        large_sequence = 'ATCG' * 1000  # 4000 bp sequence
        large_dataset = {f'seq_{i}': large_sequence for i in range(100)}
        
        # Should handle without memory issues
        assert len(large_dataset) == 100
        assert all(len(seq) == 4000 for seq in large_dataset.values())

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Empty sequence
        empty_seq = ""
        assert len(empty_seq) == 0
        
        # Single nucleotide
        single_nt = "A"
        assert len(single_nt) == 1
        assert single_nt in 'ATCG'
        
        # Very long sequence ID
        long_id = "very_long_sequence_identifier_" + "x" * 100
        assert len(long_id) > 100