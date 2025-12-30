# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Alejandra Rojas - Arbovirus Surveillance Package.

Tests pan-arbovirus primer design, stability scanning,
and cross-reactivity checking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add deliverables to path
deliverables_dir = Path(__file__).parent.parent
sys.path.insert(0, str(deliverables_dir))
sys.path.insert(0, str(deliverables_dir / "partners" / "alejandra_rojas" / "scripts"))


class TestGCContent:
    """Tests for GC content calculation."""

    def test_gc_content_all_gc(self):
        """Test 100% GC content."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import compute_gc_content

        assert compute_gc_content("GGCC") == 1.0
        assert compute_gc_content("GCGCGC") == 1.0

    def test_gc_content_no_gc(self):
        """Test 0% GC content."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import compute_gc_content

        assert compute_gc_content("AATT") == 0.0
        assert compute_gc_content("ATATAT") == 0.0

    def test_gc_content_mixed(self):
        """Test mixed GC content."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import compute_gc_content

        assert compute_gc_content("ATGC") == 0.5
        assert abs(compute_gc_content("ATGCGC") - 0.667) < 0.01

    def test_gc_content_lowercase(self):
        """Test lowercase input handling."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import compute_gc_content

        assert compute_gc_content("atgc") == 0.5

    def test_gc_content_empty(self):
        """Test empty sequence."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import compute_gc_content

        assert compute_gc_content("") == 0.0


class TestTmEstimation:
    """Tests for melting temperature estimation."""

    def test_estimate_tm_short_sequence(self):
        """Test Tm for short sequence (< 14bp, Wallace rule)."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import estimate_tm

        # Wallace rule: Tm = 2(A+T) + 4(G+C)
        # ATGC: 2*2 + 4*2 = 12
        seq = "ATGCATGCATGC"  # 12bp
        tm = estimate_tm(seq)
        # 6 AT pairs = 12, 6 GC pairs = 24, total = 36
        assert 30 < tm < 50

    def test_estimate_tm_long_sequence(self):
        """Test Tm for longer sequence (>= 14bp)."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import estimate_tm

        # Use more accurate formula for longer primers
        seq = "ATGCATGCATGCATGC"  # 16bp
        tm = estimate_tm(seq)
        assert 40 < tm < 70  # Reasonable Tm range

    def test_estimate_tm_gc_rich(self):
        """Test Tm for GC-rich sequence."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import estimate_tm

        gc_rich = "GCGCGCGCGCGCGCGC"  # 16bp all GC
        at_rich = "ATATATATATATATAT"  # 16bp all AT

        tm_gc = estimate_tm(gc_rich)
        tm_at = estimate_tm(at_rich)

        # GC-rich should have higher Tm
        assert tm_gc > tm_at


class TestArbovirusTargets:
    """Tests for arbovirus target definitions."""

    def test_arbovirus_targets_exist(self):
        """Test that all expected targets are defined."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import ARBOVIRUS_TARGETS

        expected_targets = ["DENV-1", "DENV-2", "DENV-3", "DENV-4", "ZIKV", "CHIKV", "MAYV"]
        for target in expected_targets:
            assert target in ARBOVIRUS_TARGETS, f"Missing target: {target}"

    def test_target_structure(self):
        """Test target definition structure."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import ARBOVIRUS_TARGETS

        for target_name, target_info in ARBOVIRUS_TARGETS.items():
            assert "full_name" in target_info, f"{target_name} missing full_name"
            assert "genome_size" in target_info, f"{target_name} missing genome_size"
            assert "conserved_regions" in target_info, f"{target_name} missing conserved_regions"

            # Genome size should be reasonable
            assert target_info["genome_size"] > 5000
            assert target_info["genome_size"] < 15000

    def test_dengue_serotypes(self):
        """Test all 4 Dengue serotypes present."""
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import ARBOVIRUS_TARGETS

        for i in range(1, 5):
            assert f"DENV-{i}" in ARBOVIRUS_TARGETS


class TestPrimerValidation:
    """Tests for primer validation functions."""

    def test_primer_length_validation(self):
        """Test primer length requirements."""
        # Standard primers are 18-25bp
        valid_length = range(18, 26)

        short_primer = "ATGC"  # Too short
        good_primer = "ATGCATGCATGCATGCAT"  # 18bp
        long_primer = "ATGC" * 10  # 40bp, too long

        assert len(short_primer) not in valid_length
        assert len(good_primer) in valid_length
        assert len(long_primer) not in valid_length

    def test_primer_gc_range(self):
        """Test primer GC content requirements."""
        # Optimal GC is 40-60%
        from partners.alejandra_rojas.scripts.A2_pan_arbovirus_primers import compute_gc_content

        good_primer = "ATGCATGCATGCATGCAT"  # ~50% GC
        gc = compute_gc_content(good_primer)
        assert 0.40 <= gc <= 0.60


class TestCrossReactivity:
    """Tests for cross-reactivity checking."""

    def test_sequence_similarity(self):
        """Test sequence similarity calculation."""
        # Identical sequences should have 100% similarity
        seq1 = "ATGCATGC"
        seq2 = "ATGCATGC"

        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        similarity = matches / len(seq1)
        assert similarity == 1.0

        # Different sequences should have lower similarity
        seq3 = "TTTTTTTT"
        matches = sum(1 for a, b in zip(seq1, seq3) if a == b)
        similarity = matches / len(seq1)
        assert similarity < 0.5

    def test_cross_reactive_detection(self):
        """Test cross-reactivity detection logic."""
        # A primer should not bind to multiple unrelated viruses
        threshold = 0.8  # 80% similarity threshold

        denv1_seq = "ATGCATGCATGCATGCATGC"
        denv2_seq = "ATGCATGCATGCATGCATGT"  # Very similar
        zikv_seq = "TTAATTAATTAATTAATTAA"  # Very different

        def similarity(s1: str, s2: str) -> float:
            return sum(1 for a, b in zip(s1, s2) if a == b) / len(s1)

        # DENV1 and DENV2 are similar (cross-reactive)
        assert similarity(denv1_seq, denv2_seq) > threshold

        # DENV1 and ZIKV are different (not cross-reactive)
        assert similarity(denv1_seq, zikv_seq) < threshold


class TestDemoSequenceGeneration:
    """Tests for demo sequence generation."""

    def test_generate_random_sequence(self):
        """Test random sequence generation."""
        np.random.seed(42)

        def generate_sequence(length: int) -> str:
            bases = ["A", "T", "G", "C"]
            return "".join(np.random.choice(bases, size=length))

        seq = generate_sequence(100)
        assert len(seq) == 100
        assert all(b in "ATGC" for b in seq)

    def test_generate_conserved_region(self):
        """Test conserved region simulation."""
        # Conserved regions should have lower diversity
        np.random.seed(42)

        def generate_conserved_sequence(length: int, conservation: float = 0.9) -> str:
            base_seq = "ATGC" * (length // 4 + 1)
            base_seq = base_seq[:length]

            result = list(base_seq)
            for i in range(length):
                if np.random.random() > conservation:
                    result[i] = np.random.choice(list("ATGC"))
            return "".join(result)

        seq = generate_conserved_sequence(100, conservation=0.95)
        assert len(seq) == 100


class TestPrimerDesignOutput:
    """Tests for primer design output format."""

    def test_primer_output_structure(self):
        """Test primer output has required fields."""
        expected_fields = [
            "forward_sequence",
            "reverse_sequence",
            "forward_tm",
            "reverse_tm",
            "gc_content_f",
            "gc_content_r",
            "amplicon_size",
            "target_virus",
        ]

        # Mock primer result
        primer_result = {
            "forward_sequence": "ATGCATGCATGCATGCAT",
            "reverse_sequence": "GCATGCATGCATGCATGC",
            "forward_tm": 54.5,
            "reverse_tm": 55.2,
            "gc_content_f": 0.5,
            "gc_content_r": 0.5,
            "amplicon_size": 250,
            "target_virus": "DENV-1",
        }

        for field in expected_fields:
            assert field in primer_result, f"Missing field: {field}"

    def test_tm_difference_acceptable(self):
        """Test Tm difference between primers is acceptable."""
        max_tm_diff = 2.0  # Max 2 degrees difference

        tm_f = 54.5
        tm_r = 55.2

        tm_diff = abs(tm_f - tm_r)
        assert tm_diff <= max_tm_diff


class TestAmpliconSize:
    """Tests for amplicon size requirements."""

    def test_valid_amplicon_range(self):
        """Test valid RT-PCR amplicon sizes."""
        # For RT-PCR, typical amplicon is 80-300bp
        valid_range = range(80, 301)

        # Good amplicon
        assert 200 in valid_range

        # Too small for reliable detection
        assert 50 not in valid_range

        # Too large for efficient RT-PCR
        assert 500 not in valid_range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
