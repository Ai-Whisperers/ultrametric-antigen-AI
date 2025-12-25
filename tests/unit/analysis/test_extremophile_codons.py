# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for extremophile codon usage analyzer."""

import pytest

from src.analysis.extremophile_codons import (
    CODON_TABLE,
    REFERENCE_ORGANISMS,
    CodonUsageResult,
    ExtremophileCategory,
    ExtremophileCodonAnalyzer,
    OrganismProfile,
)


class TestCodonTable:
    """Tests for codon table constants."""

    def test_codon_table_size(self):
        """Test that codon table has all 64 codons."""
        assert len(CODON_TABLE) == 64

    def test_codon_table_values(self):
        """Test specific codon translations."""
        assert CODON_TABLE["ATG"] == "M"  # Start codon / Met
        assert CODON_TABLE["TAA"] == "*"  # Stop codon
        assert CODON_TABLE["TAG"] == "*"  # Stop codon
        assert CODON_TABLE["TGA"] == "*"  # Stop codon
        assert CODON_TABLE["TGG"] == "W"  # Trp (single codon)

    def test_all_amino_acids_present(self):
        """Test that all 20 amino acids + stop are represented."""
        aas = set(CODON_TABLE.values())
        assert len(aas) == 21  # 20 amino acids + stop codon (*)


class TestReferenceOrganisms:
    """Tests for reference organism profiles."""

    def test_reference_organisms_exist(self):
        """Test that reference organisms are defined."""
        assert len(REFERENCE_ORGANISMS) > 0
        assert "ecoli_k12" in REFERENCE_ORGANISMS

    def test_ecoli_reference(self):
        """Test E. coli K-12 reference profile."""
        ecoli = REFERENCE_ORGANISMS["ecoli_k12"]
        assert ecoli.category == ExtremophileCategory.MESOPHILE
        assert ecoli.optimal_temperature == pytest.approx(37.0)
        assert ecoli.gc_content == pytest.approx(0.508)

    def test_pyrococcus_reference(self):
        """Test Pyrococcus furiosus hyperthermophile profile."""
        pf = REFERENCE_ORGANISMS["pyrococcus_furiosus"]
        assert pf.category == ExtremophileCategory.HYPERTHERMOPHILE
        assert pf.optimal_temperature == pytest.approx(100.0)

    def test_deinococcus_reference(self):
        """Test Deinococcus radiodurans radioresistant profile."""
        dr = REFERENCE_ORGANISMS["deinococcus_radiodurans"]
        assert dr.category == ExtremophileCategory.RADIORESISTANT


class TestExtremophileCodonAnalyzer:
    """Tests for ExtremophileCodonAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ExtremophileCodonAnalyzer()
        assert analyzer.p == 3
        assert len(analyzer.codon_to_aa) == 64

    def test_count_codons_simple(self):
        """Test codon counting on simple sequence."""
        analyzer = ExtremophileCodonAnalyzer()
        # ATG (Met) + TGG (Trp) + TAA (Stop)
        sequence = "ATGTGGTAA"
        counts = analyzer.count_codons(sequence)

        assert counts["ATG"] == 1
        assert counts["TGG"] == 1
        assert counts["TAA"] == 1

    def test_count_codons_incomplete_codon(self):
        """Test that incomplete codons are ignored."""
        analyzer = ExtremophileCodonAnalyzer()
        # 11 bases - last 2 should be ignored
        sequence = "ATGTGGTAACT"
        counts = analyzer.count_codons(sequence)

        assert sum(counts.values()) == 3  # Only 3 complete codons

    def test_count_codons_lowercase(self):
        """Test codon counting handles lowercase."""
        analyzer = ExtremophileCodonAnalyzer()
        sequence = "atgtggtaa"
        counts = analyzer.count_codons(sequence)

        assert counts["ATG"] == 1

    def test_count_codons_rna(self):
        """Test codon counting handles RNA (U instead of T)."""
        analyzer = ExtremophileCodonAnalyzer()
        sequence = "AUGUGGUAA"  # RNA
        counts = analyzer.count_codons(sequence)

        assert counts["ATG"] == 1
        assert counts["TGG"] == 1

    def test_compute_gc_content(self):
        """Test GC content calculation."""
        analyzer = ExtremophileCodonAnalyzer()

        # All GC
        gc, gc3 = analyzer.compute_gc_content("GCCGCCGCC")
        assert gc == pytest.approx(1.0)

        # All AT
        gc, gc3 = analyzer.compute_gc_content("ATTATTATT")
        assert gc == pytest.approx(0.0)

        # 50% GC
        gc, gc3 = analyzer.compute_gc_content("ATGCATGCATGC")
        assert gc == pytest.approx(0.5)

    def test_compute_gc3_content(self):
        """Test GC3 (third position) calculation."""
        analyzer = ExtremophileCodonAnalyzer()

        # Third positions: C, C, C
        gc, gc3 = analyzer.compute_gc_content("ATCATCATC")
        assert gc3 == pytest.approx(1.0)

        # Third positions: T, T, T
        gc, gc3 = analyzer.compute_gc_content("ATTATTAT")
        assert gc3 == pytest.approx(0.0)

    def test_compute_rscu_no_bias(self):
        """Test RSCU with equal codon usage."""
        analyzer = ExtremophileCodonAnalyzer()

        # Equal usage of Phe codons (TTT and TTC)
        counts = {"TTT": 10, "TTC": 10}
        rscu = analyzer.compute_rscu(counts)

        assert rscu["TTT"] == pytest.approx(1.0)
        assert rscu["TTC"] == pytest.approx(1.0)

    def test_compute_rscu_with_bias(self):
        """Test RSCU with biased codon usage."""
        analyzer = ExtremophileCodonAnalyzer()

        # Biased usage of Phe codons
        counts = {"TTT": 20, "TTC": 0}
        rscu = analyzer.compute_rscu(counts)

        assert rscu["TTT"] == pytest.approx(2.0)  # 2x expected
        assert rscu["TTC"] == pytest.approx(0.0)

    def test_compute_enc_range(self):
        """Test ENC is in valid range."""
        analyzer = ExtremophileCodonAnalyzer()

        # Create a sample sequence with some bias
        sequence = "ATG" * 10 + "TTT" * 20 + "TTC" * 5 + "GCT" * 15 + "GCC" * 10
        counts = analyzer.count_codons(sequence)
        enc = analyzer.compute_enc(counts)

        assert 20.0 <= enc <= 61.0

    def test_predict_temperature_high_gc(self):
        """Test temperature prediction for high GC sequence."""
        analyzer = ExtremophileCodonAnalyzer()

        # High GC should predict high temperature
        rscu = {c: 1.0 for c in CODON_TABLE.keys()}
        temp = analyzer.predict_temperature(0.70, 0.75, rscu)

        assert temp > 60  # Should predict thermophilic range

    def test_predict_temperature_low_gc(self):
        """Test temperature prediction for low GC sequence."""
        analyzer = ExtremophileCodonAnalyzer()

        rscu = {c: 1.0 for c in CODON_TABLE.keys()}
        temp = analyzer.predict_temperature(0.35, 0.30, rscu)

        assert temp < 40  # Should predict mesophilic or lower

    def test_analyze_codon_bias_result_type(self):
        """Test analyze_codon_bias returns correct type."""
        analyzer = ExtremophileCodonAnalyzer()

        # Create a simple sequence
        sequence = "ATGTTTTCAGCTGCCGGCTAA"
        result = analyzer.analyze_codon_bias(sequence)

        assert isinstance(result, CodonUsageResult)
        assert isinstance(result.codon_frequencies, dict)
        assert isinstance(result.rscu_values, dict)
        assert 0.0 <= result.gc_content <= 1.0
        assert result.enc >= 20.0

    def test_analyze_codon_bias_with_category(self):
        """Test analyze with explicit category."""
        analyzer = ExtremophileCodonAnalyzer()

        sequence = "ATGTTTTCAGCTGCCGGCTAA"
        result = analyzer.analyze_codon_bias(sequence, category=ExtremophileCategory.THERMOPHILE)

        assert result.category == ExtremophileCategory.THERMOPHILE

    def test_infer_category_hyperthermophile(self):
        """Test category inference for high temp prediction."""
        analyzer = ExtremophileCodonAnalyzer()

        category = analyzer._infer_category(gc_content=0.70, predicted_temp=95.0)
        assert category == ExtremophileCategory.HYPERTHERMOPHILE

    def test_infer_category_thermophile(self):
        """Test category inference for moderate high temp."""
        analyzer = ExtremophileCodonAnalyzer()

        category = analyzer._infer_category(gc_content=0.60, predicted_temp=60.0)
        assert category == ExtremophileCategory.THERMOPHILE

    def test_infer_category_psychrophile(self):
        """Test category inference for low temp."""
        analyzer = ExtremophileCodonAnalyzer()

        category = analyzer._infer_category(gc_content=0.40, predicted_temp=5.0)
        assert category == ExtremophileCategory.PSYCHROPHILE

    def test_infer_category_mesophile(self):
        """Test category inference for normal temp."""
        analyzer = ExtremophileCodonAnalyzer()

        category = analyzer._infer_category(gc_content=0.50, predicted_temp=37.0)
        assert category == ExtremophileCategory.MESOPHILE

    def test_compare_to_mesophile(self):
        """Test comparison to E. coli baseline."""
        analyzer = ExtremophileCodonAnalyzer()

        sequence = "ATGTTTTCAGCTGCCGGCTAA"
        result = analyzer.analyze_codon_bias(sequence)
        comparison = analyzer.compare_to_mesophile(result)

        assert "mean_deviation" in comparison
        assert "gc_deviation" in comparison

    def test_analyze_organism_known(self):
        """Test analysis with known organism profile."""
        analyzer = ExtremophileCodonAnalyzer()

        # Simple test sequence - would need real genome for accurate results
        sequence = "ATG" + "GCC" * 100 + "TAA"  # High GC
        analysis = analyzer.analyze_organism(sequence, "thermus_thermophilus")

        assert "profile" in analysis
        assert "analysis" in analysis
        assert analysis["profile"].category == ExtremophileCategory.THERMOPHILE

    def test_analyze_organism_unknown_raises(self):
        """Test that unknown organism raises error."""
        analyzer = ExtremophileCodonAnalyzer()

        with pytest.raises(ValueError, match="Unknown organism"):
            analyzer.analyze_organism("ATGTAA", "unknown_organism")


class TestExtremophileCategory:
    """Tests for ExtremophileCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert ExtremophileCategory.THERMOPHILE.value == "thermophile"
        assert ExtremophileCategory.PSYCHROPHILE.value == "psychrophile"
        assert ExtremophileCategory.MESOPHILE.value == "mesophile"

    def test_all_categories(self):
        """Test all expected categories exist."""
        categories = [c.value for c in ExtremophileCategory]
        assert "thermophile" in categories
        assert "hyperthermophile" in categories
        assert "psychrophile" in categories
        assert "radioresistant" in categories
        assert "halophile" in categories
        assert "acidophile" in categories
        assert "alkaliphile" in categories
        assert "barophile" in categories
        assert "mesophile" in categories


class TestPadicCalculations:
    """Tests for p-adic calculations in the analyzer."""

    def test_padic_valuation(self):
        """Test p-adic valuation computation."""
        analyzer = ExtremophileCodonAnalyzer(p=3)

        # v_3(9) = 2 since 9 = 3^2
        assert analyzer._compute_padic_valuation(9) == 2

        # v_3(27) = 3
        assert analyzer._compute_padic_valuation(27) == 3

        # v_3(0) should be large (infinity)
        assert analyzer._compute_padic_valuation(0) >= 100

    def test_codon_to_ternary(self):
        """Test codon to ternary conversion."""
        analyzer = ExtremophileCodonAnalyzer()

        # Same nucleotide type should give same ternary value
        val1 = analyzer._codon_to_ternary("TTT")
        val2 = analyzer._codon_to_ternary("CCC")
        val3 = analyzer._codon_to_ternary("AAA")

        # Different codons should (generally) give different values
        assert val1 != val2
        assert val2 != val3

    def test_padic_distances_computed(self):
        """Test that p-adic distances are computed."""
        analyzer = ExtremophileCodonAnalyzer()

        rscu = {"TTT": 2.0, "TTC": 0.0, "ATG": 1.0}
        distances = analyzer.compute_padic_distances(rscu)

        # Neutral codon (RSCU=1) should have distance 0
        assert distances["ATG"] == 0.0

        # Biased codons should have non-zero distances
        assert distances["TTT"] > 0
        assert distances["TTC"] > 0
