# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for extraterrestrial amino acid analyzer."""

import pytest

from src.analysis.extraterrestrial_aminoacids import (
    EARTH_AMINO_ACID_ABUNDANCE,
    MURCHISON_REFERENCE,
    NON_PROTEINOGENIC_AMINO_ACIDS,
    PROTEINOGENIC_AMINO_ACIDS,
    AminoAcidChirality,
    AminoAcidMeasurement,
    AminoAcidSource,
    AsteroidAminoAcidAnalyzer,
    CompatibilityResult,
    ExtraterrestrialSample,
)


class TestAminoAcidConstants:
    """Tests for amino acid constant data."""

    def test_proteinogenic_amino_acids(self):
        """Test that all 20 proteinogenic AAs are defined."""
        assert len(PROTEINOGENIC_AMINO_ACIDS) == 20
        assert "G" in PROTEINOGENIC_AMINO_ACIDS
        assert "A" in PROTEINOGENIC_AMINO_ACIDS
        assert "W" in PROTEINOGENIC_AMINO_ACIDS

    def test_non_proteinogenic_amino_acids(self):
        """Test non-proteinogenic AA definitions."""
        assert len(NON_PROTEINOGENIC_AMINO_ACIDS) > 0
        assert "AIB" in NON_PROTEINOGENIC_AMINO_ACIDS  # α-aminoisobutyric acid

    def test_earth_abundance_sums_to_one(self):
        """Test that Earth abundances sum approximately to 1."""
        total = sum(EARTH_AMINO_ACID_ABUNDANCE.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_murchison_reference_data(self):
        """Test Murchison meteorite reference data."""
        assert len(MURCHISON_REFERENCE) > 0
        # Glycine should be one of the most abundant
        assert "G" in MURCHISON_REFERENCE
        # AIB should be present (characteristic of meteorites)
        assert "AIB" in MURCHISON_REFERENCE


class TestAminoAcidMeasurement:
    """Tests for AminoAcidMeasurement dataclass."""

    def test_creation(self):
        """Test measurement creation."""
        m = AminoAcidMeasurement(
            amino_acid="G",
            concentration_ppb=2500,
            chirality=AminoAcidChirality.RACEMIC,
            uncertainty=250,
        )

        assert m.amino_acid == "G"
        assert m.concentration_ppb == 2500
        assert m.chirality == AminoAcidChirality.RACEMIC


class TestExtraterrestrialSample:
    """Tests for ExtraterrestrialSample dataclass."""

    def test_creation(self):
        """Test sample creation."""
        measurements = [
            AminoAcidMeasurement("G", 2500, AminoAcidChirality.RACEMIC, 250),
            AminoAcidMeasurement("A", 1800, AminoAcidChirality.RACEMIC, 180),
        ]

        sample = ExtraterrestrialSample(
            source=AminoAcidSource.MURCHISON_METEORITE,
            measurements=measurements,
            total_organic_carbon_ppm=20.0,
        )

        assert sample.source == AminoAcidSource.MURCHISON_METEORITE
        assert len(sample.measurements) == 2


class TestAsteroidAminoAcidAnalyzer:
    """Tests for AsteroidAminoAcidAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = AsteroidAminoAcidAnalyzer()
        assert analyzer.p == 3

    def test_padic_distance_same(self):
        """Test p-adic distance between same amino acids."""
        analyzer = AsteroidAminoAcidAnalyzer()
        dist = analyzer.compute_padic_distance("A", "A")
        assert dist == 0.0

    def test_padic_distance_different(self):
        """Test p-adic distance between different amino acids."""
        analyzer = AsteroidAminoAcidAnalyzer()
        dist = analyzer.compute_padic_distance("A", "C")
        assert dist > 0

    def test_normalize_concentrations(self):
        """Test concentration normalization."""
        analyzer = AsteroidAminoAcidAnalyzer()

        measurements = [
            AminoAcidMeasurement("G", 500, AminoAcidChirality.RACEMIC, 50),
            AminoAcidMeasurement("A", 500, AminoAcidChirality.RACEMIC, 50),
        ]

        freqs = analyzer.normalize_concentrations(measurements)

        assert freqs["G"] == pytest.approx(0.5)
        assert freqs["A"] == pytest.approx(0.5)

    def test_normalize_concentrations_empty(self):
        """Test normalization with empty list."""
        analyzer = AsteroidAminoAcidAnalyzer()
        freqs = analyzer.normalize_concentrations([])
        assert freqs == {}

    def test_compute_earth_compatibility(self):
        """Test Earth compatibility computation."""
        analyzer = AsteroidAminoAcidAnalyzer()

        # Use Earth abundances - should be highly compatible
        result = analyzer.compute_earth_compatibility(EARTH_AMINO_ACID_ABUNDANCE)
        assert result > 0.9

    def test_compute_earth_compatibility_empty(self):
        """Test compatibility with empty data."""
        analyzer = AsteroidAminoAcidAnalyzer()
        result = analyzer.compute_earth_compatibility({})
        assert result == 0.0

    def test_compute_padic_optimality(self):
        """Test p-adic optimality score."""
        analyzer = AsteroidAminoAcidAnalyzer()

        result = analyzer.compute_padic_optimality(EARTH_AMINO_ACID_ABUNDANCE)
        assert 0 <= result <= 1

    def test_compute_chirality_ratio_l_form(self):
        """Test chirality ratio with L-form only."""
        analyzer = AsteroidAminoAcidAnalyzer()

        measurements = [
            AminoAcidMeasurement("G", 1000, AminoAcidChirality.L_FORM, 100),
            AminoAcidMeasurement("A", 1000, AminoAcidChirality.L_FORM, 100),
        ]

        ratio = analyzer.compute_chirality_ratio(measurements)
        assert ratio == pytest.approx(1.0)

    def test_compute_chirality_ratio_racemic(self):
        """Test chirality ratio with racemic mixture."""
        analyzer = AsteroidAminoAcidAnalyzer()

        measurements = [
            AminoAcidMeasurement("G", 1000, AminoAcidChirality.RACEMIC, 100),
        ]

        ratio = analyzer.compute_chirality_ratio(measurements)
        assert ratio == pytest.approx(0.5)

    def test_compute_chirality_ratio_mixed(self):
        """Test chirality ratio with mixed chirality."""
        analyzer = AsteroidAminoAcidAnalyzer()

        measurements = [
            AminoAcidMeasurement("G", 1000, AminoAcidChirality.L_FORM, 100),
            AminoAcidMeasurement("A", 1000, AminoAcidChirality.D_FORM, 100),
        ]

        ratio = analyzer.compute_chirality_ratio(measurements)
        assert ratio == pytest.approx(0.5)

    def test_create_murchison_reference_sample(self):
        """Test Murchison reference sample creation."""
        analyzer = AsteroidAminoAcidAnalyzer()
        sample = analyzer.create_murchison_reference_sample()

        assert sample.source == AminoAcidSource.MURCHISON_METEORITE
        assert len(sample.measurements) > 0
        assert sample.total_organic_carbon_ppm > 0

    def test_analyze_sample(self):
        """Test sample analysis."""
        analyzer = AsteroidAminoAcidAnalyzer()
        sample = analyzer.create_murchison_reference_sample()

        result = analyzer.analyze_sample(sample)

        assert isinstance(result, CompatibilityResult)
        assert result.source == AminoAcidSource.MURCHISON_METEORITE
        assert 0 <= result.earth_compatibility <= 1
        assert 0 <= result.padic_optimality_score <= 1
        assert 0 <= result.chirality_ratio <= 1

    def test_analyze_sample_findings(self):
        """Test that analysis generates findings."""
        analyzer = AsteroidAminoAcidAnalyzer()
        sample = analyzer.create_murchison_reference_sample()

        result = analyzer.analyze_sample(sample)

        # Murchison should have some key findings
        assert isinstance(result.key_findings, list)

    def test_compare_sources(self):
        """Test comparing multiple sources."""
        analyzer = AsteroidAminoAcidAnalyzer()

        # Create two samples
        murchison = analyzer.create_murchison_reference_sample()

        # Simple Earth-like sample
        earth_measurements = [
            AminoAcidMeasurement(aa, int(freq * 10000), AminoAcidChirality.L_FORM, 100)
            for aa, freq in EARTH_AMINO_ACID_ABUNDANCE.items()
        ]
        earth_sample = ExtraterrestrialSample(
            source=AminoAcidSource.EARTH_BIOLOGICAL,
            measurements=earth_measurements,
            total_organic_carbon_ppm=1000.0,
        )

        results = analyzer.compare_sources([murchison, earth_sample])

        assert "murchison" in results
        assert "earth_bio" in results

        # Earth sample should have higher Earth compatibility
        assert results["earth_bio"].earth_compatibility > results["murchison"].earth_compatibility

    def test_calculate_prebiotic_padic_score(self):
        """Test prebiotic p-adic score calculation."""
        analyzer = AsteroidAminoAcidAnalyzer()

        scores = analyzer.calculate_prebiotic_padic_score(EARTH_AMINO_ACID_ABUNDANCE)

        assert "overall_score" in scores
        assert "clustering" in scores
        assert "uniformity" in scores
        assert 0 <= scores["overall_score"] <= 1

    def test_calculate_prebiotic_padic_score_empty(self):
        """Test prebiotic score with empty data."""
        analyzer = AsteroidAminoAcidAnalyzer()

        scores = analyzer.calculate_prebiotic_padic_score({})
        assert scores["overall_score"] == 0.0


class TestAminoAcidSource:
    """Tests for AminoAcidSource enum."""

    def test_source_values(self):
        """Test source enum values."""
        assert AminoAcidSource.ASTEROID_BENNU.value == "bennu"
        assert AminoAcidSource.MURCHISON_METEORITE.value == "murchison"
        assert AminoAcidSource.EARTH_BIOLOGICAL.value == "earth_bio"


class TestAminoAcidChirality:
    """Tests for AminoAcidChirality enum."""

    def test_chirality_values(self):
        """Test chirality enum values."""
        assert AminoAcidChirality.L_FORM.value == "L"
        assert AminoAcidChirality.D_FORM.value == "D"
        assert AminoAcidChirality.RACEMIC.value == "racemic"
