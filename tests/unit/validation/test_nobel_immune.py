# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Nobel Prize immune validation module."""

import pytest

from src.validation.nobel_immune import (
    REFERENCE_THRESHOLDS,
    GoldilocksZoneValidator,
    ImmuneResponse,
    ImmuneThresholdData,
    MHCClass,
    NobelImmuneValidator,
    ValidationResult,
)


class TestGoldilocksZoneValidator:
    """Tests for GoldilocksZoneValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = GoldilocksZoneValidator()
        assert validator.goldilocks_min == 0.15
        assert validator.goldilocks_max == 0.30

    def test_initialization_custom_range(self):
        """Test validator with custom range."""
        validator = GoldilocksZoneValidator(goldilocks_min=0.1, goldilocks_max=0.4)
        assert validator.goldilocks_min == 0.1
        assert validator.goldilocks_max == 0.4

    def test_is_in_goldilocks_zone(self):
        """Test Goldilocks Zone membership."""
        validator = GoldilocksZoneValidator()

        assert validator.is_in_goldilocks_zone(0.20) is True
        assert validator.is_in_goldilocks_zone(0.15) is True
        assert validator.is_in_goldilocks_zone(0.30) is True
        assert validator.is_in_goldilocks_zone(0.10) is False
        assert validator.is_in_goldilocks_zone(0.35) is False

    def test_compute_zone_score_center(self):
        """Test zone score at center."""
        validator = GoldilocksZoneValidator()

        center = (0.15 + 0.30) / 2  # 0.225
        score = validator.compute_zone_score(center)
        assert score == pytest.approx(1.0)

    def test_compute_zone_score_outside(self):
        """Test zone score outside zone."""
        validator = GoldilocksZoneValidator()

        score = validator.compute_zone_score(0.05)
        assert score == 0.0

    def test_predict_response_tolerance(self):
        """Test response prediction for low distance."""
        validator = GoldilocksZoneValidator()

        response = validator.predict_response(0.05)
        assert response == ImmuneResponse.TOLERANCE

    def test_predict_response_activation(self):
        """Test response prediction for high distance."""
        validator = GoldilocksZoneValidator()

        response = validator.predict_response(0.50)
        assert response == ImmuneResponse.ACTIVATION

    def test_predict_response_goldilocks(self):
        """Test response prediction within Goldilocks Zone."""
        validator = GoldilocksZoneValidator()

        response = validator.predict_response(0.22)
        assert response == ImmuneResponse.ACTIVATION


class TestNobelImmuneValidator:
    """Tests for NobelImmuneValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = NobelImmuneValidator()
        assert validator.p == 3

    def test_initialization_custom_range(self):
        """Test validator with custom Goldilocks range."""
        validator = NobelImmuneValidator(goldilocks_range=(0.10, 0.40))
        assert validator.goldilocks_validator.goldilocks_min == 0.10
        assert validator.goldilocks_validator.goldilocks_max == 0.40

    def test_compute_peptide_distance_same(self):
        """Test distance between identical peptides."""
        validator = NobelImmuneValidator()
        dist = validator.compute_peptide_distance("SIINFEKL", "SIINFEKL")
        assert dist == pytest.approx(0.0)

    def test_compute_peptide_distance_different(self):
        """Test distance between different peptides."""
        validator = NobelImmuneValidator()
        dist = validator.compute_peptide_distance("SIINFEKL", "XXXXXXXX")
        assert dist > 0

    def test_compute_peptide_distance_similar(self):
        """Test distance between similar peptides."""
        validator = NobelImmuneValidator()

        # One position difference
        dist = validator.compute_peptide_distance("SIINFEKL", "SIINFEKV")
        assert 0 < dist < 0.5

    def test_map_affinity_to_padic(self):
        """Test affinity to p-adic mapping."""
        validator = NobelImmuneValidator()

        # Strong binder (low IC50) -> low distance
        strong = validator.map_affinity_to_padic(10)

        # Weak binder (high IC50) -> high distance
        weak = validator.map_affinity_to_padic(10000)

        assert strong < weak
        assert 0 <= strong <= 1
        assert 0 <= weak <= 1

    def test_validate_threshold_empty(self):
        """Test validation with no data."""
        validator = NobelImmuneValidator()
        result = validator.validate_threshold([])

        assert result.n_samples == 0
        assert result.correlation == 0.0

    def test_validate_threshold_with_data(self):
        """Test validation with sample data."""
        validator = NobelImmuneValidator()

        # Generate synthetic data
        data = validator.generate_synthetic_data(n_samples=50)
        result = validator.validate_threshold(data)

        assert isinstance(result, ValidationResult)
        assert result.n_samples == 50
        assert 0 <= result.sensitivity <= 1
        assert 0 <= result.specificity <= 1

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        validator = NobelImmuneValidator()
        data = validator.generate_synthetic_data(n_samples=20)

        assert len(data) == 20
        for d in data:
            assert isinstance(d, ImmuneThresholdData)
            assert len(d.peptide_sequence) == 9
            assert d.binding_affinity_nm > 0

    def test_generate_synthetic_data_reproducible(self):
        """Test that synthetic data is reproducible."""
        validator = NobelImmuneValidator()

        data1 = validator.generate_synthetic_data(n_samples=10)
        data2 = validator.generate_synthetic_data(n_samples=10)

        # Same seed should give same data
        assert data1[0].peptide_sequence == data2[0].peptide_sequence

    def test_compute_discrimination_boundary(self):
        """Test discrimination boundary computation."""
        validator = NobelImmuneValidator()
        data = validator.generate_synthetic_data(n_samples=100)

        boundary = validator.compute_discrimination_boundary(data)

        assert "optimal_boundary" in boundary
        assert "tolerance_mean" in boundary
        assert "activation_mean" in boundary
        assert "separation" in boundary


class TestImmuneThresholdData:
    """Tests for ImmuneThresholdData dataclass."""

    def test_creation(self):
        """Test data point creation."""
        data = ImmuneThresholdData(
            peptide_sequence="SIINFEKL",
            binding_affinity_nm=50.0,
            tcr_affinity_um=5.0,
            mhc_class=MHCClass.CLASS_I,
            response=ImmuneResponse.ACTIVATION,
            molecular_distance=0.25,
        )

        assert data.peptide_sequence == "SIINFEKL"
        assert data.binding_affinity_nm == 50.0
        assert data.mhc_class == MHCClass.CLASS_I


class TestReferenceThresholds:
    """Tests for reference threshold constants."""

    def test_mhc_class_i_thresholds(self):
        """Test MHC Class I threshold values."""
        thresholds = REFERENCE_THRESHOLDS["mhc_class_i_binding"]

        assert thresholds["strong_binder"] < thresholds["moderate_binder"]
        assert thresholds["moderate_binder"] < thresholds["weak_binder"]
        assert thresholds["weak_binder"] < thresholds["non_binder"]

    def test_mhc_class_ii_thresholds(self):
        """Test MHC Class II threshold values."""
        thresholds = REFERENCE_THRESHOLDS["mhc_class_ii_binding"]

        assert thresholds["strong_binder"] < thresholds["moderate_binder"]

    def test_tcr_thresholds(self):
        """Test TCR recognition thresholds."""
        thresholds = REFERENCE_THRESHOLDS["tcr_recognition"]

        assert thresholds["high_affinity"] < thresholds["moderate_affinity"]
        assert thresholds["moderate_affinity"] < thresholds["low_affinity"]


class TestImmuneResponse:
    """Tests for ImmuneResponse enum."""

    def test_response_values(self):
        """Test response enum values."""
        assert ImmuneResponse.TOLERANCE.value == "tolerance"
        assert ImmuneResponse.ACTIVATION.value == "activation"
        assert ImmuneResponse.ANERGY.value == "anergy"


class TestMHCClass:
    """Tests for MHCClass enum."""

    def test_mhc_class_values(self):
        """Test MHC class enum values."""
        assert MHCClass.CLASS_I.value == "class_I"
        assert MHCClass.CLASS_II.value == "class_II"
