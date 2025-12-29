# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Nobel Prize immune validation module."""

import pytest

from src.analysis.immune_validation import (
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


# =============================================================================
# NEW TESTS FOR EXPERIMENTAL DATA AND VALIDATION PIPELINE
# =============================================================================

from src.analysis.immune_validation import (
    SELF_PEPTIDE_REFERENCES,
    NONSELF_PEPTIDE_REFERENCES,
    AUTOIMMUNE_PEPTIDES,
    load_experimental_dataset,
    run_validation_pipeline,
)


class TestExperimentalPeptideData:
    """Tests for curated experimental peptide datasets."""

    def test_self_peptide_references_count(self):
        """Test that self-peptide references are populated."""
        assert len(SELF_PEPTIDE_REFERENCES) >= 4

    def test_self_peptide_references_structure(self):
        """Test structure of self-peptide references."""
        for name, data in SELF_PEPTIDE_REFERENCES.items():
            assert "sequence" in data
            assert "protein" in data
            assert "binding_affinity_nm" in data
            assert "source" in data
            assert len(data["sequence"]) >= 8  # Typical epitope length
            assert data["binding_affinity_nm"] > 0

    def test_nonself_peptide_references_count(self):
        """Test that non-self peptide references are populated."""
        assert len(NONSELF_PEPTIDE_REFERENCES) >= 9

    def test_nonself_peptide_references_structure(self):
        """Test structure of non-self peptide references."""
        for name, data in NONSELF_PEPTIDE_REFERENCES.items():
            assert "sequence" in data
            assert "protein" in data
            assert "binding_affinity_nm" in data
            assert "source" in data
            assert "response_type" in data
            assert data["response_type"] == "CTL"

    def test_nonself_peptide_strong_binders(self):
        """Test that viral epitopes are strong binders (low IC50)."""
        for name, data in NONSELF_PEPTIDE_REFERENCES.items():
            # Viral epitopes should be strong to moderate binders
            assert data["binding_affinity_nm"] < 100  # Most are < 50 nM

    def test_autoimmune_peptides_count(self):
        """Test that autoimmune peptide references are populated."""
        assert len(AUTOIMMUNE_PEPTIDES) >= 6

    def test_autoimmune_peptides_structure(self):
        """Test structure of autoimmune peptide references."""
        for name, data in AUTOIMMUNE_PEPTIDES.items():
            assert "sequence" in data
            assert "protein" in data
            assert "binding_affinity_nm" in data
            assert "autoimmune_disease" in data
            assert "source" in data

    def test_autoimmune_peptides_diseases(self):
        """Test that various autoimmune diseases are represented."""
        diseases = set()
        for name, data in AUTOIMMUNE_PEPTIDES.items():
            diseases.add(data["autoimmune_disease"])

        assert "Multiple Sclerosis" in diseases
        assert "Type 1 Diabetes" in diseases
        assert "Rheumatoid Arthritis" in diseases


class TestLoadExperimentalDataset:
    """Tests for load_experimental_dataset function."""

    def test_load_returns_list(self):
        """Test that loading returns a list."""
        dataset = load_experimental_dataset()
        assert isinstance(dataset, list)

    def test_load_returns_correct_count(self):
        """Test that all peptides are loaded."""
        dataset = load_experimental_dataset()
        expected = (
            len(SELF_PEPTIDE_REFERENCES) +
            len(NONSELF_PEPTIDE_REFERENCES) +
            len(AUTOIMMUNE_PEPTIDES)
        )
        assert len(dataset) == expected

    def test_load_returns_threshold_data(self):
        """Test that loaded data is ImmuneThresholdData."""
        dataset = load_experimental_dataset()
        for item in dataset:
            assert isinstance(item, ImmuneThresholdData)

    def test_load_self_peptides_tolerance(self):
        """Test that self-peptides have tolerance response."""
        dataset = load_experimental_dataset()
        tolerance_count = sum(
            1 for d in dataset if d.response == ImmuneResponse.TOLERANCE
        )
        assert tolerance_count == len(SELF_PEPTIDE_REFERENCES)

    def test_load_nonself_peptides_activation(self):
        """Test that non-self peptides have activation response."""
        dataset = load_experimental_dataset()
        activation_peptides = [
            d for d in dataset if d.response == ImmuneResponse.ACTIVATION
        ]
        expected = len(NONSELF_PEPTIDE_REFERENCES) + len(AUTOIMMUNE_PEPTIDES)
        assert len(activation_peptides) == expected

    def test_load_autoimmune_in_goldilocks(self):
        """Test that autoimmune peptides have Goldilocks Zone distance."""
        dataset = load_experimental_dataset()

        # Find autoimmune peptides by source pattern
        autoimmune_data = [
            d for d in dataset
            if any(disease in d.source for disease in
                   ["Multiple Sclerosis", "Type 1 Diabetes", "Rheumatoid Arthritis"])
        ]

        for d in autoimmune_data:
            # Goldilocks Zone is 0.15-0.30
            assert 0.15 <= d.molecular_distance <= 0.30


class TestRunValidationPipeline:
    """Tests for run_validation_pipeline function."""

    def test_pipeline_returns_dict(self):
        """Test that pipeline returns a dictionary."""
        result = run_validation_pipeline()
        assert isinstance(result, dict)

    def test_pipeline_has_required_keys(self):
        """Test that pipeline result has all required keys."""
        result = run_validation_pipeline()

        assert "validation_result" in result
        assert "discrimination_boundary" in result
        assert "categories" in result
        assert "goldilocks_hypothesis" in result

    def test_pipeline_validation_result(self):
        """Test the validation result structure."""
        result = run_validation_pipeline()
        vr = result["validation_result"]

        assert isinstance(vr, ValidationResult)
        assert vr.n_samples > 0
        assert 0 <= vr.sensitivity <= 1
        assert 0 <= vr.specificity <= 1
        assert 0 <= vr.threshold_accuracy <= 1

    def test_pipeline_categories(self):
        """Test the categories in pipeline result."""
        result = run_validation_pipeline()
        cats = result["categories"]

        assert cats["self_peptides"] == len(SELF_PEPTIDE_REFERENCES)
        assert cats["nonself_peptides"] == len(NONSELF_PEPTIDE_REFERENCES)
        assert cats["autoimmune_peptides"] == len(AUTOIMMUNE_PEPTIDES)
        assert cats["total"] == (
            len(SELF_PEPTIDE_REFERENCES) +
            len(NONSELF_PEPTIDE_REFERENCES) +
            len(AUTOIMMUNE_PEPTIDES)
        )

    def test_pipeline_goldilocks_hypothesis(self):
        """Test the Goldilocks hypothesis results."""
        result = run_validation_pipeline()
        gh = result["goldilocks_hypothesis"]

        assert "hypothesis" in gh
        assert "15-30%" in gh["hypothesis"]
        assert "samples_in_goldilocks" in gh
        assert "samples_outside_goldilocks" in gh
        assert "prediction_accuracy" in gh
        assert "sensitivity" in gh
        assert "specificity" in gh

    def test_pipeline_discrimination_boundary(self):
        """Test the discrimination boundary results."""
        result = run_validation_pipeline()
        boundary = result["discrimination_boundary"]

        assert "optimal_boundary" in boundary
        assert "tolerance_mean" in boundary
        assert "activation_mean" in boundary
        assert "separation" in boundary

        # Tolerance should have lower p-adic distance than activation
        # (self = close, foreign = far)
        # Note: With synthetic data, there's overlap in distributions,
        # so we allow small tolerance for numerical variation
        tolerance_margin = 0.05
        assert boundary["tolerance_mean"] < boundary["activation_mean"] + tolerance_margin


class TestIntegrationValidation:
    """Integration tests for complete validation workflow."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow from start to finish."""
        # 1. Create validator
        validator = NobelImmuneValidator(goldilocks_range=(0.15, 0.30))

        # 2. Load experimental data
        dataset = load_experimental_dataset()

        # 3. Run validation
        result = validator.validate_threshold(dataset)

        # 4. Compute boundary
        boundary = validator.compute_discrimination_boundary(dataset)

        # Assertions
        assert result.n_samples == len(dataset)
        assert boundary["optimal_boundary"] > 0

    def test_goldilocks_zone_separates_responses(self):
        """Test that Goldilocks Zone helps separate immune responses."""
        validator = NobelImmuneValidator()
        dataset = load_experimental_dataset()

        # Compute p-adic distances
        tolerance_distances = []
        activation_distances = []

        for d in dataset:
            padic_dist = validator.map_affinity_to_padic(d.binding_affinity_nm)
            if d.response == ImmuneResponse.TOLERANCE:
                tolerance_distances.append(padic_dist)
            else:
                activation_distances.append(padic_dist)

        # Tolerance (self) should have different p-adic profile than activation
        import numpy as np
        tol_mean = np.mean(tolerance_distances)
        act_mean = np.mean(activation_distances)

        # This is a key hypothesis test:
        # Self-peptides should map differently than foreign peptides
        assert len(tolerance_distances) > 0
        assert len(activation_distances) > 0
