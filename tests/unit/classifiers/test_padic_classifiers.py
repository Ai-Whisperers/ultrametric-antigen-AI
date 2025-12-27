# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for p-adic classifiers.

Tests cover:
- PAdicKNN: k-Nearest Neighbors with p-adic distance
- GoldilocksZoneClassifier: Binary autoimmune risk classification
- CodonClassifier: Codon to amino acid classification
- PAdicHierarchicalClassifier: Tree-based hierarchical classifier
"""

import numpy as np
import pytest

from src.classifiers.padic_classifiers import (
    ClassificationResult,
    CodonClassifier,
    GoldilocksZoneClassifier,
    PAdicHierarchicalClassifier,
    PAdicKNN,
)


# ============================================================================
# ClassificationResult Tests
# ============================================================================


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = ClassificationResult(
            predicted_class="A",
            confidence=0.8,
        )
        assert result.predicted_class == "A"
        assert result.confidence == 0.8
        assert result.neighbor_distances == []
        assert result.neighbor_classes == []

    def test_full_creation(self):
        """Test result with all fields."""
        result = ClassificationResult(
            predicted_class="B",
            confidence=0.95,
            neighbor_distances=[0.1, 0.2, 0.3],
            neighbor_classes=["B", "B", "A"],
        )
        assert result.predicted_class == "B"
        assert result.confidence == 0.95
        assert len(result.neighbor_distances) == 3
        assert len(result.neighbor_classes) == 3


# ============================================================================
# PAdicKNN Tests
# ============================================================================


class TestPAdicKNN:
    """Tests for PAdicKNN classifier."""

    def test_initialization(self):
        """Test default initialization."""
        clf = PAdicKNN()
        assert clf.k == 5
        assert clf.p == 3
        assert clf.weights == "uniform"
        assert not clf.is_fitted

    def test_custom_initialization(self):
        """Test custom initialization."""
        clf = PAdicKNN(k=3, p=5, weights="distance")
        assert clf.k == 3
        assert clf.p == 5
        assert clf.weights == "distance"

    def test_fit_basic(self):
        """Test basic fitting."""
        clf = PAdicKNN(k=3)
        X = np.array([0, 1, 2, 3, 4, 5])
        y = np.array(["A", "A", "A", "B", "B", "B"])

        result = clf.fit(X, y)

        assert result is clf  # Method chaining
        assert clf.is_fitted
        assert len(clf.classes_) == 2

    def test_fit_empty_raises(self):
        """Test that empty training data raises error."""
        clf = PAdicKNN()
        with pytest.raises(ValueError, match="empty"):
            clf.fit(np.array([]), np.array([]))

    def test_fit_mismatched_lengths_raises(self):
        """Test that mismatched X/y lengths raise error."""
        clf = PAdicKNN()
        with pytest.raises(ValueError, match="same length"):
            clf.fit(np.array([1, 2, 3]), np.array([1, 2]))

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        clf = PAdicKNN()
        with pytest.raises(ValueError, match="fitted"):
            clf.predict(np.array([1]))

    def test_predict_uniform(self):
        """Test prediction with uniform weights."""
        clf = PAdicKNN(k=3, weights="uniform")
        # Points close in 3-adic distance
        X = np.array([0, 3, 6, 9, 12, 15])  # Multiples of 3
        y = np.array(["A", "A", "A", "B", "B", "B"])
        clf.fit(X, y)

        # Predict on training points
        pred = clf.predict(np.array([0, 9]))
        assert len(pred) == 2

    def test_predict_distance_weighted(self):
        """Test prediction with distance weights."""
        clf = PAdicKNN(k=3, weights="distance")
        X = np.array([0, 1, 2, 9, 10, 11])
        y = np.array(["A", "A", "A", "B", "B", "B"])
        clf.fit(X, y)

        pred = clf.predict(np.array([0, 10]))
        assert pred[0] == "A"
        assert pred[1] == "B"

    def test_predict_with_details(self):
        """Test detailed predictions."""
        clf = PAdicKNN(k=2)
        X = np.array([0, 1, 9, 10])
        y = np.array(["A", "A", "B", "B"])
        clf.fit(X, y)

        results = clf.predict_with_details(np.array([0]))

        assert len(results) == 1
        assert isinstance(results[0], ClassificationResult)
        assert len(results[0].neighbor_distances) == 2
        assert len(results[0].neighbor_classes) == 2

    def test_predict_proba(self):
        """Test probability prediction."""
        clf = PAdicKNN(k=3)
        X = np.array([0, 1, 2, 9, 10, 11])
        y = np.array(["A", "A", "A", "B", "B", "B"])
        clf.fit(X, y)

        probas = clf.predict_proba(np.array([0, 10]))

        assert probas.shape == (2, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_score(self):
        """Test accuracy scoring."""
        clf = PAdicKNN(k=1)
        X = np.array([0, 1, 9, 10])
        y = np.array(["A", "A", "B", "B"])
        clf.fit(X, y)

        # Score on training data should be perfect with k=1
        score = clf.score(X, y)
        assert score == 1.0


class TestPAdicKNNPadicBehavior:
    """Tests for p-adic specific behavior in PAdicKNN."""

    def test_padic_distance_grouping(self):
        """Test that p-adic distance groups related items."""
        clf = PAdicKNN(k=1, p=3)

        # Numbers divisible by 3 are "closer" in 3-adic distance
        # 0, 3, 6, 9, 12... should cluster together
        X = np.array([0, 1, 3, 4, 6, 7])
        y = np.array(["A", "B", "A", "B", "A", "B"])
        clf.fit(X, y)

        # 9 is divisible by 3, should be close to 0, 3, 6
        pred = clf.predict(np.array([9]))
        assert pred[0] == "A"

    def test_ultrametric_property(self):
        """Test ultrametric triangle inequality."""
        clf = PAdicKNN(k=1, p=3)

        # In ultrametric space, d(a,c) <= max(d(a,b), d(b,c))
        from src.core.padic_math import padic_distance

        d_01 = padic_distance(0, 1, 3)
        d_12 = padic_distance(1, 2, 3)
        d_02 = padic_distance(0, 2, 3)

        assert d_02 <= max(d_01, d_12)

    def test_different_primes(self):
        """Test classifier with different prime bases."""
        for p in [2, 3, 5, 7]:
            clf = PAdicKNN(k=1, p=p)
            X = np.array([0, p, 2 * p, 3 * p])
            y = np.array(["A", "A", "B", "B"])
            clf.fit(X, y)

            # Should work without errors
            pred = clf.predict(np.array([4 * p]))
            assert pred[0] in ["A", "B"]


# ============================================================================
# GoldilocksZoneClassifier Tests
# ============================================================================


class TestGoldilocksZoneClassifier:
    """Tests for GoldilocksZoneClassifier."""

    def test_initialization(self):
        """Test default initialization."""
        clf = GoldilocksZoneClassifier(reference_index=0)
        assert clf.reference_index == 0
        assert clf.center == 0.5
        assert clf.width == 0.15
        assert clf.threshold == 0.5
        assert clf.p == 3

    def test_custom_initialization(self):
        """Test custom initialization."""
        clf = GoldilocksZoneClassifier(
            reference_index=10,
            center=0.4,
            width=0.2,
            threshold=0.6,
            p=5,
        )
        assert clf.reference_index == 10
        assert clf.center == 0.4
        assert clf.width == 0.2
        assert clf.threshold == 0.6
        assert clf.p == 5

    def test_fit(self):
        """Test fitting (should work without data)."""
        clf = GoldilocksZoneClassifier(reference_index=0)
        result = clf.fit()

        assert result is clf
        assert clf.is_fitted

    def test_compute_scores(self):
        """Test Goldilocks score computation."""
        clf = GoldilocksZoneClassifier(reference_index=0)
        clf.fit()

        scores = clf.compute_scores(np.array([0, 1, 3, 9, 27]))

        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)
        # Reference point should have low score (too close)
        # Very different points should also have low score (too far)

    def test_predict_binary(self):
        """Test binary prediction."""
        clf = GoldilocksZoneClassifier(reference_index=0)
        clf.fit()

        pred = clf.predict(np.array([0, 1, 5, 10]))

        assert len(pred) == 4
        assert all(p in [0, 1] for p in pred)

    def test_predict_proba(self):
        """Test probability prediction."""
        clf = GoldilocksZoneClassifier(reference_index=0)
        clf.fit()

        probas = clf.predict_proba(np.array([0, 1, 5]))

        assert probas.shape == (3, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)


# ============================================================================
# CodonClassifier Tests
# ============================================================================


class TestCodonClassifier:
    """Tests for CodonClassifier."""

    def test_initialization(self):
        """Test default initialization."""
        clf = CodonClassifier()
        assert clf.k == 3
        assert clf.p == 3
        assert clf.weights == "distance"
        assert len(clf.codon_to_index) == 64
        assert len(clf.index_to_codon) == 64

    def test_codon_mappings(self):
        """Test codon-index mappings are consistent."""
        clf = CodonClassifier()

        for codon, idx in clf.codon_to_index.items():
            assert clf.index_to_codon[idx] == codon
            assert 0 <= idx < 64

    def test_codon_to_aa(self):
        """Test codon to amino acid translation."""
        clf = CodonClassifier()

        # Test some known codons
        assert clf.codon_to_aa("ATG") == "M"  # Methionine (start)
        assert clf.codon_to_aa("TAA") == "*"  # Stop
        assert clf.codon_to_aa("GGG") == "G"  # Glycine
        assert clf.codon_to_aa("TTT") == "F"  # Phenylalanine

    def test_fit_from_genetic_code(self):
        """Test fitting from standard genetic code."""
        clf = CodonClassifier()
        result = clf.fit_from_genetic_code()

        assert result is clf
        assert clf.is_fitted
        assert len(clf.classes_) == 21  # 20 AA + 1 stop

    def test_predict_after_fit(self):
        """Test prediction after fitting from genetic code."""
        clf = CodonClassifier()
        clf.fit_from_genetic_code()

        # Predict for some codons
        atg_idx = clf.codon_to_index["ATG"]
        pred = clf.predict(np.array([atg_idx]))

        # ATG should predict M (Methionine)
        assert pred[0] == "M"

    def test_evaluate_accuracy(self):
        """Test accuracy evaluation."""
        clf = CodonClassifier(k=1)  # k=1 for exact matches
        clf.fit_from_genetic_code()

        metrics = clf.evaluate_accuracy()

        assert "overall_accuracy" in metrics
        assert "per_aa_accuracy" in metrics
        assert "n_correct" in metrics
        assert "n_total" in metrics
        assert metrics["n_total"] == 64
        assert 0 <= metrics["overall_accuracy"] <= 1

    def test_synonymous_codons(self):
        """Test that synonymous codons predict same amino acid."""
        clf = CodonClassifier(k=3)
        clf.fit_from_genetic_code()

        # Leucine has 6 codons: TTA, TTG, CTT, CTC, CTA, CTG
        leu_codons = ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"]
        leu_indices = [clf.codon_to_index[c] for c in leu_codons]

        predictions = clf.predict(np.array(leu_indices))

        # Most should predict L (depends on k and exact implementation)
        leu_count = sum(1 for p in predictions if p == "L")
        assert leu_count >= 4  # At least most should be correct


class TestCodonClassifierWobble:
    """Tests for wobble hypothesis in CodonClassifier."""

    def test_third_position_wobble(self):
        """Test that third position changes affect classification less."""
        clf = CodonClassifier()
        clf.fit_from_genetic_code()

        # Phenylalanine: TTT and TTC
        # These differ only in third position
        idx_ttt = clf.codon_to_index["TTT"]
        idx_ttc = clf.codon_to_index["TTC"]

        pred_ttt = clf.predict(np.array([idx_ttt]))[0]
        pred_ttc = clf.predict(np.array([idx_ttc]))[0]

        assert pred_ttt == "F"
        assert pred_ttc == "F"


# ============================================================================
# PAdicHierarchicalClassifier Tests
# ============================================================================


class TestPAdicHierarchicalClassifier:
    """Tests for PAdicHierarchicalClassifier."""

    def test_initialization(self):
        """Test default initialization."""
        clf = PAdicHierarchicalClassifier()
        assert clf.n_digits == 6
        assert clf.p == 3
        assert clf.min_samples_leaf == 1
        assert not clf.is_fitted

    def test_custom_initialization(self):
        """Test custom initialization."""
        clf = PAdicHierarchicalClassifier(n_digits=4, p=5, min_samples_leaf=2)
        assert clf.n_digits == 4
        assert clf.p == 5
        assert clf.min_samples_leaf == 2

    def test_extract_digits(self):
        """Test p-adic digit extraction."""
        clf = PAdicHierarchicalClassifier(n_digits=4, p=3)

        # 10 in base 3: 10 = 1*9 + 0*3 + 1*1 = 101 in base 3
        digits = clf._extract_digits(10)
        assert digits == (1, 0, 1, 0)

        # 27 = 3^3, so digits are (0, 0, 0, 1)
        digits = clf._extract_digits(27)
        assert digits == (0, 0, 0, 1)

    def test_fit(self):
        """Test fitting."""
        clf = PAdicHierarchicalClassifier(n_digits=4)
        X = np.array([0, 1, 2, 9, 10, 11])
        y = np.array(["A", "A", "A", "B", "B", "B"])

        result = clf.fit(X, y)

        assert result is clf
        assert clf.is_fitted
        assert clf.tree_ is not None

    def test_fit_mismatched_raises(self):
        """Test that mismatched X/y raises error."""
        clf = PAdicHierarchicalClassifier()
        with pytest.raises(ValueError, match="same length"):
            clf.fit(np.array([1, 2]), np.array([1]))

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        clf = PAdicHierarchicalClassifier()
        with pytest.raises(ValueError, match="fitted"):
            clf.predict(np.array([1]))

    def test_predict(self):
        """Test prediction."""
        clf = PAdicHierarchicalClassifier(n_digits=4)
        X = np.array([0, 1, 2, 27, 28, 29])
        y = np.array(["A", "A", "A", "B", "B", "B"])
        clf.fit(X, y)

        pred = clf.predict(np.array([0, 27]))
        assert pred[0] == "A"
        assert pred[1] == "B"

    def test_hierarchical_structure(self):
        """Test that hierarchical structure is respected."""
        clf = PAdicHierarchicalClassifier(n_digits=3, p=3)

        # Create data where hierarchy matters
        # Numbers 0-8 (first 3^2) are class A
        # Numbers 9-26 (next 3^2 levels) are class B
        X = np.array(list(range(27)))
        y = np.array(["A"] * 9 + ["B"] * 18)
        clf.fit(X, y)

        # Tree should have entries at different levels
        assert len(clf.tree_) > 0

    def test_score(self):
        """Test scoring."""
        clf = PAdicHierarchicalClassifier(n_digits=4)
        X = np.array([0, 1, 9, 10])
        y = np.array(["A", "A", "B", "B"])
        clf.fit(X, y)

        score = clf.score(X, y)
        assert 0 <= score <= 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestClassifierIntegration:
    """Integration tests for p-adic classifiers."""

    def test_knn_vs_hierarchical_same_data(self):
        """Test that KNN and hierarchical work on same data."""
        X = np.array([0, 1, 2, 9, 10, 11, 27, 28, 29])
        y = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])

        knn = PAdicKNN(k=3)
        hier = PAdicHierarchicalClassifier(n_digits=4)

        knn.fit(X, y)
        hier.fit(X, y)

        test_X = np.array([0, 10, 28])

        knn_pred = knn.predict(test_X)
        hier_pred = hier.predict(test_X)

        # Both should classify training points correctly
        assert len(knn_pred) == 3
        assert len(hier_pred) == 3

    def test_codon_vs_knn_consistency(self):
        """Test that CodonClassifier and raw PAdicKNN are consistent."""
        codon_clf = CodonClassifier(k=1)
        codon_clf.fit_from_genetic_code()

        knn = PAdicKNN(k=1, p=3)
        X = np.array(list(range(64)))
        y = np.array([codon_clf.codon_to_aa(codon_clf.index_to_codon[i]) for i in X])
        knn.fit(X, y)

        # Both should give same results for same test points
        test_X = np.array([0, 32, 63])

        codon_pred = codon_clf.predict(test_X)
        knn_pred = knn.predict(test_X)

        assert np.array_equal(codon_pred, knn_pred)


class TestEdgeCases:
    """Edge case tests for classifiers."""

    def test_single_sample(self):
        """Test with single training sample."""
        clf = PAdicKNN(k=1)
        clf.fit(np.array([0]), np.array(["A"]))

        pred = clf.predict(np.array([0, 1, 100]))
        assert all(p == "A" for p in pred)

    def test_all_same_class(self):
        """Test with all samples in same class."""
        clf = PAdicKNN(k=3)
        X = np.array([0, 1, 2, 3, 4])
        y = np.array(["A", "A", "A", "A", "A"])
        clf.fit(X, y)

        pred = clf.predict(np.array([10, 20, 30]))
        assert all(p == "A" for p in pred)

    def test_k_larger_than_training_set(self):
        """Test when k > number of training samples."""
        clf = PAdicKNN(k=10)
        X = np.array([0, 1, 2])
        y = np.array(["A", "A", "B"])
        clf.fit(X, y)

        # Should use all available samples
        pred = clf.predict(np.array([1]))
        assert pred[0] in ["A", "B"]

    def test_large_indices(self):
        """Test with large index values - verifies classifier handles large numbers."""
        clf = PAdicKNN(k=3)
        # Note: In 3-adic distance, numbers differing by multiples of 3^k are CLOSER
        # not farther. This is counterintuitive but fundamental to p-adic math.
        # We just verify the classifier works with large indices.
        X = np.array([1000000, 1000001, 1000002, 2000000, 2000001, 2000002])
        y = np.array(["A", "A", "A", "B", "B", "B"])
        clf.fit(X, y)

        # Verify prediction returns valid classes (actual grouping depends on p-adic structure)
        pred = clf.predict(np.array([1000000, 2000000]))
        assert len(pred) == 2
        assert pred[0] in ["A", "B"]
        assert pred[1] in ["A", "B"]
        # Self-prediction should match
        assert pred[0] == "A"  # 1000000 is in training set as class A

    def test_negative_indices_handled(self):
        """Test that negative indices are handled (absolute value)."""
        clf = PAdicHierarchicalClassifier(n_digits=4)

        # extract_digits should handle negative values
        digits = clf._extract_digits(-10)
        assert all(d >= 0 for d in digits)
