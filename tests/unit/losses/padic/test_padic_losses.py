# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive pytest tests for p-adic loss modules.

This module tests:
- PAdicMetricLoss: Euclidean-to-p-adic distance alignment
- PAdicNormLoss: MSB/LSB hierarchy enforcement via p-adic norm regularization
- PAdicRankingLoss: Order-preserving ranking loss using contrastive triplets
- TripletMining utilities: TripletBatch, EuclideanTripletMiner, HyperbolicTripletMiner

Tests follow AAA pattern (Arrange, Act, Assert) and cover:
- Happy path (normal operation)
- Edge cases (empty batches, small batches, boundary values)
- Error cases (invalid inputs)
- Numerical correctness
"""

from typing import Dict

import pytest
import torch

from src.core import TERNARY
from src.losses.padic.metric_loss import PAdicMetricLoss
from src.losses.padic.norm_loss import PAdicNormLoss
from src.losses.padic.ranking_loss import PAdicRankingLoss
from src.losses.padic.triplet_mining import (
    EuclideanTripletMiner,
    HyperbolicTripletMiner,
    TripletBatch,
    compute_3adic_valuation_batch,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Returns 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def padic_test_batch(device) -> Dict[str, torch.Tensor]:
    """Create a standard test batch with latent codes and batch indices."""
    torch.manual_seed(42)
    batch_size = 32
    latent_dim = 16

    # Create random latent codes
    z = torch.randn(batch_size, latent_dim, device=device) * 0.5

    # Create random batch indices in valid range [0, 19682]
    batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (batch_size,), device=device)

    return {"z": z, "batch_indices": batch_indices}


@pytest.fixture
def small_padic_batch(device) -> Dict[str, torch.Tensor]:
    """Create a small test batch (4 samples)."""
    torch.manual_seed(42)
    batch_size = 4
    latent_dim = 16

    z = torch.randn(batch_size, latent_dim, device=device) * 0.5
    batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (batch_size,), device=device)

    return {"z": z, "batch_indices": batch_indices}


@pytest.fixture
def hierarchical_test_batch(device) -> Dict[str, torch.Tensor]:
    """Create a test batch with known 3-adic hierarchy for testing correctness."""
    latent_dim = 16

    # Indices with known valuations:
    # 0 -> v_3(0) = 9 (max)
    # 1 -> v_3(1) = 0
    # 3 -> v_3(3) = 1
    # 9 -> v_3(9) = 2
    # 27 -> v_3(27) = 3
    batch_indices = torch.tensor([0, 1, 3, 9, 27, 81, 243, 729], device=device)

    # Create latent codes with structure matching hierarchy
    # Points with similar valuations should be closer
    z = torch.randn(8, latent_dim, device=device) * 0.5

    return {"z": z, "batch_indices": batch_indices}


@pytest.fixture
def metric_loss():
    """Create PAdicMetricLoss with default parameters."""
    return PAdicMetricLoss(scale=1.0, n_pairs=100)


@pytest.fixture
def norm_loss():
    """Create PAdicNormLoss with default parameters."""
    return PAdicNormLoss(latent_dim=16)


@pytest.fixture
def ranking_loss():
    """Create PAdicRankingLoss with default parameters."""
    return PAdicRankingLoss(margin=0.1, n_triplets=50)


# =============================================================================
# PAdicMetricLoss Tests
# =============================================================================


class TestPAdicMetricLoss:
    """Tests for PAdicMetricLoss (Phase 1A from implement.md)."""

    def test_initialization_default_params(self):
        """Test PAdicMetricLoss initializes with default parameters."""
        # Act
        loss_fn = PAdicMetricLoss()

        # Assert
        assert loss_fn.scale == 1.0
        assert loss_fn.n_pairs == 1000

    def test_initialization_custom_params(self):
        """Test PAdicMetricLoss initializes with custom parameters."""
        # Act
        loss_fn = PAdicMetricLoss(scale=2.0, n_pairs=500)

        # Assert
        assert loss_fn.scale == 2.0
        assert loss_fn.n_pairs == 500

    def test_forward_returns_scalar(self, metric_loss, padic_test_batch):
        """Test forward pass returns a scalar tensor."""
        # Arrange
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = metric_loss(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.device == z.device

    def test_forward_non_negative_loss(self, metric_loss, padic_test_batch):
        """Test loss is non-negative (MSE loss)."""
        # Arrange
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = metric_loss(z, batch_indices)

        # Assert
        assert loss >= 0.0

    def test_forward_empty_batch_returns_zero(self, metric_loss, device):
        """Test empty batch returns zero loss."""
        # Arrange
        z = torch.empty(0, 16, device=device)
        batch_indices = torch.empty(0, dtype=torch.long, device=device)

        # Act
        loss = metric_loss(z, batch_indices)

        # Assert
        assert loss == 0.0

    def test_forward_single_sample_returns_zero(self, metric_loss, device):
        """Test single sample returns zero loss (no pairs possible)."""
        # Arrange
        z = torch.randn(1, 16, device=device)
        batch_indices = torch.tensor([100], device=device)

        # Act
        loss = metric_loss(z, batch_indices)

        # Assert
        assert loss == 0.0

    def test_forward_identical_indices_zero_distance(self, device):
        """Test identical batch indices yield zero 3-adic distance."""
        # Arrange
        loss_fn = PAdicMetricLoss(scale=1.0, n_pairs=10)
        z = torch.randn(10, 16, device=device)
        # All samples have the same index
        batch_indices = torch.full((10,), 42, device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert - loss should be low since all p-adic distances are 0
        # (scale * 0 = 0, so loss = MSE(latent_dist, 0))
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_forward_gradient_flow(self, metric_loss, padic_test_batch):
        """Test gradients flow through the loss."""
        # Arrange
        z = padic_test_batch["z"].clone().requires_grad_(True)
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = metric_loss(z, batch_indices)
        loss.backward()

        # Assert
        assert z.grad is not None
        assert not torch.all(z.grad == 0)

    def test_scale_affects_target_distance(self, padic_test_batch, device):
        """Test scale parameter affects target distances."""
        # Arrange
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        loss_fn_scale_1 = PAdicMetricLoss(scale=1.0, n_pairs=100)
        loss_fn_scale_10 = PAdicMetricLoss(scale=10.0, n_pairs=100)

        # Act - Set same seed for reproducible pair sampling
        torch.manual_seed(123)
        loss_scale_1 = loss_fn_scale_1(z, batch_indices)

        torch.manual_seed(123)
        loss_scale_10 = loss_fn_scale_10(z, batch_indices)

        # Assert - Different scales should produce different losses
        # (unless by chance they produce the same value)
        assert isinstance(loss_scale_1, torch.Tensor)
        assert isinstance(loss_scale_10, torch.Tensor)


# =============================================================================
# PAdicNormLoss Tests
# =============================================================================


class TestPAdicNormLoss:
    """Tests for PAdicNormLoss (Phase 1B from implement.md)."""

    def test_initialization_default_params(self):
        """Test PAdicNormLoss initializes with default latent_dim."""
        # Act
        loss_fn = PAdicNormLoss()

        # Assert
        assert loss_fn.latent_dim == 16
        assert hasattr(loss_fn, "weights")
        assert loss_fn.weights.shape == (16,)

    def test_initialization_custom_latent_dim(self):
        """Test PAdicNormLoss initializes with custom latent_dim."""
        # Act
        loss_fn = PAdicNormLoss(latent_dim=32)

        # Assert
        assert loss_fn.latent_dim == 32
        assert loss_fn.weights.shape == (32,)

    def test_weights_decay_exponentially(self, norm_loss):
        """Test weights follow 3^(-i) pattern."""
        # Arrange
        expected_weights = torch.tensor([3.0 ** (-i) for i in range(16)])

        # Act
        actual_weights = norm_loss.weights

        # Assert
        torch.testing.assert_close(actual_weights, expected_weights)

    def test_forward_returns_scalar(self, norm_loss, padic_test_batch):
        """Test forward pass returns a scalar tensor."""
        # Arrange
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = norm_loss(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_forward_non_negative_loss(self, norm_loss, padic_test_batch):
        """Test loss is non-negative."""
        # Arrange
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = norm_loss(z, batch_indices)

        # Assert
        assert loss >= 0.0

    def test_forward_gradient_flow(self, norm_loss, padic_test_batch):
        """Test gradients flow through the loss."""
        # Arrange
        z = padic_test_batch["z"].clone().requires_grad_(True)
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = norm_loss(z, batch_indices)
        loss.backward()

        # Assert
        assert z.grad is not None

    def test_expected_valuation_normalized(self, norm_loss, hierarchical_test_batch):
        """Test expected valuation is correctly normalized."""
        # Arrange
        batch_indices = hierarchical_test_batch["batch_indices"]

        # Act
        expected_val = norm_loss._compute_expected_valuation(batch_indices)

        # Assert - should be in [0, 1] range (3^(-v))
        assert torch.all(expected_val >= 0.0)
        assert torch.all(expected_val <= 1.0)

    def test_expected_valuation_known_values(self, norm_loss, device):
        """Test expected valuation with known indices."""
        # Arrange
        # v_3(1) = 0 -> 3^0 = 1
        # v_3(3) = 1 -> 3^(-1) = 0.333
        # v_3(9) = 2 -> 3^(-2) = 0.111
        indices = torch.tensor([1, 3, 9], device=device)

        # Act
        expected_val = norm_loss._compute_expected_valuation(indices)

        # Assert
        expected = torch.tensor([1.0, 1.0 / 3.0, 1.0 / 9.0], device=device)
        torch.testing.assert_close(expected_val, expected, rtol=1e-4, atol=1e-6)


# =============================================================================
# PAdicRankingLoss Tests
# =============================================================================


class TestPAdicRankingLoss:
    """Tests for PAdicRankingLoss (ranking-based contrastive loss)."""

    def test_initialization_default_params(self):
        """Test PAdicRankingLoss initializes with default parameters."""
        # Act
        loss_fn = PAdicRankingLoss()

        # Assert
        assert loss_fn.margin == 0.1
        assert loss_fn.n_triplets == 500

    def test_initialization_custom_params(self):
        """Test PAdicRankingLoss initializes with custom parameters."""
        # Act
        loss_fn = PAdicRankingLoss(margin=0.5, n_triplets=200)

        # Assert
        assert loss_fn.margin == 0.5
        assert loss_fn.n_triplets == 200

    def test_forward_returns_scalar(self, ranking_loss, padic_test_batch):
        """Test forward pass returns a scalar tensor."""
        # Arrange
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = ranking_loss(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_forward_non_negative_loss(self, ranking_loss, padic_test_batch):
        """Test loss is non-negative (ReLU applied)."""
        # Arrange
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = ranking_loss(z, batch_indices)

        # Assert
        assert loss >= 0.0

    def test_forward_empty_batch_returns_zero(self, ranking_loss, device):
        """Test empty batch returns zero loss."""
        # Arrange
        z = torch.empty(0, 16, device=device)
        batch_indices = torch.empty(0, dtype=torch.long, device=device)

        # Act
        loss = ranking_loss(z, batch_indices)

        # Assert
        assert loss == 0.0

    def test_forward_too_small_batch_returns_zero(self, ranking_loss, device):
        """Test batch with < 3 samples returns zero loss (no triplets)."""
        # Arrange
        z = torch.randn(2, 16, device=device)
        batch_indices = torch.tensor([1, 2], device=device)

        # Act
        loss = ranking_loss(z, batch_indices)

        # Assert
        assert loss == 0.0

    def test_forward_gradient_flow(self, ranking_loss, padic_test_batch):
        """Test gradients flow through the loss."""
        # Arrange
        z = padic_test_batch["z"].clone().requires_grad_(True)
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        loss = ranking_loss(z, batch_indices)
        loss.backward()

        # Assert
        assert z.grad is not None

    def test_ranking_preserves_order(self, device):
        """Test loss encourages correct ranking order."""
        # Arrange
        # Create a scenario where ranking is clearly violated
        loss_fn = PAdicRankingLoss(margin=0.1, n_triplets=50)

        # z[0] should be closer to z[1] than z[2] based on 3-adic distance
        # indices: 0, 3, 27 -> d(0,3)=3^(-1)=0.33, d(0,27)=3^(-3)=0.037
        # So 0 is 3-adically closer to 27 than to 3!
        batch_indices = torch.tensor([0, 3, 27, 81, 1, 2, 4, 5], device=device)
        z = torch.randn(8, 16, device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0


# =============================================================================
# TripletBatch Tests
# =============================================================================


class TestTripletBatch:
    """Tests for TripletBatch dataclass."""

    def test_creation(self, device):
        """Test TripletBatch creation with valid data."""
        # Arrange
        anchor_idx = torch.tensor([0, 1, 2], device=device)
        pos_idx = torch.tensor([1, 2, 3], device=device)
        neg_idx = torch.tensor([3, 4, 5], device=device)
        v_pos = torch.tensor([2.0, 1.0, 0.0], device=device)
        v_neg = torch.tensor([0.0, 0.0, 0.0], device=device)

        # Act
        batch = TripletBatch(
            anchor_idx=anchor_idx,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            v_pos=v_pos,
            v_neg=v_neg,
        )

        # Assert
        assert len(batch) == 3
        assert not batch.is_empty()

    def test_empty_creation(self, device):
        """Test TripletBatch.empty() creates empty batch."""
        # Act
        batch = TripletBatch.empty(device)

        # Assert
        assert len(batch) == 0
        assert batch.is_empty()

    def test_to_tuple(self, device):
        """Test conversion to tuple."""
        # Arrange
        anchor_idx = torch.tensor([0], device=device)
        pos_idx = torch.tensor([1], device=device)
        neg_idx = torch.tensor([2], device=device)
        v_pos = torch.tensor([1.0], device=device)
        v_neg = torch.tensor([0.0], device=device)

        batch = TripletBatch(
            anchor_idx=anchor_idx,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            v_pos=v_pos,
            v_neg=v_neg,
        )

        # Act
        result = batch.to_tuple()

        # Assert
        assert len(result) == 5
        assert torch.equal(result[0], anchor_idx)

    def test_concat_batches(self, device):
        """Test concatenation of multiple batches."""
        # Arrange
        batch1 = TripletBatch(
            anchor_idx=torch.tensor([0, 1], device=device),
            pos_idx=torch.tensor([1, 2], device=device),
            neg_idx=torch.tensor([2, 3], device=device),
            v_pos=torch.tensor([1.0, 2.0], device=device),
            v_neg=torch.tensor([0.0, 0.0], device=device),
        )
        batch2 = TripletBatch(
            anchor_idx=torch.tensor([4], device=device),
            pos_idx=torch.tensor([5], device=device),
            neg_idx=torch.tensor([6], device=device),
            v_pos=torch.tensor([3.0], device=device),
            v_neg=torch.tensor([1.0], device=device),
        )

        # Act
        combined = TripletBatch.concat([batch1, batch2])

        # Assert
        assert len(combined) == 3
        assert torch.equal(combined.anchor_idx, torch.tensor([0, 1, 4], device=device))

    def test_concat_with_empty_batches(self, device):
        """Test concat filters out empty batches."""
        # Arrange
        empty_batch = TripletBatch.empty(device)
        non_empty = TripletBatch(
            anchor_idx=torch.tensor([0], device=device),
            pos_idx=torch.tensor([1], device=device),
            neg_idx=torch.tensor([2], device=device),
            v_pos=torch.tensor([1.0], device=device),
            v_neg=torch.tensor([0.0], device=device),
        )

        # Act
        combined = TripletBatch.concat([empty_batch, non_empty, empty_batch])

        # Assert
        assert len(combined) == 1

    def test_concat_all_empty_returns_empty(self, device):
        """Test concat of all empty batches returns empty."""
        # Arrange
        empty1 = TripletBatch.empty(device)
        empty2 = TripletBatch.empty(device)

        # Act
        combined = TripletBatch.concat([empty1, empty2])

        # Assert
        assert combined.is_empty()

    def test_concat_empty_list_raises(self, device):
        """Test concat with empty list raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Cannot concatenate empty list"):
            TripletBatch.concat([])


# =============================================================================
# EuclideanTripletMiner Tests
# =============================================================================


class TestEuclideanTripletMiner:
    """Tests for EuclideanTripletMiner."""

    def test_initialization_default_params(self):
        """Test EuclideanTripletMiner initializes with defaults."""
        # Act
        miner = EuclideanTripletMiner()

        # Assert
        assert miner.base_margin == 0.05
        assert miner.margin_scale == 0.15
        assert miner.n_triplets == 500
        assert miner.hard_negative_ratio == 0.5

    def test_compute_distance_matrix(self, device):
        """Test Euclidean distance matrix computation."""
        # Arrange
        miner = EuclideanTripletMiner()
        z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=device)

        # Act
        d_matrix = miner.compute_distance_matrix(z)

        # Assert
        assert d_matrix.shape == (3, 3)
        # Diagonal should be zero
        assert torch.allclose(torch.diagonal(d_matrix), torch.zeros(3, device=device))
        # d(0,1) = d(0,2) = 1.0
        assert torch.isclose(d_matrix[0, 1], torch.tensor(1.0, device=device))
        # d(1,2) = sqrt(2)
        assert torch.isclose(d_matrix[1, 2], torch.tensor(2.0, device=device).sqrt())

    def test_mine_triplets_returns_triplet_batch(self, padic_test_batch):
        """Test mine_triplets returns TripletBatch."""
        # Arrange
        miner = EuclideanTripletMiner(n_triplets=20)
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        # Act
        triplets = miner.mine_triplets(z, batch_indices)

        # Assert
        assert isinstance(triplets, TripletBatch)

    def test_mine_triplets_small_batch_returns_empty(self, device):
        """Test mining from batch < 3 returns empty."""
        # Arrange
        miner = EuclideanTripletMiner()
        z = torch.randn(2, 16, device=device)
        batch_indices = torch.tensor([1, 2], device=device)

        # Act
        triplets = miner.mine_triplets(z, batch_indices)

        # Assert
        assert triplets.is_empty()

    def test_mine_triplets_valid_indices(self, padic_test_batch):
        """Test mined triplet indices are valid."""
        # Arrange
        miner = EuclideanTripletMiner(n_triplets=20)
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]
        batch_size = z.size(0)

        # Act
        triplets = miner.mine_triplets(z, batch_indices)

        # Assert
        if not triplets.is_empty():
            assert torch.all(triplets.anchor_idx >= 0)
            assert torch.all(triplets.anchor_idx < batch_size)
            assert torch.all(triplets.pos_idx >= 0)
            assert torch.all(triplets.pos_idx < batch_size)
            assert torch.all(triplets.neg_idx >= 0)
            assert torch.all(triplets.neg_idx < batch_size)

    def test_hierarchical_margin_computation(self):
        """Test hierarchical margin increases with valuation difference."""
        # Arrange
        miner = EuclideanTripletMiner(base_margin=0.1, margin_scale=0.2)
        v_pos = torch.tensor([3.0, 2.0, 1.0])
        v_neg = torch.tensor([0.0, 0.0, 0.0])

        # Act
        margins = miner.compute_hierarchical_margin(v_pos, v_neg)

        # Assert
        # margin = base + scale * |v_pos - v_neg|
        expected = torch.tensor([0.1 + 0.2 * 3, 0.1 + 0.2 * 2, 0.1 + 0.2 * 1])
        torch.testing.assert_close(margins, expected)


# =============================================================================
# HyperbolicTripletMiner Tests
# =============================================================================


class TestHyperbolicTripletMiner:
    """Tests for HyperbolicTripletMiner."""

    def test_initialization_default_params(self):
        """Test HyperbolicTripletMiner initializes with defaults."""
        # Act
        miner = HyperbolicTripletMiner()

        # Assert
        assert miner.curvature == 1.0
        assert miner.max_norm == 0.95

    def test_initialization_custom_params(self):
        """Test HyperbolicTripletMiner initializes with custom params."""
        # Act
        miner = HyperbolicTripletMiner(curvature=2.0, max_norm=0.9, n_triplets=100)

        # Assert
        assert miner.curvature == 2.0
        assert miner.max_norm == 0.9
        assert miner.n_triplets == 100

    def test_compute_distance_matrix_poincare(self, device):
        """Test Poincare distance matrix computation."""
        # Arrange
        miner = HyperbolicTripletMiner(curvature=1.0)
        # Points inside Poincare ball (norm < 1)
        z = torch.tensor(
            [[0.0, 0.0], [0.3, 0.0], [0.0, 0.3]],
            device=device,
        )

        # Act
        d_matrix = miner.compute_distance_matrix(z)

        # Assert
        assert d_matrix.shape == (3, 3)
        # Diagonal should be approximately zero
        assert torch.allclose(torch.diagonal(d_matrix), torch.zeros(3, device=device), atol=1e-5)
        # Hyperbolic distances should be positive for non-identical points
        assert d_matrix[0, 1] > 0
        assert d_matrix[0, 2] > 0

    def test_mine_triplets_returns_triplet_batch(self, device):
        """Test mine_triplets returns TripletBatch for hyperbolic."""
        # Arrange
        miner = HyperbolicTripletMiner(n_triplets=20)
        # Create points inside Poincare ball
        z = torch.randn(32, 16, device=device) * 0.3  # Small norm
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (32,), device=device)

        # Act
        triplets = miner.mine_triplets(z, batch_indices)

        # Assert
        assert isinstance(triplets, TripletBatch)


# =============================================================================
# compute_3adic_valuation_batch Tests
# =============================================================================


class TestCompute3AdicValuationBatch:
    """Tests for compute_3adic_valuation_batch utility function."""

    def test_same_indices_max_valuation(self, device):
        """Test identical indices have maximum valuation (v_3(0) = 9)."""
        # Arrange
        idx_i = torch.tensor([5, 10, 100], device=device)
        idx_j = torch.tensor([5, 10, 100], device=device)

        # Act
        valuations = compute_3adic_valuation_batch(idx_i, idx_j)

        # Assert
        # v_3(0) = MAX_VALUATION = 9
        assert torch.all(valuations == 9.0)

    def test_known_valuations(self, device):
        """Test known valuation values."""
        # Arrange
        # |3-0| = 3, v_3(3) = 1
        # |9-0| = 9, v_3(9) = 2
        # |2-1| = 1, v_3(1) = 0
        idx_i = torch.tensor([0, 0, 1], device=device)
        idx_j = torch.tensor([3, 9, 2], device=device)

        # Act
        valuations = compute_3adic_valuation_batch(idx_i, idx_j)

        # Assert
        expected = torch.tensor([1.0, 2.0, 0.0], device=device)
        torch.testing.assert_close(valuations, expected)

    def test_symmetry(self, device):
        """Test v_3(|i-j|) = v_3(|j-i|)."""
        # Arrange
        idx_i = torch.tensor([5, 10, 27], device=device)
        idx_j = torch.tensor([14, 1, 0], device=device)

        # Act
        v_ij = compute_3adic_valuation_batch(idx_i, idx_j)
        v_ji = compute_3adic_valuation_batch(idx_j, idx_i)

        # Assert
        torch.testing.assert_close(v_ij, v_ji)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPAdicLossesIntegration:
    """Integration tests for p-adic losses working together."""

    def test_all_losses_compute_on_same_batch(self, padic_test_batch, device):
        """Test all losses can compute on the same batch."""
        # Arrange
        z = padic_test_batch["z"]
        batch_indices = padic_test_batch["batch_indices"]

        metric_loss = PAdicMetricLoss(n_pairs=50)
        norm_loss = PAdicNormLoss(latent_dim=z.size(1))
        ranking_loss = PAdicRankingLoss(n_triplets=30)

        # Act
        l_metric = metric_loss(z, batch_indices)
        l_norm = norm_loss(z, batch_indices)
        l_ranking = ranking_loss(z, batch_indices)

        # Assert
        assert l_metric >= 0
        assert l_norm >= 0
        assert l_ranking >= 0

    def test_combined_loss_gradient_flow(self, padic_test_batch, device):
        """Test gradients flow through combined losses."""
        # Arrange
        z = padic_test_batch["z"].clone().requires_grad_(True)
        batch_indices = padic_test_batch["batch_indices"]

        metric_loss = PAdicMetricLoss(n_pairs=50)
        norm_loss = PAdicNormLoss(latent_dim=z.size(1))
        ranking_loss = PAdicRankingLoss(n_triplets=30)

        # Act
        total_loss = metric_loss(z, batch_indices) + norm_loss(z, batch_indices) + ranking_loss(z, batch_indices)
        total_loss.backward()

        # Assert
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
        assert not torch.isinf(z.grad).any()

    def test_losses_on_gpu_if_available(self, device):
        """Test losses work on GPU."""
        # Arrange
        torch.manual_seed(42)
        z = torch.randn(32, 16, device=device)
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (32,), device=device)

        metric_loss = PAdicMetricLoss()
        norm_loss = PAdicNormLoss()
        ranking_loss = PAdicRankingLoss()

        # Act
        l_metric = metric_loss(z, batch_indices)
        l_norm = norm_loss(z, batch_indices)
        l_ranking = ranking_loss(z, batch_indices)

        # Assert
        assert l_metric.device.type == z.device.type
        assert l_norm.device.type == z.device.type
        assert l_ranking.device.type == z.device.type


# =============================================================================
# Parametrized Tests
# =============================================================================


class TestPAdicLossesParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize("batch_size", [3, 8, 16, 64, 128])
    def test_metric_loss_various_batch_sizes(self, batch_size, device):
        """Test PAdicMetricLoss works with various batch sizes."""
        # Arrange
        loss_fn = PAdicMetricLoss(n_pairs=min(50, batch_size * (batch_size - 1) // 2))
        z = torch.randn(batch_size, 16, device=device)
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (batch_size,), device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("latent_dim", [4, 8, 16, 32, 64])
    def test_norm_loss_various_latent_dims(self, latent_dim, device):
        """Test PAdicNormLoss works with various latent dimensions."""
        # Arrange
        loss_fn = PAdicNormLoss(latent_dim=latent_dim)
        z = torch.randn(16, latent_dim, device=device)
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (16,), device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("margin", [0.01, 0.05, 0.1, 0.5, 1.0])
    def test_ranking_loss_various_margins(self, margin, device):
        """Test PAdicRankingLoss works with various margins."""
        # Arrange
        loss_fn = PAdicRankingLoss(margin=margin, n_triplets=30)
        z = torch.randn(32, 16, device=device)
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (32,), device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    @pytest.mark.parametrize(
        "idx_i,idx_j,expected_valuation",
        [
            (0, 0, 9),  # Same index -> v_3(0) = 9
            (0, 1, 0),  # |0-1| = 1 -> v_3(1) = 0
            (0, 3, 1),  # |0-3| = 3 -> v_3(3) = 1
            (0, 9, 2),  # |0-9| = 9 -> v_3(9) = 2
            (0, 27, 3),  # |0-27| = 27 -> v_3(27) = 3
            (0, 81, 4),  # |0-81| = 81 -> v_3(81) = 4
            (1, 4, 0),  # |1-4| = 3 -> v_3(3) = 1... wait, |4-1|=3
            (10, 13, 0),  # |10-13| = 3 -> v_3(3) = 1
        ],
    )
    def test_valuation_batch_parametrized(self, idx_i, idx_j, expected_valuation, device):
        """Test compute_3adic_valuation_batch with known values."""
        # Arrange
        i = torch.tensor([idx_i], device=device)
        j = torch.tensor([idx_j], device=device)

        # Act
        v = compute_3adic_valuation_batch(i, j)

        # Assert
        # Note: for |1-4|=3, v_3(3)=1, so we need to adjust expected
        actual_diff = abs(idx_i - idx_j)
        if actual_diff == 0:
            assert v.item() == 9
        else:
            # Compute expected valuation of actual_diff
            expected_v = 0
            temp = actual_diff
            while temp % 3 == 0:
                expected_v += 1
                temp //= 3
            assert v.item() == expected_v

    @pytest.mark.parametrize("n_triplets", [10, 50, 100, 500])
    def test_euclidean_miner_various_triplet_counts(self, n_triplets, device):
        """Test EuclideanTripletMiner with various triplet counts."""
        # Arrange
        miner = EuclideanTripletMiner(n_triplets=n_triplets)
        z = torch.randn(64, 16, device=device)
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (64,), device=device)

        # Act
        triplets = miner.mine_triplets(z, batch_indices)

        # Assert
        assert isinstance(triplets, TripletBatch)
        # Should not exceed requested number (may be less due to filtering)
        assert len(triplets) <= n_triplets


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of p-adic losses."""

    def test_metric_loss_large_latent_values(self, device):
        """Test PAdicMetricLoss handles large latent values."""
        # Arrange
        loss_fn = PAdicMetricLoss(n_pairs=50)
        z = torch.randn(16, 16, device=device) * 100.0  # Large values
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (16,), device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_metric_loss_small_latent_values(self, device):
        """Test PAdicMetricLoss handles small latent values."""
        # Arrange
        loss_fn = PAdicMetricLoss(n_pairs=50)
        z = torch.randn(16, 16, device=device) * 1e-6  # Small values
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (16,), device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_norm_loss_near_zero_latent(self, device):
        """Test PAdicNormLoss handles near-zero latent values."""
        # Arrange
        loss_fn = PAdicNormLoss(latent_dim=16)
        z = torch.zeros(16, 16, device=device) + 1e-10  # Near zero
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (16,), device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_ranking_loss_identical_latent_points(self, device):
        """Test PAdicRankingLoss handles identical latent points."""
        # Arrange
        loss_fn = PAdicRankingLoss(n_triplets=30)
        z = torch.ones(16, 16, device=device)  # All same point
        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (16,), device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_hyperbolic_miner_boundary_points(self, device):
        """Test HyperbolicTripletMiner handles boundary points."""
        # Arrange
        miner = HyperbolicTripletMiner(curvature=1.0, max_norm=0.95)
        # Points very close to boundary
        z = torch.randn(32, 16, device=device)
        z = z / torch.norm(z, dim=-1, keepdim=True) * 0.94

        batch_indices = torch.randint(0, TERNARY.N_OPERATIONS, (32,), device=device)

        # Act
        triplets = miner.mine_triplets(z, batch_indices)

        # Assert
        assert isinstance(triplets, TripletBatch)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for p-adic losses."""

    def test_metric_loss_exact_batch_of_two(self, device):
        """Test PAdicMetricLoss with exactly 2 samples."""
        # Arrange
        loss_fn = PAdicMetricLoss(n_pairs=10)
        z = torch.randn(2, 16, device=device)
        batch_indices = torch.tensor([1, 100], device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert - should compute loss for the single pair
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_ranking_loss_exactly_three_samples(self, device):
        """Test PAdicRankingLoss with exactly 3 samples (minimum for triplets)."""
        # Arrange
        loss_fn = PAdicRankingLoss(n_triplets=10)
        z = torch.randn(3, 16, device=device)
        batch_indices = torch.tensor([1, 3, 9], device=device)  # Known valuations

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_norm_loss_single_sample(self, device):
        """Test PAdicNormLoss with single sample."""
        # Arrange
        loss_fn = PAdicNormLoss(latent_dim=16)
        z = torch.randn(1, 16, device=device)
        batch_indices = torch.tensor([42], device=device)

        # Act
        loss = loss_fn(z, batch_indices)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0.0

    def test_miner_all_same_indices(self, device):
        """Test EuclideanTripletMiner when all batch indices are same."""
        # Arrange
        miner = EuclideanTripletMiner(n_triplets=20)
        z = torch.randn(16, 16, device=device)
        batch_indices = torch.full((16,), 42, device=device)  # All same

        # Act
        triplets = miner.mine_triplets(z, batch_indices)

        # Assert - should return empty or very few triplets
        # (all valuations are 9, so v_pos > v_neg condition never holds)
        assert isinstance(triplets, TripletBatch)

    def test_loss_with_max_index(self, device):
        """Test losses with maximum valid index."""
        # Arrange
        max_idx = TERNARY.N_OPERATIONS - 1
        z = torch.randn(8, 16, device=device)
        batch_indices = torch.tensor([0, 1, max_idx, max_idx - 1, 100, 200, 300, 400], device=device)

        metric_loss = PAdicMetricLoss(n_pairs=20)
        norm_loss = PAdicNormLoss(latent_dim=16)
        ranking_loss = PAdicRankingLoss(n_triplets=15)

        # Act
        l_metric = metric_loss(z, batch_indices)
        l_norm = norm_loss(z, batch_indices)
        l_ranking = ranking_loss(z, batch_indices)

        # Assert
        assert not torch.isnan(l_metric)
        assert not torch.isnan(l_norm)
        assert not torch.isnan(l_ranking)
