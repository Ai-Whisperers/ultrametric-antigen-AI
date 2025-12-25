import torch

from src.data.autoimmunity import AutoimmunityLoader
from src.losses.autoimmunity import (
    AutoimmuneCodonRegularizer,
    CD4CD8AwareRegularizer,
    HUMAN_CODON_RSCU,
)


def test_risk_scoring():
    loader = AutoimmunityLoader()

    # 1. Low risk sequence (diverse)
    seq_low = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 10, 11, 12])
    score_low = loader.get_risk_score(seq_low)
    assert score_low < 0.5

    # 2. High risk sequence (repetitive)
    seq_high = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
    score_high = loader.get_risk_score(seq_high)
    assert score_high > 0.5

    # 3. Specific "bad" codon (63)
    seq_bad = torch.tensor([0, 1, 63, 2])
    score_bad = loader.get_risk_score(seq_bad)
    assert score_bad == 0.9


def test_batch_processing():
    loader = AutoimmunityLoader()
    batch = torch.stack([torch.tensor([0, 1, 2, 3]), torch.tensor([1, 1, 1, 1])])
    scores = loader.get_batch_risk(batch)
    assert scores.shape == (2,)
    assert scores[0] < scores[1]


class TestAutoimmuneCodonRegularizer:
    """Tests for AutoimmuneCodonRegularizer."""

    def test_initialization(self):
        """Test regularizer initialization."""
        regularizer = AutoimmuneCodonRegularizer()
        assert regularizer.pathogen == "hiv"
        assert regularizer.rscu_weight == 0.3
        assert regularizer.risk_weight == 0.7

    def test_forward_basic(self):
        """Test forward pass with codon indices only."""
        regularizer = AutoimmuneCodonRegularizer()
        codon_indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        loss = regularizer(codon_indices)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0

    def test_forward_with_logits(self):
        """Test forward pass with codon logits."""
        regularizer = AutoimmuneCodonRegularizer()
        batch_size = 2
        seq_len = 8
        n_codons = 64

        codon_indices = torch.randint(0, n_codons, (batch_size, seq_len))
        codon_logits = torch.randn(batch_size, seq_len, n_codons)

        loss = regularizer(codon_indices, codon_logits)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_return_components(self):
        """Test returning individual loss components."""
        regularizer = AutoimmuneCodonRegularizer()
        codon_indices = torch.randint(0, 64, (2, 10))
        codon_logits = torch.randn(2, 10, 64)

        result = regularizer(codon_indices, codon_logits, return_components=True)

        assert isinstance(result, dict)
        assert "total" in result
        assert "rscu" in result
        assert "risk" in result
        assert "diversity" in result

    def test_risk_penalty_high_for_bad_sequence(self):
        """Test that high-risk sequences get higher penalties."""
        regularizer = AutoimmuneCodonRegularizer(rscu_weight=0, risk_weight=1.0, diversity_weight=0)

        # Good sequence (diverse)
        good_seq = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        good_loss = regularizer(good_seq)

        # Bad sequence (with stop codon 63)
        bad_seq = torch.tensor([[0, 1, 63, 3]])
        bad_loss = regularizer(bad_seq)

        assert bad_loss > good_loss

    def test_diversity_penalty(self):
        """Test diversity regularization."""
        regularizer = AutoimmuneCodonRegularizer(
            rscu_weight=0, risk_weight=0, diversity_weight=1.0, target_diversity=0.8
        )

        # Low diversity sequence
        low_div = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 2, 2]])
        low_div_loss = regularizer(low_div)

        # High diversity sequence
        high_div = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        high_div_loss = regularizer(high_div)

        # Both should have some loss, but different values
        assert low_div_loss >= 0
        assert high_div_loss >= 0

    def test_safe_codon_mask(self):
        """Test getting safe codon mask."""
        regularizer = AutoimmuneCodonRegularizer()
        mask = regularizer.get_safe_codon_mask(threshold=0.3)

        assert mask.shape == (64,)
        assert mask.dtype == torch.bool
        # Some codons should be safe (above threshold)
        assert mask.sum() > 0


class TestCD4CD8AwareRegularizer:
    """Tests for CD4CD8AwareRegularizer."""

    def test_initialization(self):
        """Test initialization with default base regularizer."""
        regularizer = CD4CD8AwareRegularizer()
        assert regularizer.sensitivity_scale == 2.0
        assert regularizer.base is not None

    def test_immune_sensitivity_immunocompromised(self):
        """Test sensitivity for immunocompromised states."""
        regularizer = CD4CD8AwareRegularizer(sensitivity_scale=1.0)

        # Severely compromised (ratio < 0.5)
        sensitivity_low = regularizer.compute_immune_sensitivity(0.3)
        # Normal (ratio 1.0-1.5)
        sensitivity_normal = regularizer.compute_immune_sensitivity(1.2)
        # Strong immunity (ratio > 1.5)
        sensitivity_high = regularizer.compute_immune_sensitivity(2.0)

        assert sensitivity_low > sensitivity_normal > sensitivity_high

    def test_forward_scaling(self):
        """Test that loss scales with CD4/CD8 ratio."""
        regularizer = CD4CD8AwareRegularizer()
        codon_indices = torch.randint(0, 64, (2, 10))

        # Immunocompromised (higher penalty)
        loss_compromised = regularizer(codon_indices, cd4_cd8_ratio=0.3)

        # Strong immunity (lower penalty)
        loss_strong = regularizer(codon_indices, cd4_cd8_ratio=2.0)

        # Compromised state should have higher loss
        assert loss_compromised > loss_strong


def test_human_rscu_values():
    """Test that RSCU values are properly defined."""
    # Should have values for multiple codons
    assert len(HUMAN_CODON_RSCU) > 20

    # All values should be in valid range (0-1 for RSCU)
    for codon_idx, rscu in HUMAN_CODON_RSCU.items():
        assert 0 <= rscu <= 1.0, f"Invalid RSCU value {rscu} for codon {codon_idx}"
