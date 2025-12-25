import torch

from src.encoders.codon_encoder import CodonEncoder


def test_encoder_initialization():
    encoder = CodonEncoder(embedding_dim=16, use_padic_init=True)
    assert encoder.embedding.weight.shape == (64, 16)


def test_forward_pass():
    encoder = CodonEncoder(embedding_dim=8)
    # Batch of 4 sequences, length 10
    x = torch.randint(0, 64, (4, 10))
    emb = encoder(x)

    assert emb.shape == (4, 10, 8)


def test_padic_init_structure():
    """Verify that hierarchical initialization logic (bases) applied correctly."""
    encoder = CodonEncoder(embedding_dim=16, use_padic_init=True)
    w = encoder.embedding.weight.data

    # Check Codon 0 (AAA -> 0, 0, 0)
    # Should have weight in first 4 dims (1,0,0,0), next 4 (1,0,0,0), next 4 (1,0,0,0)
    # Scaled by 0.5

    # Index 0
    # b1=0 -> [0.5, 0, 0, 0]
    expected_slice = torch.tensor([0.5, 0.0, 0.0, 0.0])

    assert torch.allclose(w[0, 0:4], expected_slice)
    assert torch.allclose(w[0, 4:8], expected_slice)
    assert torch.allclose(w[0, 8:12], expected_slice)


def test_distance_matrix():
    encoder = CodonEncoder(embedding_dim=16)
    dist_mat = encoder.get_distance_matrix()

    assert dist_mat.shape == (64, 64)
    # Diagonal should be 0
    # Diagonal should be close to 0
    diag_max = dist_mat.diag().abs().max()
    assert diag_max < 1e-2, f"Diagonal max was {diag_max}"
