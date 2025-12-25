import torch

from src.models.spectral_encoder import SpectralGraphEncoder


def test_spectral_laplacian_shape():
    encoder = SpectralGraphEncoder(hidden_dim=4)
    # Batch=2, Nodes=5
    adj = torch.rand(2, 5, 5)
    # Symmetrize
    adj = (adj + adj.transpose(1, 2)) / 2

    L = encoder.compute_laplacian(adj)
    assert L.shape == (2, 5, 5)


def test_forward_pass_dims():
    encoder = SpectralGraphEncoder(hidden_dim=4, curvature=1.0)
    # Batch=2, Nodes=10
    adj = torch.rand(2, 10, 10)
    adj = (adj + adj.transpose(1, 2)) / 2

    output = encoder(adj)

    # Should perform eigendecomposition, pooling, and projection
    assert output.shape == (2, 4)

    # Check Poincaré norm constraint (default 0.95)
    # Note: geoopt/poincare projection ensures this
    assert torch.all(torch.norm(output, dim=-1) < 1.0)


def test_forward_padding():
    # If nodes < hidden_dim
    encoder = SpectralGraphEncoder(hidden_dim=8)
    # Batch=1, Nodes=4 (less than 8)
    adj = torch.rand(1, 4, 4)
    adj = (adj + adj.transpose(1, 2)) / 2

    output = encoder(adj)
    assert output.shape == (1, 8)
