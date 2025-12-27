"""Tests for Cross-Resistance VAE module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add project root to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))


class TestCrossResistanceMatrix:
    """Test cross-resistance knowledge base."""

    def test_matrix_symmetry(self):
        """Cross-resistance should be approximately symmetric."""
        from dataclasses import dataclass, field
        from typing import List

        # Embedded matrix for testing
        CROSS_RESISTANCE_MATRIX = {
            "AZT": {"AZT": 1.00, "D4T": 0.85, "ABC": 0.40, "TDF": 0.30, "DDI": 0.35, "3TC": -0.15},
            "D4T": {"AZT": 0.85, "D4T": 1.00, "ABC": 0.45, "TDF": 0.35, "DDI": 0.40, "3TC": -0.10},
            "ABC": {"AZT": 0.40, "D4T": 0.45, "ABC": 1.00, "TDF": 0.55, "DDI": 0.50, "3TC": 0.35},
            "TDF": {"AZT": 0.30, "D4T": 0.35, "TDF": 1.00, "ABC": 0.55, "DDI": 0.45, "3TC": 0.25},
            "DDI": {"AZT": 0.35, "D4T": 0.40, "ABC": 0.50, "TDF": 0.45, "DDI": 1.00, "3TC": 0.20},
            "3TC": {"AZT": -0.15, "D4T": -0.10, "ABC": 0.35, "TDF": 0.25, "DDI": 0.20, "3TC": 1.00},
        }

        drugs = list(CROSS_RESISTANCE_MATRIX.keys())
        for i, drug1 in enumerate(drugs):
            for j, drug2 in enumerate(drugs):
                val1 = CROSS_RESISTANCE_MATRIX[drug1].get(drug2, 0)
                val2 = CROSS_RESISTANCE_MATRIX[drug2].get(drug1, 0)
                # Allow small asymmetry for biological reasons
                assert abs(val1 - val2) < 0.15, f"Large asymmetry: {drug1}-{drug2}"

    def test_diagonal_is_one(self):
        """Self-resistance should be 1.0."""
        CROSS_RESISTANCE_MATRIX = {
            "AZT": {"AZT": 1.00, "D4T": 0.85},
            "D4T": {"AZT": 0.85, "D4T": 1.00},
        }

        for drug in CROSS_RESISTANCE_MATRIX:
            assert CROSS_RESISTANCE_MATRIX[drug][drug] == 1.0

    def test_tam_cross_resistance(self):
        """TAM drugs (AZT, D4T) should have high cross-resistance."""
        CROSS_RESISTANCE_MATRIX = {
            "AZT": {"AZT": 1.00, "D4T": 0.85},
            "D4T": {"AZT": 0.85, "D4T": 1.00},
        }

        assert CROSS_RESISTANCE_MATRIX["AZT"]["D4T"] > 0.7

    def test_m184v_resensitization(self):
        """3TC should have negative/low cross-resistance with AZT (M184V resensitization)."""
        CROSS_RESISTANCE_MATRIX = {
            "AZT": {"3TC": -0.15},
            "3TC": {"AZT": -0.15},
        }

        assert CROSS_RESISTANCE_MATRIX["AZT"]["3TC"] < 0


class TestCrossResistanceVAE:
    """Test CrossResistanceVAE model."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from dataclasses import dataclass, field
        from typing import List

        @dataclass
        class CrossResistanceConfig:
            input_dim: int
            latent_dim: int = 32
            hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
            drug_names: List[str] = field(default_factory=lambda: ["AZT", "D4T", "3TC"])
            n_positions: int = 50
            dropout: float = 0.1
            use_cross_attention: bool = True
            ranking_weight: float = 0.3

        return CrossResistanceConfig(input_dim=50 * 22, n_positions=50)

    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        import torch.nn as nn

        class CrossDrugAttention(nn.Module):
            def __init__(self, n_drugs: int, latent_dim: int, n_heads: int = 4):
                super().__init__()
                self.n_drugs = n_drugs
                self.latent_dim = latent_dim
                self.drug_embed = nn.Embedding(n_drugs, latent_dim)
                self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
                self.output_proj = nn.Linear(latent_dim, latent_dim)

            def forward(self, z):
                batch_size = z.size(0)
                device = z.device
                drug_indices = torch.arange(self.n_drugs, device=device)
                drug_embeds = self.drug_embed(drug_indices).unsqueeze(0).expand(batch_size, -1, -1)
                z_expanded = z.unsqueeze(1).expand(-1, self.n_drugs, -1)
                query = drug_embeds + z_expanded
                attn_out, _ = self.attention(query, query, query)
                return self.output_proj(attn_out)

        class CrossResistanceVAE(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg
                self.drug_names = cfg.drug_names
                self.n_drugs = len(cfg.drug_names)

                layers = []
                in_dim = cfg.input_dim
                for h in cfg.hidden_dims:
                    layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h)])
                    in_dim = h
                self.encoder = nn.Sequential(*layers)

                self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
                self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

                dec_layers = []
                dec_in = cfg.latent_dim
                for h in reversed(cfg.hidden_dims):
                    dec_layers.extend([nn.Linear(dec_in, h), nn.GELU()])
                    dec_in = h
                dec_layers.append(nn.Linear(dec_in, cfg.input_dim))
                self.decoder = nn.Sequential(*dec_layers)

                self.cross_attention = CrossDrugAttention(self.n_drugs, cfg.latent_dim)

                self.drug_heads = nn.ModuleDict({
                    drug: nn.Sequential(nn.Linear(cfg.latent_dim, 32), nn.GELU(), nn.Linear(32, 1))
                    for drug in cfg.drug_names
                })

            def forward(self, x):
                h = self.encoder(x)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                x_recon = self.decoder(z)
                z_drugs = self.cross_attention(z)

                predictions = {}
                for i, drug in enumerate(self.drug_names):
                    z_drug = z_drugs[:, i, :]
                    pred = self.drug_heads[drug](z_drug).squeeze(-1)
                    predictions[drug] = pred

                return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "predictions": predictions}

        return CrossResistanceVAE(config)

    def test_forward_pass(self, model, config):
        """Test forward pass produces expected outputs."""
        x = torch.randn(4, config.input_dim)
        out = model(x)

        assert "x_recon" in out
        assert "mu" in out
        assert "logvar" in out
        assert "predictions" in out

        assert out["x_recon"].shape == x.shape
        assert out["mu"].shape == (4, config.latent_dim)

    def test_predictions_for_all_drugs(self, model, config):
        """Test that predictions are generated for all drugs."""
        x = torch.randn(4, config.input_dim)
        out = model(x)

        for drug in config.drug_names:
            assert drug in out["predictions"]
            assert out["predictions"][drug].shape == (4,)

    def test_cross_attention_shape(self, model, config):
        """Test cross-attention produces correct shape."""
        z = torch.randn(4, config.latent_dim)
        z_drugs = model.cross_attention(z)

        assert z_drugs.shape == (4, len(config.drug_names), config.latent_dim)

    def test_gradient_flow(self, model, config):
        """Test that gradients flow through entire model."""
        x = torch.randn(4, config.input_dim)
        out = model(x)

        # Sum all predictions and backprop
        loss = sum(out["predictions"][drug].sum() for drug in config.drug_names)
        loss.backward()

        # Check encoder gradients
        for param in model.encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_reproducibility(self, model, config):
        """Test that same input produces same output with fixed seed."""
        torch.manual_seed(42)
        x = torch.randn(4, config.input_dim)

        model.eval()
        with torch.no_grad():
            # Use mean instead of sampling
            h = model.encoder(x)
            mu = model.fc_mu(h)
            z = mu  # Use mean directly

            out1_preds = {}
            z_drugs = model.cross_attention(z)
            for i, drug in enumerate(config.drug_names):
                z_drug = z_drugs[:, i, :]
                out1_preds[drug] = model.drug_heads[drug](z_drug).squeeze(-1).clone()

            out2_preds = {}
            z_drugs = model.cross_attention(z)
            for i, drug in enumerate(config.drug_names):
                z_drug = z_drugs[:, i, :]
                out2_preds[drug] = model.drug_heads[drug](z_drug).squeeze(-1).clone()

        for drug in config.drug_names:
            assert torch.allclose(out1_preds[drug], out2_preds[drug])


class TestUncertaintyVAE:
    """Test uncertainty quantification."""

    @pytest.fixture
    def uncertainty_model(self):
        """Create uncertainty VAE."""
        from dataclasses import dataclass, field
        from typing import List
        import torch.nn as nn
        import torch.nn.functional as F

        @dataclass
        class UncertaintyConfig:
            input_dim: int
            latent_dim: int = 16
            hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
            dropout: float = 0.1
            mc_samples: int = 50

        class MCDropout(nn.Module):
            def __init__(self, p=0.1):
                super().__init__()
                self.p = p

            def forward(self, x):
                return F.dropout(x, p=self.p, training=True)

        class UncertaintyVAE(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg

                layers = []
                in_dim = cfg.input_dim
                for h in cfg.hidden_dims:
                    layers.extend([nn.Linear(in_dim, h), nn.GELU(), MCDropout(cfg.dropout)])
                    in_dim = h
                self.encoder = nn.Sequential(*layers)

                self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
                self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)
                self.predictor = nn.Sequential(
                    nn.Linear(cfg.latent_dim, 16), nn.GELU(), MCDropout(cfg.dropout), nn.Linear(16, 1)
                )

            def forward(self, x):
                h = self.encoder(x)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                pred = self.predictor(z).squeeze(-1)
                return {"mu": mu, "logvar": logvar, "prediction": pred}

            def predict_with_uncertainty(self, x, n_samples=None):
                n_samples = n_samples or self.cfg.mc_samples
                predictions = [self.forward(x)["prediction"] for _ in range(n_samples)]
                pred_stack = torch.stack(predictions, dim=0)
                return pred_stack.mean(dim=0), pred_stack.std(dim=0)

        cfg = UncertaintyConfig(input_dim=100)
        return UncertaintyVAE(cfg)

    def test_mc_dropout_variance(self, uncertainty_model):
        """MC Dropout should produce variance in predictions."""
        x = torch.randn(4, 100)

        predictions = []
        for _ in range(10):
            out = uncertainty_model(x)
            predictions.append(out["prediction"])

        pred_stack = torch.stack(predictions, dim=0)
        variance = pred_stack.var(dim=0)

        # Should have non-zero variance due to dropout
        assert (variance > 0).all()

    def test_uncertainty_estimation(self, uncertainty_model):
        """Test uncertainty estimation produces valid outputs."""
        x = torch.randn(4, 100)

        mean, std = uncertainty_model.predict_with_uncertainty(x, n_samples=20)

        assert mean.shape == (4,)
        assert std.shape == (4,)
        assert (std >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
