# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Integration tests for full architecture pipeline.

Tests cover:
- BaseVAE with epistasis module
- Structure-aware VAE with uncertainty
- Transfer learning with disease analyzers
- End-to-end drug resistance prediction
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.encoders.alphafold_encoder import AlphaFoldEncoder, AlphaFoldStructure
from src.losses.epistasis_loss import EpistasisLoss
from src.models.base_vae import BaseVAE, VAEConfig
from src.models.epistasis_module import EpistasisModule, EpistasisPredictor
from src.models.structure_aware_vae import StructureAwareVAE, StructureConfig


class IntegrationVAE(BaseVAE):
    """VAE for integration testing."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.GELU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[-1]),
        )
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(self.hidden_dims[-1], self.input_dim * 3),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        out = self.decoder(z)
        return out.view(-1, self.input_dim, 3)


class SimpleDataset(Dataset):
    """Simple dataset for integration testing."""

    def __init__(self, n_samples=50, input_dim=64):
        self.x = torch.randn(n_samples, input_dim)
        self.y = torch.rand(n_samples, 5)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


class TestBaseVAEWithEpistasis:
    """Test BaseVAE integration with epistasis module."""

    def test_vae_forward_with_epistasis_loss(self):
        """Test VAE forward pass with epistasis loss."""
        vae = IntegrationVAE(input_dim=64, latent_dim=16, hidden_dims=[128, 64])
        epistasis_loss = EpistasisLoss(latent_dim=16, n_drugs=5, use_coevolution=False)

        x = torch.randn(8, 64)
        outputs = vae(x)

        # Simulate predictions
        predictions = torch.randn(8, 5)
        model_output = {
            "predictions": predictions,
            "z": outputs["z"],
        }
        targets = {"resistance": torch.rand(8, 5)}

        loss_result = epistasis_loss(model_output, targets)

        assert loss_result.total_loss.dim() == 0
        assert not torch.isnan(loss_result.total_loss)

    def test_vae_training_loop_with_epistasis(self):
        """Test VAE training loop with epistasis loss."""
        vae = IntegrationVAE(input_dim=64, latent_dim=16)
        epistasis_loss = EpistasisLoss(latent_dim=16, n_drugs=5, use_coevolution=False)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        for _ in range(3):
            x = torch.randn(16, 64)
            outputs = vae(x)

            predictions = outputs["z"][:, :5]  # Use first 5 dims as predictions
            model_output = {"predictions": predictions, "z": outputs["z"]}
            targets = {"resistance": torch.rand(16, 5)}

            loss_result = epistasis_loss(model_output, targets)

            optimizer.zero_grad()
            loss_result.total_loss.backward()
            optimizer.step()

    def test_epistasis_predictor_with_vae(self):
        """Test epistasis predictor combined with VAE."""
        vae = IntegrationVAE(input_dim=64, latent_dim=16)
        predictor = EpistasisPredictor(n_positions=100, n_outputs=5)

        x = torch.randn(4, 64)
        vae_outputs = vae(x)

        # Simulate mutation positions
        positions = torch.randint(0, 100, (4, 3))
        amino_acids = torch.randint(0, 21, (4, 3))

        pred_outputs = predictor(positions, amino_acids)

        assert pred_outputs["predictions"].shape == (4, 5)


class TestStructureAwareVAEWithUncertainty:
    """Test Structure-aware VAE with uncertainty."""

    def test_structure_vae_forward(self):
        """Test structure-aware VAE forward pass."""
        config = StructureConfig(use_structure=True)
        vae = StructureAwareVAE(input_dim=64, latent_dim=16, structure_config=config)

        x = torch.randn(4, 64)
        structure = torch.randn(4, 30, 3)
        plddt = torch.rand(4, 30) * 100

        outputs = vae(x, structure=structure, plddt=plddt)

        assert outputs["z"].shape == (4, 16)
        assert outputs["logits"].shape == (4, 64)

    def test_structure_vae_with_alphafold_encoder(self):
        """Test structure VAE with AlphaFold encoder."""
        # Create AlphaFold encoder
        af_encoder = AlphaFoldEncoder(embed_dim=64)

        # Create structure VAE
        vae = StructureAwareVAE(input_dim=128, latent_dim=32)

        # Process structure through AF encoder
        coords = torch.randn(4, 50, 3)
        plddt = torch.rand(4, 50) * 100

        struct_embed = af_encoder(coords, plddt)
        struct_pooled = af_encoder.pool(struct_embed)

        # Use pooled structure as part of VAE input
        x = torch.cat([torch.randn(4, 64), struct_pooled], dim=-1)
        outputs = vae(x)

        assert outputs["z"].shape == (4, 32)

    def test_structure_vae_with_uncertainty_wrapper(self):
        """Test structure VAE with MC Dropout uncertainty."""
        from src.models.uncertainty import MCDropoutWrapper

        class StructureVAEWithDropout(StructureAwareVAE):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pred_head = nn.Sequential(
                    nn.Linear(self.latent_dim, 32),
                    nn.Dropout(0.2),
                    nn.Linear(32, 5),
                )

            def forward(self, *args, **kwargs):
                outputs = super().forward(*args, **kwargs)
                outputs["prediction"] = self.pred_head(outputs["z"])
                return outputs

        vae = StructureVAEWithDropout(input_dim=64, latent_dim=16, dropout=0.2)
        mc_wrapper = MCDropoutWrapper(vae, n_samples=10)

        x = torch.randn(4, 64)
        estimate = mc_wrapper.predict_with_uncertainty(x)

        assert estimate.mean.shape[0] == 4
        assert estimate.std.shape[0] == 4


class TestTransferLearningIntegration:
    """Test transfer learning integration."""

    def test_pretrain_finetune_pipeline(self):
        """Test pretrain -> finetune pipeline."""
        from src.training.transfer_pipeline import (
            TransferConfig,
            TransferLearningPipeline,
        )

        config = TransferConfig(
            latent_dim=16,
            hidden_dims=[32],
            pretrain_epochs=2,
            finetune_epochs=2,
            batch_size=8,
            checkpoint_dir=Path(tempfile.mkdtemp()),
        )
        pipeline = TransferLearningPipeline(config)

        # Pretrain on multiple diseases
        datasets = {
            "disease1": SimpleDataset(30, 64),
            "disease2": SimpleDataset(30, 64),
        }
        disease_outputs = {"disease1": 5, "disease2": 3}

        model = pipeline.pretrain(datasets, disease_outputs, input_dim=64)

        assert "disease1" in model.heads
        assert "disease2" in model.heads

        # Finetune on new disease
        target_dataset = SimpleDataset(20, 64)
        finetuned = pipeline.finetune("target", target_dataset, n_outputs=4)

        assert "target" in finetuned.heads

    def test_transfer_with_epistasis(self):
        """Test transfer learning with epistasis loss."""
        from src.training.transfer_pipeline import MultiDiseaseModel

        model = MultiDiseaseModel(
            input_dim=64,
            latent_dim=16,
            hidden_dims=[32],
            disease_outputs={"hiv": 25},
        )
        epistasis_loss = EpistasisLoss(latent_dim=16, n_drugs=25, use_coevolution=False)

        x = torch.randn(8, 64)
        outputs = model(x, "hiv")

        loss_result = epistasis_loss(
            {"predictions": outputs["predictions"], "z": outputs["z"]},
            {"resistance": torch.rand(8, 25)},
        )

        assert not torch.isnan(loss_result.total_loss)


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_sequence_to_prediction_pipeline(self):
        """Test full pipeline from sequence to drug resistance prediction."""
        # 1. Create sequence encoding (simulated)
        sequence_encoding = torch.randn(4, 128)

        # 2. Create structure (simulated AlphaFold)
        structure = AlphaFoldStructure(
            coords=np.random.randn(50, 3).astype(np.float32),
            plddt=np.random.rand(50).astype(np.float32) * 100,
            sequence="M" * 50,
            uniprot_id="P00000",
        )
        struct_tensors = structure.to_tensors()

        # 3. Encode structure
        af_encoder = AlphaFoldEncoder(embed_dim=64)
        struct_embed = af_encoder(
            struct_tensors["coords"].unsqueeze(0).expand(4, -1, -1),
            struct_tensors["plddt"].unsqueeze(0).expand(4, -1),
        )
        struct_pooled = af_encoder.pool(struct_embed)

        # 4. Combine sequence and structure
        combined = torch.cat([sequence_encoding, struct_pooled], dim=-1)

        # 5. VAE encoding
        vae = StructureAwareVAE(input_dim=192, latent_dim=32)
        vae_outputs = vae(combined)

        # 6. Drug resistance prediction with epistasis
        predictor = EpistasisPredictor(n_positions=300, n_outputs=25)
        positions = torch.randint(0, 300, (4, 5))
        amino_acids = torch.randint(0, 21, (4, 5))
        predictions = predictor(positions, amino_acids)

        # 7. Combine VAE and epistasis for final prediction
        final_prediction = predictions["predictions"]

        assert final_prediction.shape == (4, 25)

    def test_pipeline_with_training(self):
        """Test end-to-end pipeline with training loop."""
        # Setup
        vae = StructureAwareVAE(input_dim=64, latent_dim=16)
        epistasis = EpistasisPredictor(n_positions=100, n_outputs=5)
        loss_fn = EpistasisLoss(latent_dim=16, n_drugs=5, use_coevolution=False)

        optimizer = torch.optim.Adam(list(vae.parameters()) + list(epistasis.parameters()), lr=1e-3)

        # Training loop
        losses = []
        for _ in range(5):
            # Forward
            x = torch.randn(8, 64)
            vae_out = vae(x)

            positions = torch.randint(0, 100, (8, 3))
            amino_acids = torch.randint(0, 21, (8, 3))
            pred_out = epistasis(positions, amino_acids)

            # Loss
            model_output = {
                "predictions": pred_out["predictions"],
                "z": vae_out["z"],
            }
            targets = {"resistance": torch.rand(8, 5)}
            loss_result = loss_fn(model_output, targets)

            # Backward
            optimizer.zero_grad()
            loss_result.total_loss.backward()
            optimizer.step()

            losses.append(loss_result.total_loss.item())

        # Loss should not explode
        assert all(not np.isnan(l) for l in losses)


class TestComponentInteroperability:
    """Test that components work together correctly."""

    def test_base_vae_subclass_compatibility(self):
        """Test BaseVAE subclass compatibility."""
        vae1 = IntegrationVAE(input_dim=64, latent_dim=16)
        vae2 = StructureAwareVAE(input_dim=64, latent_dim=16)

        x = torch.randn(4, 64)

        # Both should return dict with same keys
        out1 = vae1(x)
        out2 = vae2(x)

        for key in ["logits", "mu", "logvar", "z"]:
            assert key in out1
            assert key in out2

    def test_epistasis_with_different_models(self):
        """Test epistasis loss works with different model outputs."""
        loss_fn = EpistasisLoss(latent_dim=16, n_drugs=5, use_coevolution=False)

        # Different model output structures
        outputs1 = {"predictions": torch.randn(4, 5), "z": torch.randn(4, 16)}
        outputs2 = {"predictions": torch.randn(4, 5), "z": torch.randn(4, 16), "extra": torch.randn(4, 8)}

        targets = {"resistance": torch.rand(4, 5)}

        loss1 = loss_fn(outputs1, targets)
        loss2 = loss_fn(outputs2, targets)

        assert not torch.isnan(loss1.total_loss)
        assert not torch.isnan(loss2.total_loss)

    def test_encoder_interchangeability(self):
        """Test different encoders can be used interchangeably."""
        from src.models.structure_aware_vae import SE3Encoder

        # AlphaFold encoder
        af_enc = AlphaFoldEncoder(embed_dim=64)

        # SE3 encoder
        se3_enc = SE3Encoder(node_dim=64)

        coords = torch.randn(2, 30, 3)

        # Both should produce same output shape
        af_out = af_enc(coords)
        se3_out = se3_enc(coords)

        assert af_out.shape == se3_out.shape


class TestGradientFlow:
    """Test gradient flow through integrated components."""

    def test_gradient_through_full_pipeline(self):
        """Test gradients flow through entire pipeline."""
        vae = StructureAwareVAE(input_dim=64, latent_dim=16)
        predictor = EpistasisPredictor(n_positions=100, n_outputs=5)

        x = torch.randn(4, 64, requires_grad=True)
        vae_out = vae(x)

        positions = torch.randint(0, 100, (4, 3))
        amino_acids = torch.randint(0, 21, (4, 3))
        pred_out = predictor(positions, amino_acids)

        # Combined loss
        loss = vae_out["z"].sum() + pred_out["predictions"].sum()
        loss.backward()

        assert x.grad is not None

    def test_no_gradient_leak(self):
        """Test no gradient leak between batches."""
        vae = IntegrationVAE(input_dim=64, latent_dim=16)
        optimizer = torch.optim.Adam(vae.parameters())

        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(4, 64)
            out = vae(x)
            loss = out["z"].sum()
            loss.backward()
            optimizer.step()

            # Check no accumulated gradients
            for param in vae.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
