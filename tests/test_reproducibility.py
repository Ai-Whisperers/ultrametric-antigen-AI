"""Test reproducibility of Ternary VAE v5.5."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ternary_vae_v5_5 import DualNeuralVAEV5
from src.utils.data import generate_all_ternary_operations, sample_operations


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestReproducibility:
    """Test suite for reproducibility."""

    @pytest.fixture
    def model(self):
        """Create a model instance."""
        set_seed(42)
        model = DualNeuralVAEV5(
            input_dim=9,
            latent_dim=16,
            rho_min=0.1,
            rho_max=0.7,
            lambda3_base=0.3,
            lambda3_amplitude=0.15,
            eps_kl=0.0005
        )
        return model

    def test_weight_initialization(self):
        """Test that weight initialization is deterministic."""
        set_seed(42)
        model1 = DualNeuralVAEV5(input_dim=9, latent_dim=16)

        set_seed(42)
        model2 = DualNeuralVAEV5(input_dim=9, latent_dim=16)

        # Compare weights
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
            assert torch.allclose(param1, param2), f"Weights differ for {name1}"

    def test_forward_determinism(self, model):
        """Test that forward pass is deterministic."""
        set_seed(42)
        x = torch.randn(10, 9)

        # First forward pass
        set_seed(100)
        out1 = model(x, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)

        # Second forward pass with same seed
        set_seed(100)
        out2 = model(x, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)

        # Compare outputs
        assert torch.allclose(out1['logits_A'], out2['logits_A']), "VAE-A logits differ"
        assert torch.allclose(out1['logits_B'], out2['logits_B']), "VAE-B logits differ"
        assert torch.allclose(out1['z_A'], out2['z_A']), "VAE-A latent codes differ"
        assert torch.allclose(out1['z_B'], out2['z_B']), "VAE-B latent codes differ"

    def test_sampling_determinism(self, model):
        """Test that sampling is deterministic."""
        model.eval()

        # First sampling
        set_seed(42)
        samples1_A = model.sample(100, 'cpu', 'A')

        set_seed(42)
        samples1_B = model.sample(100, 'cpu', 'B')

        # Second sampling with same seed
        set_seed(42)
        samples2_A = model.sample(100, 'cpu', 'A')

        set_seed(42)
        samples2_B = model.sample(100, 'cpu', 'B')

        # Compare
        assert torch.allclose(samples1_A, samples2_A), "VAE-A samples differ"
        assert torch.allclose(samples1_B, samples2_B), "VAE-B samples differ"

    def test_training_step_determinism(self, model):
        """Test that a training step is deterministic."""
        set_seed(42)
        x = torch.FloatTensor(sample_operations(32, replacement=True, seed=42))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # First training step
        set_seed(100)
        model.train()
        out1 = model(x, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)
        loss1 = model.loss_function(x, out1)
        optimizer.zero_grad()
        loss1['loss'].backward()
        optimizer.step()

        # Reset model
        set_seed(42)
        model2 = DualNeuralVAEV5(input_dim=9, latent_dim=16)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

        # Second training step with same seed
        set_seed(100)
        model2.train()
        out2 = model2(x, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)
        loss2 = model2.loss_function(x, out2)
        optimizer2.zero_grad()
        loss2['loss'].backward()
        optimizer2.step()

        # Compare losses
        assert torch.allclose(loss1['loss'], loss2['loss']), "Losses differ"

        # Compare updated weights
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(param1, param2, atol=1e-6), f"Updated weights differ for {name1}"

    def test_data_generation_determinism(self):
        """Test that data generation is deterministic."""
        # Generate twice
        data1 = generate_all_ternary_operations()
        data2 = generate_all_ternary_operations()

        # Should be identical (no randomness)
        assert np.array_equal(data1, data2), "Data generation is not deterministic"

        # Check size
        assert data1.shape == (19683, 9), f"Expected shape (19683, 9), got {data1.shape}"

    def test_checkpoint_reproducibility(self, model, tmp_path):
        """Test that saving and loading checkpoints preserves state."""
        # Generate some data
        x = torch.randn(10, 9)

        # Forward pass before save
        model.eval()
        with torch.no_grad():
            out_before = model(x, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save({'model': model.state_dict()}, checkpoint_path)

        # Create new model and load
        model2 = DualNeuralVAEV5(input_dim=9, latent_dim=16)
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['model'])
        model2.eval()

        # Forward pass after load
        with torch.no_grad():
            out_after = model2(x, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)

        # Compare
        assert torch.allclose(out_before['logits_A'], out_after['logits_A']), "Logits differ after checkpoint load"
        assert torch.allclose(out_before['logits_B'], out_after['logits_B']), "Logits differ after checkpoint load"

    def test_phase_schedule_determinism(self, model):
        """Test that phase scheduling is deterministic."""
        # Phase 1
        rho1 = model.compute_phase_scheduled_rho(20, phase_4_start=250)
        rho2 = model.compute_phase_scheduled_rho(20, phase_4_start=250)
        assert rho1 == rho2, "Phase 1 rho is not deterministic"

        # Phase 2
        rho1 = model.compute_phase_scheduled_rho(60, phase_4_start=250)
        rho2 = model.compute_phase_scheduled_rho(60, phase_4_start=250)
        assert rho1 == rho2, "Phase 2 rho is not deterministic"

        # Phase 3
        model.grad_balance_achieved = True
        rho1 = model.compute_phase_scheduled_rho(150, phase_4_start=250)
        rho2 = model.compute_phase_scheduled_rho(150, phase_4_start=250)
        assert rho1 == rho2, "Phase 3 rho is not deterministic"

        # Phase 4
        rho1 = model.compute_phase_scheduled_rho(300, phase_4_start=250)
        rho2 = model.compute_phase_scheduled_rho(300, phase_4_start=250)
        assert rho1 == rho2, "Phase 4 rho is not deterministic"

    def test_cyclic_lambda_determinism(self, model):
        """Test that cyclic lambda is deterministic."""
        lambda1 = model.compute_cyclic_lambda3(50, period=30)
        lambda2 = model.compute_cyclic_lambda3(50, period=30)
        assert lambda1 == lambda2, "Cyclic lambda is not deterministic"

    def test_gradient_norm_tracking_determinism(self):
        """Test that gradient norm tracking is deterministic."""
        # First run
        set_seed(42)
        model1 = DualNeuralVAEV5(input_dim=9, latent_dim=16)
        model1.train()
        x1 = torch.FloatTensor(sample_operations(32, replacement=True, seed=100))

        set_seed(200)
        out1 = model1(x1, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)
        loss1 = model1.loss_function(x1, out1)
        loss1['loss'].backward()
        model1.update_gradient_norms()

        # Second run with same seeds
        set_seed(42)
        model2 = DualNeuralVAEV5(input_dim=9, latent_dim=16)
        model2.train()
        x2 = torch.FloatTensor(sample_operations(32, replacement=True, seed=100))

        set_seed(200)
        out2 = model2(x2, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)
        loss2 = model2.loss_function(x2, out2)
        loss2['loss'].backward()
        model2.update_gradient_norms()

        # Compare
        assert torch.allclose(model1.grad_norm_A_ema, model2.grad_norm_A_ema), "Gradient norm A EMA differs"
        assert torch.allclose(model1.grad_norm_B_ema, model2.grad_norm_B_ema), "Gradient norm B EMA differs"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
