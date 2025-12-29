# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for HyboWaveNet model.

Tests cover:
- Node-level encoding
- Graph-level encoding
- Multi-scale processing
- Attention aggregation
- Gradient flow
- Edge cases
"""

from __future__ import annotations

import pytest
import torch

from src.graphs import HyboWaveNet


class TestHyboWaveNetInit:
    """Tests for HyboWaveNet initialization."""

    def test_default_init(self):
        """Test default initialization."""
        model = HyboWaveNet(
            in_channels=16,
            hidden_channels=32,
            out_channels=8,
        )

        assert model.in_channels == 16
        assert model.hidden_channels == 32
        assert model.out_channels == 8
        assert model.n_scales == 4
        assert model.curvature == 1.0

    def test_custom_scales(self):
        """Test initialization with custom scales."""
        model = HyboWaveNet(
            in_channels=16,
            hidden_channels=32,
            out_channels=8,
            n_scales=6,
        )

        assert model.n_scales == 6
        assert len(model.scale_encoders) == 6
        assert len(model.scale_gnns) == 6

    def test_custom_layers(self):
        """Test initialization with custom layer count."""
        model = HyboWaveNet(
            in_channels=16,
            hidden_channels=32,
            out_channels=8,
            n_layers=4,
        )

        # Each scale should have 4 GNN layers
        for scale_gnns in model.scale_gnns:
            assert len(scale_gnns) == 4

    def test_attention_disabled(self):
        """Test initialization without attention."""
        model = HyboWaveNet(
            in_channels=16,
            hidden_channels=32,
            out_channels=8,
            use_attention=False,
        )

        assert model.use_attention is False
        assert not hasattr(model, "scale_attention") or model.scale_attention is None


class TestHyboWaveNetForward:
    """Tests for HyboWaveNet forward pass."""

    def test_output_shape(self, hybowave_net, small_graph):
        """Test output has correct shape."""
        model = hybowave_net.to(small_graph["x"].device)
        output = model(small_graph["x"], small_graph["edge_index"])

        assert output.shape == (small_graph["n_nodes"], 8)

    def test_output_in_ball(self, hybowave_net, small_graph):
        """Test output stays inside Poincare ball."""
        model = hybowave_net.to(small_graph["x"].device)
        output = model(small_graph["x"], small_graph["edge_index"])

        norms = output.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_output_finite(self, hybowave_net, small_graph):
        """Test output is always finite."""
        model = hybowave_net.to(small_graph["x"].device)
        output = model(small_graph["x"], small_graph["edge_index"])

        assert torch.isfinite(output).all()

    def test_deterministic_in_eval(self, hybowave_net, small_graph):
        """Test deterministic output in eval mode."""
        model = hybowave_net.to(small_graph["x"].device)
        model.eval()

        with torch.no_grad():
            output1 = model(small_graph["x"], small_graph["edge_index"])
            output2 = model(small_graph["x"], small_graph["edge_index"])

        assert torch.allclose(output1, output2)


class TestHyboWaveNetGraphEncoding:
    """Tests for graph-level encoding."""

    def test_encode_graph_shape(self, hybowave_net, small_graph):
        """Test graph encoding has correct shape."""
        model = hybowave_net.to(small_graph["x"].device)
        graph_emb = model.encode_graph(small_graph["x"], small_graph["edge_index"])

        assert graph_emb.shape == (1, 8)

    def test_encode_graph_in_ball(self, hybowave_net, small_graph):
        """Test graph embedding stays inside Poincare ball."""
        model = hybowave_net.to(small_graph["x"].device)
        graph_emb = model.encode_graph(small_graph["x"], small_graph["edge_index"])

        norm = graph_emb.norm()
        assert norm < 1.0

    def test_encode_graph_finite(self, hybowave_net, small_graph):
        """Test graph embedding is finite."""
        model = hybowave_net.to(small_graph["x"].device)
        graph_emb = model.encode_graph(small_graph["x"], small_graph["edge_index"])

        assert torch.isfinite(graph_emb).all()

    def test_encode_batched_graphs(self, hybowave_net, batched_graphs):
        """Test encoding multiple batched graphs."""
        model = hybowave_net.to(batched_graphs["x"].device)
        graph_embs = model.encode_graph(
            batched_graphs["x"],
            batched_graphs["edge_index"],
            batch=batched_graphs["batch"],
        )

        assert graph_embs.shape == (batched_graphs["n_graphs"], 8)
        assert torch.isfinite(graph_embs).all()


class TestHyboWaveNetGradients:
    """Tests for gradient flow through HyboWaveNet."""

    def test_gradients_flow_to_input(self, hybowave_net, small_graph):
        """Test gradients flow to input features."""
        model = hybowave_net.to(small_graph["x"].device)

        x = small_graph["x"].clone().requires_grad_(True)
        output = model(x, small_graph["edge_index"])
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_parameters_have_gradients(self, hybowave_net, small_graph):
        """Test all parameters receive gradients."""
        model = hybowave_net.to(small_graph["x"].device)

        output = model(small_graph["x"], small_graph["edge_index"])
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_training_step(self, small_graph):
        """Test complete training step."""
        device = small_graph["x"].device
        model = HyboWaveNet(16, 32, 8, n_scales=2, n_layers=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_params = {name: p.clone() for name, p in model.named_parameters()}

        optimizer.zero_grad()
        output = model(small_graph["x"], small_graph["edge_index"])
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Check at least some parameters changed
        params_changed = False
        for name, p in model.named_parameters():
            if not torch.allclose(p, initial_params[name]):
                params_changed = True
                break

        assert params_changed, "Some parameters should change after training step"

    def test_multiple_training_steps(self, small_graph):
        """Test multiple training steps maintain valid outputs."""
        device = small_graph["x"].device
        model = HyboWaveNet(16, 32, 8, n_scales=2, n_layers=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(10):
            optimizer.zero_grad()
            output = model(small_graph["x"], small_graph["edge_index"])
            loss = output.sum()
            loss.backward()
            optimizer.step()

        # Final output should still be valid
        with torch.no_grad():
            final_output = model(small_graph["x"], small_graph["edge_index"])
            assert torch.isfinite(final_output).all()
            assert (final_output.norm(dim=-1) < 1.0).all()


class TestHyboWaveNetEdgeCases:
    """Tests for edge cases."""

    def test_single_node(self, hybowave_net, device):
        """Test with single-node graph."""
        model = hybowave_net.to(device)

        x = torch.randn(1, 16, device=device) * 0.2
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        output = model(x, edge_index)

        assert output.shape == (1, 8)
        assert torch.isfinite(output).all()

    def test_disconnected_graph(self, hybowave_net, device):
        """Test with disconnected graph."""
        model = hybowave_net.to(device)

        # Two disconnected components
        x = torch.randn(6, 16, device=device) * 0.2
        edge_index = torch.tensor([[0, 1, 3, 4],
                                   [1, 0, 4, 3]], device=device)

        output = model(x, edge_index)

        assert output.shape == (6, 8)
        assert torch.isfinite(output).all()

    def test_sparse_graph(self, hybowave_net, device):
        """Test with very sparse graph."""
        model = hybowave_net.to(device)

        # 50 nodes with only 5 edges
        x = torch.randn(50, 16, device=device) * 0.2
        edge_index = torch.tensor([[0, 10, 20, 30, 40],
                                   [1, 11, 21, 31, 41]], device=device)

        output = model(x, edge_index)

        assert output.shape == (50, 8)
        assert torch.isfinite(output).all()


class TestHyboWaveNetConfigurations:
    """Tests for different configurations."""

    @pytest.mark.parametrize("n_scales", [1, 2, 4])
    def test_scale_configurations(self, n_scales, device):
        """Test different scale configurations."""
        model = HyboWaveNet(16, 32, 8, n_scales=n_scales).to(device)

        x = torch.randn(10, 16, device=device) * 0.2
        edge_index = torch.randint(0, 10, (2, 30), device=device)

        output = model(x, edge_index)

        assert output.shape == (10, 8)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_layer_configurations(self, n_layers, device):
        """Test different layer depths."""
        model = HyboWaveNet(16, 32, 8, n_layers=n_layers, n_scales=2).to(device)

        x = torch.randn(10, 16, device=device) * 0.2
        edge_index = torch.randint(0, 10, (2, 30), device=device)

        output = model(x, edge_index)

        assert output.shape == (10, 8)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("use_attention", [True, False])
    def test_attention_configurations(self, use_attention, device):
        """Test with and without attention."""
        model = HyboWaveNet(
            16, 32, 8,
            n_scales=2,
            use_attention=use_attention,
        ).to(device)

        x = torch.randn(10, 16, device=device) * 0.2
        edge_index = torch.randint(0, 10, (2, 30), device=device)

        output = model(x, edge_index)

        assert output.shape == (10, 8)
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0])
    def test_curvature_configurations(self, curvature, device):
        """Test different curvatures."""
        model = HyboWaveNet(16, 32, 8, curvature=curvature, n_scales=2).to(device)

        x = torch.randn(10, 16, device=device) * 0.2
        edge_index = torch.randint(0, 10, (2, 30), device=device)

        output = model(x, edge_index)

        assert output.shape == (10, 8)
        assert torch.isfinite(output).all()


class TestHyboWaveNetIntegration:
    """Integration tests."""

    def test_full_pipeline(self, device):
        """Test full training pipeline."""
        model = HyboWaveNet(32, 64, 16, n_scales=3, n_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Simulate training loop
        for epoch in range(5):
            # Generate random graph data
            n_nodes = 20
            x = torch.randn(n_nodes, 32, device=device) * 0.2
            edge_index = torch.randint(0, n_nodes, (2, 60), device=device)

            model.train()
            optimizer.zero_grad()

            # Forward pass
            node_emb = model(x, edge_index)
            graph_emb = model.encode_graph(x, edge_index)

            # Dummy loss
            loss = node_emb.sum() + graph_emb.sum()
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_x = torch.randn(15, 32, device=device) * 0.2
            test_edge = torch.randint(0, 15, (2, 40), device=device)

            node_out = model(test_x, test_edge)
            graph_out = model.encode_graph(test_x, test_edge)

            assert node_out.shape == (15, 16)
            assert graph_out.shape == (1, 16)
            assert torch.isfinite(node_out).all()
            assert torch.isfinite(graph_out).all()

    def test_serialization(self, hybowave_net, tmp_path, device):
        """Test model save/load."""
        model = hybowave_net.to(device)

        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        # Load into new model
        new_model = HyboWaveNet(16, 32, 8, n_scales=3, n_layers=2).to(device)
        new_model.load_state_dict(torch.load(save_path, weights_only=True))

        # Compare outputs
        x = torch.randn(5, 16, device=device) * 0.2
        edge_index = torch.randint(0, 5, (2, 10), device=device)

        model.eval()
        new_model.eval()

        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = new_model(x, edge_index)

        assert torch.allclose(out1, out2)
