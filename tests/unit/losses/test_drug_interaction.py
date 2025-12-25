import torch

from src.losses.drug_interaction import DrugInteractionPenalty


def test_drug_interaction_forward():
    # Batch of 4 pairs
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    interaction = torch.tensor([1, 1, 0, 0], dtype=torch.float32)

    criterion = DrugInteractionPenalty(margin=1.0)
    loss = criterion(z1, z2, interaction)

    assert loss.dim() == 0
    assert loss.item() >= 0


def test_contrastive_logic():
    # Helper to check if interacting pairs are pulled closer than non-interacting
    loss_fn = DrugInteractionPenalty(margin=10.0)

    # Case 1: Perfect interaction (d=0) -> Loss should be 0
    z_close = torch.randn(1, 4)
    loss_match = loss_fn(z_close, z_close, torch.tensor([1.0]))
    assert torch.isclose(loss_match, torch.tensor(0.0))

    # Case 2: Non-interacting but close (d=0) -> Loss should be margin^2
    loss_mismatch = loss_fn(z_close, z_close, torch.tensor([0.0]))
    assert torch.isclose(loss_mismatch, torch.tensor(100.0))
