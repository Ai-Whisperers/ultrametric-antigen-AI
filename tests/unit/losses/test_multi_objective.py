import pytest
import torch

# Skip all tests if module not available
pytest.importorskip("src.training.optimizers.multi_objective")

from src.training.optimizers.multi_objective import ParetoFrontOptimizer


def test_pareto_dominance():
    optimizer = ParetoFrontOptimizer()

    # Population scores (Minimize both)
    # A: [1, 1]
    # B: [2, 2] -> Dominated by A
    # C: [0.5, 3] -> Not dominated by A, not dominated by B
    population = torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])

    # Test candidate dominated by A
    cand_bad = torch.tensor([1.5, 1.5])
    assert optimizer.is_dominated(cand_bad, population) is True

    # Test candidate dominating A (should NOT be dominated by pop)
    cand_good = torch.tensor([0.9, 0.9])
    assert optimizer.is_dominated(cand_good, population) is False

    # Test candidate on frontier (not dominated)
    cand_mix = torch.tensor([0.4, 4.0])
    assert optimizer.is_dominated(cand_mix, population) is False


def test_identify_front():
    optimizer = ParetoFrontOptimizer()

    candidates = torch.arange(4).unsqueeze(1)  # IDs 0, 1, 2, 3

    # Scores:
    # 0: [1, 10] (Front)
    # 1: [2, 11] (Dominated by 0)
    # 2: [10, 1] (Front)
    # 3: [5, 5] (Dominated by nothing? No, dominated by nothing here. Wait.
    #    [1, 10] vs [5, 5] -> 1<5, 10>5. No.
    #    [10, 1] vs [5, 5] -> 10>5, 1<5. No.
    #    So 0, 2, 3 are front. 1 is dominated by 0.

    scores = torch.tensor([[1.0, 10.0], [2.0, 11.0], [10.0, 1.0], [5.0, 5.0]])

    front_cands, front_scores = optimizer.identify_pareto_front(candidates, scores)

    assert len(front_cands) == 3
    # Check that index 1 (value 2.0, 11.0) is NOT in front
    # candidates are [[0], [1], [2], [3]]

    front_indices = front_cands.flatten().tolist()
    assert 1 not in front_indices
    assert 0 in front_indices
    assert 2 in front_indices
    assert 3 in front_indices
