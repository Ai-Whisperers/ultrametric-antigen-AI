import torch

from src.data.autoimmunity import AutoimmunityLoader


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
