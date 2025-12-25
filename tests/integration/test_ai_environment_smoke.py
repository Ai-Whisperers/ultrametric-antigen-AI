import pytest
import torch
from src.models.ternary_vae import TernaryVAEV5_11


def test_environment_import_and_instantiation():
    """
    Smoke test to verify:
    1. PYTHONPATH is correctly set (can import src).
    2. Model definitions are valid (can access TernaryVAEV5_11).
    3. Dependencies (geoopt, torch) are installed.
    """
    try:
        model = TernaryVAEV5_11(latent_dim=16)
        assert model is not None
        print("\n[Smoke Test] Model instantiated successfully.")
    except Exception as e:
        pytest.fail(f"Environment check failed: {e}")


if __name__ == "__main__":
    test_environment_import_and_instantiation()
