import sys
from pathlib import Path

try:
    from src.models.ternary_vae import TernaryVAE

    print("SUCCESS: src imported successfully")
except ImportError as e:
    print(f"FAILURE: {e}")
    print(f"Current Path: {sys.path}")
