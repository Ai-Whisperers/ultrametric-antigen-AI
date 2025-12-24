"""Mathematical Proofs Verification Script.

Implements the "Mathematical Stress Tests" defined in
DOCUMENTATION/01_STAKEHOLDER_RESOURCES/validation_suite/02_MATHEMATICAL_STRESS_TESTS.md

Validates:
1. Delta-Hyperbolicity (Gromov 4-point condition)
2. Ultrametricity Score (Isosceles triangle condition)
3. Zero-Structure / P-Adic Valuation (via analyze_zero_structure logic)
"""

import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import TernaryVAEV5_11
from src.data.generation import generate_all_ternary_operations
from src.analysis.geometry import (
    compute_delta_hyperbolicity,
    compute_ultrametricity_score,
)

# Import the existing specific zero analysis logic
try:
    from scripts.analysis.analyze_zero_structure import (
        analyze_checkpoint as analyze_zeros_internal,
    )
except ImportError:
    # If import fails (path issues), we'll reimplement the light version here
    analyze_zeros_internal = None


def verify_proofs(
    checkpoint_path: str = None,
    model_config: Dict[str, Any] = None,
    device: str = "cuda",
):
    print(f"\n{'='*60}")
    print("MATHEMATICAL PROOFS VERIFICATION")
    print(f"{'='*60}\n")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load or Initialize Model
    if model_config is None:
        model_config = {
            "latent_dim": 16,
            "hidden_dim": 64,
            "use_dual_projection": True,
            "use_controller": True,
            "curvature": -1.0,
            # Note: Model definition expects curvature as float magnitude usually,
            # or check implementation. V5.11 init takes 'curvature' float.
            # Usually we pass positive curvature parameter c where K = -c^2 or similar.
            # Default to 1.0 (K=-1).
        }

    model = TernaryVAEV5_11(**model_config)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        # Use simple load for now, assuming standard save format
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Strict processing failed ({e}), trying loose...")
            pass
    else:
        print("No checkpoint found/provided. Using random initialization.")
        print(
            "(Note: Random models are unlikely to satisfy proofs unless architecture enforces them)"
        )

    model.to(device)
    model.eval()

    # 2. generating Data
    print("Generating complete ternary operation set (3^9 = 19683)...")
    ops = generate_all_ternary_operations()
    ops_tensor = torch.tensor(ops, dtype=torch.float32, device=device)

    # 3. Compute Embeddings
    print("Computing hyperbolic embeddings...")
    with torch.no_grad():
        outputs = model(ops_tensor)
        z_A = outputs["z_A_hyp"]  # (N, D)
        outputs["z_B_hyp"]

        # We can analyze both or just A. Proofs usually target the "primary" manifold.
        # Let's check A.

    embeddings = z_A
    print(f"Embedding shape: {embeddings.shape}")

    # 4. Verify Delta-Hyperbolicity
    print("\n[Test 1] Delta-Hyperbolicity (Target < 0.1)")
    delta = compute_delta_hyperbolicity(embeddings, sample_size=200)
    print(f"  Result: delta = {delta:.4f}")
    if delta < 0.1:
        print("  Status: PASS ✅")
    elif delta < 0.5:
        print("  Status: MARGINAL ⚠️")
    else:
        print("  Status: FAIL ❌")

    # 5. Verify Ultrametricity
    print("\n[Test 2] Ultrametricity Score (Target > 0.99)")
    ultra_score = compute_ultrametricity_score(embeddings, sample_size=200)
    print(f"  Result: score = {ultra_score:.4f}")
    if ultra_score > 0.99:
        print("  Status: PASS ✅")
    elif ultra_score > 0.90:
        print("  Status: MARGINAL ⚠️")
    else:
        print("  Status: FAIL ❌")

    # 6. Zero Structure (P-adic)
    # We can calculate correlation quickly here
    print("\n[Test 3] P-adic Valuation Correlation")

    # Calculate valuation (trailing zeros)
    def valuation(op):
        cnt = 0
        for x in op:
            if x == 0:
                cnt += 1
            else:
                break
        return cnt

    vals = np.array([valuation(op) for op in ops])
    radii = torch.norm(embeddings, dim=1).cpu().numpy()

    # Check correlation
    from scipy import stats

    corr, p_val = stats.pearsonr(vals, radii)

    print(f"  Correlation (Valuation vs Radius): r={corr:.4f}")
    print(
        "  Expected: Negative correlation (higher valuation = closer to origin = smaller radius)"
    )

    if corr < -0.5:
        print("  Status: PASS ✅ (Strong p-adic structure)")
    elif corr < -0.1:
        print("  Status: MARGINAL ⚠️ (Weak p-adic structure)")
    else:
        print("  Status: FAIL ❌ (No/Inverted p-adic structure)")

    # Return summary
    return {"delta": delta, "ultrametricity": ultra_score, "padic_correlation": corr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--dim", type=int, default=16, help="Latent dimension")
    args = parser.parse_args()

    verify_proofs(
        checkpoint_path=args.checkpoint, model_config={"latent_dim": args.dim}
    )
