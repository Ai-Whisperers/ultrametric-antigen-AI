#!/usr/bin/env python3
"""Test script for AMP Mechanism-Based Design API.

Validates:
1. MechanismDesignService works correctly
2. FastAPI endpoint models are properly structured
3. Integration with peptide_encoder_service works
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths
API_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = API_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR.parent.parent.parent))

from deliverables.partners.carlos_brizuela.api.mechanism_service import (
    get_mechanism_service,
    MECHANISM_FEATURES,
    ARROW_FLIP_THRESHOLDS,
    PATHOGEN_PROPERTIES,
)


def test_mechanism_service():
    """Test the MechanismDesignService."""
    print("=" * 70)
    print("TESTING MechanismDesignService")
    print("=" * 70)

    service = get_mechanism_service()

    # Test 1: Mechanism classification
    print("\n1. Mechanism Classification")
    print("-" * 50)

    test_cases = [
        # (length, hydro, charge) -> expected mechanism
        (14, 0.29, 1.6, "detergent"),  # cluster 3
        (22, 0.01, 3.6, "barrel_stave"),  # cluster 1/4
        (12, -0.5, 2.0, "carpet"),  # high charge
        (10, 0.0, 1.5, "detergent"),  # short
    ]

    for length, hydro, charge, expected in test_cases:
        result = service.classify_mechanism(length, hydro, charge)
        status = "PASS" if result["mechanism"] == expected else "FAIL"
        print(f"  {status}: ({length:2}, {hydro:+.2f}, {charge:+.1f}) -> {result['mechanism']:<12} (conf={result['confidence']:.2f})")

    # Test 2: Regime routing
    print("\n2. Regime Routing (Arrow-Flip Threshold)")
    print("-" * 50)

    test_hydros = [0.05, 0.107, 0.15, 0.30]
    for hydro in test_hydros:
        result = service.route_regime(hydro)
        print(f"  hydro={hydro:+.3f} -> {result['regime']:<20} (separation={result['expected_separation']:.3f})")

    # Test 3: Design rules per pathogen
    print("\n3. Design Rules by Pathogen")
    print("-" * 50)

    for pathogen in ["P_aeruginosa", "S_aureus", "Enterobacteriaceae", "A_baumannii"]:
        result = service.get_design_rules(pathogen)
        conf = result.get("confidence", "N/A")
        length = result.get("recommended_length", "N/A")
        mech = result.get("recommended_mechanism", ["N/A"])
        warn = result.get("warning")
        print(f"  {pathogen:20} | conf={conf:<6} | len={length or 'N/A':10} | mech={mech}")
        if warn:
            print(f"      WARNING: {warn}")

    # Test 4: Pathogen ranking for a sequence
    print("\n4. Pathogen Ranking for Sequence")
    print("-" * 50)

    test_sequences = [
        "KKLFKKILKYL",  # Short, cationic -> detergent
        "GIGKFLHSAKKFGKAFVGEIMNS",  # Long -> barrel_stave
    ]

    for seq in test_sequences:
        result = service.rank_pathogens(seq)
        print(f"\n  Sequence: {seq}")
        print(f"  Mechanism: {result['mechanism']} (cluster {result['cluster_id']})")
        print(f"  Hydrophobicity: {result['hydrophobicity']:.3f}")
        print("  Ranking:")
        for r in result["pathogen_ranking"][:3]:
            print(f"    {r['pathogen']:20} | efficacy={r['relative_efficacy']:.2f} | conf={r['confidence']}")

    # Test 5: Arrow-flip thresholds
    print("\n5. Arrow-Flip Thresholds")
    print("-" * 50)

    thresholds = service.get_thresholds()
    for name, info in thresholds["arrow_flip_thresholds"].items():
        sig = "***" if info["significant"] else ""
        print(f"  {name:15} | value={info['value']:.3f} | improvement={info['improvement']:.3f} {sig}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


def test_api_models():
    """Test that Pydantic models are properly defined."""
    print("\n" + "=" * 70)
    print("TESTING Pydantic Models")
    print("=" * 70)

    from deliverables.partners.carlos_brizuela.api.models import (
        MechanismClassifyRequest,
        MechanismClassifyResponse,
        RegimeRouteRequest,
        DesignRulesRequest,
        PathogenRankRequest,
    )

    # Test request models
    print("\n1. Request Model Validation")
    print("-" * 50)

    try:
        req = MechanismClassifyRequest(length=14, hydrophobicity=0.29, net_charge=1.6)
        print(f"  PASS: MechanismClassifyRequest -> {req.length}, {req.hydrophobicity}")
    except Exception as e:
        print(f"  FAIL: MechanismClassifyRequest -> {e}")

    try:
        req = RegimeRouteRequest(hydrophobicity=0.15)
        print(f"  PASS: RegimeRouteRequest -> {req.hydrophobicity}")
    except Exception as e:
        print(f"  FAIL: RegimeRouteRequest -> {e}")

    try:
        req = DesignRulesRequest(target_pathogen="P_aeruginosa")
        print(f"  PASS: DesignRulesRequest -> {req.target_pathogen}")
    except Exception as e:
        print(f"  FAIL: DesignRulesRequest -> {e}")

    try:
        req = PathogenRankRequest(sequence="KKLFKKILKYL")
        print(f"  PASS: PathogenRankRequest -> {req.sequence}")
    except Exception as e:
        print(f"  FAIL: PathogenRankRequest -> {e}")

    # Test validation boundaries
    print("\n2. Validation Boundary Tests")
    print("-" * 50)

    # Should fail: length out of range
    try:
        req = MechanismClassifyRequest(length=60, hydrophobicity=0.0, net_charge=0)
        print("  FAIL: Length=60 should be rejected")
    except Exception:
        print("  PASS: Length=60 correctly rejected (max 50)")

    # Should fail: sequence too short
    try:
        req = PathogenRankRequest(sequence="KK")
        print("  FAIL: 2-AA sequence should be rejected")
    except Exception:
        print("  PASS: 2-AA sequence correctly rejected (min 5)")

    print("\n" + "=" * 70)
    print("MODEL TESTS COMPLETED")
    print("=" * 70)


def test_peptide_encoder_integration():
    """Test integration with peptide_encoder_service."""
    print("\n" + "=" * 70)
    print("TESTING Peptide Encoder Integration")
    print("=" * 70)

    try:
        from deliverables.partners.carlos_brizuela.src.peptide_encoder_service import (
            get_peptide_encoder_service,
        )

        service = get_peptide_encoder_service()

        test_seq = "KKLFKKILKYL"
        print(f"\nTest sequence: {test_seq}")

        # Test mechanism-aware methods
        print("\n1. get_mechanism()")
        mech = service.get_mechanism(test_seq)
        print(f"   Mechanism: {mech['mechanism']}")
        print(f"   Confidence: {mech['confidence']:.2f}")
        print(f"   Cluster: {mech['cluster_id']}")

        print("\n2. get_pathogen_ranking()")
        ranking = service.get_pathogen_ranking(test_seq)
        for r in ranking[:3]:
            print(f"   {r['pathogen']:20} | {r['relative_efficacy']:.2f}")

        print("\n3. get_design_rules('P_aeruginosa')")
        rules = service.get_design_rules("P_aeruginosa")
        print(f"   Length: {rules.get('recommended_length')}")
        print(f"   Mechanism: {rules.get('recommended_mechanism')}")
        print(f"   Confidence: {rules.get('confidence')}")

        print("\n4. route_regime()")
        regime = service.route_regime(test_seq)
        print(f"   Regime: {regime['regime']}")
        print(f"   Threshold: {regime['threshold_used']}")

        print("\n" + "=" * 70)
        print("INTEGRATION TESTS COMPLETED")
        print("=" * 70)

    except ImportError as e:
        print(f"\nWARNING: Could not import peptide_encoder_service: {e}")
        print("Integration test skipped.")


if __name__ == "__main__":
    test_mechanism_service()
    test_api_models()
    test_peptide_encoder_integration()
