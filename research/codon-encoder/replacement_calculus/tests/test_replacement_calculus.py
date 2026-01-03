#!/usr/bin/env python3
"""Test suite for Replacement Calculus framework.

Tests the core components:
1. Invariant computation (valuation, redundancy, symmetry)
2. Local minimum construction
3. Morphism validity
4. Groupoid construction and escape paths

Usage:
    python -m pytest tests/test_replacement_calculus.py
    # or
    python tests/test_replacement_calculus.py
"""

import sys
from pathlib import Path

# Add parent to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from replacement_calculus.invariants import (
    valuation,
    redundancy,
    symmetry_rank,
    invariant_tuple,
    InvariantTuple,
    compare_invariants,
)
from replacement_calculus.groups import LocalMinimum, Constraint, create_codon_local_minimum
from replacement_calculus.morphisms import (
    Morphism,
    MorphismType,
    is_valid_morphism,
    create_identity_morphism,
    compose_morphisms,
)
from replacement_calculus.groupoids import Groupoid, find_escape_path, analyze_groupoid_structure


# =============================================================================
# Test Invariants
# =============================================================================


def test_valuation():
    """Test p-adic valuation computation."""
    # 3-adic valuation
    assert valuation(9, 3) == 2   # 9 = 3^2
    assert valuation(27, 3) == 3  # 27 = 3^3
    assert valuation(6, 3) == 1   # 6 = 2 * 3
    assert valuation(5, 3) == 0   # 5 not divisible by 3
    assert valuation(0, 3) == 100  # Infinity

    # 2-adic valuation
    assert valuation(8, 2) == 3   # 8 = 2^3
    assert valuation(12, 2) == 2  # 12 = 4 * 3

    print("✓ Valuation tests passed")


def test_redundancy():
    """Test redundancy (coset index) computation."""
    equivalence = {
        "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],  # 6 codons
        "M": ["ATG"],  # 1 codon
        "W": ["TGG"],  # 1 codon
        "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],  # 6 codons
    }

    assert redundancy("L", equivalence) == 6
    assert redundancy("M", equivalence) == 1
    assert redundancy("X", equivalence) == 1  # Unknown defaults to 1

    print("✓ Redundancy tests passed")


def test_symmetry_rank():
    """Test symmetry rank computation."""
    # Fully symmetric matrix (identity-like) should have high symmetry
    symmetric = np.eye(4)
    σ_sym = symmetry_rank(symmetric)

    # Random matrix should have low symmetry
    np.random.seed(42)
    asymmetric = np.random.rand(4, 4)
    asymmetric = (asymmetric + asymmetric.T) / 2  # Make symmetric
    σ_asym = symmetry_rank(asymmetric)

    assert σ_sym >= 0
    assert σ_asym >= 0
    print(f"  Symmetric matrix σ = {σ_sym}")
    print(f"  Random matrix σ = {σ_asym}")

    print("✓ Symmetry rank tests passed")


def test_invariant_tuple_ordering():
    """Test partial ordering of invariant tuples."""
    I1 = InvariantTuple(valuation=2, redundancy=3, symmetry_rank=1)
    I2 = InvariantTuple(valuation=3, redundancy=4, symmetry_rank=2)
    I3 = InvariantTuple(valuation=1, redundancy=5, symmetry_rank=0)

    # I2 dominates I1 (all components greater)
    assert I2 >= I1
    assert I2 > I1
    assert compare_invariants(I2, I1) == "dominates"

    # I1 and I3 are incomparable
    assert not (I1 >= I3)
    assert not (I3 >= I1)
    assert compare_invariants(I1, I3) == "incomparable"

    print("✓ Invariant ordering tests passed")


# =============================================================================
# Test Groups
# =============================================================================


def test_local_minimum_creation():
    """Test LocalMinimum creation and properties."""
    minimum = LocalMinimum(
        name="test_group",
        generators=[0, 1, 2, 3],
        relations=[
            Constraint(
                name="sum_constraint",
                variables=(0, 1),
                predicate=lambda a, b: a + b < 10,
                strength=1.0,
            )
        ],
        center=np.array([0.5, 0.5]),
        members=[np.array([0.4, 0.5]), np.array([0.5, 0.6])],
    )

    assert minimum.n_generators == 4
    assert minimum.n_relations == 1
    assert not minimum.is_over_constrained
    assert minimum.constraint_ratio == 0.25

    print("✓ LocalMinimum creation tests passed")


def test_over_constrained_detection():
    """Test detection of over-constrained groups."""
    # Over-constrained: more relations than generators
    over_constrained = LocalMinimum(
        name="over",
        generators=[0, 1],
        relations=[
            Constraint("c1", (0,), lambda x: x > 0, 1.0),
            Constraint("c2", (1,), lambda x: x > 0, 1.0),
            Constraint("c3", (0, 1), lambda a, b: a < b, 1.0),
        ],
    )

    # Under-constrained: fewer relations than generators
    under_constrained = LocalMinimum(
        name="under",
        generators=[0, 1, 2, 3],
        relations=[Constraint("c1", (0,), lambda x: x > 0, 1.0)],
    )

    assert over_constrained.is_over_constrained
    assert not under_constrained.is_over_constrained

    print("✓ Over-constrained detection tests passed")


def test_codon_local_minimum():
    """Test creation of codon-based local minimum."""
    codon_to_index = {
        "TTA": 0, "TTG": 1, "CTT": 2, "CTC": 3, "CTA": 4, "CTG": 5,
        "ATG": 6, "TGG": 7,
    }

    leucine = create_codon_local_minimum(
        amino_acid="L",
        codons=["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
        codon_to_index=codon_to_index,
    )

    assert leucine.n_generators == 6
    assert leucine.metadata["amino_acid"] == "L"

    print("✓ Codon LocalMinimum tests passed")


# =============================================================================
# Test Morphisms
# =============================================================================


def test_morphism_validity():
    """Test morphism validity checking."""
    source = LocalMinimum(
        name="source",
        generators=[3, 6, 9],  # All divisible by 3
        members=[np.array([0.1, 0.1]) for _ in range(3)],
    )

    # Valid target: higher valuations
    valid_target = LocalMinimum(
        name="valid_target",
        generators=[9, 18, 27],  # Higher 3-adic valuations
        members=[np.array([0.2, 0.2]) for _ in range(3)],
    )

    # Invalid target: lower valuations
    invalid_target = LocalMinimum(
        name="invalid_target",
        generators=[1, 2, 4],  # Not divisible by 3
        members=[np.array([0.2, 0.2]) for _ in range(3)],
    )

    # Valid morphism
    valid_morph = Morphism(
        source=source,
        target=valid_target,
        map_function=lambda x: x * 3,  # Multiply by 3
    )

    is_valid, reason = is_valid_morphism(valid_morph, p=3)
    print(f"  Valid morphism: {is_valid} - {reason}")

    # Invalid morphism (valuation decreases)
    invalid_morph = Morphism(
        source=source,
        target=invalid_target,
        map_function=lambda x: x // 3 + 1,  # Decrease valuation
    )

    is_invalid, reason = is_valid_morphism(invalid_morph, p=3)
    print(f"  Invalid morphism: {is_invalid} - {reason}")

    print("✓ Morphism validity tests passed")


def test_morphism_composition():
    """Test morphism composition."""
    a = LocalMinimum(name="a", generators=[1, 2], members=[np.zeros(2)])
    b = LocalMinimum(name="b", generators=[3, 6], members=[np.zeros(2)])
    c = LocalMinimum(name="c", generators=[9, 18], members=[np.zeros(2)])

    f = Morphism(source=a, target=b, map_function=lambda x: x * 3)
    g = Morphism(source=b, target=c, map_function=lambda x: x * 3)

    composed = compose_morphisms(f, g)

    assert composed is not None
    assert composed.source == a
    assert composed.target == c
    assert composed.apply(1) == 9  # 1 * 3 * 3 = 9

    print("✓ Morphism composition tests passed")


# =============================================================================
# Test Groupoids
# =============================================================================


def test_groupoid_construction():
    """Test groupoid construction."""
    groupoid = Groupoid(name="test_groupoid")

    # Add objects
    a = LocalMinimum(name="a", generators=[1, 2, 3], members=[np.zeros(2)])
    b = LocalMinimum(name="b", generators=[3, 6, 9], members=[np.zeros(2)])
    c = LocalMinimum(name="c", generators=[9, 18, 27], members=[np.zeros(2)])

    idx_a = groupoid.add_object(a)
    idx_b = groupoid.add_object(b)
    idx_c = groupoid.add_object(c)

    assert groupoid.n_objects() == 3

    # Add morphisms
    f = Morphism(source=a, target=b, map_function=lambda x: x * 3)
    g = Morphism(source=b, target=c, map_function=lambda x: x * 3)

    groupoid.add_morphism(idx_a, idx_b, f)
    groupoid.add_morphism(idx_b, idx_c, g)

    assert groupoid.n_morphisms() == 2
    assert groupoid.has_morphism(idx_a, idx_b)
    assert groupoid.has_morphism(idx_b, idx_c)
    assert not groupoid.has_morphism(idx_a, idx_c)

    print("✓ Groupoid construction tests passed")


def test_escape_path():
    """Test escape path finding."""
    groupoid = Groupoid(name="escape_test")

    # Create chain: a → b → c
    a = LocalMinimum(name="a", generators=[1], members=[np.zeros(2)])
    b = LocalMinimum(name="b", generators=[3], members=[np.zeros(2)])
    c = LocalMinimum(name="c", generators=[9], members=[np.zeros(2)])

    idx_a = groupoid.add_object(a)
    idx_b = groupoid.add_object(b)
    idx_c = groupoid.add_object(c)

    f = Morphism(source=a, target=b, map_function=lambda x: x * 3, cost=1.0)
    g = Morphism(source=b, target=c, map_function=lambda x: x * 3, cost=1.0)

    groupoid.add_morphism(idx_a, idx_b, f)
    groupoid.add_morphism(idx_b, idx_c, g)

    # Find path a → c
    path = find_escape_path(groupoid, idx_a, idx_c)

    assert path is not None
    assert len(path) == 2
    print(f"  Escape path: {[m.source.name + '→' + m.target.name for m in path]}")

    print("✓ Escape path tests passed")


def test_groupoid_analysis():
    """Test groupoid structure analysis."""
    groupoid = Groupoid(name="analysis_test")

    # Create simple groupoid
    for i in range(5):
        minimum = LocalMinimum(
            name=f"node_{i}",
            generators=[3**i],
            members=[np.array([i * 0.1, i * 0.1])],
        )
        groupoid.add_object(minimum)

    # Add chain morphisms
    for i in range(4):
        f = Morphism(
            source=groupoid.objects[i],
            target=groupoid.objects[i+1],
            map_function=lambda x: x * 3,
            cost=1.0,
        )
        groupoid.add_morphism(i, i+1, f)

    analysis = analyze_groupoid_structure(groupoid)

    print(f"  Objects: {analysis['n_objects']}")
    print(f"  Morphisms: {analysis['n_morphisms']}")
    print(f"  Connected: {analysis['is_connected']}")
    print(f"  Maximal elements: {analysis['maximal_indices']}")
    print(f"  Minimal elements: {analysis['minimal_indices']}")

    assert analysis['n_objects'] == 5
    assert analysis['n_morphisms'] == 4
    assert analysis['is_connected']
    assert 4 in analysis['maximal_indices']
    assert 0 in analysis['minimal_indices']

    print("✓ Groupoid analysis tests passed")


# =============================================================================
# Main
# =============================================================================


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("REPLACEMENT CALCULUS TEST SUITE")
    print("=" * 60)

    print("\n--- Invariant Tests ---")
    test_valuation()
    test_redundancy()
    test_symmetry_rank()
    test_invariant_tuple_ordering()

    print("\n--- Group Tests ---")
    test_local_minimum_creation()
    test_over_constrained_detection()
    test_codon_local_minimum()

    print("\n--- Morphism Tests ---")
    test_morphism_validity()
    test_morphism_composition()

    print("\n--- Groupoid Tests ---")
    test_groupoid_construction()
    test_escape_path()
    test_groupoid_analysis()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
