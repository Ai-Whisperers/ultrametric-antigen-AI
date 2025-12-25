# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for quantum-chemical descriptors."""

import pytest
import torch

from src.quantum.descriptors import (
    AminoAcidQuantumProperties,
    QuantumBioDescriptor,
    QuantumDescriptorResult,
)


class TestAminoAcidQuantumProperties:
    """Tests for amino acid quantum property tables."""

    def test_all_amino_acids_have_homo_lumo(self):
        """Test that all 20 amino acids have HOMO-LUMO values."""
        aa_codes = "ACDEFGHIKLMNPQRSTVWY"
        for aa in aa_codes:
            assert aa in AminoAcidQuantumProperties.HOMO_LUMO_GAP

    def test_homo_lumo_values_reasonable(self):
        """Test that HOMO-LUMO gaps are in reasonable range."""
        for aa, gap in AminoAcidQuantumProperties.HOMO_LUMO_GAP.items():
            assert 4.0 <= gap <= 8.0, f"HOMO-LUMO gap for {aa} is {gap}"

    def test_dipole_moments_reasonable(self):
        """Test that dipole moments are in reasonable range."""
        for aa, dipole in AminoAcidQuantumProperties.DIPOLE_MOMENT.items():
            assert 0.0 <= dipole <= 5.0, f"Dipole for {aa} is {dipole}"

    def test_aromatic_lower_homo_lumo(self):
        """Test that aromatic amino acids have lower HOMO-LUMO gaps."""
        aromatic = ["F", "W", "Y", "H"]
        aliphatic = ["A", "V", "L", "I"]

        avg_aromatic = sum(AminoAcidQuantumProperties.HOMO_LUMO_GAP[aa] for aa in aromatic) / len(aromatic)
        avg_aliphatic = sum(AminoAcidQuantumProperties.HOMO_LUMO_GAP[aa] for aa in aliphatic) / len(aliphatic)

        assert avg_aromatic < avg_aliphatic

    def test_tryptophan_lowest_gap(self):
        """Test that tryptophan has one of the lowest HOMO-LUMO gaps."""
        trp_gap = AminoAcidQuantumProperties.HOMO_LUMO_GAP["W"]
        assert trp_gap == min(AminoAcidQuantumProperties.HOMO_LUMO_GAP.values())


class TestQuantumBioDescriptor:
    """Tests for QuantumBioDescriptor."""

    def test_initialization(self):
        """Test descriptor initialization."""
        descriptor = QuantumBioDescriptor()
        assert descriptor.p == 3

    def test_initialization_custom_p(self):
        """Test descriptor with custom prime."""
        descriptor = QuantumBioDescriptor(p=5)
        assert descriptor.p == 5

    def test_get_residue_homo_lumo(self):
        """Test getting HOMO-LUMO for single residue."""
        descriptor = QuantumBioDescriptor()

        assert descriptor.get_residue_homo_lumo("W") == pytest.approx(4.8)
        assert descriptor.get_residue_homo_lumo("G") == pytest.approx(7.5)

    def test_get_residue_homo_lumo_unknown(self):
        """Test getting HOMO-LUMO for unknown residue."""
        descriptor = QuantumBioDescriptor()

        # Should return default value
        assert descriptor.get_residue_homo_lumo("X") == pytest.approx(6.5)

    def test_get_residue_dipole(self):
        """Test getting dipole moment for single residue."""
        descriptor = QuantumBioDescriptor()

        assert descriptor.get_residue_dipole("R") == pytest.approx(4.5)  # Charged
        assert descriptor.get_residue_dipole("F") == pytest.approx(0.8)  # Symmetric

    def test_compute_sequence_descriptors_basic(self):
        """Test computing descriptors for simple sequence."""
        descriptor = QuantumBioDescriptor()
        result = descriptor.compute_sequence_descriptors("WWW")

        assert isinstance(result, QuantumDescriptorResult)
        assert result.sequence == "WWW"
        assert result.homo_lumo_gap == pytest.approx(4.8)  # All W

    def test_compute_sequence_descriptors_mixed(self):
        """Test computing descriptors for mixed sequence."""
        descriptor = QuantumBioDescriptor()
        result = descriptor.compute_sequence_descriptors("MVLSPADKTNVKAAW")

        assert len(result.per_residue_homo_lumo) == 15
        assert result.homo_lumo_gap > 0
        assert result.dipole_moment > 0
        assert result.polarizability > 0

    def test_derived_descriptors(self):
        """Test derived quantum descriptors."""
        descriptor = QuantumBioDescriptor()
        result = descriptor.compute_sequence_descriptors("ACDEFG")

        # Chemical hardness = (IP - EA) / 2
        expected_hardness = (result.ionization_potential - result.electron_affinity) / 2
        assert result.hardness == pytest.approx(expected_hardness)

        # Electronegativity = (IP + EA) / 2
        expected_chi = (result.ionization_potential + result.electron_affinity) / 2
        assert result.electronegativity == pytest.approx(expected_chi)

    def test_padic_signature_computed(self):
        """Test that p-adic signature is computed."""
        descriptor = QuantumBioDescriptor()
        result = descriptor.compute_sequence_descriptors("ACDEFGHIKL")

        assert "valuation_sum" in result.padic_signature
        assert "normalized_distance" in result.padic_signature
        assert "gap_variance" in result.padic_signature

    def test_binding_site_descriptors(self):
        """Test computing descriptors for binding site."""
        descriptor = QuantumBioDescriptor()
        sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

        # Residues 0, 5, 10, 15
        binding_sites = [0, 5, 10, 15]
        result = descriptor.compute_binding_site_descriptors(sequence, binding_sites)

        assert len(result.sequence) == 4  # 4 binding site residues

    def test_binding_site_empty(self):
        """Test binding site with no valid positions."""
        descriptor = QuantumBioDescriptor()
        result = descriptor.compute_binding_site_descriptors("ACDEF", [100, 200])

        assert result.sequence == ""
        assert result.homo_lumo_gap == 0.0

    def test_estimate_tunneling_barrier(self):
        """Test tunneling barrier estimation."""
        descriptor = QuantumBioDescriptor()

        # Lower gap = easier tunneling
        prob_low_gap = descriptor.estimate_tunneling_barrier(4.0, 3.0)
        prob_high_gap = descriptor.estimate_tunneling_barrier(8.0, 3.0)

        assert prob_low_gap > prob_high_gap
        assert 0 <= prob_low_gap <= 1
        assert 0 <= prob_high_gap <= 1

    def test_estimate_tunneling_distance_effect(self):
        """Test that longer distance reduces tunneling probability."""
        descriptor = QuantumBioDescriptor()

        prob_short = descriptor.estimate_tunneling_barrier(5.0, 2.0)
        prob_long = descriptor.estimate_tunneling_barrier(5.0, 5.0)

        assert prob_short > prob_long

    def test_to_feature_vector(self):
        """Test conversion to feature vector."""
        descriptor = QuantumBioDescriptor()
        result = descriptor.compute_sequence_descriptors("ACDEFGHIKL")

        vector = descriptor.to_feature_vector(result)

        assert isinstance(vector, torch.Tensor)
        assert vector.dim() == 1
        assert vector.shape[0] == 11  # Number of features

    def test_batch_compute(self):
        """Test batch computation."""
        descriptor = QuantumBioDescriptor()
        sequences = ["ACDEF", "GHIKL", "MNPQR", "STVWY"]

        results = descriptor.batch_compute(sequences)

        assert len(results) == 4
        assert all(isinstance(r, QuantumDescriptorResult) for r in results)


class TestPadicSignature:
    """Tests for p-adic signature computation."""

    def test_padic_valuation(self):
        """Test p-adic valuation calculation."""
        descriptor = QuantumBioDescriptor(p=3)

        assert descriptor._compute_padic_valuation(9) == 2
        assert descriptor._compute_padic_valuation(27) == 3
        assert descriptor._compute_padic_valuation(10) == 0

    def test_padic_signature_variance(self):
        """Test that p-adic signature captures variance."""
        descriptor = QuantumBioDescriptor()

        # Uniform sequence should have low variance
        uniform = descriptor.compute_sequence_descriptors("AAAAAAAAAA")

        # Mixed sequence should have higher variance
        mixed = descriptor.compute_sequence_descriptors("AWFYHCDEFG")

        assert uniform.padic_signature["gap_variance"] < mixed.padic_signature["gap_variance"]
