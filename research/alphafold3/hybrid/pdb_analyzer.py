# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""PDB Structure Analyzer using BioPython.

Provides structural context for HIV integrase mutations without
requiring AlphaFold3's C++ extensions.
"""

import gzip
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

try:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.NeighborSearch import NeighborSearch
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    # Define placeholder for type hints when BioPython not available
    Structure = None
    Residue = None


class ResidueContact(NamedTuple):
    """Contact between two residues."""

    chain_id: str
    res_id: int
    res_name: str
    distance: float
    contact_type: str  # 'backbone', 'sidechain', 'interface'


class StructuralContext(NamedTuple):
    """Structural context for a residue position."""

    position: int
    residue_name: str
    chain_id: str
    secondary_structure: str  # 'helix', 'sheet', 'coil'
    solvent_accessibility: float  # 0-1, estimated
    contacts: List[ResidueContact]
    interface_residue: bool  # True if at protein-protein interface
    catalytic_site: bool  # True if near catalytic residues


class PDBAnalyzer:
    """Analyze PDB structures for HIV integrase context.

    Uses BioPython for parsing instead of AlphaFold3's C++ extensions.
    """

    # HIV-1 Integrase catalytic triad
    CATALYTIC_RESIDUES = {64, 116, 152}  # D64, D116, E152

    # LEDGF interface residues (from 2B4J structure)
    LEDGF_INTERFACE = {
        128,
        129,
        130,
        131,
        132,  # Loop region
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,  # Helix region
    }

    def __init__(self, pdb_dir: Optional[Path] = None):
        """Initialize the PDB analyzer.

        Args:
            pdb_dir: Directory containing PDB/mmCIF files
        """
        if not HAS_BIOPYTHON:
            raise ImportError("BioPython is required for PDB analysis. " "Install with: pip install biopython")

        self.pdb_dir = pdb_dir or Path(__file__).parent.parent / "data" / "pdb"
        self.mmcif_parser = MMCIFParser(QUIET=True)
        self.pdb_parser = PDBParser(QUIET=True)
        self._structure_cache: Dict[str, "Structure"] = {}

    def load_structure(self, pdb_id: str) -> "Structure":
        """Load a PDB structure from file.

        Args:
            pdb_id: PDB identifier (e.g., '2B4J')

        Returns:
            BioPython Structure object
        """
        if pdb_id in self._structure_cache:
            return self._structure_cache[pdb_id]

        # Try mmCIF format first (preferred)
        cif_path = self.pdb_dir / f"{pdb_id}.cif.gz"
        if cif_path.exists():
            with gzip.open(cif_path, "rt") as f:
                structure = self.mmcif_parser.get_structure(pdb_id, f)
                self._structure_cache[pdb_id] = structure
                return structure

        # Try uncompressed mmCIF
        cif_path = self.pdb_dir / f"{pdb_id}.cif"
        if cif_path.exists():
            structure = self.mmcif_parser.get_structure(pdb_id, str(cif_path))
            self._structure_cache[pdb_id] = structure
            return structure

        # Try PDB format
        pdb_path = self.pdb_dir / f"{pdb_id}.pdb"
        if pdb_path.exists():
            structure = self.pdb_parser.get_structure(pdb_id, str(pdb_path))
            self._structure_cache[pdb_id] = structure
            return structure

        raise FileNotFoundError(f"Structure {pdb_id} not found in {self.pdb_dir}. " f"Run download_integrase_structures.py first.")

    def get_residue_contacts(
        self,
        structure: "Structure",
        chain_id: str,
        res_id: int,
        radius: float = 8.0,
    ) -> List[ResidueContact]:
        """Find residues in contact with a given position.

        Args:
            structure: BioPython Structure
            chain_id: Chain identifier
            res_id: Residue number
            radius: Contact distance cutoff in Angstroms

        Returns:
            List of contacting residues
        """
        # Get the target residue
        model = structure[0]  # First model
        chain = model[chain_id]
        target_res = chain[(" ", res_id, " ")]

        # Get CA atom for distance calculation
        if "CA" not in target_res:
            return []

        target_atom = target_res["CA"]

        # Build neighbor search
        all_atoms = list(model.get_atoms())
        ns = NeighborSearch(all_atoms)

        # Find nearby atoms
        nearby = ns.search(target_atom.get_coord(), radius, level="R")

        contacts = []
        for res in nearby:
            if res == target_res:
                continue

            # Calculate distance
            if "CA" in res:
                dist = target_atom - res["CA"]
            else:
                # Use first atom if no CA
                dist = target_atom - list(res.get_atoms())[0]

            # Determine contact type
            res_chain = res.get_parent().get_id()
            if res_chain != chain_id:
                contact_type = "interface"
            elif dist < 4.5:
                contact_type = "backbone"
            else:
                contact_type = "sidechain"

            contacts.append(
                ResidueContact(
                    chain_id=res_chain,
                    res_id=res.get_id()[1],
                    res_name=res.get_resname(),
                    distance=dist,
                    contact_type=contact_type,
                )
            )

        return sorted(contacts, key=lambda c: c.distance)

    def get_structural_context(self, pdb_id: str, position: int, chain_id: str = "A") -> StructuralContext:
        """Get structural context for a residue position.

        Args:
            pdb_id: PDB identifier
            position: Residue position
            chain_id: Chain identifier

        Returns:
            StructuralContext with local environment info
        """
        structure = self.load_structure(pdb_id)
        model = structure[0]

        try:
            chain = model[chain_id]
            residue = chain[(" ", position, " ")]
            res_name = residue.get_resname()
        except KeyError:
            # Position not found in structure
            return StructuralContext(
                position=position,
                residue_name="UNK",
                chain_id=chain_id,
                secondary_structure="coil",
                solvent_accessibility=0.5,
                contacts=[],
                interface_residue=False,
                catalytic_site=False,
            )

        contacts = self.get_residue_contacts(structure, chain_id, position)

        # Check if at interface (contacts with different chain)
        interface_residue = any(c.contact_type == "interface" for c in contacts)

        # Check if near catalytic site
        catalytic_site = position in self.CATALYTIC_RESIDUES or any(c.res_id in self.CATALYTIC_RESIDUES and c.distance < 8.0 for c in contacts)

        # Estimate solvent accessibility from contact count
        # Fewer contacts = more exposed
        buried_cutoff = 15
        n_contacts = len([c for c in contacts if c.distance < 5.0])
        solvent_accessibility = max(0.0, 1.0 - (n_contacts / buried_cutoff))

        # Simple secondary structure estimation from backbone contacts
        # (A proper DSSP analysis would be more accurate)
        backbone_contacts = [c for c in contacts if c.contact_type == "backbone"]
        if len(backbone_contacts) >= 4:
            secondary_structure = "helix"
        elif len(backbone_contacts) >= 2:
            secondary_structure = "sheet"
        else:
            secondary_structure = "coil"

        return StructuralContext(
            position=position,
            residue_name=res_name,
            chain_id=chain_id,
            secondary_structure=secondary_structure,
            solvent_accessibility=solvent_accessibility,
            contacts=contacts[:10],  # Top 10 contacts
            interface_residue=interface_residue,
            catalytic_site=catalytic_site,
        )

    def is_ledgf_interface(self, position: int) -> bool:
        """Check if position is at the LEDGF binding interface."""
        return position in self.LEDGF_INTERFACE

    def analyze_mutation_site(
        self,
        pdb_id: str,
        position: int,
        wt_aa: str,
        mut_aa: str,
        chain_id: str = "A",
    ) -> Dict:
        """Analyze the structural impact of a mutation.

        Args:
            pdb_id: PDB identifier
            position: Residue position
            wt_aa: Wild-type amino acid (one-letter)
            mut_aa: Mutant amino acid (one-letter)
            chain_id: Chain identifier

        Returns:
            Dictionary with structural analysis
        """
        context = self.get_structural_context(pdb_id, position, chain_id)

        # Amino acid properties for impact estimation
        CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}
        HYDROPHOBIC = {"A", "V", "L", "I", "M", "F", "W", "Y", "P"}
        AROMATIC = {"F", "W", "Y", "H"}

        wt_charge = CHARGE.get(wt_aa, 0)
        mut_charge = CHARGE.get(mut_aa, 0)
        charge_change = mut_charge - wt_charge

        wt_hydrophobic = wt_aa in HYDROPHOBIC
        mut_hydrophobic = mut_aa in HYDROPHOBIC
        hydrophobicity_change = int(mut_hydrophobic) - int(wt_hydrophobic)

        # Determine impact mechanism
        mechanisms = []

        if abs(charge_change) >= 1:
            if charge_change > 0:
                mechanisms.append("charge_introduction")
            else:
                mechanisms.append("charge_removal")

        if wt_aa in AROMATIC and mut_aa not in AROMATIC:
            mechanisms.append("aromatic_loss")
        elif mut_aa in AROMATIC and wt_aa not in AROMATIC:
            mechanisms.append("aromatic_gain")

        if context.interface_residue:
            mechanisms.append("interface_disruption")

        if context.catalytic_site:
            mechanisms.append("catalytic_site_adjacent")

        if self.is_ledgf_interface(position):
            mechanisms.append("ledgf_interface")

        return {
            "position": position,
            "mutation": f"{wt_aa}{position}{mut_aa}",
            "structural_context": context._asdict(),
            "charge_change": charge_change,
            "hydrophobicity_change": hydrophobicity_change,
            "mechanisms": mechanisms,
            "is_ledgf_interface": self.is_ledgf_interface(position),
            "is_catalytic_adjacent": context.catalytic_site,
            "solvent_accessibility": context.solvent_accessibility,
            "n_contacts": len(context.contacts),
        }
