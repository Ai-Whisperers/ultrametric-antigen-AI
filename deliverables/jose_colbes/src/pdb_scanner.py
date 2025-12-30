from pathlib import Path
from typing import Iterator


class RotamerExtractor:
    """Extracts rotamer angles from PDB structures."""

    def extract_from_file(self, pdb_path: str):
        """Parse a single PDB file."""
        # Placeholder: Call Biopython PDBParser and calc_dihedral
        print(f"Extracting rotamers from {pdb_path}...")
        return []


class PDBScanner:
    """Iterates over directories of PDB files (e.g., AlphaFold outputs)."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def scan(self, pattern: str = "*.pdb") -> Iterator[Path]:
        """Yield PDB files matching the pattern."""
        if not self.root_dir.exists():
            return

        for p in self.root_dir.rglob(pattern):
            yield p
