# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Unified Data Loader for HIV Dataset Integration

Provides consistent loading interfaces for all HIV datasets:
- Stanford HIVDB drug resistance data (PI, NRTI, NNRTI, INSTI)
- LANL CTL epitope database
- CATNAP antibody neutralization data
- V3 coreceptor tropism data
- Human-HIV protein-protein interactions
- gp120 alignments with tropism labels

All loaders return pandas DataFrames with standardized column names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Get project root by traversing up from script location
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[5]  # hiv -> codon_encoder_research -> bioinformatics -> research -> root

# Data directories
RESEARCH_DATASETS_DIR = _PROJECT_ROOT / "research" / "datasets"
EXTERNAL_DATA_DIR = _PROJECT_ROOT / "data" / "external"
GITHUB_DATA_DIR = EXTERNAL_DATA_DIR / "github"
HUGGINGFACE_DATA_DIR = EXTERNAL_DATA_DIR / "huggingface"
ZENODO_DATA_DIR = EXTERNAL_DATA_DIR / "zenodo"
KAGGLE_DATA_DIR = EXTERNAL_DATA_DIR / "kaggle"


def get_data_paths() -> dict[str, Path]:
    """Return dictionary of all data directory paths."""
    return {
        "research_datasets": RESEARCH_DATASETS_DIR,
        "external": EXTERNAL_DATA_DIR,
        "github": GITHUB_DATA_DIR,
        "huggingface": HUGGINGFACE_DATA_DIR,
        "zenodo": ZENODO_DATA_DIR,
        "kaggle": KAGGLE_DATA_DIR,
    }


# ============================================================================
# STANFORD HIVDB DRUG RESISTANCE LOADERS
# ============================================================================


def load_stanford_hivdb(drug_class: str = "all") -> pd.DataFrame:
    """
    Load Stanford HIV Drug Resistance Database data.

    Args:
        drug_class: One of 'pi', 'nrti', 'nnrti', 'ini', or 'all'

    Returns:
        DataFrame with columns:
        - SeqID: Sequence identifier
        - Drug columns: Fold-change values (e.g., FPV, ATV, etc.)
        - Position columns: Amino acid at each position (P1-P99, RT1-RT560, IN1-IN288)
        - CompMutList: Composite mutation list string
        - drug_class: Added column indicating source (if 'all')

    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If invalid drug_class specified
    """
    valid_classes = {"pi", "nrti", "nnrti", "ini", "all"}
    drug_class = drug_class.lower()

    if drug_class not in valid_classes:
        raise ValueError(f"Invalid drug_class '{drug_class}'. Must be one of: {valid_classes}")

    file_mapping = {
        "pi": "stanford_hivdb_pi.txt",
        "nrti": "stanford_hivdb_nrti.txt",
        "nnrti": "stanford_hivdb_nnrti.txt",
        "ini": "stanford_hivdb_ini.txt",
    }

    if drug_class == "all":
        dfs = []
        for cls, filename in file_mapping.items():
            filepath = RESEARCH_DATASETS_DIR / filename
            if filepath.exists():
                df = pd.read_csv(filepath, sep="\t", low_memory=False)
                df["drug_class"] = cls.upper()
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"No Stanford HIVDB files found in {RESEARCH_DATASETS_DIR}")
        return pd.concat(dfs, ignore_index=True)

    filepath = RESEARCH_DATASETS_DIR / file_mapping[drug_class]
    if not filepath.exists():
        raise FileNotFoundError(f"Stanford HIVDB file not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t", low_memory=False)
    df["drug_class"] = drug_class.upper()
    return df


def get_stanford_drug_columns(drug_class: str) -> list[str]:
    """Get drug column names for a specific drug class."""
    drug_columns = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "CAB", "DTG", "EVG", "RAL"],
    }
    return drug_columns.get(drug_class.lower(), [])


def parse_mutation_list(mut_string: str) -> list[dict]:
    """
    Parse Stanford HIVDB CompMutList format.

    Args:
        mut_string: Comma-separated mutation string like "D30N, M46I, R57G"

    Returns:
        List of dicts with 'position', 'wild_type', 'mutant' keys
    """
    if pd.isna(mut_string) or not mut_string:
        return []

    mutations = []
    pattern = r"([A-Z])(\d+)([A-Z*])"

    for mut in str(mut_string).split(","):
        mut = mut.strip()
        match = re.match(pattern, mut)
        if match:
            mutations.append(
                {
                    "wild_type": match.group(1),
                    "position": int(match.group(2)),
                    "mutant": match.group(3),
                }
            )
    return mutations


def extract_stanford_positions(df: pd.DataFrame, protein: str = "PR") -> pd.DataFrame:
    """
    Extract position columns from Stanford data as a clean matrix.

    Args:
        df: Stanford HIVDB DataFrame
        protein: 'PR' (protease), 'RT' (reverse transcriptase), or 'IN' (integrase)

    Returns:
        DataFrame with position columns only (P1-P99, RT1-RT560, or IN1-IN288)
    """
    prefix_map = {"PR": "P", "RT": "RT", "IN": "IN"}
    prefix = prefix_map.get(protein.upper(), "P")

    position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
    position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

    return df[["SeqID"] + position_cols].copy()


# ============================================================================
# LANL CTL EPITOPE LOADER
# ============================================================================


def load_lanl_ctl() -> pd.DataFrame:
    """
    Load LANL CTL epitope database.

    Returns:
        DataFrame with columns:
        - Epitope: Peptide sequence
        - Protein: HIV protein (Gag, Pol, Env, etc.)
        - HXB2_start: Start position in HXB2 reference
        - HXB2_end: End position in HXB2 reference
        - Subprotein: Subprotein region (e.g., p17, RT)
        - HXB2_DNA_Contig: DNA coordinates
        - Subtype: HIV subtype(s)
        - Species: Host species
        - HLA: HLA restriction(s)
    """
    filepath = RESEARCH_DATASETS_DIR / "ctl_summary.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"CTL summary file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Standardize column names
    column_mapping = {
        "HXB2 start": "HXB2_start",
        "HXB2 end": "HXB2_end",
        "HXB2 DNA Contig": "HXB2_DNA_Contig",
    }
    df = df.rename(columns=column_mapping)

    # Parse numeric positions
    df["HXB2_start"] = pd.to_numeric(df["HXB2_start"], errors="coerce")
    df["HXB2_end"] = pd.to_numeric(df["HXB2_end"], errors="coerce")

    return df


def parse_hla_restrictions(hla_string: str) -> list[str]:
    """
    Parse HLA restriction string into individual HLA types.

    Args:
        hla_string: Comma-separated HLA string like "A*02:01, B*57, B27"

    Returns:
        List of standardized HLA alleles
    """
    if pd.isna(hla_string) or not hla_string:
        return []

    hla_list = []
    for hla in str(hla_string).split(","):
        hla = hla.strip()
        if hla:
            # Standardize format (e.g., A3 -> A*03)
            hla_list.append(hla)
    return hla_list


def get_epitopes_by_protein(df: Optional[pd.DataFrame] = None, protein: str = "Gag") -> pd.DataFrame:
    """Get epitopes for a specific HIV protein."""
    if df is None:
        df = load_lanl_ctl()
    return df[df["Protein"].str.contains(protein, case=False, na=False)].copy()


def get_epitopes_by_hla(df: Optional[pd.DataFrame] = None, hla: str = "A*02:01") -> pd.DataFrame:
    """Get epitopes restricted by a specific HLA type."""
    if df is None:
        df = load_lanl_ctl()
    return df[df["HLA"].str.contains(hla, case=False, na=False)].copy()


# ============================================================================
# CATNAP NEUTRALIZATION LOADER
# ============================================================================


def load_catnap() -> pd.DataFrame:
    """
    Load CATNAP antibody neutralization data.

    Returns:
        DataFrame with columns:
        - Antibody: Antibody name (e.g., VRC01, PG9)
        - Virus: Virus strain identifier
        - Reference: Publication reference
        - Pubmed_ID: PubMed identifier
        - IC50: 50% inhibitory concentration (ug/mL)
        - IC80: 80% inhibitory concentration
        - ID50: 50% neutralization dilution
    """
    filepath = RESEARCH_DATASETS_DIR / "catnap_assay.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"CATNAP file not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    # Standardize column names
    column_mapping = {
        "Pubmed ID": "Pubmed_ID",
    }
    df = df.rename(columns=column_mapping)

    # Parse IC50 values (handle ">", "<" prefixes)
    df["IC50_numeric"] = df["IC50"].apply(_parse_ic_value)
    df["IC50_censored"] = df["IC50"].apply(lambda x: ">" in str(x) or "<" in str(x))

    return df


def _parse_ic_value(value) -> Optional[float]:
    """Parse IC50/IC80 values, handling censored values."""
    if pd.isna(value) or value == "":
        return None
    val_str = str(value).strip()
    # Remove > or < prefix
    val_str = val_str.lstrip(">< ")
    try:
        return float(val_str)
    except ValueError:
        return None


def get_catnap_by_antibody(df: Optional[pd.DataFrame] = None, antibody: str = "VRC01") -> pd.DataFrame:
    """Get neutralization data for a specific antibody."""
    if df is None:
        df = load_catnap()
    return df[df["Antibody"].str.contains(antibody, case=False, na=False)].copy()


def get_catnap_sensitive_viruses(
    df: Optional[pd.DataFrame] = None, antibody: str = "VRC01", ic50_threshold: float = 1.0
) -> pd.DataFrame:
    """Get viruses sensitive to an antibody (IC50 below threshold)."""
    if df is None:
        df = load_catnap()
    ab_data = df[df["Antibody"].str.contains(antibody, case=False, na=False)].copy()
    return ab_data[(ab_data["IC50_numeric"].notna()) & (ab_data["IC50_numeric"] <= ic50_threshold)]


def get_catnap_resistant_viruses(
    df: Optional[pd.DataFrame] = None, antibody: str = "VRC01", ic50_threshold: float = 50.0
) -> pd.DataFrame:
    """Get viruses resistant to an antibody (IC50 above threshold)."""
    if df is None:
        df = load_catnap()
    ab_data = df[df["Antibody"].str.contains(antibody, case=False, na=False)].copy()
    return ab_data[(ab_data["IC50_numeric"].notna()) & (ab_data["IC50_numeric"] >= ic50_threshold)]


def calculate_antibody_breadth(df: Optional[pd.DataFrame] = None, ic50_threshold: float = 50.0) -> pd.DataFrame:
    """
    Calculate neutralization breadth for each antibody.

    Args:
        df: CATNAP DataFrame
        ic50_threshold: IC50 threshold for considering a virus neutralized

    Returns:
        DataFrame with antibody, n_tested, n_neutralized, breadth_pct
    """
    if df is None:
        df = load_catnap()

    # Filter to records with valid IC50
    valid_df = df[df["IC50_numeric"].notna()].copy()

    results = []
    for antibody, group in valid_df.groupby("Antibody"):
        n_tested = len(group)
        n_neutralized = (group["IC50_numeric"] <= ic50_threshold).sum()
        breadth = 100 * n_neutralized / n_tested if n_tested > 0 else 0

        results.append(
            {
                "Antibody": antibody,
                "n_tested": n_tested,
                "n_neutralized": n_neutralized,
                "breadth_pct": breadth,
            }
        )

    return pd.DataFrame(results).sort_values("breadth_pct", ascending=False)


# ============================================================================
# V3 CORECEPTOR TROPISM LOADER
# ============================================================================


def load_v3_coreceptor() -> pd.DataFrame:
    """
    Load V3 loop coreceptor tropism dataset.

    Returns:
        DataFrame with columns:
        - sequence: V3 loop amino acid sequence
        - tropism: CCR5 or CXCR4
        - Additional columns from original dataset
    """
    parquet_path = HUGGINGFACE_DATA_DIR / "HIV_V3_coreceptor"

    # Try parquet files first
    parquet_files = list(parquet_path.glob("**/*.parquet"))
    if parquet_files:
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(f"V3 coreceptor dataset not found in: {parquet_path}")


def get_ccr5_sequences(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Get CCR5-tropic V3 sequences."""
    if df is None:
        df = load_v3_coreceptor()
    # Column names may vary - check for common patterns
    for col in ["tropism", "Tropism", "label", "Label", "target"]:
        if col in df.columns:
            return df[df[col].astype(str).str.upper().str.contains("R5|CCR5")].copy()
    return pd.DataFrame()


def get_cxcr4_sequences(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Get CXCR4-tropic V3 sequences."""
    if df is None:
        df = load_v3_coreceptor()
    for col in ["tropism", "Tropism", "label", "Label", "target"]:
        if col in df.columns:
            return df[df[col].astype(str).str.upper().str.contains("X4|CXCR4")].copy()
    return pd.DataFrame()


# ============================================================================
# HUMAN-HIV PROTEIN-PROTEIN INTERACTION LOADER
# ============================================================================


def load_human_hiv_ppi() -> pd.DataFrame:
    """
    Load human-HIV protein-protein interaction dataset.

    Returns:
        DataFrame with columns:
        - hiv_protein_name: HIV protein (e.g., gp120, Tat)
        - human_protein_name: Human protein (e.g., CD4, CCR5)
        - interaction_type: Type of interaction
        - hiv_protein_sequence: HIV protein sequence
        - human_protein_sequence: Human protein sequence
    """
    parquet_path = HUGGINGFACE_DATA_DIR / "human_hiv_ppi"

    parquet_files = list(parquet_path.glob("**/*.parquet"))
    if parquet_files:
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(f"Human-HIV PPI dataset not found in: {parquet_path}")


def get_ppi_by_hiv_protein(df: Optional[pd.DataFrame] = None, protein: str = "gp120") -> pd.DataFrame:
    """Get interactions for a specific HIV protein."""
    if df is None:
        df = load_human_hiv_ppi()
    col = "hiv_protein_name" if "hiv_protein_name" in df.columns else df.columns[0]
    return df[df[col].str.contains(protein, case=False, na=False)].copy()


def get_ppi_by_human_protein(df: Optional[pd.DataFrame] = None, protein: str = "CD4") -> pd.DataFrame:
    """Get interactions for a specific human protein."""
    if df is None:
        df = load_human_hiv_ppi()
    col = "human_protein_name" if "human_protein_name" in df.columns else df.columns[1]
    return df[df[col].str.contains(protein, case=False, na=False)].copy()


# ============================================================================
# GP120 ALIGNMENT LOADER
# ============================================================================


def load_gp120_alignments() -> dict[str, str]:
    """
    Load aligned gp120 sequences from Zenodo dataset.

    Returns:
        Dictionary mapping sequence ID to aligned sequence
    """
    gp120_path = ZENODO_DATA_DIR / "cview_gp120"

    fasta_files = list(gp120_path.glob("**/*.fasta")) + list(gp120_path.glob("**/*.fa"))
    if not fasta_files:
        raise FileNotFoundError(f"No gp120 FASTA files found in: {gp120_path}")

    sequences = {}
    for fasta_file in fasta_files:
        sequences.update(_parse_fasta(fasta_file))

    return sequences


def load_gp120_tropism_labels() -> dict[str, str]:
    """
    Load tropism labels for gp120 sequences.

    Returns:
        Dictionary mapping sequence ID to tropism (CCR5 or CXCR4)
    """
    gp120_path = ZENODO_DATA_DIR / "cview_gp120"

    labels = {}

    # Look for CCR5 titles
    ccr5_file = list(gp120_path.glob("**/CCR5*.txt"))
    if ccr5_file:
        with open(ccr5_file[0]) as f:
            for line in f:
                seq_id = line.strip()
                if seq_id:
                    labels[seq_id] = "CCR5"

    # Look for CXCR4 titles
    cxcr4_file = list(gp120_path.glob("**/CXCR4*.txt"))
    if cxcr4_file:
        with open(cxcr4_file[0]) as f:
            for line in f:
                seq_id = line.strip()
                if seq_id:
                    labels[seq_id] = "CXCR4"

    return labels


def _parse_fasta(filepath: Path) -> dict[str, str]:
    """Parse a FASTA file into a dictionary."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]  # Take first word as ID
                current_seq = []
            else:
                current_seq.append(line)

        if current_id:
            sequences[current_id] = "".join(current_seq)

    return sequences


# ============================================================================
# HIV SEQUENCE LOADERS (GITHUB)
# ============================================================================


def load_hiv_env_sequences(species: str = "hiv1") -> dict[str, str]:
    """
    Load HIV envelope sequences from GitHub dataset.

    Args:
        species: 'hiv1', 'hiv2', or 'siv'

    Returns:
        Dictionary mapping sequence ID to sequence
    """
    hiv_data_path = GITHUB_DATA_DIR / "HIV-data"

    file_mapping = {
        "hiv1": "HIV1_ALL_2021_env_DNA.2000.fasta",
        "hiv2": "HIV2_ALL_2021_env_DNA.2000.fasta",
        "siv": "SIV_ALL_2021_env_DNA.2000.fasta",
    }

    filename = file_mapping.get(species.lower())
    if not filename:
        raise ValueError(f"Unknown species: {species}. Use 'hiv1', 'hiv2', or 'siv'")

    fasta_files = list(hiv_data_path.glob(f"**/{filename}"))
    if not fasta_files:
        raise FileNotFoundError(f"HIV env file not found: {filename}")

    return _parse_fasta(fasta_files[0])


def load_hiv_genome_sequences() -> dict[str, str]:
    """
    Load HIV-1 full genome sequences.

    Returns:
        Dictionary mapping sequence ID to full genome sequence
    """
    hiv_data_path = GITHUB_DATA_DIR / "HIV-data"

    fasta_files = list(hiv_data_path.glob("**/HIV1_ALL_2021_genome_DNA.5000.fasta"))
    if not fasta_files:
        raise FileNotFoundError("HIV genome file not found")

    return _parse_fasta(fasta_files[0])


# ============================================================================
# EPIDEMIOLOGICAL DATA LOADERS
# ============================================================================


def load_pmtct_data() -> pd.DataFrame:
    """
    Load Prevention of Mother-to-Child Transmission data.

    Returns:
        DataFrame with country-level PMTCT statistics
    """
    filepath = KAGGLE_DATA_DIR / "hiv-aids-dataset" / "prevention_of_mother_to_child_transmission_by_country_clean.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"PMTCT data not found: {filepath}")

    return pd.read_csv(filepath)


def load_corgis_aids() -> pd.DataFrame:
    """
    Load CORGIS AIDS/HIV statistics.

    Returns:
        DataFrame with country-year HIV statistics
    """
    filepath = EXTERNAL_DATA_DIR / "csv" / "corgis_aids.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"CORGIS AIDS data not found: {filepath}")

    return pd.read_csv(filepath)


# ============================================================================
# UNIFIED DATASET SUMMARY
# ============================================================================


def get_dataset_summary() -> pd.DataFrame:
    """
    Get summary of all available datasets.

    Returns:
        DataFrame with dataset name, location, record count, status
    """
    datasets = []

    # Stanford HIVDB
    for drug_class in ["pi", "nrti", "nnrti", "ini"]:
        filepath = RESEARCH_DATASETS_DIR / f"stanford_hivdb_{drug_class}.txt"
        datasets.append(
            {
                "Dataset": f"Stanford HIVDB ({drug_class.upper()})",
                "Location": str(filepath),
                "Exists": filepath.exists(),
                "Records": _count_lines(filepath) - 1 if filepath.exists() else 0,
            }
        )

    # LANL CTL
    filepath = RESEARCH_DATASETS_DIR / "ctl_summary.csv"
    datasets.append(
        {
            "Dataset": "LANL CTL Epitopes",
            "Location": str(filepath),
            "Exists": filepath.exists(),
            "Records": _count_lines(filepath) - 1 if filepath.exists() else 0,
        }
    )

    # CATNAP
    filepath = RESEARCH_DATASETS_DIR / "catnap_assay.txt"
    datasets.append(
        {
            "Dataset": "CATNAP Neutralization",
            "Location": str(filepath),
            "Exists": filepath.exists(),
            "Records": _count_lines(filepath) - 1 if filepath.exists() else 0,
        }
    )

    # V3 Coreceptor
    v3_path = HUGGINGFACE_DATA_DIR / "HIV_V3_coreceptor"
    datasets.append(
        {
            "Dataset": "V3 Coreceptor Tropism",
            "Location": str(v3_path),
            "Exists": v3_path.exists(),
            "Records": _count_parquet_rows(v3_path),
        }
    )

    # Human-HIV PPI
    ppi_path = HUGGINGFACE_DATA_DIR / "human_hiv_ppi"
    datasets.append(
        {
            "Dataset": "Human-HIV PPI",
            "Location": str(ppi_path),
            "Exists": ppi_path.exists(),
            "Records": _count_parquet_rows(ppi_path),
        }
    )

    # gp120 alignments
    gp120_path = ZENODO_DATA_DIR / "cview_gp120"
    datasets.append(
        {
            "Dataset": "gp120 Alignments",
            "Location": str(gp120_path),
            "Exists": gp120_path.exists(),
            "Records": _count_fasta_sequences(gp120_path),
        }
    )

    return pd.DataFrame(datasets)


def _count_lines(filepath: Path) -> int:
    """Count lines in a file."""
    if not filepath.exists():
        return 0
    with open(filepath) as f:
        return sum(1 for _ in f)


def _count_parquet_rows(dirpath: Path) -> int:
    """Count rows across parquet files in a directory."""
    if not dirpath.exists():
        return 0
    try:
        total = 0
        for f in dirpath.glob("**/*.parquet"):
            df = pd.read_parquet(f)
            total += len(df)
        return total
    except Exception:
        return 0


def _count_fasta_sequences(dirpath: Path) -> int:
    """Count sequences across FASTA files in a directory."""
    if not dirpath.exists():
        return 0
    try:
        total = 0
        for f in list(dirpath.glob("**/*.fasta")) + list(dirpath.glob("**/*.fa")):
            with open(f) as fh:
                total += sum(1 for line in fh if line.startswith(">"))
        return total
    except Exception:
        return 0


# ============================================================================
# MAIN - TEST LOADERS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HIV Dataset Summary")
    print("=" * 60)

    summary = get_dataset_summary()
    print(summary.to_string(index=False))

    print("\nTotal records across all datasets:", summary["Records"].sum())
