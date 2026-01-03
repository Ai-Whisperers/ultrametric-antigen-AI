"""
Common configuration for codon-encoder research scripts.

Centralizes paths to embeddings, results, and data directories.
"""

from pathlib import Path

# Base directories
CODON_ENCODER_DIR = Path(__file__).parent
RESEARCH_DIR = CODON_ENCODER_DIR.parent
PROJECT_ROOT = RESEARCH_DIR.parent

# P-adic codon embeddings (from trained VAE)
# Data now stored locally within codon-encoder directory
DATA_DIR = CODON_ENCODER_DIR / "data"

# Alias for backwards compatibility (scripts may reference GENETIC_CODE_DIR)
GENETIC_CODE_DIR = DATA_DIR

# Legacy path (for reference, can be removed after full migration)
_LEGACY_GENETIC_CODE_DIR = RESEARCH_DIR / "bioinformatics" / "genetic_code" / "data"

# Embedding files
CODON_MAPPING_FILE = DATA_DIR / "codon_mapping_3adic.json"
EMBEDDINGS_FILE = DATA_DIR / "v5_11_3_embeddings.pt"

# Results directories (organized by category)
RESULTS_DIR = CODON_ENCODER_DIR / "results"
BENCHMARK_RESULTS_DIR = RESULTS_DIR / "benchmarks"
TRAINING_RESULTS_DIR = RESULTS_DIR / "training"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"
PIPELINE_RESULTS_DIR = RESULTS_DIR / "pipelines"

# Ensure directories exist
for d in [RESULTS_DIR, BENCHMARK_RESULTS_DIR, TRAINING_RESULTS_DIR,
          ANALYSIS_RESULTS_DIR, PIPELINE_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Codon to amino acid mapping
CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Amino acid properties
AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'volume': 88.6, 'mass': 89.09, 'charge': 0, 'flexibility': 1.0},
    'R': {'hydropathy': -4.5, 'volume': 173.4, 'mass': 174.20, 'charge': 1, 'flexibility': 0.9},
    'N': {'hydropathy': -3.5, 'volume': 114.1, 'mass': 132.12, 'charge': 0, 'flexibility': 0.95},
    'D': {'hydropathy': -3.5, 'volume': 111.1, 'mass': 133.10, 'charge': -1, 'flexibility': 0.95},
    'C': {'hydropathy': 2.5, 'volume': 108.5, 'mass': 121.16, 'charge': 0, 'flexibility': 0.9},
    'Q': {'hydropathy': -3.5, 'volume': 143.8, 'mass': 146.15, 'charge': 0, 'flexibility': 0.9},
    'E': {'hydropathy': -3.5, 'volume': 138.4, 'mass': 147.13, 'charge': -1, 'flexibility': 0.9},
    'G': {'hydropathy': -0.4, 'volume': 60.1, 'mass': 75.07, 'charge': 0, 'flexibility': 1.5},
    'H': {'hydropathy': -3.2, 'volume': 153.2, 'mass': 155.16, 'charge': 0, 'flexibility': 0.85},
    'I': {'hydropathy': 4.5, 'volume': 166.7, 'mass': 131.17, 'charge': 0, 'flexibility': 0.7},
    'L': {'hydropathy': 3.8, 'volume': 166.7, 'mass': 131.17, 'charge': 0, 'flexibility': 0.8},
    'K': {'hydropathy': -3.9, 'volume': 168.6, 'mass': 146.19, 'charge': 1, 'flexibility': 0.85},
    'M': {'hydropathy': 1.9, 'volume': 162.9, 'mass': 149.21, 'charge': 0, 'flexibility': 0.85},
    'F': {'hydropathy': 2.8, 'volume': 189.9, 'mass': 165.19, 'charge': 0, 'flexibility': 0.75},
    'P': {'hydropathy': -1.6, 'volume': 112.7, 'mass': 115.13, 'charge': 0, 'flexibility': 0.3},
    'S': {'hydropathy': -0.8, 'volume': 89.0, 'mass': 105.09, 'charge': 0, 'flexibility': 1.0},
    'T': {'hydropathy': -0.7, 'volume': 116.1, 'mass': 119.12, 'charge': 0, 'flexibility': 0.9},
    'W': {'hydropathy': -0.9, 'volume': 227.8, 'mass': 204.23, 'charge': 0, 'flexibility': 0.7},
    'Y': {'hydropathy': -1.3, 'volume': 193.6, 'mass': 181.19, 'charge': 0, 'flexibility': 0.75},
    'V': {'hydropathy': 4.2, 'volume': 140.0, 'mass': 117.15, 'charge': 0, 'flexibility': 0.75},
}

# Force constants derived from vibrational data (relative scale)
AA_FORCE_CONSTANTS = {
    'G': 0.50, 'A': 0.65, 'S': 0.70, 'P': 0.75, 'V': 0.80,
    'T': 0.82, 'C': 0.85, 'I': 0.88, 'L': 0.88, 'N': 0.90,
    'D': 0.92, 'Q': 0.95, 'K': 0.98, 'E': 1.00, 'M': 1.02,
    'H': 1.05, 'F': 1.10, 'R': 1.15, 'Y': 1.18, 'W': 1.25,
}


def poincare_distance_from_origin(x: "np.ndarray", c: float = 1.0) -> float:
    """Compute hyperbolic distance from origin in Poincaré ball.

    Formula: d(0, x) = (2 / sqrt(c)) * arctanh(|x|)

    Args:
        x: Point in Poincaré ball (numpy array)
        c: Curvature parameter (default 1.0)

    Returns:
        Hyperbolic distance from origin
    """
    import numpy as np
    euclidean_norm = np.linalg.norm(x)
    # Clamp to avoid numerical issues at boundary
    euclidean_norm = np.clip(euclidean_norm, 0, 1 - 1e-7)
    return (2.0 / np.sqrt(c)) * np.arctanh(euclidean_norm)


def load_padic_embeddings(curvature: float = 1.0):
    """Load p-adic embeddings from trained VAE.

    Args:
        curvature: Poincaré ball curvature (default 1.0)

    Returns:
        Tuple of (radii dict, embeddings dict) where radii are
        hyperbolic distances from origin (NOT Euclidean norms).
    """
    import torch
    import json
    import numpy as np

    if not CODON_MAPPING_FILE.exists() or not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embedding files not found at {DATA_DIR}")

    with open(CODON_MAPPING_FILE) as f:
        mapping = json.load(f)

    codon_to_pos = mapping['codon_to_position']
    emb_data = torch.load(EMBEDDINGS_FILE, map_location='cpu', weights_only=False)
    z = emb_data['z_B_hyp'].numpy()

    # Group by amino acid
    aa_embs = {}
    for codon, pos in codon_to_pos.items():
        aa = CODON_TO_AA.get(codon)
        if aa:
            if aa not in aa_embs:
                aa_embs[aa] = []
            aa_embs[aa].append(z[pos])

    # Average embeddings per amino acid
    # V5.12.2 FIX: Use hyperbolic distance, not Euclidean norm
    radii = {}
    embeddings = {}
    for aa in aa_embs:
        mean_emb = np.mean(aa_embs[aa], axis=0)
        radii[aa] = poincare_distance_from_origin(mean_emb, c=curvature)
        embeddings[aa] = mean_emb

    return radii, embeddings
