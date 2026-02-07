"""
HIV Sequence Analysis Package

A comprehensive bioinformatics package for HIV sequence analysis including:
- Multiple sequence alignment with MAFFT
- Conservation scoring (Shannon entropy, Simpson index, property-based)
- Interactive visualization (HTML and text formats)
- Format export (FASTA, ClustalW, PHYLIP, NEXUS, MSF)

Author: AI Whisperers
License: MIT
"""

__version__ = "0.1.0"
__author__ = "AI Whisperers"
__email__ = "research@ai-whisperers.com"

# Main package imports
from .scripts import *

__all__ = [
    "alignment_viewer",
    "conservation_scorer", 
    "format_exporter",
    "mafft_wrapper",
    "setup_hiv_data"
]