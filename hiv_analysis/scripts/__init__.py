"""
HIV Analysis Scripts Package

Collection of bioinformatics scripts for HIV sequence analysis:
- alignment_viewer: Interactive alignment visualization
- conservation_scorer: Conservation analysis with multiple metrics
- format_exporter: Export alignments to standard formats  
- mafft_wrapper: MAFFT multiple sequence alignment
- setup_hiv_data: Data structure initialization

Each script can be run standalone or imported as a module.
"""

# Import all main script modules for easy access
try:
    from . import alignment_viewer
    from . import conservation_scorer
    from . import format_exporter  
    from . import mafft_wrapper
    from . import setup_hiv_data
    
    __all__ = [
        "alignment_viewer",
        "conservation_scorer",
        "format_exporter", 
        "mafft_wrapper",
        "setup_hiv_data"
    ]
except ImportError:
    # Graceful fallback if dependencies aren't installed
    __all__ = []