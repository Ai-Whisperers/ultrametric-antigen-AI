#!/usr/bin/env python3
"""
HIV Analysis Command Line Interface

Main entry point for the hiv-analysis command defined in pyproject.toml.
Provides access to all pipeline components from the command line.
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HIV Sequence Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hiv-analysis setup                    # Initialize data structure
  hiv-analysis align input.fasta        # Run MAFFT alignment
  hiv-analysis score aligned.fasta      # Calculate conservation scores
  hiv-analysis view aligned.fasta       # Create visualization  
  hiv-analysis export aligned.fasta     # Export to multiple formats
  
For more information, see: https://github.com/Ai-Whisperers/ultrametric-antigen-AI
        """
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "align", "score", "view", "export", "help"],
        help="Pipeline command to run"
    )
    
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Input FASTA file (required for align, score, view, export)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file/directory"
    )
    
    parser.add_argument(
        "--algorithm",
        default="auto",
        choices=["auto", "linsi", "ginsi", "einsi", "fftns", "fftnsi"],
        help="MAFFT alignment algorithm (default: auto)"
    )
    
    parser.add_argument(
        "--format",
        default="fasta,clustal",
        help="Export formats (comma-separated): fasta,clustal,phylip,nexus,msf"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of threads for alignment (default: 2)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", 
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file requirement
    if args.command in ["align", "score", "view", "export"] and not args.input_file:
        parser.error(f"Command '{args.command}' requires an input file")
        
    # Execute commands
    try:
        if args.command == "help":
            parser.print_help()
            return 0
        elif args.command == "setup":
            from .scripts.setup_hiv_data import main as setup_main
            return setup_main()
        elif args.command == "align":
            print(f"🔄 Aligning sequences: {args.input_file}")
            print("💡 Use python3 hiv-analysis/scripts/mafft_wrapper.py for now")
            print("📋 Full CLI integration coming in future version")
            return 0
        elif args.command == "score":
            print(f"📊 Scoring conservation: {args.input_file}")
            print("💡 Use python3 hiv-analysis/scripts/conservation_scorer.py for now")
            print("📋 Full CLI integration coming in future version")
            return 0
        elif args.command == "view":
            print(f"👁️ Creating visualization: {args.input_file}")
            print("💡 Use python3 hiv-analysis/scripts/alignment_viewer.py for now")
            print("📋 Full CLI integration coming in future version")
            return 0
        elif args.command == "export":
            print(f"📤 Exporting formats: {args.input_file}")
            print("💡 Use python3 hiv-analysis/scripts/format_exporter.py for now")
            print("📋 Full CLI integration coming in future version")
            return 0
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())