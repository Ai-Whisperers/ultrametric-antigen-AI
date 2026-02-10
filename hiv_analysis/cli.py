#!/usr/bin/env python3
"""
HIV Analysis Command Line Interface

Main entry point for the hiv-analysis command defined in pyproject.toml.
Provides access to all pipeline components from the command line.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Import package components
# Using absolute imports for consistency within the package
from hiv_analysis.scripts.setup_hiv_data import main as setup_main
from hiv_analysis.scripts.mafft_wrapper import MAFFTWrapper, MAFFTConfig
from hiv_analysis.scripts.conservation_scorer import ConservationScorer, ConservationConfig
from hiv_analysis.scripts.alignment_viewer import AlignmentViewer, ViewerConfig, SequenceAlignment as ViewerAlignment
from hiv_analysis.scripts.format_exporter import AlignmentExporter, ExportConfig, SequenceAlignment as ExporterAlignment
from hiv_analysis._version import __version__

def check_setup() -> bool:
    """Check if the project structure is properly set up"""
    # Check for core directories
    base_dirs = ["data/raw", "data/processed", "results"]
    for d in base_dirs:
        if not Path(d).exists():
            return False
    return True

def run_setup(args):
    """Run the setup command"""
    print("🚀 Initializing HIV analysis data structure...")
    return setup_main(base_dir=Path.cwd())

def run_align(args):
    """Run sequence alignment"""
    config = MAFFTConfig(
        algorithm=args.algorithm,
        threads=args.threads,
        quiet=not args.verbose
    )
    wrapper = MAFFTWrapper(config)
    
    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else None
    
    print(f"🔄 Aligning sequences: {input_path}")
    result = wrapper.align_sequences(input_path, output_path)
    
    if result["status"] in ["success", "mock_success"]:
        print(f"✅ Alignment successful: {result['output_file']}")
        if "statistics" in result and result["statistics"]:
            stats = result["statistics"]
            if stats.get("sequences"):
                print(f"📈 Sequences processed: {stats['sequences']}")
        return 0
    else:
        print(f"❌ Alignment failed: {result.get('error', 'Unknown error')}")
        return 1

def run_score(args):
    """Run conservation scoring"""
    score_types = args.score_types.split(",") if args.score_types else ['shannon', 'simpson', 'property']
    config = ConservationConfig(
        score_types=score_types,
        output_format=args.format if args.format in ['json', 'csv', 'tsv'] else 'json'
    )
    scorer = ConservationScorer(config)
    
    input_path = Path(args.input_file)
    print(f"📊 Calculating conservation: {input_path}")
    
    try:
        alignment = scorer.load_alignment(input_path)
        results = scorer.calculate_all_scores(alignment)
        
        output_prefix = args.output or str(Path("results/conservation/hiv_env_conservation"))
        output_path = Path(output_prefix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        scorer.export_scores(results, output_path)
        
        # Also generate report
        report_file = output_path.parent / "conservation_report.txt"
        scorer.generate_conservation_report(results, report_file)
        
        print(f"✅ Conservation scores exported to: {output_path}")
        print(f"📋 Analysis report saved to: {report_file}")
        return 0
    except Exception as e:
        print(f"❌ Conservation scoring failed: {e}")
        return 1

def run_view(args):
    """Run alignment visualization"""
    config = ViewerConfig(
        output_format=args.format if args.format in ['html', 'text', 'both'] else 'html',
        line_width=args.line_width
    )
    viewer = AlignmentViewer(config)
    
    input_path = Path(args.input_file)
    output_path = Path(args.output) if args.output else Path("results/alignment_view")
    
    print(f"👁️ Creating visualization for: {input_path}")
    try:
        alignment = ViewerAlignment.from_fasta(input_path)
        viewer.view_alignment(alignment, output_path)
        print(f"✅ Visualization created: {output_path}")
        return 0
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return 1

def run_export(args):
    """Run format export"""
    formats = args.format.split(",")
    config = ExportConfig(
        output_formats=formats
    )
    exporter = AlignmentExporter(config)
    
    input_path = Path(args.input_file)
    output_prefix = args.output or str(Path("results/exports/hiv_env_alignment"))
    
    print(f"📤 Exporting {input_path} to formats: {', '.join(formats)}")
    try:
        alignment = ExporterAlignment.from_fasta(input_path)
        output_files = exporter.export_alignment(alignment, output_prefix)
        print(f"✅ Export completed to: {Path(output_prefix).parent}")
        return 0
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1

def setup_logging(verbose: bool):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HIV Sequence Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
HIV Antigen AI v{__version__}
Examples:
  hiv-analysis setup                    # Initialize data structure
  hiv-analysis align input.fasta        # Run MAFFT alignment
  hiv-analysis score aligned.fasta      # Calculate conservation scores
  hiv-analysis view aligned.fasta       # Create visualization  
  hiv-analysis export aligned.fasta     # Export to multiple formats
  
For more information, see: https://github.com/Ai-Whisperers/hiv-antigen-ai
        """
    )
    
    parser.add_argument(
        "--version", action="version", version=f"hiv-antigen-ai {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    subparsers.add_parser("setup", help="Initialize project data structure")
    
    # Align command
    align_parser = subparsers.add_parser("align", help="Run sequence alignment")
    align_parser.add_argument("input_file", help="Input FASTA file")
    align_parser.add_argument("-o", "--output", help="Output alignment file")
    align_parser.add_argument("--algorithm", default="auto", choices=["auto", "linsi", "ginsi", "einsi", "fftns", "fftnsi"], help="MAFFT algorithm")
    align_parser.add_argument("--threads", type=int, default=2, help="Number of threads")
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Calculate conservation scores")
    score_parser.add_argument("input_file", help="Aligned FASTA file")
    score_parser.add_argument("-o", "--output", help="Output prefix")
    score_parser.add_argument("--score-types", default="shannon,simpson,property", help="Comma-separated score types")
    score_parser.add_argument("--format", default="json", choices=["json", "csv", "tsv"], help="Output format")
    
    # View command
    view_parser = subparsers.add_parser("view", help="Create alignment visualization")
    view_parser.add_argument("input_file", help="Aligned FASTA file")
    view_parser.add_argument("-o", "--output", help="Output file prefix")
    view_parser.add_argument("--format", default="html", choices=["html", "text", "both"], help="Output format")
    view_parser.add_argument("--line-width", type=int, default=80, help="Line width for visualization")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export alignment to multiple formats")
    export_parser.add_argument("input_file", help="Aligned FASTA file")
    export_parser.add_argument("-o", "--output", help="Output prefix")
    export_parser.add_argument("--format", default="fasta,clustal,phylip", help="Comma-separated formats")
    
    # Verbose global flag
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize logging
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return 0
        
    # Check setup for analysis commands
    if args.command in ["align", "score", "view", "export"]:
        if not check_setup():
            print("⚠️  Project structure not found. Running setup first...")
            setup_main(base_dir=Path.cwd())
            
    # Execute commands
    try:
        if args.command == "setup":
            return run_setup(args)
        elif args.command == "align":
            return run_align(args)
        elif args.command == "score":
            return run_score(args)
        elif args.command == "view":
            return run_view(args)
        elif args.command == "export":
            return run_export(args)
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
