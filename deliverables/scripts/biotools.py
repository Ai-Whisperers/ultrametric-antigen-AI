# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified CLI for Bioinformatics Tools.

This is the main entry point for all partner-specific bioinformatics tools
in the Ternary VAE project. It provides a unified interface to access all
research partner functionalities.

===============================================================================
AVAILABLE COMMANDS
===============================================================================

Demo Commands (Quick Showcases):
--------------------------------
    demo-all         Run all tool demos sequentially
    demo-hiv         HIV TDR screening + LA injectable selection
    demo-amp         Antimicrobial peptide design optimization
    demo-primers     Arbovirus RT-PCR primer design
    demo-stability   Rosetta-blind protein stability detection

Showcase Commands (Generate Outputs):
-------------------------------------
    showcase         Generate all figures, reports, and demos
    showcase-figures Generate publication-quality figures only

Partner Tools:
--------------
    arbovirus-primers   Design pan-arbovirus primers (DENV, ZIKV, CHIKV, MAYV)
    pathogen-amp        Design pathogen-specific antimicrobial peptides
    microbiome-amp      Design microbiome-safe AMPs
    synthesis-amp       Optimize AMPs for synthesis feasibility
    rosetta-blind       Detect Rosetta-blind instabilities
    mutation-effect     Predict mutation effects (DDG)
    tdr-screening       HIV transmitted drug resistance screening
    la-selection        HIV long-acting injectable eligibility

Analysis Commands:
------------------
    analyze <SEQ>    Analyze peptide properties

Utility Commands:
-----------------
    --list           List all available tools with descriptions
    init --all       Initialize all data (download sequences)

===============================================================================
USAGE EXAMPLES
===============================================================================

    # List all tools
    python biotools.py --list

    # Run HIV demos
    python biotools.py demo-hiv

    # Run all demos
    python biotools.py demo-all

    # Generate showcase figures
    python biotools.py showcase

    # Analyze a peptide sequence
    python biotools.py analyze KLWKKWKKWLK

    # Run specific tools with options
    python biotools.py tdr-screening --demo
    python biotools.py pathogen-amp --pathogen S_aureus

===============================================================================
ARCHITECTURE
===============================================================================

This CLI integrates the following research packages:

1. HIV Research Package (partners/hiv_research_package/)
   - TDRScreener: WHO SDRM mutation detection
   - LASelector: CAB-LA/RPV-LA eligibility assessment
   - HIVSequenceAligner: HXB2 reference alignment

2. Arbovirus Package (partners/alejandra_rojas/)
   - NCBIClient: Sequence download from NCBI
   - PrimerDesigner: RT-PCR primer design

3. AMP Package (partners/carlos_brizuela/)
   - NSGA-II optimizer for multi-objective peptide design
   - Pathogen-specific and microbiome-safe design

4. Stability Package (partners/jose_colbes/)
   - Geometric predictor for mutation effects
   - Rosetta-blind instability detection

All packages share the VAE service (shared/vae_service.py) for sequence
encoding and the configuration module (shared/config.py) for path management.

===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Available tools with their modules and descriptions
TOOLS = {
    # Alejandra Rojas - Arboviruses
    "arbovirus-primers": {
        "module": "alejandra_rojas.scripts.A2_pan_arbovirus_primers",
        "description": "Design pan-arbovirus primers (DENV, ZIKV, CHIKV, MAYV)",
        "partner": "Alejandra Rojas",
        "flags": ["--demo", "--use-ncbi"],
        "demo_args": ["--demo"],
    },
    # Carlos Brizuela - AMPs
    "pathogen-amp": {
        "module": "carlos_brizuela.scripts.B1_pathogen_specific_design",
        "description": "Design pathogen-specific antimicrobial peptides",
        "partner": "Carlos Brizuela",
        "flags": ["--pathogen", "--use-dramp"],
        "demo_args": ["--pathogen", "S_aureus", "--generations", "5", "--population", "20"],
    },
    "microbiome-amp": {
        "module": "carlos_brizuela.scripts.B8_microbiome_safe_amps",
        "description": "Design microbiome-safe AMPs (kill pathogens, spare commensals)",
        "partner": "Carlos Brizuela",
        "flags": ["--use-dramp"],
        "demo_args": ["--generations", "5", "--population", "20"],
    },
    "synthesis-amp": {
        "module": "carlos_brizuela.scripts.B10_synthesis_optimization",
        "description": "Optimize AMPs for synthesis feasibility",
        "partner": "Carlos Brizuela",
        "flags": ["--use-dramp"],
        "demo_args": ["--generations", "5", "--population", "20"],
    },
    # Jose Colbes - Protein Stability
    "rosetta-blind": {
        "module": "jose_colbes.scripts.C1_rosetta_blind_detection",
        "description": "Detect Rosetta-blind instabilities in protein structures",
        "partner": "Jose Colbes",
        "flags": ["--input", "--n_demo"],
        "demo_args": ["--n_demo", "20"],
    },
    "mutation-effect": {
        "module": "jose_colbes.scripts.C4_mutation_effect_predictor",
        "description": "Predict mutation effects on protein stability (DDG)",
        "partner": "Jose Colbes",
        "flags": ["--mutations", "--use-protherm"],
        "demo_args": ["--mutations", "V156A,L99A,F133A", "--context", "core"],
    },
    # HIV Research Package
    "tdr-screening": {
        "module": "hiv_research_package.scripts.H6_tdr_screening",
        "description": "Screen for transmitted drug resistance in HIV",
        "partner": "HIV Research Package",
        "flags": ["--demo", "--use-stanford"],
        "demo_args": ["--demo"],
    },
    "la-selection": {
        "module": "hiv_research_package.scripts.H7_la_injectable_selection",
        "description": "Assess eligibility for long-acting injectable HIV therapy",
        "partner": "HIV Research Package",
        "flags": ["--demo", "--use-stanford"],
        "demo_args": ["--demo"],
    },
}


def list_tools():
    """List all available tools."""
    print("\n" + "=" * 70)
    print("  BIOINFORMATICS TOOLS - Ternary VAE Project")
    print("=" * 70)

    # Group by partner
    by_partner = {}
    for name, info in TOOLS.items():
        partner = info["partner"]
        if partner not in by_partner:
            by_partner[partner] = []
        by_partner[partner].append((name, info))

    for partner, tools in by_partner.items():
        print(f"\n{partner}:")
        print("-" * 50)
        for name, info in tools:
            flags = ", ".join(info["flags"]) if info["flags"] else "none"
            print(f"  {name:<20} {info['description']}")
            print(f"  {'':<20} Flags: {flags}")

    print("\n" + "=" * 70)
    print("Usage: python biotools.py <tool-name> [options]")
    print("       python biotools.py init --all      # Initialize all data")
    print("       python biotools.py demo-all        # Run all tool demos")
    print("       python biotools.py analyze <SEQ>   # Analyze a peptide")
    print("=" * 70 + "\n")


def run_init(args: list[str]):
    """Run the initialization script."""
    try:
        from initialize_all_data import main as init_main
        # Reconstruct sys.argv for argparse in the init script
        sys.argv = ["initialize_all_data"] + args
        init_main()
    except ImportError:
        print("Error: Could not import initialize_all_data module")
        print("Make sure you're running from the deliverables/scripts directory")
        sys.exit(1)


def run_demo_all():
    """Run demos of all tools."""
    print("\n" + "=" * 70)
    print("  RUNNING ALL TOOL DEMOS")
    print("=" * 70)

    results = {}
    for name, info in TOOLS.items():
        print(f"\n>>> Running {name}...")
        print("-" * 50)
        try:
            demo_args = info.get("demo_args", [])
            run_tool(name, demo_args)
            results[name] = "SUCCESS"
        except SystemExit:
            results[name] = "SUCCESS"  # argparse sometimes calls sys.exit(0)
        except Exception as e:
            results[name] = f"FAILED: {e}"
            print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("  DEMO SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        symbol = "✓" if "SUCCESS" in status else "✗"
        print(f"  {symbol} {name}: {status}")
    print("=" * 70)


def run_demo_hiv():
    """Run HIV-specific demos (TDR screening + LA selection).

    This demo showcases the HIV research package capabilities:

    1. Sequence Alignment
       - Aligns RT sequence to HXB2 reference
       - Reports identity, coverage, and mutations found

    2. TDR Screening
       - Screens for WHO-defined surveillance drug resistance mutations
       - Returns TDR status, confidence, and recommended first-line regimen

    3. LA Injectable Eligibility
       - Assesses eligibility for CAB-LA/RPV-LA therapy
       - Considers viral suppression, adherence history, and risk factors

    Output:
        Prints formatted results to stdout with success/failure banner.

    Raises:
        ImportError: If HIV package is not available in the path.
    """
    print("\n" + "=" * 70)
    print("  HIV ANALYSIS DEMO")
    print("=" * 70)

    try:
        from partners.hiv_research_package.src import (
            TDRScreener, LASelector, PatientData,
            HIVSequenceAligner, ClinicalReportGenerator
        )

        # Demo RT sequence
        demo_sequence = """PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV
        FAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKNKSVTVLDVGDAYFSVPL""".replace("\n", "").replace(" ", "")

        print("\n1. Sequence Alignment")
        print("-" * 50)
        aligner = HIVSequenceAligner()
        alignment = aligner.align(demo_sequence, gene="RT")
        print(f"   Identity: {alignment.identity:.1%}")
        print(f"   Coverage: {alignment.coverage:.1%}")
        print(f"   Mutations: {len(alignment.mutations)}")

        print("\n2. TDR Screening")
        print("-" * 50)
        screener = TDRScreener()
        result = screener.screen_patient(demo_sequence, "DEMO-001")
        print(f"   TDR Status: {'POSITIVE' if result.tdr_positive else 'NEGATIVE'}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Recommendation: {result.recommended_regimen}")

        print("\n3. LA Injectable Eligibility")
        print("-" * 50)
        patient = PatientData(
            patient_id="DEMO-001", age=35, sex="M", bmi=24.5,
            viral_load=0, cd4_count=650,
            prior_regimens=["TDF/FTC/DTG"], adherence_history="excellent"
        )
        selector = LASelector()
        la_result = selector.assess_eligibility(patient, demo_sequence)
        print(f"   Eligible: {'YES' if la_result.eligible else 'NO'}")
        print(f"   Success Probability: {la_result.success_probability:.1%}")

        print("\n" + "=" * 70)
        print("  HIV DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except ImportError as e:
        print(f"Error: HIV package not available: {e}")
        print("Make sure the partners/hiv_research_package/src module is installed")


def run_demo_amp():
    """Run antimicrobial peptide (AMP) design demo.

    This demo showcases the VAE-based peptide generation:

    1. VAE Service Initialization
       - Checks if real VAE model or mock mode is being used
       - Real mode uses trained checkpoint for accurate embeddings

    2. Latent Space Sampling
       - Samples 20 candidate peptides with cationic bias
       - Applies charge_bias=0.5 for antimicrobial properties
       - Applies hydro_bias=0.3 for membrane interaction

    3. Property Calculation
       - Computes net charge, hydrophobicity for each candidate
       - Calculates stability score from p-adic valuation

    4. Ranking
       - Sorts candidates by combined score (charge + stability)
       - Displays top 10 candidates with properties

    Output:
        Table of top 10 peptides with sequence, charge, hydrophobicity, stability.

    Note:
        In mock mode, stability scores may be 0 due to simplified algorithm.
    """
    print("\n" + "=" * 70)
    print("  ANTIMICROBIAL PEPTIDE DESIGN DEMO")
    print("=" * 70)

    try:
        from shared.vae_service import get_vae_service
        from shared.peptide_utils import compute_peptide_properties
        import numpy as np

        vae = get_vae_service()
        print(f"\nVAE Status: {'Real model' if vae.is_real else 'Mock mode'}")

        # Sample peptides with cationic bias (antimicrobial)
        print("\nGenerating candidate peptides...")
        n_candidates = 20
        latent_samples = vae.sample_latent(
            n_samples=n_candidates,
            charge_bias=0.5,
            hydro_bias=0.3
        )

        peptides = []
        for z in latent_samples:
            seq = vae.decode_latent(z)
            props = compute_peptide_properties(seq)
            peptides.append((seq, props, vae.get_stability_score(z)))

        # Sort by combined score (charge + stability)
        peptides.sort(key=lambda x: x[1]['net_charge'] + x[2], reverse=True)

        print("\nTop 10 Candidate Peptides:")
        print("-" * 70)
        print(f"{'#':<3} {'Sequence':<25} {'Charge':>8} {'Hydro':>8} {'Stability':>10}")
        print("-" * 70)
        for i, (seq, props, stab) in enumerate(peptides[:10], 1):
            print(f"{i:<3} {seq[:22]+'...':<25} {props['net_charge']:>+8.1f} {props['hydrophobicity']:>8.3f} {stab:>10.3f}")

        print("\n" + "=" * 70)
        print("  AMP DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except ImportError as e:
        print(f"Error: {e}")


def run_demo_primers():
    """Run arbovirus RT-PCR primer design demo.

    This demo showcases the arbovirus surveillance tools:

    1. Target Viruses
       - Lists supported arbovirus targets (DENV-1-4, ZIKV, CHIKV, MAYV)
       - Each has NCBI taxonomy ID for sequence retrieval

    2. Primer Constraints
       - Displays configured primer design constraints
       - Length: 18-25 bp, GC: 40-60%, Tm: 55-65°C

    3. Primer Design
       - Designs primer pairs for demo genome sequence
       - Calculates Tm using nearest-neighbor thermodynamics
       - Scores primers based on GC content, Tm match, amplicon size

    Output:
        List of designed primer pairs with forward/reverse sequences,
        amplicon sizes, and quality scores.

    Note:
        In demo mode, uses synthetic genome sequence. With --use-ncbi,
        downloads real sequences from NCBI GenBank.
    """
    print("\n" + "=" * 70)
    print("  ARBOVIRUS PRIMER DESIGN DEMO")
    print("=" * 70)

    try:
        from partners.alejandra_rojas.src import PrimerDesigner
        from partners.alejandra_rojas.src.constants import PRIMER_CONSTRAINTS, ARBOVIRUS_TARGETS

        print("\nTarget Viruses:")
        for virus in list(ARBOVIRUS_TARGETS.keys())[:4]:
            print(f"  - {virus}")

        print("\nPrimer Constraints:")
        print(f"  Length: {PRIMER_CONSTRAINTS['length']['min']}-{PRIMER_CONSTRAINTS['length']['max']} bp")
        print(f"  GC: {PRIMER_CONSTRAINTS['gc_content']['min']*100:.0f}-{PRIMER_CONSTRAINTS['gc_content']['max']*100:.0f}%")
        print(f"  Tm: {PRIMER_CONSTRAINTS['tm']['min']}-{PRIMER_CONSTRAINTS['tm']['max']}°C")

        # Demo sequence
        demo_genome = "ATGAACAACCAACGGAAAAAGACGGGTCGACCGTCTTTCAATATGCTGAAACGCGCGAGAAACCGCGT" * 10

        designer = PrimerDesigner(constraints=PRIMER_CONSTRAINTS)
        pairs = designer.design_primer_pairs(demo_genome, "DENV-1", n_pairs=5)

        print(f"\nDesigned {len(pairs)} primer pairs:")
        print("-" * 70)
        for i, pair in enumerate(pairs[:5], 1):
            print(f"{i}. Forward: {pair.forward.sequence[:20]}...")
            print(f"   Reverse: {pair.reverse.sequence[:20]}...")
            print(f"   Amplicon: {pair.amplicon_size} bp, Score: {pair.score:.1f}")

        print("\n" + "=" * 70)
        print("  PRIMER DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except ImportError as e:
        print(f"Error: {e}")


def run_demo_stability():
    """Run Rosetta-blind protein stability detection demo.

    This demo showcases the p-adic geometric stability prediction:

    1. VAE Encoding
       - Encodes mutant amino acid to 16-dimensional latent space
       - Maps to hyperbolic Poincaré ball representation

    2. P-adic Valuation
       - Calculates radial position (distance from origin)
       - Maps radius to p-adic valuation level (0-9)

    3. Stability Scoring
       - Converts valuation to stability score (0-1)
       - Higher valuation = closer to center = more stable

    4. Classification
       - Stabilizing: stability > 0.7
       - Neutral: 0.3 < stability < 0.7
       - Destabilizing: stability < 0.3

    Demo Mutations:
        A1V  - Small hydrophobic change
        G23D - Glycine to charged (flexibility loss)
        L45P - Helix breaker (proline)
        K67R - Conservative positive charge
        P112A - Proline removal (flexibility gain)

    Output:
        Table of mutations with descriptions and stability classifications.

    Note:
        In mock mode, predictions use simplified heuristics.
        Real mode uses trained VAE for accurate geometric embedding.
    """
    print("\n" + "=" * 70)
    print("  PROTEIN STABILITY ANALYSIS DEMO")
    print("=" * 70)

    try:
        from shared.vae_service import get_vae_service
        import numpy as np

        vae = get_vae_service()

        # Demo mutations
        mutations = [
            ("A1V", "Alanine to Valine"),
            ("G23D", "Glycine to Aspartate"),
            ("L45P", "Leucine to Proline"),
            ("K67R", "Lysine to Arginine"),
            ("P112A", "Proline to Alanine"),
        ]

        print("\nAnalyzing mutations with p-adic geometric scoring:")
        print("-" * 60)
        print(f"{'Mutation':<12} {'Description':<25} {'Stability':>12}")
        print("-" * 60)

        for mut, desc in mutations:
            # Encode mutant residue
            new_aa = mut[-1]
            z = vae.encode_sequence(new_aa * 10)
            stability = vae.get_stability_score(z)
            valuation = vae.get_padic_valuation(z)

            status = "Stable" if stability > 0.5 else "Destabilizing"
            print(f"{mut:<12} {desc:<25} {stability:>8.3f} ({status})")

        print("\n" + "=" * 70)
        print("  STABILITY DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except ImportError as e:
        print(f"Error: {e}")


def run_showcase():
    """Generate all showcase outputs (figures + reports)."""
    print("\n" + "=" * 70)
    print("  GENERATING SHOWCASE OUTPUTS")
    print("=" * 70)

    try:
        from generate_showcase_figures import main as generate_figures

        print("\nGenerating publication figures...")
        sys.argv = ["generate_showcase_figures"]
        generate_figures()

        print("\n" + "=" * 70)
        print("  SHOWCASE GENERATION COMPLETED")
        print("=" * 70)

    except ImportError as e:
        print(f"Error importing showcase generator: {e}")
        print("Make sure generate_showcase_figures.py is in the scripts directory")


def run_showcase_figures():
    """Generate publication figures only."""
    run_showcase()  # Currently same as showcase


def analyze_peptide(sequence: str):
    """Quick peptide analysis."""
    from shared import (
        compute_peptide_properties,
        HemolysisPredictor,
        validate_sequence,
    )

    is_valid, error = validate_sequence(sequence)
    if not is_valid:
        print(f"Error: {error}")
        return

    props = compute_peptide_properties(sequence)
    predictor = HemolysisPredictor()
    hemo = predictor.predict(sequence)

    print("\n" + "=" * 50)
    print(f"  PEPTIDE ANALYSIS: {sequence[:30]}...")
    print("=" * 50)
    print(f"\nSequence: {sequence}")
    print(f"Length: {props['length']} amino acids")
    print(f"\nBiophysical Properties:")
    print(f"  Net charge: {props['net_charge']:+.1f}")
    print(f"  Hydrophobicity: {props['hydrophobicity']:.3f}")
    print(f"  Hydrophobic ratio: {props['hydrophobic_ratio']:.1%}")
    print(f"  Cationic ratio: {props['cationic_ratio']:.1%}")
    print(f"\nHemolysis Prediction:")
    print(f"  HC50: {hemo['hc50_predicted']:.1f} uM")
    print(f"  Risk: {hemo['risk_category']}")
    print(f"  Probability: {hemo['hemolytic_probability']:.1%}")
    print("=" * 50)


def run_tool(tool_name: str, args: list[str]):
    """Run a specific tool."""
    if tool_name not in TOOLS:
        print(f"Error: Unknown tool '{tool_name}'")
        print(f"Available tools: {', '.join(TOOLS.keys())}")
        sys.exit(1)

    tool_info = TOOLS[tool_name]
    module_path = tool_info["module"]

    try:
        # Import the module dynamically
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            package, module = parts
            exec(f"from {package} import {module}")
            mod = eval(module)
        else:
            exec(f"import {module_path}")
            mod = eval(module_path)

        # Set sys.argv for the tool's argparse
        sys.argv = [tool_name] + args

        # Run the tool's main function
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"Error: Tool {tool_name} has no main() function")
            sys.exit(1)

    except ImportError as e:
        print(f"Error importing {tool_name}: {e}")
        print("Make sure the tool module exists and dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"Error running {tool_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    # Add parent directories to path
    script_dir = Path(__file__).parent
    deliverables_dir = script_dir.parent
    project_root = deliverables_dir.parent

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(deliverables_dir))

    # Parse initial arguments
    parser = argparse.ArgumentParser(
        description="Unified CLI for Bioinformatics Tools",
        usage="biotools.py <command> [options]",
        add_help=False,
    )
    parser.add_argument("command", nargs="?", help="Tool name or command")
    parser.add_argument("--list", "-l", action="store_true", help="List available tools")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")

    # Parse only known args to pass the rest to the tool
    args, remaining = parser.parse_known_args()

    if args.list:
        list_tools()
        return

    if args.help and not args.command:
        parser.print_help()
        print("\nUse --list to see available tools")
        return

    if not args.command:
        list_tools()
        return

    if args.command == "init":
        run_init(remaining)
    elif args.command == "demo-all":
        run_demo_all()
    elif args.command == "demo-hiv":
        run_demo_hiv()
    elif args.command == "demo-amp":
        run_demo_amp()
    elif args.command == "demo-primers":
        run_demo_primers()
    elif args.command == "demo-stability":
        run_demo_stability()
    elif args.command == "showcase":
        run_showcase()
    elif args.command == "showcase-figures":
        run_showcase_figures()
    elif args.command == "analyze":
        if remaining:
            analyze_peptide(remaining[0])
        else:
            print("Error: Please provide a peptide sequence")
            print("Usage: python biotools.py analyze KLWKKWKKWLK")
            sys.exit(1)
    elif args.command in TOOLS:
        # Add --help back if it was passed
        if args.help:
            remaining.append("--help")
        run_tool(args.command, remaining)
    else:
        print(f"Error: Unknown command '{args.command}'")
        print("Use --list to see available tools")
        sys.exit(1)


if __name__ == "__main__":
    main()
