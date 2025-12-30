# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified CLI for Bioinformatics Tools.

This is the main entry point for all partner-specific bioinformatics tools
in the Ternary VAE project.

Usage:
    # List all available tools
    python biotools.py --list

    # Run a specific tool
    python biotools.py arbovirus-primers --use-ncbi
    python biotools.py pathogen-amp --use-dramp --pathogen A_baumannii
    python biotools.py mutation-effect --use-protherm --mutations G45A,A123G
    python biotools.py tdr-screening --use-stanford --demo
    python biotools.py la-selection --use-stanford --demo

    # Initialize all data (download sequences, train models)
    python biotools.py init --all
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
        "flags": ["--use-ncbi"],
    },
    # Carlos Brizuela - AMPs
    "pathogen-amp": {
        "module": "carlos_brizuela.scripts.B1_pathogen_specific_design",
        "description": "Design pathogen-specific antimicrobial peptides",
        "partner": "Carlos Brizuela",
        "flags": ["--use-dramp"],
    },
    "microbiome-amp": {
        "module": "carlos_brizuela.scripts.B8_microbiome_safe_amps",
        "description": "Design microbiome-safe AMPs (kill pathogens, spare commensals)",
        "partner": "Carlos Brizuela",
        "flags": ["--use-dramp"],
    },
    "synthesis-amp": {
        "module": "carlos_brizuela.scripts.B10_synthesis_optimization",
        "description": "Optimize AMPs for synthesis feasibility",
        "partner": "Carlos Brizuela",
        "flags": ["--use-dramp"],
    },
    # Jose Colbes - Protein Stability
    "mutation-effect": {
        "module": "jose_colbes.scripts.C4_mutation_effect_predictor",
        "description": "Predict mutation effects on protein stability (DDG)",
        "partner": "Jose Colbes",
        "flags": ["--use-protherm"],
    },
    # HIV Research Package
    "tdr-screening": {
        "module": "hiv_research_package.scripts.H6_tdr_screening",
        "description": "Screen for transmitted drug resistance in HIV",
        "partner": "HIV Research Package",
        "flags": ["--use-stanford"],
    },
    "la-selection": {
        "module": "hiv_research_package.scripts.H7_la_injectable_selection",
        "description": "Assess eligibility for long-acting injectable HIV therapy",
        "partner": "HIV Research Package",
        "flags": ["--use-stanford"],
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
    print("       python biotools.py init --all  # Initialize all data")
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
