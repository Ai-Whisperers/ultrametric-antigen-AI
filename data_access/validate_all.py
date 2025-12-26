"""
Comprehensive validation script for data access module.

This script tests all API connections, validates data retrieval,
and generates a detailed report of the data access capabilities.

Usage:
    python -m data_access.validate_all
    python -m data_access.validate_all --live  # Include live API tests
    python -m data_access.validate_all --save  # Save results to files
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n--- {text} ---")


def print_result(name: str, status: str, message: str = "") -> None:
    """Print a formatted result."""
    symbols = {"OK": "[OK]", "FAIL": "[!!]", "SKIP": "[--]", "INFO": "[..]"}
    symbol = symbols.get(status, "[??]")
    msg = f": {message}" if message else ""
    print(f"  {symbol} {name}{msg}")


class DataAccessValidator:
    """Comprehensive validator for data access module."""

    def __init__(self, live_tests: bool = False, output_dir: Optional[Path] = None):
        """
        Initialize validator.

        Args:
            live_tests: Whether to run live API tests
            output_dir: Directory for saving results
        """
        self.live_tests = live_tests
        self.output_dir = output_dir
        self.results = []

    def validate_imports(self) -> bool:
        """Validate all imports work correctly."""
        print_section("Validating Imports")

        imports_ok = True

        # Core imports
        try:
            from data_access import DataHub, settings
            print_result("DataHub", "OK")
        except ImportError as e:
            print_result("DataHub", "FAIL", str(e))
            imports_ok = False

        # Client imports
        clients = [
            ("NCBIClient", "data_access.clients"),
            ("HIVDBClient", "data_access.clients"),
            ("CBioPortalClient", "data_access.clients"),
            ("CARDClient", "data_access.clients"),
            ("BVBRCClient", "data_access.clients"),
            ("MalariaGENClient", "data_access.clients"),
        ]

        for client_name, module in clients:
            try:
                exec(f"from {module} import {client_name}")
                print_result(client_name, "OK")
            except ImportError as e:
                print_result(client_name, "FAIL", str(e))
                # MalariaGEN may fail if package not installed
                if client_name != "MalariaGENClient":
                    imports_ok = False

        # Integration imports
        try:
            from data_access.integration import (
                HIVAnalysisIntegration,
                SequenceProcessor,
                ResultsExtractor,
            )
            print_result("Integration modules", "OK")
        except ImportError as e:
            print_result("Integration modules", "FAIL", str(e))
            imports_ok = False

        return imports_ok

    def validate_configuration(self) -> bool:
        """Validate configuration settings."""
        print_section("Validating Configuration")

        from data_access import settings

        config_ok = True

        # Check NCBI config
        if settings.ncbi.email:
            print_result("NCBI Email", "OK", settings.ncbi.email)
        else:
            print_result("NCBI Email", "SKIP", "Not configured")

        if settings.ncbi.api_key:
            print_result("NCBI API Key", "OK", "Configured")
        else:
            print_result("NCBI API Key", "SKIP", "Not configured (limited rate)")

        # Check other endpoints
        print_result("HIVDB Endpoint", "OK", settings.hivdb.endpoint)
        print_result("cBioPortal URL", "OK", settings.cbioportal.url)
        print_result("CARD API URL", "OK", settings.card.api_url)
        print_result("BV-BRC API URL", "OK", settings.bvbrc.api_url)
        print_result("Timeout", "INFO", f"{settings.timeout}s")

        return config_ok

    def validate_datahub(self) -> bool:
        """Validate DataHub functionality."""
        print_section("Validating DataHub")

        from data_access import DataHub

        hub = DataHub()

        # Test lazy loading
        print_result("Lazy loading", "INFO", "Testing client instantiation")

        clients_tested = 0
        for client_name in ["hivdb", "cbioportal", "card", "bvbrc"]:
            try:
                client = getattr(hub, client_name)
                print_result(f"  {client_name}", "OK")
                clients_tested += 1
            except Exception as e:
                print_result(f"  {client_name}", "FAIL", str(e))

        return clients_tested >= 3

    def validate_connections(self) -> pd.DataFrame:
        """Validate API connections."""
        print_section("Validating API Connections")

        from data_access import DataHub

        hub = DataHub()

        if self.live_tests:
            print_result("Mode", "INFO", "Live API testing enabled")
            results = hub.test_connections()

            for _, row in results.iterrows():
                status = "OK" if row["status"] == "OK" else (
                    "SKIP" if row["status"] == "SKIP" else "FAIL"
                )
                print_result(row["api"], status, row["message"][:50])

            self.results.append(("connections", results))
            return results
        else:
            print_result("Mode", "INFO", "Mock testing only (use --live for API tests)")
            return pd.DataFrame()

    def validate_hivdb_analysis(self) -> bool:
        """Validate HIVDB analysis capabilities."""
        print_section("Validating HIVDB Analysis")

        from data_access.integration import HIVAnalysisIntegration

        integration = HIVAnalysisIntegration()

        # Test mutation parsing
        test_mutations = ["M184V", "K103N", "D30N", "RT:M41L"]
        mutations_ok = True

        for mut_str in test_mutations:
            try:
                mut = integration.parse_mutation(mut_str)
                print_result(f"Parse '{mut_str}'", "OK", f"pos={mut.position}")
            except Exception as e:
                print_result(f"Parse '{mut_str}'", "FAIL", str(e))
                mutations_ok = False

        return mutations_ok

    def validate_sequence_processor(self) -> bool:
        """Validate sequence processing capabilities."""
        print_section("Validating Sequence Processor")

        from data_access.integration import SequenceProcessor

        processor = SequenceProcessor()

        # Test sequence
        test_seq = "ATGTTTAGAACAGGAGGTTTTAAA"

        # Test cleaning
        cleaned = processor.clean_sequence("atg ttt\naga aca gga ggt ttt aaa")
        print_result("Sequence cleaning", "OK" if cleaned else "FAIL")

        # Test validation
        is_valid, messages = processor.validate_sequence(test_seq)
        print_result("Sequence validation", "OK" if is_valid else "INFO",
                     messages[0] if messages else "Valid")

        # Test codon extraction
        codons = list(processor.extract_codons(test_seq))
        print_result("Codon extraction", "OK", f"{len(codons)} codons")

        # Test frame detection
        best_frame = processor.find_best_frame(test_seq)
        print_result("Frame detection", "OK", f"frame={best_frame}")

        # Test statistics
        stats = processor.calculate_codon_statistics(test_seq)
        print_result("Codon statistics", "OK" if not stats.empty else "FAIL")

        return True

    def validate_results_extractor(self) -> bool:
        """Validate results extraction capabilities."""
        print_section("Validating Results Extractor")

        from data_access.integration import ResultsExtractor, ComprehensiveResults

        extractor = ResultsExtractor()

        # Test HIVDB extraction
        mock_hivdb = {
            "data": {
                "viewer": {
                    "sequenceAnalysis": [{
                        "drugResistance": [{
                            "drugClass": {"name": "NRTI"},
                            "gene": {"name": "RT"},
                            "drugScores": [
                                {"drug": {"name": "ABC"}, "score": 10, "text": "Low"}
                            ]
                        }]
                    }]
                }
            }
        }

        result = extractor.extract_hivdb_resistance(mock_hivdb, "test")
        print_result("HIVDB extraction", "OK" if not result.is_empty else "FAIL",
                     f"{result.record_count} records")

        # Test comprehensive results
        results = ComprehensiveResults()
        results.add_result(result)
        print_result("Comprehensive results", "OK",
                     f"summary: {results.summary.get('total_records', 0)} records")

        # Test report generation
        report = extractor.create_analysis_report(results)
        print_result("Report generation", "OK" if report else "FAIL")

        return True

    def run_all(self) -> dict:
        """Run all validations."""
        print_header("DATA ACCESS MODULE VALIDATION")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Live Tests: {'Enabled' if self.live_tests else 'Disabled'}")

        validation_results = {}

        # Run validations
        validation_results["imports"] = self.validate_imports()
        validation_results["configuration"] = self.validate_configuration()
        validation_results["datahub"] = self.validate_datahub()

        if self.live_tests:
            self.validate_connections()

        validation_results["hivdb_analysis"] = self.validate_hivdb_analysis()
        validation_results["sequence_processor"] = self.validate_sequence_processor()
        validation_results["results_extractor"] = self.validate_results_extractor()

        # Summary
        print_header("VALIDATION SUMMARY")

        passed = sum(1 for v in validation_results.values() if v)
        total = len(validation_results)

        for name, result in validation_results.items():
            print_result(name.replace("_", " ").title(),
                        "OK" if result else "FAIL")

        print(f"\nTotal: {passed}/{total} validations passed")

        # Save results if output dir specified
        if self.output_dir and self.results:
            print_section("Saving Results")
            self.output_dir.mkdir(parents=True, exist_ok=True)

            for name, df in self.results:
                path = self.output_dir / f"validation_{name}.csv"
                df.to_csv(path, index=False)
                print_result(f"Saved {name}", "OK", str(path))

        return validation_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate data access module functionality"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Include live API connection tests"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./validation_results"),
        help="Output directory for saved results"
    )

    args = parser.parse_args()

    output_dir = args.output_dir if args.save else None

    validator = DataAccessValidator(
        live_tests=args.live,
        output_dir=output_dir,
    )

    results = validator.run_all()

    # Exit with error code if any validation failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
