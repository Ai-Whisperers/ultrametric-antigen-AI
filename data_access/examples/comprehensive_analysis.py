"""
Comprehensive analysis example demonstrating data access capabilities.

This script shows how to:
1. Connect to multiple biological databases
2. Extract and process data
3. Integrate with HIV p-adic analysis
4. Generate comprehensive results

Usage:
    python -m data_access.examples.comprehensive_analysis
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd


def main():
    """Run comprehensive analysis example."""
    print("=" * 60)
    print("COMPREHENSIVE DATA ACCESS ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # =========================================================================
    # 1. Initialize DataHub and check connections
    # =========================================================================
    print("\n--- Initializing Data Access ---")

    from data_access import DataHub
    from data_access.integration import (
        HIVAnalysisIntegration,
        SequenceProcessor,
        ResultsExtractor,
        ComprehensiveResults,
    )

    hub = DataHub()

    # Validate configuration
    warnings = hub.validate()
    if warnings:
        print("Configuration warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Configuration: OK")

    # =========================================================================
    # 2. Test API connections
    # =========================================================================
    print("\n--- Testing API Connections ---")

    try:
        connection_results = hub.test_connections()
        print(connection_results.to_string())
    except Exception as e:
        print(f"Connection test failed: {e}")
        connection_results = pd.DataFrame()

    # =========================================================================
    # 3. Demonstrate sequence processing
    # =========================================================================
    print("\n--- Sequence Processing Demo ---")

    processor = SequenceProcessor()

    # Example HIV RT sequence fragment
    sample_sequence = (
        "ATGCCTCAGGTCACTCTTTGGCAACGACCCCTCGTCACAATAAAGATAGGGGGGCAACTAA"
        "AGGAAGCTCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGAAATGAGTTTGCCAGG"
    )

    # Process sequence
    seq_info = processor.process_sequence(sample_sequence, "hiv_rt_fragment", "pol")

    print(f"Sequence ID: {seq_info.sequence_id}")
    print(f"Length: {seq_info.length} nt, {seq_info.codon_count} codons")
    print(f"Translation: {seq_info.translated_sequence[:20]}...")

    # Codon statistics
    stats = processor.calculate_codon_statistics(sample_sequence)
    print("\nCodon usage (top 5):")
    print(stats.head().to_string())

    # =========================================================================
    # 4. HIV drug resistance analysis integration
    # =========================================================================
    print("\n--- HIV Resistance Analysis Demo ---")

    hiv_integration = HIVAnalysisIntegration(hivdb_client=hub.hivdb)

    # Parse some mutations
    test_mutations = ["M184V", "K103N", "D30N", "T215Y"]
    print("Parsing mutations:")
    for mut_str in test_mutations:
        mut = hiv_integration.parse_mutation(mut_str)
        print(f"  {mut_str}: position={mut.position}, {mut.wild_type}->{mut.mutant}")

    # =========================================================================
    # 5. Results extraction and aggregation
    # =========================================================================
    print("\n--- Results Extraction Demo ---")

    extractor = ResultsExtractor()
    results = ComprehensiveResults()

    # Simulate HIVDB extraction
    mock_hivdb_result = {
        "data": {
            "viewer": {
                "sequenceAnalysis": [{
                    "drugResistance": [{
                        "drugClass": {"name": "NRTI"},
                        "gene": {"name": "RT"},
                        "drugScores": [
                            {"drug": {"name": "ABC"}, "score": 0, "text": "Susceptible"},
                            {"drug": {"name": "3TC"}, "score": 60, "text": "High-Level"},
                        ]
                    }]
                }]
            }
        }
    }

    hivdb_extraction = extractor.extract_hivdb_resistance(mock_hivdb_result, "demo")
    results.add_result(hivdb_extraction)

    # Simulate NCBI extraction
    ncbi_data = [
        {"AccessionVersion": "KY123456.1", "Title": "HIV-1 RT gene", "Length": 1500},
        {"AccessionVersion": "KY789012.1", "Title": "HIV-1 PR gene", "Length": 300},
    ]
    ncbi_extraction = extractor.extract_ncbi_sequences(ncbi_data)
    results.add_result(ncbi_extraction)

    # Print summary
    print(f"Total sources: {results.summary['total_sources']}")
    print(f"Total records: {results.summary['total_records']}")

    # =========================================================================
    # 6. Generate analysis report
    # =========================================================================
    print("\n--- Analysis Report ---")

    report = extractor.create_analysis_report(results)
    print(report[:1000])  # Print first 1000 chars

    # =========================================================================
    # 7. Live API demos (optional)
    # =========================================================================
    print("\n--- Live API Demos (if available) ---")

    # HIVDB - Get drug classes
    try:
        drug_classes = hub.hivdb.get_drug_classes()
        if drug_classes:
            print(f"HIVDB Drug Classes: {drug_classes}")
    except Exception as e:
        print(f"HIVDB: {e}")

    # cBioPortal - Get cancer types count
    try:
        cancer_types = hub.cbioportal.get_cancer_types()
        print(f"cBioPortal: {len(cancer_types)} cancer types available")
    except Exception as e:
        print(f"cBioPortal: {e}")

    # CARD - Get ESKAPE pathogens
    try:
        eskape = hub.card.get_eskape_pathogens()
        print(f"CARD: {len(eskape)} ESKAPE pathogens")
    except Exception as e:
        print(f"CARD: {e}")

    # BV-BRC - Get data summary
    try:
        bvbrc_summary = hub.bvbrc.get_data_summary()
        print("BV-BRC: Data summary retrieved")
    except Exception as e:
        print(f"BV-BRC: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
