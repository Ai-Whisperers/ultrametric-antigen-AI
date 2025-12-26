"""
Results extraction utilities for comprehensive data analysis.

Provides structured extraction and export of results from all data sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import pandas as pd


@dataclass
class ExtractionResult:
    """Container for extraction results."""
    source: str
    data_type: str
    data: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    errors: list[str] = field(default_factory=list)

    @property
    def record_count(self) -> int:
        """Number of records extracted."""
        return len(self.data)

    @property
    def is_empty(self) -> bool:
        """Check if extraction returned no data."""
        return self.data.empty

    @property
    def has_errors(self) -> bool:
        """Check if extraction had errors."""
        return len(self.errors) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "data_type": self.data_type,
            "record_count": self.record_count,
            "columns": list(self.data.columns),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
        }


@dataclass
class ComprehensiveResults:
    """Container for multi-source extraction results."""
    results: list[ExtractionResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_result(self, result: ExtractionResult) -> None:
        """Add an extraction result."""
        self.results.append(result)
        self._update_summary()

    def _update_summary(self) -> None:
        """Update summary statistics."""
        self.summary = {
            "total_sources": len(self.results),
            "total_records": sum(r.record_count for r in self.results),
            "sources_with_data": sum(1 for r in self.results if not r.is_empty),
            "sources_with_errors": sum(1 for r in self.results if r.has_errors),
            "by_source": {
                r.source: {
                    "records": r.record_count,
                    "type": r.data_type,
                    "has_errors": r.has_errors,
                }
                for r in self.results
            },
        }

    def get_combined_dataframe(self, normalize: bool = True) -> pd.DataFrame:
        """
        Get combined DataFrame from all results.

        Args:
            normalize: Whether to normalize column names

        Returns:
            Combined DataFrame with source column
        """
        dfs = []
        for result in self.results:
            if not result.is_empty:
                df = result.data.copy()
                df["_source"] = result.source
                df["_data_type"] = result.data_type
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        if normalize:
            # Normalize column names to lowercase with underscores
            for df in dfs:
                df.columns = [
                    c.lower().replace(" ", "_").replace("-", "_")
                    for c in df.columns
                ]

        return pd.concat(dfs, ignore_index=True)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }, indent=2, default=str)


class ResultsExtractor:
    """
    Extract and structure results from multiple data sources.

    Provides comprehensive extraction from:
    - HIVDB drug resistance analysis
    - NCBI sequence data
    - cBioPortal cancer genomics
    - CARD antibiotic resistance
    - BV-BRC bacterial/viral genomes
    - MalariaGEN malaria genomics
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize extractor.

        Args:
            output_dir: Directory for saving results (optional)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_hivdb_resistance(
        self,
        analysis_result: dict,
        sequence_id: str = "query",
    ) -> ExtractionResult:
        """
        Extract structured results from HIVDB analysis.

        Args:
            analysis_result: Raw HIVDB API response
            sequence_id: Sequence identifier

        Returns:
            Structured extraction result
        """
        records = []
        errors = []
        metadata = {"sequence_id": sequence_id}

        try:
            if "data" in analysis_result and "viewer" in analysis_result["data"]:
                viewer = analysis_result["data"]["viewer"]
                if "sequenceAnalysis" in viewer:
                    for analysis in viewer["sequenceAnalysis"]:
                        # Extract drug resistance
                        if "drugResistance" in analysis:
                            for dr in analysis["drugResistance"]:
                                drug_class = dr.get("drugClass", {}).get("name", "")
                                gene = dr.get("gene", {}).get("name", "")

                                if "drugScores" in dr:
                                    for ds in dr["drugScores"]:
                                        records.append({
                                            "sequence_id": sequence_id,
                                            "gene": gene,
                                            "drug_class": drug_class,
                                            "drug": ds.get("drug", {}).get("name", ""),
                                            "score": ds.get("score", 0),
                                            "level": ds.get("text", ""),
                                        })

        except Exception as e:
            errors.append(str(e))

        return ExtractionResult(
            source="HIVDB",
            data_type="drug_resistance",
            data=pd.DataFrame(records),
            metadata=metadata,
            errors=errors,
        )

    def extract_ncbi_sequences(
        self,
        sequences: list[dict],
    ) -> ExtractionResult:
        """
        Extract structured results from NCBI sequence data.

        Args:
            sequences: List of sequence records

        Returns:
            Structured extraction result
        """
        records = []
        errors = []

        for seq in sequences:
            try:
                records.append({
                    "accession": seq.get("AccessionVersion", seq.get("Id", "")),
                    "title": seq.get("Title", ""),
                    "length": seq.get("Length", 0),
                    "organism": seq.get("Organism", ""),
                    "mol_type": seq.get("MolType", ""),
                    "create_date": seq.get("CreateDate", ""),
                    "update_date": seq.get("UpdateDate", ""),
                })
            except Exception as e:
                errors.append(f"Error processing sequence: {e}")

        return ExtractionResult(
            source="NCBI",
            data_type="sequences",
            data=pd.DataFrame(records),
            errors=errors,
        )

    def extract_cbioportal_mutations(
        self,
        mutations_df: pd.DataFrame,
        study_id: str = "",
    ) -> ExtractionResult:
        """
        Extract structured results from cBioPortal mutation data.

        Args:
            mutations_df: Raw mutations DataFrame
            study_id: Study identifier

        Returns:
            Structured extraction result
        """
        metadata = {"study_id": study_id}

        # Standardize column names
        if not mutations_df.empty:
            column_mapping = {
                "sampleId": "sample_id",
                "entrezGeneId": "entrez_gene_id",
                "proteinChange": "protein_change",
                "mutationType": "mutation_type",
                "variantType": "variant_type",
            }

            for old, new in column_mapping.items():
                if old in mutations_df.columns:
                    mutations_df = mutations_df.rename(columns={old: new})

        return ExtractionResult(
            source="cBioPortal",
            data_type="mutations",
            data=mutations_df,
            metadata=metadata,
        )

    def extract_card_resistance(
        self,
        resistance_df: pd.DataFrame,
    ) -> ExtractionResult:
        """
        Extract structured results from CARD resistance data.

        Args:
            resistance_df: Raw CARD DataFrame

        Returns:
            Structured extraction result
        """
        return ExtractionResult(
            source="CARD",
            data_type="antibiotic_resistance",
            data=resistance_df,
        )

    def extract_bvbrc_genomes(
        self,
        genomes_df: pd.DataFrame,
        organism: str = "",
    ) -> ExtractionResult:
        """
        Extract structured results from BV-BRC genome data.

        Args:
            genomes_df: Raw genomes DataFrame
            organism: Organism name

        Returns:
            Structured extraction result
        """
        metadata = {"organism": organism}

        return ExtractionResult(
            source="BV-BRC",
            data_type="genomes",
            data=genomes_df,
            metadata=metadata,
        )

    def extract_malariagen_samples(
        self,
        samples_df: pd.DataFrame,
        dataset: str = "Pf7",
    ) -> ExtractionResult:
        """
        Extract structured results from MalariaGEN sample data.

        Args:
            samples_df: Raw samples DataFrame
            dataset: Dataset name (Pf7, Pv4, Ag3)

        Returns:
            Structured extraction result
        """
        metadata = {"dataset": dataset}

        return ExtractionResult(
            source="MalariaGEN",
            data_type="samples",
            data=samples_df,
            metadata=metadata,
        )

    def save_results(
        self,
        results: ComprehensiveResults,
        prefix: str = "extraction",
    ) -> dict[str, Path]:
        """
        Save extraction results to files.

        Args:
            results: Comprehensive results to save
            prefix: Filename prefix

        Returns:
            Dictionary mapping data types to file paths
        """
        if not self.output_dir:
            raise ValueError("Output directory not configured")

        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_path = self.output_dir / f"{prefix}_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            f.write(results.to_json())
        saved_files["summary"] = summary_path

        # Save each result as CSV
        for result in results.results:
            if not result.is_empty:
                filename = f"{prefix}_{result.source.lower()}_{result.data_type}_{timestamp}.csv"
                file_path = self.output_dir / filename
                result.data.to_csv(file_path, index=False)
                saved_files[f"{result.source}_{result.data_type}"] = file_path

        # Save combined data
        combined = results.get_combined_dataframe()
        if not combined.empty:
            combined_path = self.output_dir / f"{prefix}_combined_{timestamp}.csv"
            combined.to_csv(combined_path, index=False)
            saved_files["combined"] = combined_path

        return saved_files

    def create_analysis_report(
        self,
        results: ComprehensiveResults,
    ) -> str:
        """
        Create a text report of extraction results.

        Args:
            results: Comprehensive results

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 60,
            "DATA EXTRACTION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Sources: {results.summary.get('total_sources', 0)}",
            f"Total Records: {results.summary.get('total_records', 0)}",
            f"Sources with Data: {results.summary.get('sources_with_data', 0)}",
            f"Sources with Errors: {results.summary.get('sources_with_errors', 0)}",
            "",
            "DETAILS BY SOURCE",
            "-" * 40,
        ]

        for result in results.results:
            lines.extend([
                f"\n{result.source} ({result.data_type})",
                f"  Records: {result.record_count}",
                f"  Columns: {', '.join(result.data.columns[:5])}...",
            ])

            if result.has_errors:
                lines.append(f"  Errors: {len(result.errors)}")
                for error in result.errors[:3]:
                    lines.append(f"    - {error[:50]}...")

            if result.metadata:
                for key, value in result.metadata.items():
                    lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])

        return "\n".join(lines)
