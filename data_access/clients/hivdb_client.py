"""
Stanford HIVDB Sierra API client for HIV drug resistance analysis.

Provides access to:
- Drug resistance interpretation
- Mutation analysis
- Algorithm scoring
- Subtype detection

Uses the Sierra GraphQL API: https://hivdb.stanford.edu/graphql
"""

from typing import Optional

import pandas as pd
import requests

from ..config import settings


class HIVDBClient:
    """Client for Stanford HIVDB Sierra GraphQL API."""

    def __init__(self, endpoint: Optional[str] = None):
        """
        Initialize HIVDB client.

        Args:
            endpoint: GraphQL endpoint URL (defaults to public HIVDB)
        """
        self.endpoint = endpoint or settings.hivdb.endpoint
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _execute_query(self, query: str, variables: Optional[dict] = None) -> dict:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query response data
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = self.session.post(
            self.endpoint,
            json=payload,
            timeout=settings.timeout,
        )
        response.raise_for_status()

        result = response.json()
        if "errors" in result:
            raise RuntimeError(f"GraphQL errors: {result['errors']}")

        return result.get("data", {})

    def analyze_sequence(self, sequence: str, sequence_name: str = "query") -> dict:
        """
        Analyze an HIV sequence for drug resistance.

        Args:
            sequence: Nucleotide or amino acid sequence
            sequence_name: Name for the sequence

        Returns:
            Complete analysis results including resistance scores
        """
        query = """
        query AnalyzeSequence($sequences: [UnalignedSequenceInput]!) {
            sequenceAnalysis(sequences: $sequences) {
                inputSequence {
                    header
                    sequence
                }
                strain {
                    name
                }
                subtypeText
                validationResults {
                    level
                    message
                }
                alignedGeneSequences {
                    gene {
                        name
                    }
                    firstAA
                    lastAA
                    mutations {
                        text
                        position
                        AAs
                        isInsertion
                        isDeletion
                        isApobecMutation
                        isUnsequenced
                    }
                }
                drugResistance {
                    gene {
                        name
                    }
                    drugScores {
                        drug {
                            name
                            displayAbbr
                            drugClass {
                                name
                            }
                        }
                        score
                        level
                        text
                        partialScores {
                            mutations {
                                text
                            }
                            score
                        }
                    }
                }
            }
        }
        """

        variables = {"sequences": [{"header": sequence_name, "sequence": sequence}]}

        return self._execute_query(query, variables)

    def get_mutations_analysis(self, mutations: list[str], gene: str = "RT") -> dict:
        """
        Analyze a list of mutations directly.

        Args:
            mutations: List of mutation strings (e.g., ["M184V", "K65R"])
            gene: Gene name (PR, RT, IN)

        Returns:
            Analysis results for the mutations
        """
        query = """
        query MutationsAnalysis($mutations: [String]!) {
            mutationsAnalysis(mutations: $mutations) {
                validationResults {
                    level
                    message
                }
                drugResistance {
                    gene {
                        name
                    }
                    drugScores {
                        drug {
                            name
                            displayAbbr
                            drugClass {
                                name
                            }
                        }
                        score
                        level
                        text
                    }
                }
            }
        }
        """

        # Format mutations with gene prefix
        formatted_mutations = [f"{gene}:{m}" if ":" not in m else m for m in mutations]

        variables = {"mutations": formatted_mutations}
        return self._execute_query(query, variables)

    def get_drug_classes(self) -> pd.DataFrame:
        """
        Get all available drug classes.

        Returns:
            DataFrame with drug class information
        """
        query = """
        query {
            genes {
                name
                drugClasses {
                    name
                    fullName
                    drugs {
                        name
                        displayAbbr
                        fullName
                    }
                }
            }
        }
        """

        data = self._execute_query(query)
        genes = data.get("genes", [])

        rows = []
        seen = set()  # Avoid duplicates
        for gene in genes:
            for dc in gene.get("drugClasses", []):
                for drug in dc.get("drugs", []):
                    key = (dc["name"], drug["name"])
                    if key not in seen:
                        seen.add(key)
                        rows.append(
                            {
                                "gene": gene["name"],
                                "drug_class": dc["name"],
                                "drug_class_full": dc["fullName"],
                                "drug_name": drug["name"],
                                "drug_abbr": drug["displayAbbr"],
                                "drug_full_name": drug["fullName"],
                            }
                        )

        return pd.DataFrame(rows)

    def get_genes(self) -> pd.DataFrame:
        """
        Get all available genes with mutation information.

        Returns:
            DataFrame with gene information
        """
        query = """
        query {
            genes {
                name
                length
                strain {
                    name
                }
                drugClasses {
                    name
                }
            }
        }
        """

        data = self._execute_query(query)
        genes = data.get("genes", [])

        rows = []
        for gene in genes:
            rows.append(
                {
                    "gene": gene["name"],
                    "length": gene["length"],
                    "strain": gene["strain"]["name"],
                    "drug_classes": ", ".join(dc["name"] for dc in gene.get("drugClasses", [])),
                }
            )

        return pd.DataFrame(rows)

    def get_mutation_types(self, gene: str = "RT") -> pd.DataFrame:
        """
        Get mutation type classifications for a gene.

        Note: This method may not be available in all HIVDB versions.

        Args:
            gene: Gene name (PR, RT, IN)

        Returns:
            DataFrame with mutation classifications
        """
        # This endpoint may vary - return empty if not available
        try:
            query = """
            query {
                genes {
                    name
                    mutationTypes {
                        position
                        primaryType
                    }
                }
            }
            """
            data = self._execute_query(query)
            genes = data.get("genes", [])
            for g in genes:
                if g["name"] == gene:
                    return pd.DataFrame(g.get("mutationTypes", []))
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_algorithms(self) -> dict:
        """
        Get available resistance interpretation algorithms.

        Returns:
            Dictionary with algorithm version information
        """
        query = """
        query {
            currentVersion {
                text
                publishDate
            }
        }
        """

        data = self._execute_query(query)
        return data.get("currentVersion", {})

    def batch_analyze_sequences(self, sequences: list[tuple[str, str]]) -> list[dict]:
        """
        Analyze multiple sequences in batch.

        Args:
            sequences: List of (name, sequence) tuples

        Returns:
            List of analysis results
        """
        query = """
        query AnalyzeSequences($sequences: [UnalignedSequenceInput]!) {
            sequenceAnalysis(sequences: $sequences) {
                inputSequence {
                    header
                }
                subtypeText
                drugResistance {
                    gene {
                        name
                    }
                    drugScores {
                        drug {
                            displayAbbr
                            drugClass {
                                name
                            }
                        }
                        score
                        level
                        text
                    }
                }
            }
        }
        """

        variables = {
            "sequences": [{"header": name, "sequence": seq} for name, seq in sequences]
        }

        data = self._execute_query(query, variables)
        return data.get("sequenceAnalysis", [])

    def get_resistance_summary(self, sequence: str) -> pd.DataFrame:
        """
        Get a simplified resistance summary for a sequence.

        Args:
            sequence: HIV nucleotide or amino acid sequence

        Returns:
            DataFrame with drug resistance levels
        """
        result = self.analyze_sequence(sequence)

        analyses = result.get("sequenceAnalysis", [])
        if not analyses:
            return pd.DataFrame()

        analysis = analyses[0]
        rows = []

        for dr in analysis.get("drugResistance", []):
            gene = dr["gene"]["name"]
            for ds in dr.get("drugScores", []):
                rows.append(
                    {
                        "gene": gene,
                        "drug_class": ds["drug"]["drugClass"]["name"],
                        "drug": ds["drug"]["displayAbbr"],
                        "score": ds["score"],
                        "level": ds["level"],
                        "interpretation": ds["text"],
                    }
                )

        return pd.DataFrame(rows)

    def get_mutations_list(self, sequence: str) -> pd.DataFrame:
        """
        Extract all mutations from a sequence.

        Args:
            sequence: HIV nucleotide or amino acid sequence

        Returns:
            DataFrame with detected mutations
        """
        result = self.analyze_sequence(sequence)

        analyses = result.get("sequenceAnalysis", [])
        if not analyses:
            return pd.DataFrame()

        analysis = analyses[0]
        rows = []

        for ags in analysis.get("alignedGeneSequences", []):
            gene = ags["gene"]["name"]
            for mut in ags.get("mutations", []):
                rows.append(
                    {
                        "gene": gene,
                        "position": mut["position"],
                        "mutation": mut["text"],
                        "amino_acids": mut["AAs"],
                        "is_insertion": mut["isInsertion"],
                        "is_deletion": mut["isDeletion"],
                        "is_apobec": mut["isApobecMutation"],
                    }
                )

        return pd.DataFrame(rows)
