# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Peptide Linguistics Module.

Implements linguistic analysis of protein and peptide sequences,
treating amino acids as a language with grammar and syntax.

Key Components:
- PeptideGrammar: Context-free grammar for peptide motifs
- TreeLSTM: Tree-structured LSTM for hierarchical sequence modeling
- MotifParser: Parse peptide sequences into grammatical structures

Key Insights:
1. Proteins have syntax: secondary structure rules, domain boundaries
2. Amino acids have semantics: physicochemical properties as meaning
3. Codon-protein mapping is analogous to morphology in linguistics

Mathematical Connections:
- Chomsky hierarchy applies to biological sequences
- Regular motifs (Type 3) -> simple patterns (NxS/T glycosylation)
- Context-free (Type 2) -> secondary structure (palindromic stems)
- Context-sensitive (Type 1) -> tertiary contacts

References:
- Searls (2002): The language of genes
- Tai (2015): Improved Semantic Representations From Tree-Structured LSTMs
- Dyer et al. (2016): Recurrent Neural Network Grammars
"""

from src._experimental.linguistics.peptide_grammar import (
    GrammarRule,
    MotifParser,
    ParseTree,
    PeptideGrammar,
    ProteinMotif,
)
from src._experimental.linguistics.tree_lstm import (
    ChildSumTreeLSTM,
    NaryTreeLSTM,
    ProteinTreeEncoder,
    TreeLSTM,
    TreeNode,
)

__all__ = [
    # Grammar
    "PeptideGrammar",
    "GrammarRule",
    "MotifParser",
    "ParseTree",
    "ProteinMotif",
    # TreeLSTM
    "TreeLSTM",
    "ChildSumTreeLSTM",
    "NaryTreeLSTM",
    "ProteinTreeEncoder",
    "TreeNode",
]
