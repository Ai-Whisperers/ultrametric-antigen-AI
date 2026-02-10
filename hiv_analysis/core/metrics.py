#!/usr/bin/env python3
"""
p-adic Metrics for Sequence Analysis
Core mathematical functions for p-adic distances and ultrametric spaces
"""

import math
from typing import List, Union, Optional

def get_p_adic_valuation(n: int, p: int) -> int:
    """
    Calculate the p-adic valuation of an integer n
    The largest power of p that divides n
    """
    if n == 0:
        return float('inf')
    
    valuation = 0
    while n % p == 0:
        valuation += 1
        n //= p
        
    return valuation

def p_adic_distance(x: int, y: int, p: int) -> float:
    """
    Calculate the p-adic distance between two integers x and y
    d_p(x, y) = p^(-v_p(x-y))
    """
    if x == y:
        return 0.0
    
    diff = abs(x - y)
    valuation = get_p_adic_valuation(diff, p)
    
    return math.pow(p, -valuation)

class SequenceEncoder:
    """Encodes biological sequences into p-adic integers"""
    
    # Standard amino acid mapping (alphabetical)
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    AA_MAP = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
    @classmethod
    def encode_to_int(cls, sequence: str) -> int:
        """
        Encode a sequence as a base-20 integer
        Each amino acid is a digit in base 20
        """
        val = 0
        for i, aa in enumerate(reversed(sequence)):
            if aa in cls.AA_MAP:
                val += cls.AA_MAP[aa] * (20 ** i)
        return val

def sequence_ultrametric_distance(seq1: str, seq2: str, p: int = 2) -> float:
    """
    Calculate the p-adic ultrametric distance between two sequences
    """
    # Simple integer encoding for demonstration
    val1 = SequenceEncoder.encode_to_int(seq1)
    val2 = SequenceEncoder.encode_to_int(seq2)
    
    return p_adic_distance(val1, val2, p)
