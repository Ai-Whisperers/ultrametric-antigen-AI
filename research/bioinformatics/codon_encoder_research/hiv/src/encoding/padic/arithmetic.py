"""
P-adic arithmetic operations.

Implements addition, subtraction, and multiplication in p-adic integers.
"""
from .number import PadicNumber


def padic_add(a: PadicNumber, b: PadicNumber) -> PadicNumber:
    """
    Add two p-adic numbers.

    Args:
        a: First p-adic number
        b: Second p-adic number

    Returns:
        Sum a + b

    Raises:
        ValueError: If primes don't match
    """
    if a.prime != b.prime:
        raise ValueError(f"Prime mismatch: {a.prime} vs {b.prime}")

    max_len = max(len(a), len(b))
    precision = max(a.precision, b.precision)

    result_digits = []
    carry = 0

    for i in range(min(max_len + 1, precision)):
        total = a[i] + b[i] + carry
        result_digits.append(total % a.prime)
        carry = total // a.prime

    return PadicNumber(digits=result_digits, prime=a.prime, precision=precision)


def padic_subtract(a: PadicNumber, b: PadicNumber) -> PadicNumber:
    """
    Subtract two p-adic numbers (a - b).

    Uses complement arithmetic for subtraction.

    Args:
        a: First p-adic number
        b: Second p-adic number

    Returns:
        Difference a - b
    """
    if a.prime != b.prime:
        raise ValueError(f"Prime mismatch: {a.prime} vs {b.prime}")

    max_len = max(len(a), len(b))
    precision = max(a.precision, b.precision)

    result_digits = []
    borrow = 0

    for i in range(min(max_len, precision)):
        diff = a[i] - b[i] - borrow
        if diff < 0:
            diff += a.prime
            borrow = 1
        else:
            borrow = 0
        result_digits.append(diff)

    return PadicNumber(digits=result_digits, prime=a.prime, precision=precision)


def padic_multiply(a: PadicNumber, b: PadicNumber) -> PadicNumber:
    """
    Multiply two p-adic numbers.

    Uses standard polynomial multiplication with carry.

    Args:
        a: First p-adic number
        b: Second p-adic number

    Returns:
        Product a * b
    """
    if a.prime != b.prime:
        raise ValueError(f"Prime mismatch: {a.prime} vs {b.prime}")

    precision = max(a.precision, b.precision)
    result_len = min(len(a) + len(b), precision)
    result_digits = [0] * result_len

    for i, da in enumerate(a.digits):
        for j, db in enumerate(b.digits):
            if i + j < result_len:
                result_digits[i + j] += da * db

    return PadicNumber(digits=result_digits, prime=a.prime, precision=precision)


def padic_negate(a: PadicNumber) -> PadicNumber:
    """
    Negate a p-adic number (additive inverse).

    In p-adic integers, -a is represented as (p-1, p-1, ...) - a + 1.

    Args:
        a: P-adic number to negate

    Returns:
        -a in p-adic representation
    """
    # For finite precision, we compute p^n - a
    complement = [a.prime - 1 - d for d in a.digits]
    result = PadicNumber(digits=complement, prime=a.prime, precision=a.precision)

    # Add 1
    one = PadicNumber(digits=[1], prime=a.prime, precision=a.precision)
    return padic_add(result, one)


def padic_scale(a: PadicNumber, scalar: int) -> PadicNumber:
    """
    Multiply p-adic number by integer scalar.

    Args:
        a: P-adic number
        scalar: Integer multiplier

    Returns:
        a * scalar
    """
    if scalar == 0:
        return PadicNumber(digits=[0], prime=a.prime, precision=a.precision)
    if scalar == 1:
        return a

    b = PadicNumber.from_integer(abs(scalar), prime=a.prime, precision=a.precision)
    result = padic_multiply(a, b)

    if scalar < 0:
        result = padic_negate(result)

    return result
