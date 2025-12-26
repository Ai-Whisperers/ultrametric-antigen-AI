"""
P-adic number representation.

P-adic numbers are a completion of rationals under the p-adic metric,
where "closeness" is determined by divisibility by prime p.

For genetic encoding, p=3 is natural (3 nucleotides per codon).
"""
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True)
class PadicNumber:
    """
    Represents a p-adic number.

    P-adic numbers are represented as a sequence of digits in base p,
    potentially infinite to the "left" (toward higher powers of p).

    Attributes:
        digits: List of digits in base p (index 0 = units, 1 = p's place, etc.)
        prime: The prime p
        precision: Number of digits stored
    """

    digits: list[int]
    prime: int = 3
    precision: int = field(default=10, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize."""
        if self.prime < 2:
            raise ValueError(f"Prime must be >= 2: {self.prime}")

        # Normalize digits to be within [0, p-1]
        self._normalize()

        # Trim trailing zeros (high order)
        while len(self.digits) > 1 and self.digits[-1] == 0:
            self.digits.pop()

    def _normalize(self) -> None:
        """Normalize digits to base p with carry."""
        carry = 0
        for i in range(len(self.digits)):
            total = self.digits[i] + carry
            self.digits[i] = total % self.prime
            carry = total // self.prime

        while carry > 0:
            self.digits.append(carry % self.prime)
            carry //= self.prime

    @classmethod
    def from_integer(cls, n: int, prime: int = 3, precision: int = 10) -> "PadicNumber":
        """
        Create p-adic number from integer.

        Args:
            n: Integer to convert
            prime: Prime base
            precision: Number of digits

        Returns:
            PadicNumber representation
        """
        if n == 0:
            return cls(digits=[0], prime=prime, precision=precision)

        digits = []
        remaining = abs(n)
        while remaining > 0 and len(digits) < precision:
            digits.append(remaining % prime)
            remaining //= prime

        return cls(digits=digits, prime=prime, precision=precision)

    @classmethod
    def from_string(cls, s: str, prime: int = 3) -> "PadicNumber":
        """
        Create from string representation (digits from right to left).

        Example: "102" in base 3 = 1*3^2 + 0*3 + 2 = 11

        Args:
            s: String of digits
            prime: Prime base

        Returns:
            PadicNumber
        """
        digits = [int(c) for c in reversed(s)]
        return cls(digits=digits, prime=prime)

    def valuation(self) -> int:
        """
        Calculate p-adic valuation (order of p in number).

        The valuation is the index of the first nonzero digit.
        Returns infinity (represented as precision) if zero.

        Returns:
            p-adic valuation v_p(n)
        """
        for i, d in enumerate(self.digits):
            if d != 0:
                return i
        return self.precision  # Represents infinity

    def norm(self) -> float:
        """
        Calculate p-adic norm |n|_p = p^(-v_p(n)).

        Returns:
            p-adic absolute value
        """
        v = self.valuation()
        if v >= self.precision:
            return 0.0
        return self.prime ** (-v)

    def to_integer(self) -> int:
        """Convert to integer (truncates if too large)."""
        result = 0
        for i, d in enumerate(self.digits):
            result += d * (self.prime ** i)
        return result

    def __iter__(self) -> Iterator[int]:
        return iter(self.digits)

    def __len__(self) -> int:
        return len(self.digits)

    def __getitem__(self, index: int) -> int:
        if index < len(self.digits):
            return self.digits[index]
        return 0  # Implicit zeros

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PadicNumber):
            return False
        if self.prime != other.prime:
            return False
        # Compare digits
        max_len = max(len(self.digits), len(other.digits))
        for i in range(max_len):
            if self[i] != other[i]:
                return False
        return True

    def __hash__(self) -> int:
        return hash((tuple(self.digits), self.prime))

    def __str__(self) -> str:
        return "".join(str(d) for d in reversed(self.digits))

    def __repr__(self) -> str:
        return f"PadicNumber({self!s}, p={self.prime})"
