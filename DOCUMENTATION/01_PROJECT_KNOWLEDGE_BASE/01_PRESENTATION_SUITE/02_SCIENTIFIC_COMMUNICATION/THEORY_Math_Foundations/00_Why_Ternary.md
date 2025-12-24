# Why Ternary? The Theoretical Advantage

> **Source**: Based on _Ternary Computing: Theoretically Better than Binary_ (Asianometry).

## 1. The Radix Economy

The efficiency of a number system is defined by its **Radix Economy** ($E$):
$$ E(r, N) = r \times \lceil \log_r N \rceil $$
Where $r$ is the radix (base) and $N$ is the number to represent.

- **Base 2 (Binary)**: Simple switches (ON/OFF).
- **Base 10 (Decimal)**: Human intuition.
- **Base $e$ (2.718...)**: The theoretical optimum.

Since we cannot have a fractional base of transistors, **Base 3 (Ternary)** is the closest integer to $e$, making it mathematically more efficient than Binary.

## 2. Trits vs. Bits

- **1 Trit** $\approx 1.585$ Bits ($\log_2 3$).
- A 10-digit decimal number requires:
  - **40 Bits** (Binary)
  - **21 Trits** (Ternary) -> **47% hardware savings** in terms of "digits".

## 3. History: The Setun Computer (1958)

Built at Moscow State University, Setun was a balanced ternary computer (-1, 0, +1).

- **Logic**: Used "Fair" logic (Yes, No, Unknown).
- **Efficiency**: Simpler architecture for arithmetic operations compared to binary counterparts of the era.
- **Demise**: Lack of native 3-state hardware switches (transistors are naturally binary).

## 4. Modern Relevance: "1.58-bit" LLMs

We are seeing a resurgence of ternary logic in AI:

- **Quantization**: Weights in modern Large Language Models (LLMs) are often quantized to {-1, 0, 1}.
- **Sparsity**: The '0' state allows for massive compute savings (skip multiplications).
- **Ternary VAEs**: Our project leverages this "Natural Trinity" ($3^N$) to map biological states (Wildtype, Variant, Deleted) more naturally than binary encoding.
