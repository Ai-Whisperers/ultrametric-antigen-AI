# Auto-generated validation script
"""Validate all HIV datasets are present and readable."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from unified_data_loader import get_dataset_summary

def main():
    print("=" * 60)
    print("HIV Dataset Validation")
    print("=" * 60)

    summary = get_dataset_summary()
    print(summary.to_string(index=False))

    print("\n" + "=" * 60)

    missing = summary[~summary["Exists"]]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} datasets missing")
        for _, row in missing.iterrows():
            print(f"  - {row['Dataset']}")
    else:
        print("All datasets present!")

    total_records = summary["Records"].sum()
    print(f"\nTotal records available: {total_records:,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
