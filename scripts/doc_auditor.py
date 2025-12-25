import os


def audit_docs():
    inventory = {
        "Executive": [],
        "Scientific": [],
        "Developer": [],
        "Uncategorized": [],
    }

    # Files to ignore (e.g. within .git, .mypy_cache, etc.)
    ignore_dirs = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "venv",
        ".venv",
    }

    for root, dirs, files in os.walk("."):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            if file.endswith((".md", ".txt")):
                file_path = os.path.join(root, file)

                # Basic categorization logic
                if any(kw in file_path.lower() for kw in ["pitch", "executive", "vision", "strategy"]):
                    inventory["Executive"].append(file_path)
                elif any(
                    kw in file_path.lower()
                    for kw in [
                        "theory",
                        "math",
                        "scientific",
                        "research",
                        "paper",
                        "foundations",
                        "alphafold",
                    ]
                ):
                    inventory["Scientific"].append(file_path)
                elif any(
                    kw in file_path.lower()
                    for kw in [
                        "setup",
                        "api",
                        "dev",
                        "developer",
                        "readme",
                        "install",
                        "tech",
                        "workflow",
                        "conductor",
                        "rules",
                        "scripts",
                        "src",
                        "tests",
                    ]
                ):
                    inventory["Developer"].append(file_path)
                else:
                    inventory["Uncategorized"].append(file_path)

    # Ensure reports directory exists
    os.makedirs("DOCUMENTATION/reports", exist_ok=True)

    with open("DOCUMENTATION/reports/inventory.md", "w", encoding="utf-8") as f:
        f.write("# Documentation Inventory\n\n")
        for tier, files in inventory.items():
            f.write(f"## {tier} Tier\n")
            if not files:
                f.write("- No files found.\n")
            else:
                for path in sorted(files):
                    f.write(f"- {path}\n")
            f.write("\n")

    print("Inventory created at DOCUMENTATION/reports/inventory.md")


if __name__ == "__main__":
    audit_docs()
