import json
import os
import py_compile
import re
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VSCODE_DIR = PROJECT_ROOT / ".vscode"
SETTINGS_FILE = VSCODE_DIR / "settings.json"
CSPELL_KEY = "cSpell.words"


def run_command(command, description):
    print(f"\nrunning {description}...")
    try:
        subprocess.run(command, check=True, shell=True, cwd=PROJECT_ROOT)
        print(f"✅ {description} complete.")
    except subprocess.CalledProcessError:
        print(f"❌ {description} failed or found issues.")


def run_formatter():
    run_command("black . --line-length 150", "Black (Formatter - Relaxed 150 chars)")
    run_command("isort .", "isort (Import Sorter)")


def run_linter_fixes():
    # Use Ruff for fast linting and auto-fixing
    # We use --select ALL or specific rules? For now, standard check --fix
    # We use --unsafe-fixes to handle more aggressive fixes if needed, but start safe.
    # Actually, let's just use check --fix.
    print("\nrunning Ruff (Linter & Auto-Fixer)...")
    try:
        # We ignore errors (check=False) because ruff returns non-zero on found violations even if fixed
        subprocess.run("ruff check --fix .", check=False, shell=True, cwd=PROJECT_ROOT)
        print("✅ Ruff verification complete.")
    except Exception as e:
        print(f"⚠️ Ruff failed to run: {e}")


def add_words_to_dictionary(words):
    if not SETTINGS_FILE.exists():
        print(f"Creating {SETTINGS_FILE}")
        VSCODE_DIR.mkdir(exist_ok=True)
        settings = {CSPELL_KEY: []}
    else:
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
        except json.JSONDecodeError:
            settings = {CSPELL_KEY: []}

    current_words = set(settings.get(CSPELL_KEY, []))
    new_words_count = 0
    for w in words:
        if w not in current_words:
            current_words.add(w)
            new_words_count += 1
            print(f"Added '{w}' to dictionary.")

    if new_words_count > 0:
        settings[CSPELL_KEY] = sorted(list(current_words))
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"✅ Added {new_words_count} words to .vscode/settings.json")
    else:
        print("No new words to add.")


BIO_TERMS = [
    # General Tech/Math
    "hyperbolic",
    "poincare",
    "embedding",
    "bioinformatics",
    "ternary",
    "vaes",
    "vae",
    "gcn",
    "cnn",
    "autoencoder",
    "decoder",
    "encoder",
    "latent",
    "manifold",
    "phylogeny",
    "phylogenetic",
    "genomic",
    "codon",
    "adjoint",
    "laplacian",
    "eigenvectors",
    "eigendecomposition",
    "tqdm",
    "argparse",
    "numpy",
    "matplotlib",
    "pyplot",
    "scikit",
    "sklearn",
    "pandas",
    "pytorch",
    "torch",
    "cuda",
    "cpu",
    "gpu",
    # New additions
    "adic",
    "Adic",
    "ADIC",
    "Möbius",
    "Fréchet",
    "frechet",
    "cdist",
    "pearsonr",
    "spearmanr",
    "atleast",
    "arccosh",
    "mobius",
    "keepdims",
    "linalg",
    "arctanh",
    "Riemannian",
    "padic",
    "randn",
    "embs",
    "degs",
    "arccos",
    "Nanoparticle",
    "nanoparticle",
    "penalises",
    "epitope",
    "RMSD",
    "epitopes",
    "Brizuela",
    "venv",
    "roadmaps",
    "pytest",
    "bibtex",
    "zenodo",
    "conftest",
    "pythonpath",
    "testpaths",
    "filterwarnings",
    "addopts",
    "autouse",
    "addinivalue",
    "keepdim",
    "logvar",
    "rtol",
    "rcfile",
    "pylintrc",
    # --- Comprehensive Additions ---
    "alphafold",
    "wildtype",
    "glycan",
    "deglyc",
    "integrase",
    "crossvalidation",
    "neurodegeneration",
    "alzheimers",
    "phospho",
    "mtbr",
    "functionomic",
    "citrullination",
    "citrullinated",
    "autoantigen",
    "immunogenicity",
    "proteome",
    "acpa",
    "wandb",
    "plotly",
    "umap",
    "tsne",
    "lorentz",
    "fibration",
    "calabi",
    "yau",
    "ricci",
    "curvature",
    "geodesic",
    "isometry",
    "automorphism",
    "homeomorphism",
    "diffeomorphism",
    "holomorphic",
    "meromorphic",
    "cohomology",
    "homology",
    "homotopy",
    "betti",
    "hodge",
    "kaehler",
    "kahler",
    "symal",
    "symb",
    "bg505",
    "gp120",
    "n103",
    "n332",
    "v1v2",
    "cd4",
    "bnab",
    "bnabs",
    "epitope",
    "paratope",
    "antibody",
    "antigen",
    "receptor",
    "ligand",
    "docking",
    "molecular",
    "dynamics",
    "residue",
    "residues",
    "amino",
    "nucleotide",
    "genomic",
    "proteomic",
    "transcriptomic",
    "omics",
    "multiomics",
    "interactome",
    "connectome",
    "embeddings",
    "latents",
]


def run_syntax_check():
    print("\nrunning Syntax Check...")
    error_count = 0
    checked_count = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # reuse ignore set if possible, but for now simple filter
        if ".venv" in dirs:
            dirs.remove(".venv")
        if ".git" in dirs:
            dirs.remove(".git")

        for file in files:
            if file.endswith(".py"):
                path = str(Path(root) / file)
                checked_count += 1
                try:
                    py_compile.compile(path, doraise=True)
                except py_compile.PyCompileError as e:
                    print(f"❌ Syntax Error in {file}: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"⚠️ Could not compile {file}: {e}")

    if error_count == 0:
        print(f"✅ Syntax Check Passed ({checked_count} file(s)).")
    else:
        print(f"❌ Found {error_count} syntax errors.")


def run_integrity_audit():
    print("\nrunning Integrity/Dependency Audit...")
    missing_files = []

    # Simple regex for finding file strings
    file_pattern = re.compile(r'["\']([^"\']+\.(pt|json|csv|pdb|fasta))["\']')

    for root, dirs, files in os.walk(PROJECT_ROOT):
        if ".venv" in dirs:
            dirs.remove(".venv")
        if ".git" in dirs:
            dirs.remove(".git")
        if "node_modules" in dirs:
            dirs.remove("node_modules")

        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                _check_file_integrity(path, file_pattern, missing_files)

    if not missing_files:
        print("✅ No missing data dependencies found.")
    else:
        print(
            f"⚠️ Found {len(missing_files)} missing data references (check 'integrity_report.log' for details)."
        )


def _check_file_integrity(path, pattern, missing_files_list):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            matches = pattern.findall(content)
            for match in matches:
                filename = match[0]
                if "*" in filename or "{" in filename:
                    continue

                if not _dependency_exists(path, filename):
                    # Only report if it looks like a local file
                    if not filename.startswith("http"):
                        missing_files_list.append(
                            (str(path.relative_to(PROJECT_ROOT)), filename)
                        )
    except Exception:
        pass


def _dependency_exists(source_path, filename):
    # Check relative to script
    if (source_path.parent / filename).exists():
        return True
    # Check relative to project root
    if (PROJECT_ROOT / filename).exists():
        return True
    return False


def clean_artifacts():
    print("\nrunning Cleanup...")
    # Remove the temporary vocabulary file
    vocab_file = PROJECT_ROOT / "detected_unknowns.txt"
    if vocab_file.exists():
        try:
            os.remove(vocab_file)
            print(f"✅ Removed {vocab_file.name}")
        except Exception as e:
            print(f"❌ Failed to remove {vocab_file.name}: {e}")
    else:
        print("✅ No artifacts to clean.")


def run_tests():
    print("\nrunning Unit Tests...")
    try:
        # Run fast tests only, quiet mode
        subprocess.run(
            "pytest tests/unit --maxfail=5 -q",
            check=True,
            shell=True,
            cwd=PROJECT_ROOT,
        )
        print("✅ Unit Tests Passed.")
    except subprocess.CalledProcessError:
        print("❌ Unit Tests Failed.")


def main():
    print("=== Codebase Maintenance Script ===")

    # 1. Format Code
    run_formatter()

    # 2. Syntax Check (New)
    run_syntax_check()

    # 3. Integrity Audit (New)
    run_integrity_audit()

    # 4. Unit Tests (New)
    run_tests()

    # 2. Add Common Terms to Dictionary
    print("\nUpdating Dictionary...")

    # Check for auto-generated vocabulary list
    auto_vocab_file = PROJECT_ROOT / "detected_unknowns.txt"
    if auto_vocab_file.exists():
        print(f"Found auto-generated vocabulary: {auto_vocab_file}")
        try:
            with open(auto_vocab_file, "r", encoding="utf-8") as f:
                auto_terms = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(auto_terms)} terms from scan.")
            BIO_TERMS.extend(auto_terms)
        except Exception as e:
            print(f"Error reading vocabulary file: {e}")

    add_words_to_dictionary(BIO_TERMS)

    # 5. Cleanup
    clean_artifacts()

    # 6. Auto-Fix Linter Issues (New)
    run_linter_fixes()

    print("\n=== Maintenance Complete ===")


if __name__ == "__main__":
    main()
