import sys
import subprocess
from datetime import datetime

# List of 20+ libraries/tools to research/check
TOOLS = [
    {
        "name": "pylint",
        "category": "Linter",
        "description": "Highly configurable linter",
    },
    {
        "name": "flake8",
        "category": "Linter",
        "description": "Wrapper for pyflakes, pycodestyle, mccabe",
    },
    {
        "name": "ruff",
        "category": "Linter",
        "description": "Fast Rust-based linter/formatter",
    },
    {"name": "mypy", "category": "Type Checker", "description": "Static type checker"},
    {
        "name": "pyright",
        "category": "Type Checker",
        "description": "Fast type checker by Microsoft",
    },
    {
        "name": "radon",
        "category": "Complexity",
        "description": "Cyclomatic complexity metrics",
    },
    {
        "name": "xenon",
        "category": "Complexity",
        "description": "Asserts code complexity requirements",
    },
    {
        "name": "mccabe",
        "category": "Complexity",
        "description": "McCabe complexity checker",
    },
    {
        "name": "bandit",
        "category": "Security",
        "description": "Security vulnerability scanner",
    },
    {
        "name": "safety",
        "category": "Security",
        "description": "Checks installed dependencies for known vulnerabilities",
    },
    {"name": "vulture", "category": "Dead Code", "description": "Finds unused code"},
    {
        "name": "eradicate",
        "category": "Dead Code",
        "description": "Removes commented-out code",
    },
    {
        "name": "black",
        "category": "Formatter",
        "description": "The uncompromising code formatter",
    },
    {"name": "isort", "category": "Formatter", "description": "Sorts imports"},
    {"name": "yapf", "category": "Formatter", "description": "Google's formatter"},
    {
        "name": "coverage",
        "category": "Testing",
        "description": "Code coverage measurement",
    },
    {"name": "pytest", "category": "Testing", "description": "Testing framework"},
    {
        "name": "hypothesis",
        "category": "Testing",
        "description": "Property-based testing",
    },
    {"name": "mutmut", "category": "Testing", "description": "Mutation testing"},
    {
        "name": "deptry",
        "category": "Dependencies",
        "description": "Finds unused/missing dependencies",
    },
    {
        "name": "pip-audit",
        "category": "Dependencies",
        "description": "Audits dependencies for vulnerabilities",
    },
    {"name": "pygount", "category": "Metrics", "description": "Lines of code counter"},
]


def check_install(tool_name):
    """Check if a tool is installed."""
    try:
        subprocess.run(
            [sys.executable, "-m", tool_name, "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Some tools don't support python -m tool --version, try direct import
        try:
            __import__(tool_name)
            return True
        except ImportError:
            return False
    except Exception:
        return False


def generate_report():
    print("# External Tools Analysis Report")
    print(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")

    print("## Tool Availability Codebase Audit")
    print("| Tool | Category | Status | Description |")
    print("| :--- | :--- | :--- | :--- |")

    installed_count = 0
    missing = []

    for tool in TOOLS:
        is_installed = check_install(tool["name"])
        status = "✅ Installed" if is_installed else "❌ Missing"
        if is_installed:
            installed_count += 1
        else:
            missing.append(tool)

        print(
            f"| **{tool['name']}** | {tool['category']} | {status} | {tool['description']} |"
        )

    print(f"\n**Summary:** {installed_count}/{len(TOOLS)} tools detected.\n")

    print("## Recommendations for Implementation")
    print(
        "Based on the 'Missing' list, the following high-value tools are recommended for immediate integration:\n"
    )

    for tool in missing[:10]:  # Top 10 recommendations
        print(f"### 🔹 Implement `{tool['name']}` ({tool['category']})")
        print(f"- **Why:** {tool['description']}")
        print(
            f"- **Action:** Create `scripts/analysis/run_{tool['name']}.py` to automate this check."
        )
        print("")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    generate_report()
