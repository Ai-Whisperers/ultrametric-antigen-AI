"""Setup script for Ternary VAE v5.5."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="ternary-vae",
    version="5.5.0",
    description="Dual-Pathway Variational Autoencoder for Complete Ternary Operation Coverage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Whisperers",
    author_email="support@aiwhisperers.com",
    url="https://github.com/ai-whisperers/ternary-vae",
    license="MIT",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.8",

    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],

    extras_require={
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
        "analysis": [
            "scikit-learn>=1.2.0",
            "pandas>=2.0.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "all": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "scikit-learn>=1.2.0",
            "pandas>=2.0.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "ternary-train=scripts.train.train_ternary_v5_5:main",
            "ternary-eval=scripts.eval.evaluate_coverage:main",
            "ternary-benchmark=scripts.benchmark.run_benchmark:main",
        ],
    },

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],

    keywords="variational-autoencoder vae ternary-logic meta-learning neural-networks",

    project_urls={
        "Documentation": "https://github.com/ai-whisperers/ternary-vae/docs",
        "Source": "https://github.com/ai-whisperers/ternary-vae",
        "Bug Reports": "https://github.com/ai-whisperers/ternary-vae/issues",
    },
)
