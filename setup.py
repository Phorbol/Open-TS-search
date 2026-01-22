from setuptools import setup, find_packages
import os

# Read the long description from README.md
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="open-ts-search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.15.0",
        "ase>=3.26.0",
        "matplotlib>=3.10.6",
        "torch>=2.8.0",  # Required for GPU components
    ],
    extras_require={
        "irc": ["sella>=2.3.5"],  # For True IRC functionality
        "ml": ["mace-torch>=0.3.14"],  # For ML potential examples
        "dev": ["pytest>=9.0.1", "black", "flake8"],
    },
    author="Trae Research",
    description="A modular transition state search library for scientific computing and AI4S",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)
