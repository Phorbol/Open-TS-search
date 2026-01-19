from setuptools import setup, find_packages

setup(
    name="open-ts-search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "ase",
        "matplotlib",
        "sella",  # Optional, for IRC
        # "pymatgen", # Optional, for IDPP
    ],
    author="Trae Research",
    description="A modular transition state search library",
    python_requires=">=3.8",
)
