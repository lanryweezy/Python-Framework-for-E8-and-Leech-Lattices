#!/usr/bin/env python3
"""
E8 Leech Lattice Framework - Setup Script
"""

from setuptools import setup, find_packages
import os

# Read the requirements file
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "E8 Leech Lattice Framework - A comprehensive mathematical framework for exceptional lattices"

setup(
    name="e8leech",
    version="1.0.0",
    description="E8 Leech Lattice Framework - Comprehensive mathematical framework for exceptional lattices",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="E8 Leech Framework Team",
    author_email="contact@e8leech.com",
    url="https://github.com/e8leech/framework",
    
    # Package configuration
    packages=find_packages(),
    package_dir={"e8leech": "e8leech_project"},
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Entry points for CLI
    entry_points={
        'console_scripts': [
            'e8leech=e8leech.cli:main',
        ],
    },
    
    # Package data
    package_data={
        'e8leech': [
            'configs/*.yaml',
            'configs/*.json',
        ],
    },
    
    # Additional metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    
    # Keywords
    keywords="lattice mathematics cryptography ai machine-learning e8 leech quantum",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/e8leech/framework/issues",
        "Source": "https://github.com/e8leech/framework",
        "Documentation": "https://e8leech.readthedocs.io/",
    },
    
    # Optional dependencies
    extras_require={
        'gpu': ['cupy-cuda11x>=10.0.0'],
        'quantum': ['qiskit>=0.39.0', 'cirq>=1.0.0'],
        'visualization': ['plotly>=5.0.0', 'dash>=2.0.0'],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.991',
        ],
        'all': [
            'cupy-cuda11x>=10.0.0',
            'qiskit>=0.39.0',
            'cirq>=1.0.0',
            'plotly>=5.0.0',
            'dash>=2.0.0',
        ],
    },
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
)

