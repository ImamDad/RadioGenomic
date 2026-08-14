

"""
Setup configuration for MS-HGNN
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ms-hgnn",
    version="1.0.0",
    author="Imam Dad, Jianfeng He",
    author_email="jfenghe@kust.edu.cn",
    description="MS-HGNN: Multi-Scale Hierarchical Graph Neural Network for NSCLC Prognosis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ImamDad/MS-HGNN",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "lifelines>=0.27.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "medical": [
            "PyRadiomics>=3.0.1",
            "SimpleITK>=2.0.0",
            "nibabel>=3.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ms-hgnn=main:main",
        ],
    },
)
