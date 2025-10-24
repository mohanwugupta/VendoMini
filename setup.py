from setuptools import setup, find_packages

setup(
    name="vendomini",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "cluster": [
            "ray[default]>=2.7.0",  # Linux/Mac only
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
        ],
        "analysis": [
            "lifelines>=0.27.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "lifelines>=0.27.0",
        ]
    },
)
