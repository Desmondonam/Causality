from setuptools import setup, find_packages

setup(
    name="breast-cancer-causal-ml",
    version="1.0.0",
    description="Causal Machine Learning Analysis for Breast Cancer Prediction",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'shap>=0.42.0',
    ],
    python_requires='>=3.8',
)