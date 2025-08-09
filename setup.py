from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mlops-pipeline-demo",
    version="1.0.0",
    author="MLOps Demo Team",
    author_email="demo@mlops.com",
    description="End-to-End MLOps Pipeline Demo for Customer Churn Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/mlops-pipeline-demo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlops-train=src.models.train:main",
            "mlops-predict=src.models.predict:main",
            "mlops-serve=src.api.app:main",
            "mlops-monitor=src.monitoring.performance:main",
        ],
    },
)
