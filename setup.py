from setuptools import setup, find_packages

setup(
    name="LongevAI",
    version="0.1.0",
    author="Tolstoy Justin",
    author_email="tolstoy.justin@example.com",
    description="Transformer-based forecasting for global life expectancy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DPSDevops/LongevAI-Transformer-Based-Forecasting-for-Global-Life-Expectancy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "scikit-learn>=1.2.2",
        "torch>=2.0.1",
        "PyQt5>=5.15.9",
    ],
    entry_points={
        "console_scripts": [
            "longevai-cli=life_expectancy_analysis:main",
            "longevai-gui=life_expectancy_gui:main",
        ],
    },
) 