from setuptools import setup, find_packages

setup(
    name="weatherflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9",
        "numpy>=1.20",
        "xarray>=0.19",
        "matplotlib>=3.4",
        "cartopy>=0.21",
        "wandb",
        "tqdm"
    ],
    python_requires=">=3.8",
)
