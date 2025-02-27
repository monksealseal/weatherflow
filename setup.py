from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="weatherflow",
    version="0.3.2",  # Updated to match pyproject.toml
    author="Eduardo Siman",
    author_email="esiman@msn.com",
    description="A Deep Learning Library for Weather Prediction with Flow Matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monksealseal/weatherflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "xarray>=0.19.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "cartopy>=0.20.0",
        "wandb>=0.12.0",
        "tqdm>=4.60.0",
        "fsspec>=2023.9.0",
        "gcsfs>=2023.9.0",
        "zarr>=2.14.2,<3.0.0",
        "scipy>=1.7.0",
        "netCDF4>=1.5.0",
        "h5py>=3.0.0",
        "torchdiffeq>=0.2.3",  # Added critical dependency for ODE solvers
        "bottleneck>=1.3.6",
    ]
)