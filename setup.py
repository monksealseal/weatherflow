from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="weatherflow",
    version="0.2.16",  # Explicitly set to 0.2.1
    author="Eduardo Siman",
    author_email="esiman@msn.com",
    description="A Deep Learning Library for Weather Prediction",
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
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "xarray>=0.19.0",
        "matplotlib>=3.4.0",
        "cartopy>=0.20.0",
        "wandb>=0.12.0",
        "tqdm>=4.60.0",
        "fsspec>=2021.6.0",
        "gcsfs>=2021.6.0",
        "scipy>=1.7.0",
        "netCDF4>=1.5.0",
        "h5py>=3.0.0",
    ]
)
