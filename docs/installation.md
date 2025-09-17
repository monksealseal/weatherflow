# Installation

WeatherFlow targets Python 3.8+ and PyTorch 2.x. The quickest way to get
started is to create an isolated environment, install the package in editable
mode, and pull in the optional extras that match your workflow.

## 1. Prerequisites

- Python 3.8 or newer (3.10/3.11 recommended).
- A recent PyTorch build with CUDA if you plan to train on the GPU. The examples
  default to CPU when CUDA is not available.
- Cartopy, PROJ, and GEOS system libraries are required for the map plotting
  utilities. On Ubuntu/Debian you can install them with:

  ```bash
  sudo apt-get install libproj-dev proj-data proj-bin libgeos-dev
  ```

- Optional but recommended: `conda` or `python -m venv` for environment
  management.

## 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
python -m pip install --upgrade pip
```

You can substitute `conda create -n weatherflow python=3.10` if you prefer the
Conda stack.

## 3. Install WeatherFlow from source

Clone the repository and install it in editable mode so that local changes are
immediately reflected when you run the examples:

```bash
git clone https://github.com/monksealseal/weatherflow.git
cd weatherflow
pip install -e .
```

This pulls in the core runtime dependencies declared in `pyproject.toml`,
including PyTorch, xarray, TorchDiffEq, FastAPI, and the plotting stack.

### Optional extras

Install the optional dependency groups when you need them:

| Extra | Command | Purpose |
| --- | --- | --- |
| Development | `pip install -e .[dev]` | Installs pytest, coverage, Black, isort, flake8, and mypy. |
| Documentation | `pip install -e .[docs]` | Brings in `mkdocs-material` and `mkdocstrings` for building this documentation site. |
| Notebooks | `python setup_notebook_env.py` | Creates a ready-to-use notebook environment with WeatherFlow pre-installed. |

You can combine extras, e.g. `pip install -e .[dev,docs]`.

## 4. Configure access to ERA5/WeatherBench2

The default `ERA5Dataset` streams data from the public WeatherBench2 bucket
(`gs://weatherbench2/...`). Anonymous GCS access works out of the box on most
networks. If you need to route through a proxy or provide explicit credentials,
set the environment variables consumed by `gcsfs`, for example:

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
export GCSFS_TOKEN="~/.config/gcloud/application_default_credentials.json"
```

Alternatively, download the data locally and point the dataset at the NetCDF or
Zarr archive via the `data_path` argument.

## 5. Verify the installation

Run a quick import test and the unit suite to make sure everything works in your
environment:

```bash
python - <<'PY'
import weatherflow
print("WeatherFlow version", weatherflow.__version__)
PY

pytest
```

The tests exercise the lightweight synthetic data loaders and model components,
so they finish quickly even on CPU-only machines.

## 6. Troubleshooting

- **Cartopy import errors** – ensure the PROJ and GEOS libraries are installed
  system-wide before installing WeatherFlow. Reinstall `cartopy` after the
  libraries are present.
- **`torchdiffeq` missing** – the ODE-based components depend on
  `torchdiffeq>=0.2.3`. It is installed automatically via the package metadata,
  but double-check with `pip show torchdiffeq` if you see import errors.
- **GCS access timeouts** – the dataset tries several access strategies in
  sequence. If you are behind a firewall, set `data_path` to a local file to
  avoid repeated retries.
- **Plotly optional dependencies** – modules such as the educational dashboard
  and SKEW-T visualiser require Plotly. The base package already depends on
  `plotly>=5.18.0`; if you trimmed dependencies manually, reinstall with the
  default extras.

With the environment ready, continue to the
[Getting Started guide](getting_started.md) for a runnable walkthrough.
