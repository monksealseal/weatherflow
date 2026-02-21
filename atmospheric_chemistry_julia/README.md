# Atmospheric Chemistry Transport Model (Julia)

A GPU-enabled global atmospheric chemistry transport model at 1° resolution,
written in pure Julia. Uses ERA5 reanalysis for meteorological fields and EDGAR
for anthropogenic emission inventories.

## Features

- **1° global resolution** on a regular lat-lon grid (360×181)
- **47 hybrid sigma-pressure levels** from surface to ~0.01 hPa
- **ERA5 integration** — automated download of wind, temperature, humidity, and surface fields via CDS API
- **EDGAR emissions** — automated download of NOx, CO, SO₂, CH₄, NMVOC, NH₃, PM2.5 inventories
- **Simplified tropospheric chemistry** — 14 species, 17 reactions covering O₃-NOx-CO-VOC-HOx
- **Photolysis** — parameterised j-values dependent on solar zenith angle, altitude, and cloud cover
- **Transport** — Piecewise Parabolic Method (PPM) advection, implicit vertical diffusion, mass-flux convection
- **Deposition** — dry deposition (resistance model) and wet deposition (Henry's law scavenging)
- **GPU acceleration** — CUDA.jl / KernelAbstractions.jl for all compute kernels
- **ERA5 spectral transforms** — spherical harmonic to grid-point conversion
- **NetCDF output** with mass budget diagnostics

## Quick Start

```bash
# 1. Clone and enter the project
cd atmospheric_chemistry_julia

# 2. Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Download data (requires CDS API key in ~/.cdsapirc)
julia --project=. scripts/download_data.jl configs/default.toml

# 4. Run the model
julia --project=. scripts/run.jl configs/default.toml

# 5. Visualize output
julia --project=. scripts/visualize.jl output/actm_20230101_00.nc O3 NO2
```

## GPU Mode

To run on GPU, set `gpu = true` in your config file or use the pre-made GPU config:

```bash
julia --project=. scripts/run.jl configs/gpu_highres.toml
```

Requires an NVIDIA GPU with CUDA support. The model automatically falls back to
CPU if CUDA is not available.

## Configuration

Configuration files are TOML format. See `configs/` for examples:

| Config | Description |
|--------|-------------|
| `default.toml` | Standard 1° / 47-level / 24-hour run |
| `gpu_highres.toml` | GPU-accelerated 1° / 60-level / 1-week run |
| `quick_test.toml` | Fast 2° / 20-level / 6-hour test run |

### Key settings

```toml
[grid]
resolution = 1.0    # degrees (1.0, 2.0, 2.5, 5.0)
nlevels = 47        # vertical levels

[time]
start_date = "2023-01-01T00:00:00"
end_date = "2023-01-02T00:00:00"
dt_advection = 900.0    # seconds
dt_chemistry = 600.0    # seconds

[numerics]
gpu = false         # true for GPU acceleration
```

## ERA5 Data Setup

1. Register at [CDS](https://cds.climate.copernicus.eu/)
2. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-uid>:<your-api-key>
   ```
3. The model downloads data automatically, or run `scripts/download_data.jl`

## Chemical Mechanism

Simplified tropospheric chemistry with 14 species:

| Species | Description |
|---------|-------------|
| O₃ | Ozone |
| NO, NO₂ | Nitrogen oxides |
| CO | Carbon monoxide |
| SO₂ | Sulfur dioxide |
| CH₄ | Methane |
| HCHO | Formaldehyde |
| HNO₃ | Nitric acid |
| H₂O₂ | Hydrogen peroxide |
| OH, HO₂ | Hydroxyl radicals |
| ISOP | Isoprene |
| PAN | Peroxyacetyl nitrate |
| PM2.5 | Fine particulate matter |

17 reactions including NOx photochemistry, HOx cycling, SO₂ oxidation,
and simplified VOC chemistry.

## Project Structure

```
atmospheric_chemistry_julia/
├── Project.toml              # Julia package manifest
├── configs/                  # TOML configuration files
│   ├── default.toml
│   ├── gpu_highres.toml
│   └── quick_test.toml
├── scripts/
│   ├── run.jl               # Main entry point
│   ├── download_data.jl     # Data download utility
│   └── visualize.jl         # Output visualization
├── src/
│   ├── AtmosphericChemistry.jl  # Module entry point
│   ├── model.jl             # Main driver & time integration
│   ├── grid/
│   │   ├── horizontal.jl    # Lat-lon grid
│   │   ├── vertical.jl      # Hybrid sigma-pressure levels
│   │   └── spectral.jl      # Spherical harmonic transforms
│   ├── data/
│   │   ├── cds_api.jl       # CDS API client
│   │   ├── era5.jl          # ERA5 data management
│   │   └── edgar.jl         # EDGAR emissions inventory
│   ├── chemistry/
│   │   ├── species.jl       # Chemical species definitions
│   │   ├── reactions.jl     # Reaction mechanism
│   │   ├── photolysis.jl    # Photolysis rate calculations
│   │   └── solver.jl        # Chemistry ODE solver
│   ├── transport/
│   │   ├── advection.jl     # PPM advection scheme
│   │   ├── diffusion.jl     # Vertical turbulent diffusion
│   │   ├── convection.jl    # Convective transport
│   │   └── deposition.jl    # Dry & wet deposition
│   ├── gpu/
│   │   ├── kernels.jl       # CUDA/KA compute kernels
│   │   └── backend.jl       # GPU backend management
│   ├── diagnostics/
│   │   ├── output.jl        # NetCDF output writer
│   │   └── budget.jl        # Mass budget diagnostics
│   └── utils/
│       ├── constants.jl     # Physical constants
│       └── arrays.jl        # Device-agnostic array utilities
└── test/
    └── runtests.jl          # Unit tests
```

## Testing

```bash
julia --project=. test/runtests.jl
```

## Requirements

- Julia ≥ 1.9
- CUDA.jl (optional, for GPU)
- NCDatasets.jl, HTTP.jl, JSON3.jl, KernelAbstractions.jl, etc. (see Project.toml)
