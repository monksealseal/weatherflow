"""
    AtmosphericChemistry

GPU-enabled Atmospheric Chemistry Transport Model at 1° resolution.

Uses ERA5 reanalysis for meteorological fields and EDGAR for emissions inventories.
Solves coupled chemistry-transport equations on a global lat-lon grid with
operator splitting (transport → chemistry → deposition).

# Components
- `Grid`: Lat-lon grid with ERA5 spectral grid support and vertical hybrid-sigma levels
- `ERA5`: Automated download and processing of ERA5 reanalysis data via CDS API
- `EDGAR`: EDGAR emissions inventory download and gridding
- `Chemistry`: Gas-phase chemistry solver (simplified tropospheric mechanism)
- `Transport`: Advection (PPM), vertical diffusion, and convective transport
- `GPU`: CUDA kernel abstractions for all computational kernels
- `Diagnostics`: Output, mass conservation checks, budget analysis
"""
module AtmosphericChemistry

using Dates
using Printf
using LinearAlgebra
using Logging

# Optional GPU support — fall back to CPU arrays when CUDA is unavailable
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

using KernelAbstractions
using Interpolations
using NCDatasets
using HTTP
using JSON3
using CSV
using DataFrames
using ProgressMeter
using TOML

# ---------------------------------------------------------------------------
# Sub-modules (order matters — later modules depend on earlier ones)
# ---------------------------------------------------------------------------
include("utils/constants.jl")
include("utils/arrays.jl")

include("grid/horizontal.jl")
include("grid/vertical.jl")
include("grid/spectral.jl")

include("data/cds_api.jl")
include("data/era5.jl")
include("data/edgar.jl")

include("chemistry/species.jl")
include("chemistry/reactions.jl")
include("chemistry/photolysis.jl")
include("chemistry/solver.jl")

include("transport/advection.jl")
include("transport/diffusion.jl")
include("transport/convection.jl")
include("transport/deposition.jl")

include("gpu/kernels.jl")
include("gpu/backend.jl")

include("diagnostics/output.jl")
include("diagnostics/budget.jl")

include("model.jl")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
export ModelConfig, ModelState, run_simulation!
export HorizontalGrid, VerticalGrid, SpectralTransform
export ERA5DataManager, download_era5!, load_era5_fields
export EDGARManager, download_edgar!, load_emissions
export ChemicalMechanism, Species, Reaction
export solve_chemistry!, compute_photolysis_rates
export advect!, diffuse_vertical!, convective_transport!
export dry_deposition!, wet_deposition!
export GPUBackend, to_device, synchronize_device
export DiagnosticOutput, write_diagnostics, compute_mass_budget

end # module
