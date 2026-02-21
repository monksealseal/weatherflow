#!/usr/bin/env julia
"""
    run.jl — Main entry point for the Atmospheric Chemistry Transport Model

Usage:
    julia --project=. scripts/run.jl [config_file.toml]

If no config file is given, uses configs/default.toml.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
using TOML

# Load the model
include(joinpath(@__DIR__, "..", "src", "AtmosphericChemistry.jl"))
using .AtmosphericChemistry

function parse_config(filepath::String)
    cfg = TOML.parsefile(filepath)

    grid = get(cfg, "grid", Dict())
    time = get(cfg, "time", Dict())
    data = get(cfg, "data", Dict())
    output = get(cfg, "output", Dict())
    numerics = get(cfg, "numerics", Dict())
    physics = get(cfg, "physics", Dict())

    return ModelConfig(
        resolution      = get(grid, "resolution", 1.0),
        nlevels         = get(grid, "nlevels", 47),
        p_top           = get(grid, "p_top", 1.0),
        start_date      = DateTime(get(time, "start_date", "2023-01-01T00:00:00")),
        end_date        = DateTime(get(time, "end_date", "2023-01-02T00:00:00")),
        dt_advection    = get(time, "dt_advection", 900.0),
        dt_chemistry    = get(time, "dt_chemistry", 600.0),
        dt_diffusion    = get(time, "dt_diffusion", 900.0),
        dt_emission     = get(time, "dt_emission", 3600.0),
        met_update_freq = get(time, "met_update_freq", 6),
        era5_data_dir   = get(data, "era5_data_dir", "data/era5"),
        edgar_data_dir  = get(data, "edgar_data_dir", "data/edgar"),
        edgar_version   = get(data, "edgar_version", "v8.0"),
        edgar_year      = get(data, "edgar_year", 2022),
        output_dir      = get(output, "output_dir", "output"),
        output_freq_h   = get(output, "output_freq_h", 6),
        output_prefix   = get(output, "output_prefix", "actm"),
        chem_substeps   = get(numerics, "chem_substeps", 4),
        gpu             = get(numerics, "gpu", false),
        do_advection    = get(physics, "do_advection", true),
        do_chemistry    = get(physics, "do_chemistry", true),
        do_diffusion    = get(physics, "do_diffusion", true),
        do_convection   = get(physics, "do_convection", true),
        do_dry_dep      = get(physics, "do_dry_dep", true),
        do_wet_dep      = get(physics, "do_wet_dep", true),
        do_emissions    = get(physics, "do_emissions", true),
    )
end

function main()
    config_file = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "configs", "default.toml")

    if !isfile(config_file)
        error("Config file not found: $config_file")
    end

    @info "Loading configuration from $config_file"
    config = parse_config(config_file)

    @info "Initialising model..."
    state = ModelState(config)

    @info "Running simulation..."
    run_simulation!(state)

    @info "Done!"
end

main()
