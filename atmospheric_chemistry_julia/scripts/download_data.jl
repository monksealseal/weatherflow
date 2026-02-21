#!/usr/bin/env julia
"""
    download_data.jl — Download ERA5 and EDGAR data for a simulation period

Usage:
    julia --project=. scripts/download_data.jl [config_file.toml]

Downloads all required ERA5 meteorological fields and EDGAR emissions
inventories. Requires a valid CDS API key in ~/.cdsapirc.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosphericChemistry.jl"))
using .AtmosphericChemistry

function main()
    config_file = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "configs", "default.toml")
    cfg = TOML.parsefile(config_file)

    time_cfg = get(cfg, "time", Dict())
    data_cfg = get(cfg, "data", Dict())
    grid_cfg = get(cfg, "grid", Dict())

    start_date = DateTime(get(time_cfg, "start_date", "2023-01-01T00:00:00"))
    end_date   = DateTime(get(time_cfg, "end_date", "2023-01-02T00:00:00"))
    resolution = get(grid_cfg, "resolution", 1.0)

    # --- ERA5 ---
    @info "=== Downloading ERA5 data ==="
    era5_dir = get(data_cfg, "era5_data_dir", "data/era5")
    era5 = ERA5DataManager(; data_dir=era5_dir, resolution=resolution)

    t = start_date
    while t <= end_date
        year = Dates.year(t)
        month = Dates.month(t)
        @info "Downloading ERA5 for $year-$month..."
        try
            download_era5!(era5, year, month)
        catch e
            @warn "Failed to download ERA5 for $year-$month: $e"
        end
        # Advance to next month
        t = t + Dates.Month(1)
        if Dates.month(t) == Dates.month(start_date) && Dates.year(t) > Dates.year(end_date)
            break
        end
    end

    # --- EDGAR ---
    @info "=== Downloading EDGAR emissions ==="
    edgar_dir = get(data_cfg, "edgar_data_dir", "data/edgar")
    edgar_version = get(data_cfg, "edgar_version", "v8.0")
    edgar_year = get(data_cfg, "edgar_year", 2022)

    edgar = EDGARManager(; data_dir=edgar_dir, version=edgar_version,
                          base_year=edgar_year, resolution=resolution)
    download_edgar!(edgar)

    @info "=== Data download complete ==="
end

main()
