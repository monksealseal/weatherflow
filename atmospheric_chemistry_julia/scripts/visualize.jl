#!/usr/bin/env julia
"""
    visualize.jl — Quick visualization of model output

Usage:
    julia --project=. scripts/visualize.jl output/actm_20230101_00.nc [species]

Generates PNG plots of specified species (default: O3, NO2, CO, SO2).
Requires the Plots and GeoMakie packages (optional dependencies).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NCDatasets
using Printf
using Dates

function plot_species_ascii(filepath::String, species::String)
    """Generate an ASCII contour plot for terminals without graphics."""
    ds = NCDatasets.Dataset(filepath)

    if !haskey(ds, species)
        println("Species '$species' not found in $filepath")
        println("Available: $(join(keys(ds), ", "))")
        close(ds)
        return
    end

    data = ds[species][:, :, :]
    lon = ds["lon"][:]
    lat = ds["lat"][:]
    close(ds)

    # Take surface level (last index)
    field = data[:, :, end]

    # Simple statistics
    nlon, nlat = size(field)
    fmin = minimum(filter(!isnan, field))
    fmax = maximum(filter(!isnan, field))
    fmean = sum(filter(!isnan, field)) / count(!isnan, field)

    println("=" ^ 60)
    println("  $species from $(basename(filepath))")
    println("  Surface level | Grid: $(nlon)×$(nlat)")
    @printf("  Min: %.2e  Max: %.2e  Mean: %.2e\n", fmin, fmax, fmean)
    println("  Units: molec cm⁻³")
    println("=" ^ 60)

    # ASCII rendering: subsample to 72 × 36 characters
    chars = [' ', '·', '░', '▒', '▓', '█']
    ncols = min(72, nlon)
    nrows = min(36, nlat)

    for jj in nrows:-1:1
        j = round(Int, 1 + (jj - 1) * (nlat - 1) / (nrows - 1))
        line = ""
        for ii in 1:ncols
            i = round(Int, 1 + (ii - 1) * (nlon - 1) / (ncols - 1))
            val = field[i, j]
            if isnan(val) || fmax == fmin
                line *= ' '
            else
                idx = clamp(round(Int, (val - fmin) / (fmax - fmin) * 5) + 1, 1, 6)
                line *= string(chars[idx])
            end
        end
        lat_label = @sprintf("%+6.1f", lat[j])
        println("  $lat_label |$line|")
    end
    println()
end

function main()
    if length(ARGS) < 1
        # Find latest output file
        output_dir = "output"
        if !isdir(output_dir)
            println("Usage: julia scripts/visualize.jl <output_file.nc> [species]")
            return
        end
        files = filter(f -> endswith(f, ".nc"), readdir(output_dir; join=true))
        if isempty(files)
            println("No output files found in $output_dir/")
            return
        end
        filepath = sort(files)[end]
    else
        filepath = ARGS[1]
    end

    species_list = length(ARGS) > 1 ? ARGS[2:end] : ["O3", "NO2", "CO", "SO2"]

    println("\nVisualizing: $filepath\n")

    for sp in species_list
        plot_species_ascii(filepath, sp)
    end
end

main()
