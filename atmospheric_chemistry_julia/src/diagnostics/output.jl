# ---------------------------------------------------------------------------
# Diagnostic output — NetCDF writer for model state
# ---------------------------------------------------------------------------

"""
    DiagnosticOutput(; output_dir, prefix, species_list, hgrid, vgrid)

Manages writing model output to NetCDF files.
"""
mutable struct DiagnosticOutput
    output_dir    :: String
    prefix        :: String
    species_list  :: Vector{Species}
    hgrid         :: HorizontalGrid
    vgrid         :: VerticalGrid
    write_counter :: Int
    output_freq_h :: Int   # Hours between outputs
end

function DiagnosticOutput(; output_dir::String="output",
                           prefix::String="actm",
                           species_list::Vector{Species},
                           hgrid::HorizontalGrid,
                           vgrid::VerticalGrid,
                           output_freq_h::Int=6)
    mkpath(output_dir)
    return DiagnosticOutput(output_dir, prefix, species_list, hgrid, vgrid,
                            0, output_freq_h)
end

"""
    write_diagnostics(diag, conc, met, datetime)

Write a snapshot of concentrations and selected meteorological fields
to a NetCDF file.
"""
function write_diagnostics(diag::DiagnosticOutput,
                           conc::AbstractArray{Float64, 4},
                           met::NamedTuple,
                           datetime::DateTime)
    diag.write_counter += 1
    datestr = Dates.format(datetime, "yyyymmdd_HH")
    filename = joinpath(diag.output_dir,
                        "$(diag.prefix)_$(datestr).nc")

    nlon = diag.hgrid.nlon
    nlat = diag.hgrid.nlat
    nlevels = diag.vgrid.nlevels

    NCDatasets.Dataset(filename, "c") do ds
        # Define dimensions
        defDim(ds, "lon", nlon)
        defDim(ds, "lat", nlat)
        defDim(ds, "level", nlevels)

        # Coordinate variables
        lon_var = defVar(ds, "lon", Float64, ("lon",),
                         attrib=Dict("units" => "degrees_east",
                                     "long_name" => "longitude"))
        lat_var = defVar(ds, "lat", Float64, ("lat",),
                         attrib=Dict("units" => "degrees_north",
                                     "long_name" => "latitude"))
        lev_var = defVar(ds, "level", Float64, ("level",),
                         attrib=Dict("units" => "1",
                                     "long_name" => "model level index"))

        lon_var[:] = Array(diag.hgrid.lon)
        lat_var[:] = Array(diag.hgrid.lat)
        lev_var[:] = collect(1.0:nlevels)

        # Global attributes
        ds.attrib["title"] = "Atmospheric Chemistry Transport Model Output"
        ds.attrib["model"] = "AtmosphericChemistry.jl"
        ds.attrib["datetime"] = string(datetime)
        ds.attrib["resolution_deg"] = diag.hgrid.resolution
        ds.attrib["nlevels"] = nlevels
        ds.attrib["history"] = "Created $(Dates.now())"

        # Species concentrations
        conc_cpu = Array(conc)
        for (s, sp) in enumerate(diag.species_list)
            varname = String(sp.name)
            v = defVar(ds, varname, Float64, ("lon", "lat", "level"),
                       attrib=Dict("units" => "molec cm-3",
                                   "long_name" => "$(sp.name) number density",
                                   "molecular_weight_kg_mol" => sp.mw))
            v[:, :, :] = conc_cpu[:, :, :, s]
        end

        # Selected meteorological fields
        if hasfield(typeof(met), :T)
            T_var = defVar(ds, "temperature", Float64, ("lon", "lat", "level"),
                           attrib=Dict("units" => "K", "long_name" => "Temperature"))
            T_var[:, :, :] = Array(met.T)
        end

        if hasfield(typeof(met), :ps)
            ps_var = defVar(ds, "surface_pressure", Float64, ("lon", "lat"),
                            attrib=Dict("units" => "Pa",
                                        "long_name" => "Surface pressure"))
            ps_var[:, :] = Array(met.ps)
        end

        if hasfield(typeof(met), :u)
            u_var = defVar(ds, "u_wind", Float64, ("lon", "lat", "level"),
                           attrib=Dict("units" => "m s-1",
                                       "long_name" => "Zonal wind"))
            u_var[:, :, :] = Array(met.u)
        end

        if hasfield(typeof(met), :v)
            v_var = defVar(ds, "v_wind", Float64, ("lon", "lat", "level"),
                           attrib=Dict("units" => "m s-1",
                                       "long_name" => "Meridional wind"))
            v_var[:, :, :] = Array(met.v)
        end
    end

    @info "Wrote diagnostics: $filename"
    return filename
end

"""
    write_timeseries(diag, ts_data, filename)

Write a time series DataFrame to CSV.
"""
function write_timeseries(diag::DiagnosticOutput, ts_data::DataFrame,
                          filename::String="timeseries.csv")
    filepath = joinpath(diag.output_dir, filename)
    CSV.write(filepath, ts_data)
    @info "Wrote time series: $filepath"
end
