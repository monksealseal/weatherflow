# ---------------------------------------------------------------------------
# ERA5 reanalysis data download and loading
# ---------------------------------------------------------------------------

"""
    ERA5DataManager(; data_dir, resolution, client)

Manages downloading, caching, and loading ERA5 reanalysis fields needed
for the atmospheric chemistry-transport model.

Required fields:
- Wind components: u, v (horizontal), w (vertical / omega)
- Temperature, specific humidity
- Surface pressure, boundary-layer height
- Precipitation (for wet deposition)
- Surface solar radiation (for photolysis)

All data are downloaded at the model's horizontal resolution on ERA5
pressure levels and then interpolated to the model's hybrid-sigma grid.
"""
mutable struct ERA5DataManager
    data_dir   :: String
    resolution :: Float64   # degrees
    client     :: CDSClient
    cache      :: Dict{String, Any}
end

function ERA5DataManager(; data_dir::String="data/era5",
                          resolution::Float64=1.0,
                          client::Union{CDSClient, Nothing}=nothing)
    if client === nothing
        client = CDSClient()
    end
    mkpath(data_dir)
    return ERA5DataManager(data_dir, resolution, client, Dict{String, Any}())
end

# Variables we need from ERA5
const ERA5_PRESSURE_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "temperature",
    "specific_humidity",
]

const ERA5_SINGLE_VARS = [
    "surface_pressure",
    "boundary_layer_height",
    "total_precipitation",
    "surface_net_solar_radiation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]

const ERA5_PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
    "100", "125", "150", "175", "200", "225", "250", "300",
    "350", "400", "450", "500", "550", "600", "650", "700",
    "750", "775", "800", "825", "850", "875", "900", "925",
    "950", "975", "1000"
]

"""
    download_era5!(mgr, year, month; days=1:31, hours=0:6:18)

Download ERA5 data for a given year/month from the CDS API.
Downloads both pressure-level and single-level fields.
"""
function download_era5!(mgr::ERA5DataManager, year::Int, month::Int;
                        days=nothing, hours=[0, 6, 12, 18])
    if days === nothing
        days = 1:Dates.daysinmonth(year, month)
    end

    day_strs = [@sprintf("%02d", d) for d in days]
    hour_strs = [@sprintf("%02d:00", h) for h in hours]
    grid_str = "$(mgr.resolution)/$(mgr.resolution)"

    # Pressure-level data
    pl_file = joinpath(mgr.data_dir,
                       @sprintf("era5_pl_%04d%02d.nc", year, month))
    if !isfile(pl_file)
        request = Dict(
            "product_type" => "reanalysis",
            "format"       => "netcdf",
            "variable"     => ERA5_PRESSURE_VARS,
            "pressure_level" => ERA5_PRESSURE_LEVELS,
            "year"         => string(year),
            "month"        => @sprintf("%02d", month),
            "day"          => day_strs,
            "time"         => hour_strs,
            "grid"         => grid_str,
        )
        submit_request(mgr.client, "reanalysis-era5-pressure-levels",
                       request; output_path=pl_file)
    end

    # Single-level data
    sl_file = joinpath(mgr.data_dir,
                       @sprintf("era5_sl_%04d%02d.nc", year, month))
    if !isfile(sl_file)
        request = Dict(
            "product_type" => "reanalysis",
            "format"       => "netcdf",
            "variable"     => ERA5_SINGLE_VARS,
            "year"         => string(year),
            "month"        => @sprintf("%02d", month),
            "day"          => day_strs,
            "time"         => hour_strs,
            "grid"         => grid_str,
        )
        submit_request(mgr.client, "reanalysis-era5-single-levels",
                       request; output_path=sl_file)
    end

    @info "ERA5 data ready for $year-$(@sprintf("%02d", month))"
    return pl_file, sl_file
end

"""
    load_era5_fields(mgr, year, month, day, hour; vgrid, gpu)

Load ERA5 meteorological fields for a single time step and interpolate
to the model's vertical grid.

Returns a `NamedTuple` with fields: u, v, w, T, q, ps, blh, precip, ssrd, u10, v10, t2m
All 3-D fields have shape (nlon, nlat, nlevels).
"""
function load_era5_fields(mgr::ERA5DataManager, year::Int, month::Int,
                          day::Int, hour::Int;
                          vgrid::VerticalGrid, gpu::Bool=false)
    pl_file = joinpath(mgr.data_dir,
                       @sprintf("era5_pl_%04d%02d.nc", year, month))
    sl_file = joinpath(mgr.data_dir,
                       @sprintf("era5_sl_%04d%02d.nc", year, month))

    if !isfile(pl_file) || !isfile(sl_file)
        error("ERA5 data not found. Run download_era5! first.")
    end

    # Target time index — ERA5 files have time dimension
    target_time = DateTime(year, month, day, hour)

    fields_3d = Dict{Symbol, Array{Float64, 3}}()
    fields_2d = Dict{Symbol, Array{Float64, 2}}()

    # Load pressure-level data
    NCDatasets.Dataset(pl_file) do ds
        times = ds["time"][:]
        tidx = _find_time_index(times, target_time)

        # Variable name mapping (ERA5 short names)
        varmap = Dict(
            :u => "u", :v => "v", :w => "w",
            :T => "t", :q => "q"
        )

        for (sym, varname) in varmap
            if haskey(ds, varname)
                data = ds[varname][:, :, :, tidx]  # (lon, lat, level)
                fields_3d[sym] = Float64.(data)
            end
        end
    end

    # Load single-level data
    NCDatasets.Dataset(sl_file) do ds
        times = ds["time"][:]
        tidx = _find_time_index(times, target_time)

        varmap_2d = Dict(
            :ps    => "sp",
            :blh   => "blh",
            :precip => "tp",
            :ssrd  => "ssr",
            :u10   => "u10",
            :v10   => "v10",
            :t2m   => "t2m",
        )

        for (sym, varname) in varmap_2d
            if haskey(ds, varname)
                data = ds[varname][:, :, tidx]
                fields_2d[sym] = Float64.(data)
            end
        end
    end

    # Interpolate 3-D fields from ERA5 pressure levels to model hybrid levels
    ps = get(fields_2d, :ps, fill(Constants.P_REF, 360, 181))
    p_model = pressure_at_levels(vgrid, ps)

    era5_p = [parse(Float64, p) * 100.0 for p in ERA5_PRESSURE_LEVELS]  # hPa → Pa

    interp_3d = Dict{Symbol, Array{Float64, 3}}()
    for (sym, data_era5) in fields_3d
        interp_3d[sym] = _interp_to_model_levels(data_era5, era5_p, p_model)
    end

    # Move to device
    result_3d = Dict{Symbol, Any}()
    result_2d = Dict{Symbol, Any}()
    for (k, v) in interp_3d
        result_3d[k] = to_device(v; gpu)
    end
    for (k, v) in fields_2d
        result_2d[k] = to_device(v; gpu)
    end

    return (
        u      = get(result_3d, :u, device_array(Float64, size(p_model)...; gpu)),
        v      = get(result_3d, :v, device_array(Float64, size(p_model)...; gpu)),
        w      = get(result_3d, :w, device_array(Float64, size(p_model)...; gpu)),
        T      = get(result_3d, :T, device_array(Float64, size(p_model)...; gpu)),
        q      = get(result_3d, :q, device_array(Float64, size(p_model)...; gpu)),
        ps     = get(result_2d, :ps, device_array(Float64, 360, 181; gpu)),
        blh    = get(result_2d, :blh, device_array(Float64, 360, 181; gpu)),
        precip = get(result_2d, :precip, device_array(Float64, 360, 181; gpu)),
        ssrd   = get(result_2d, :ssrd, device_array(Float64, 360, 181; gpu)),
        u10    = get(result_2d, :u10, device_array(Float64, 360, 181; gpu)),
        v10    = get(result_2d, :v10, device_array(Float64, 360, 181; gpu)),
        t2m    = get(result_2d, :t2m, device_array(Float64, 360, 181; gpu)),
    )
end

"""Find the closest time index in a NetCDF time variable."""
function _find_time_index(times, target::DateTime)
    # ERA5 times are typically hours since 1900-01-01
    ref = DateTime(1900, 1, 1)
    target_hours = Dates.value(target - ref) / 3_600_000  # ms → hours

    best_idx = 1
    best_diff = Inf
    for (i, t) in enumerate(times)
        d = abs(Float64(t) - target_hours)
        if d < best_diff
            best_diff = d
            best_idx = i
        end
    end
    return best_idx
end

"""
Interpolate ERA5 pressure-level data to model hybrid-sigma levels.
"""
function _interp_to_model_levels(data_era5::Array{Float64, 3},
                                  era5_p::Vector{Float64},
                                  p_model::Array{Float64, 3})
    nlon, nlat, nlevels_model = size(p_model)
    result = zeros(Float64, nlon, nlat, nlevels_model)
    nlevels_era5 = size(data_era5, 3)

    for i in 1:nlon, j in 1:nlat
        # Build interpolant in log-pressure space
        lp_era5 = log.(era5_p[1:nlevels_era5])
        col = @view data_era5[i, j, :]

        for k in 1:nlevels_model
            lp_target = log(max(p_model[i, j, k], 1.0))

            # Linear interpolation in log-p
            if lp_target <= lp_era5[1]
                result[i, j, k] = col[1]
            elseif lp_target >= lp_era5[end]
                result[i, j, k] = col[end]
            else
                # Find bracketing levels
                idx = searchsortedlast(lp_era5, lp_target)
                idx = clamp(idx, 1, nlevels_era5 - 1)
                frac = (lp_target - lp_era5[idx]) / (lp_era5[idx+1] - lp_era5[idx])
                result[i, j, k] = col[idx] * (1 - frac) + col[idx+1] * frac
            end
        end
    end

    return result
end
