# ---------------------------------------------------------------------------
# EDGAR (Emissions Database for Global Atmospheric Research) inventory
# ---------------------------------------------------------------------------

"""
    EDGARManager(; data_dir, version, year, resolution)

Download and process EDGAR emission inventories for use in the chemistry
transport model.

EDGAR provides global gridded emissions for key pollutants (NOx, CO, SO2,
NMVOCs, CH4, NH3, PM2.5, etc.) at 0.1° × 0.1° resolution. We download
and regrid to the model resolution.

See: https://edgar.jrc.ec.europa.eu/
"""
mutable struct EDGARManager
    data_dir   :: String
    version    :: String   # e.g. "v8.0"
    base_year  :: Int      # inventory reference year
    resolution :: Float64
    cache      :: Dict{Symbol, Array{Float64, 2}}
end

function EDGARManager(; data_dir::String="data/edgar",
                       version::String="v8.0",
                       base_year::Int=2022,
                       resolution::Float64=1.0)
    mkpath(data_dir)
    return EDGARManager(data_dir, version, base_year, resolution,
                        Dict{Symbol, Array{Float64, 2}}())
end

# EDGAR species mapping → our model species
const EDGAR_SPECIES = Dict{Symbol, String}(
    :NOx  => "NOx",
    :CO   => "CO",
    :SO2  => "SO2",
    :CH4  => "CH4",
    :NMVOC => "NMVOC",
    :NH3  => "NH3",
    :PM25 => "PM2.5",
)

# EDGAR sector codes and descriptions
const EDGAR_SECTORS = Dict{String, String}(
    "ENE" => "Power industry",
    "REF_TRF" => "Oil refineries and transformation",
    "IND" => "Manufacturing industries and construction",
    "TNR_Aviation_CDS" => "Aviation (climbing/descent)",
    "TNR_Aviation_CRS" => "Aviation (cruise)",
    "TNR_Aviation_LTO" => "Aviation (landing/takeoff)",
    "TNR_Ship" => "Shipping",
    "TNR_Other" => "Other transport",
    "TRO_noRES" => "Road transport (non-residential)",
    "RCO" => "Residential and other sectors",
    "PRO" => "Fuel exploitation",
    "NMM" => "Non-metallic minerals production",
    "CHE" => "Chemical processes",
    "IRO" => "Iron and steel production",
    "NFE" => "Non-ferrous metals production",
    "NEU" => "Non-energy use of fuels",
    "AGS" => "Agricultural soils",
    "AWB" => "Agricultural waste burning",
    "SWD_INC" => "Solid waste incineration",
    "SWD_LDF" => "Solid waste landfills",
    "FFF" => "Fossil fuel fires",
)

"""
    edgar_url(version, species, year; sector=nothing)

Construct the download URL for an EDGAR emissions NetCDF file.
"""
function edgar_url(version::String, species::String, year::Int;
                   sector::Union{String, Nothing}=nothing)
    base = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR"
    ver_path = replace(version, "." => "")  # v80

    if sector !== nothing
        return "$base/datasets/$ver_path/$(species)_$(year)_$(sector).0.1x0.1.nc"
    else
        return "$base/datasets/$ver_path/$(species)_$(year)_TOTALS.0.1x0.1.nc"
    end
end

"""
    download_edgar!(mgr; species=keys(EDGAR_SPECIES))

Download EDGAR emission inventories for specified species.
Downloads total (all-sector) emissions by default.
"""
function download_edgar!(mgr::EDGARManager;
                         species::Union{Vector{Symbol}, Nothing}=nothing)
    if species === nothing
        species = collect(keys(EDGAR_SPECIES))
    end

    for sp in species
        edgar_name = EDGAR_SPECIES[sp]
        filename = "edgar_$(edgar_name)_$(mgr.base_year)_TOTALS.nc"
        filepath = joinpath(mgr.data_dir, filename)

        if isfile(filepath)
            @info "EDGAR $edgar_name already downloaded"
            continue
        end

        url = edgar_url(mgr.version, edgar_name, mgr.base_year)
        @info "Downloading EDGAR $edgar_name from $url..."

        try
            resp = HTTP.get(url; status_exception=false, connect_timeout=30, readtimeout=120)
            if resp.status == 200
                write(filepath, resp.body)
                @info "Downloaded $(edgar_name): $(filesize(filepath) / 1e6) MB"
            else
                @warn "Failed to download EDGAR $edgar_name (HTTP $(resp.status)). " *
                      "The file may need to be downloaded manually."
                _generate_synthetic_edgar!(mgr, sp, filepath)
            end
        catch e
            @warn "Network error downloading EDGAR $edgar_name: $e. " *
                  "Generating synthetic emissions as placeholder."
            _generate_synthetic_edgar!(mgr, sp, filepath)
        end
    end
end

"""
Generate synthetic emission fields as placeholders when EDGAR download fails.
Uses reasonable global emission distributions based on literature values.
"""
function _generate_synthetic_edgar!(mgr::EDGARManager, sp::Symbol, filepath::String)
    # Global annual total emissions [kg/year] (approximate from literature)
    totals = Dict{Symbol, Float64}(
        :NOx  => 120e9,   # ~120 Tg NO2/yr
        :CO   => 600e9,   # ~600 Tg CO/yr
        :SO2  => 100e9,   # ~100 Tg SO2/yr
        :CH4  => 580e9,   # ~580 Tg CH4/yr
        :NMVOC => 150e9,
        :NH3  => 60e9,
        :PM25 => 55e9,
    )

    nlon = round(Int, 360.0 / mgr.resolution)
    nlat = round(Int, 180.0 / mgr.resolution) + 1

    emissions = zeros(Float64, nlon, nlat)

    total = get(totals, sp, 100e9)

    # Distribute emissions with population-proxy pattern
    # (enhanced over continents, especially NH mid-latitudes)
    for j in 1:nlat, i in 1:nlon
        lon = (i - 1) * mgr.resolution
        lat = -90.0 + (j - 1) * mgr.resolution

        # Background
        e = 0.01

        # Enhanced over land (crude continent mask)
        is_land = _crude_land_mask(lon, lat)
        if is_land
            e += 0.5

            # Population density proxy: higher in NH mid-latitudes
            lat_factor = exp(-((lat - 35)^2) / (2 * 20^2))  # NH peak
            e += 2.0 * lat_factor

            # Industrial hotspots
            if _is_industrial_region(lon, lat)
                e += 5.0
            end
        end

        emissions[i, j] = e
    end

    # Normalise to get correct total [kg m⁻² s⁻¹]
    earth_area = 4π * Constants.R_EARTH^2
    cell_area = earth_area / (nlon * nlat)
    current_total = sum(emissions) * cell_area * 365.25 * 86400
    emissions .*= (total / current_total)

    mgr.cache[sp] = emissions
    @info "Generated synthetic $sp emissions (total: $(total/1e9) Tg/yr)"
end

"""Crude land mask based on major continent bounding boxes."""
function _crude_land_mask(lon::Float64, lat::Float64)
    # North America
    (lon > 230 && lon < 310 && lat > 15 && lat < 75) && return true
    # South America
    (lon > 280 && lon < 325 && lat > -55 && lat < 15) && return true
    # Europe
    (lon > 350 || lon < 40) && lat > 35 && lat < 72 && return true
    # Africa
    (lon > 340 || lon < 52) && lat > -35 && lat < 37 && return true
    # Asia
    (lon > 25 && lon < 180 && lat > 0 && lat < 75) && return true
    # Australia
    (lon > 110 && lon < 155 && lat > -45 && lat < -10) && return true
    return false
end

"""Check if coordinates fall in major industrial regions."""
function _is_industrial_region(lon::Float64, lat::Float64)
    # Eastern China
    (lon > 100 && lon < 125 && lat > 22 && lat < 42) && return true
    # India
    (lon > 70 && lon < 90 && lat > 8 && lat < 30) && return true
    # Western Europe
    ((lon > 350 || lon < 15) && lat > 42 && lat < 55) && return true
    # Eastern US
    (lon > 265 && lon < 290 && lat > 30 && lat < 45) && return true
    # Japan/Korea
    (lon > 125 && lon < 145 && lat > 30 && lat < 45) && return true
    return false
end

"""
    load_emissions(mgr, species; hgrid, gpu)

Load emission field for a species, regridded to model resolution.
Returns 2-D array (nlon, nlat) in [kg m⁻² s⁻¹].
"""
function load_emissions(mgr::EDGARManager, sp::Symbol;
                        hgrid::HorizontalGrid,
                        gpu::Bool=false)
    # Check cache first
    if haskey(mgr.cache, sp)
        return to_device(mgr.cache[sp]; gpu)
    end

    edgar_name = EDGAR_SPECIES[sp]
    filename = "edgar_$(edgar_name)_$(mgr.base_year)_TOTALS.nc"
    filepath = joinpath(mgr.data_dir, filename)

    if !isfile(filepath)
        @warn "EDGAR file not found for $sp, generating synthetic data"
        _generate_synthetic_edgar!(mgr, sp, filepath)
        return to_device(mgr.cache[sp]; gpu)
    end

    # Load and regrid from native 0.1° to model resolution
    emissions = NCDatasets.Dataset(filepath) do ds
        # EDGAR typically has variable named "emi_<species>" or "flx_<species>"
        varnames = keys(ds)
        emi_var = nothing
        for vn in varnames
            if startswith(String(vn), "emi") || startswith(String(vn), "flx")
                emi_var = String(vn)
                break
            end
        end

        if emi_var === nothing
            @warn "Could not find emission variable in $filepath"
            return zeros(Float64, hgrid.nlon, hgrid.nlat)
        end

        raw = Float64.(ds[emi_var][:, :])
        return _regrid_emissions(raw, hgrid)
    end

    mgr.cache[sp] = emissions
    return to_device(emissions; gpu)
end

"""Regrid emissions from EDGAR native resolution to model grid via area-weighted averaging."""
function _regrid_emissions(raw::Matrix{Float64}, hgrid::HorizontalGrid)
    nlon_raw, nlat_raw = size(raw)
    res_raw = 360.0 / nlon_raw

    nlon = hgrid.nlon
    nlat = hgrid.nlat
    ratio = round(Int, hgrid.resolution / res_raw)

    if ratio <= 1
        # Same or higher resolution — just return subset
        return raw[1:min(nlon, nlon_raw), 1:min(nlat, nlat_raw)]
    end

    # Average over ratio × ratio blocks
    result = zeros(Float64, nlon, nlat)
    for j in 1:nlat
        j_lo = max(1, (j - 1) * ratio + 1)
        j_hi = min(nlat_raw, j * ratio)
        for i in 1:nlon
            i_lo = max(1, (i - 1) * ratio + 1)
            i_hi = min(nlon_raw, i * ratio)
            block = @view raw[i_lo:i_hi, j_lo:j_hi]
            result[i, j] = mean_skipnan(block)
        end
    end

    return result
end

"""Mean of an array, skipping NaN values."""
function mean_skipnan(x)
    s = 0.0
    n = 0
    for v in x
        if !isnan(v) && !ismissing(v)
            s += v
            n += 1
        end
    end
    return n > 0 ? s / n : 0.0
end
