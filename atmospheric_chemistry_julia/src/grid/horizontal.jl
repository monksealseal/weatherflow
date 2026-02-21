# ---------------------------------------------------------------------------
# Horizontal grid — regular lat-lon at configurable resolution
# ---------------------------------------------------------------------------

"""
    HorizontalGrid(; resolution=1.0, gpu=false)

Regular latitude-longitude grid. Default is 1° × 1° (360 × 181 for 0–360, -90–90).

Fields:
- `nlon`, `nlat`: grid dimensions
- `lon`, `lat`: 1-D coordinate vectors [degrees]
- `λ`, `φ`: 1-D coordinate vectors [radians]
- `dx`, `dy`: 2-D grid spacing arrays [m]
- `area`: 2-D cell area [m²]
- `coslat`: cos(latitude) array (used everywhere in transport)
"""
struct HorizontalGrid{T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    resolution :: T
    nlon       :: Int
    nlat       :: Int
    lon        :: V          # degrees
    lat        :: V          # degrees
    λ          :: V          # radians
    φ          :: V          # radians
    dx         :: M          # [m] — varies with latitude
    dy         :: M          # [m] — constant for regular grid
    area       :: M          # [m²]
    coslat     :: V          # cos(latitude)
end

function HorizontalGrid(; resolution::Float64=1.0, gpu::Bool=false)
    nlon = round(Int, 360.0 / resolution)
    nlat = round(Int, 180.0 / resolution) + 1

    lon = collect(range(0.0, step=resolution, length=nlon))
    lat = collect(range(-90.0, stop=90.0, length=nlat))

    λ = deg2rad.(lon)
    φ = deg2rad.(lat)

    R = Constants.R_EARTH
    dφ = deg2rad(resolution)
    dλ = deg2rad(resolution)

    coslat = cos.(φ)
    # Clamp to avoid division by zero at poles
    coslat_safe = max.(coslat, 1e-10)

    # Grid spacing in metres
    dx_mat = zeros(Float64, nlon, nlat)
    dy_mat = zeros(Float64, nlon, nlat)
    area_mat = zeros(Float64, nlon, nlat)

    for j in 1:nlat
        for i in 1:nlon
            dx_mat[i, j] = R * coslat_safe[j] * dλ
            dy_mat[i, j] = R * dφ
            # Area of grid cell
            lat_lo = φ[j] - dφ / 2
            lat_hi = φ[j] + dφ / 2
            area_mat[i, j] = R^2 * abs(sin(lat_hi) - sin(lat_lo)) * dλ
        end
    end

    if gpu
        lon      = to_device(lon; gpu)
        lat      = to_device(lat; gpu)
        λ        = to_device(λ; gpu)
        φ        = to_device(φ; gpu)
        coslat   = to_device(coslat; gpu)
        dx_mat   = to_device(dx_mat; gpu)
        dy_mat   = to_device(dy_mat; gpu)
        area_mat = to_device(area_mat; gpu)
    end

    return HorizontalGrid(resolution, nlon, nlat, lon, lat, λ, φ,
                           dx_mat, dy_mat, area_mat, coslat)
end

Base.show(io::IO, g::HorizontalGrid) =
    print(io, "HorizontalGrid($(g.nlon)×$(g.nlat) @ $(g.resolution)°)")
