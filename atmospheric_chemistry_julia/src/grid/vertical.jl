# ---------------------------------------------------------------------------
# Vertical grid — hybrid sigma-pressure coordinates (ERA5 L137 / L60 / custom)
# ---------------------------------------------------------------------------

"""
    VerticalGrid(; nlevels=47, p_top=1.0, gpu=false)

Hybrid sigma-pressure vertical coordinate.

The pressure at each level is: p(k) = a(k) + b(k) * p_surface

For simplicity we provide a default 47-level configuration that spans
from ~1 Pa (top) to the surface, similar to a reduced ERA5 grid.

Fields:
- `nlevels`: number of model levels
- `a_half`, `b_half`: hybrid coefficients at level interfaces (nlevels+1)
- `a_full`, `b_full`: hybrid coefficients at level centres (nlevels)
- `p_top`: pressure at model top [Pa]
"""
struct VerticalGrid{T <: AbstractFloat, V <: AbstractVector{T}}
    nlevels :: Int
    a_half  :: V   # interface coefficients [Pa]
    b_half  :: V   # interface coefficients [dimensionless]
    a_full  :: V   # mid-level coefficients [Pa]
    b_full  :: V   # mid-level coefficients [dimensionless]
    p_top   :: T
end

function VerticalGrid(; nlevels::Int=47, p_top::Float64=1.0, gpu::Bool=false)
    # Generate default hybrid coefficients
    # Use a smooth transition: upper levels are pure pressure (b≈0),
    # lower levels are pure sigma (a≈0)
    a_half = zeros(Float64, nlevels + 1)
    b_half = zeros(Float64, nlevels + 1)

    for k in 1:(nlevels + 1)
        # Normalised level index [0 = top, 1 = surface]
        η = (k - 1) / nlevels

        # Smooth transition using a cubic profile
        b_half[k] = η^3
        a_half[k] = p_top + (Constants.P_REF - p_top) * (η - η^3)
    end

    # Ensure boundary conditions
    a_half[1]           = p_top
    b_half[1]           = 0.0
    a_half[nlevels + 1] = 0.0
    b_half[nlevels + 1] = 1.0

    # Mid-level values (simple average of interfaces)
    a_full = 0.5 .* (a_half[1:end-1] .+ a_half[2:end])
    b_full = 0.5 .* (b_half[1:end-1] .+ b_half[2:end])

    if gpu
        a_half = to_device(a_half; gpu)
        b_half = to_device(b_half; gpu)
        a_full = to_device(a_full; gpu)
        b_full = to_device(b_full; gpu)
    end

    return VerticalGrid(nlevels, a_half, b_half, a_full, b_full, p_top)
end

"""
    pressure_at_levels(vg::VerticalGrid, ps)

Compute 3-D pressure field from surface pressure `ps` (2-D).
Returns full-level pressures with shape (nlon, nlat, nlevels).
"""
function pressure_at_levels(vg::VerticalGrid, ps::AbstractMatrix{T}) where T
    nlon, nlat = size(ps)
    p = similar(ps, nlon, nlat, vg.nlevels)
    for k in 1:vg.nlevels
        @views p[:, :, k] .= vg.a_full[k] .+ vg.b_full[k] .* ps
    end
    return p
end

"""
    pressure_at_interfaces(vg::VerticalGrid, ps)

Compute interface pressures. Returns shape (nlon, nlat, nlevels+1).
"""
function pressure_at_interfaces(vg::VerticalGrid, ps::AbstractMatrix{T}) where T
    nlon, nlat = size(ps)
    p = similar(ps, nlon, nlat, vg.nlevels + 1)
    for k in 1:(vg.nlevels + 1)
        @views p[:, :, k] .= vg.a_half[k] .+ vg.b_half[k] .* ps
    end
    return p
end

"""
    layer_thickness_dp(vg::VerticalGrid, ps)

Pressure thickness Δp of each layer [Pa]. Shape (nlon, nlat, nlevels).
"""
function layer_thickness_dp(vg::VerticalGrid, ps::AbstractMatrix)
    pi = pressure_at_interfaces(vg, ps)
    nlon, nlat = size(ps)
    dp = similar(ps, nlon, nlat, vg.nlevels)
    for k in 1:vg.nlevels
        @views dp[:, :, k] .= pi[:, :, k+1] .- pi[:, :, k]
    end
    return dp
end

Base.show(io::IO, vg::VerticalGrid) =
    print(io, "VerticalGrid($(vg.nlevels) levels, p_top=$(vg.p_top) Pa)")
