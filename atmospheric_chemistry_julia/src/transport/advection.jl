# ---------------------------------------------------------------------------
# Advection — Piecewise Parabolic Method (PPM) for tracer transport
# ---------------------------------------------------------------------------

"""
    advect!(conc, u, v, w, hgrid, vgrid, ps, dt; gpu=false)

Perform 3-D advection of all tracer species using operator splitting:
1. Horizontal advection (lon direction) with PPM
2. Horizontal advection (lat direction) with PPM
3. Vertical advection with PPM

Arguments:
- `conc`: 4-D array (nlon, nlat, nlevels, nspecies) — modified in-place
- `u, v`: horizontal wind components [m s⁻¹]
- `w`: vertical velocity (omega) [Pa s⁻¹]
- `hgrid`: HorizontalGrid
- `vgrid`: VerticalGrid
- `ps`: surface pressure [Pa]
- `dt`: advection timestep [s]
"""
function advect!(conc::AbstractArray{Float64, 4},
                  u::AbstractArray{Float64, 3},
                  v::AbstractArray{Float64, 3},
                  w::AbstractArray{Float64, 3},
                  hgrid::HorizontalGrid,
                  vgrid::VerticalGrid,
                  ps::AbstractMatrix{Float64},
                  dt::Float64;
                  gpu::Bool=false)
    nlon, nlat, nlevels, nspec = size(conc)

    # Compute CFL-limited sub-steps
    max_u = maximum(abs, u)
    max_v = maximum(abs, v)
    dx_min = minimum(hgrid.dx)
    dy_min = minimum(hgrid.dy)

    cfl_limit = 0.8
    n_sub_x = max(1, ceil(Int, max_u * dt / (cfl_limit * dx_min)))
    n_sub_y = max(1, ceil(Int, max_v * dt / (cfl_limit * dy_min)))
    dt_sub_x = dt / n_sub_x
    dt_sub_y = dt / n_sub_y

    # Operator splitting: X → Y → Z
    for _ in 1:n_sub_x
        _advect_x!(conc, u, hgrid, dt_sub_x)
    end

    for _ in 1:n_sub_y
        _advect_y!(conc, v, hgrid, dt_sub_y)
    end

    # Vertical advection
    dp = layer_thickness_dp(vgrid, ps)
    _advect_z!(conc, w, dp, dt)
end

"""
Zonal (x-direction) advection using PPM with periodic boundary conditions.
"""
function _advect_x!(conc::AbstractArray{Float64, 4},
                     u::AbstractArray{Float64, 3},
                     hgrid::HorizontalGrid,
                     dt::Float64)
    nlon, nlat, nlevels, nspec = size(conc)

    Threads.@threads for k in 1:nlevels
        for j in 1:nlat
            for s in 1:nspec
                # Extract 1-D slice
                q = zeros(Float64, nlon)
                for i in 1:nlon
                    q[i] = conc[i, j, k, s]
                end

                # Compute fluxes using PPM
                flux = _ppm_flux_periodic(q, view(u, :, j, k),
                                          view(hgrid.dx, :, j), dt, nlon)

                # Update concentrations
                for i in 1:nlon
                    ip = mod1(i + 1, nlon)
                    conc[i, j, k, s] = q[i] - dt / hgrid.dx[i, j] * (flux[ip] - flux[i])
                    conc[i, j, k, s] = max(conc[i, j, k, s], 0.0)
                end
            end
        end
    end
end

"""
Meridional (y-direction) advection using PPM with zero-flux boundary at poles.
"""
function _advect_y!(conc::AbstractArray{Float64, 4},
                     v::AbstractArray{Float64, 3},
                     hgrid::HorizontalGrid,
                     dt::Float64)
    nlon, nlat, nlevels, nspec = size(conc)

    Threads.@threads for k in 1:nlevels
        for i in 1:nlon
            for s in 1:nspec
                q = zeros(Float64, nlat)
                for j in 1:nlat
                    q[j] = conc[i, j, k, s]
                end

                flux = _ppm_flux_bounded(q, view(v, i, :, k),
                                          view(hgrid.dy, i, :), dt, nlat)

                for j in 1:nlat
                    jp = min(j + 1, nlat)
                    conc[i, j, k, s] = q[j] - dt / hgrid.dy[i, j] * (flux[jp] - flux[j])
                    conc[i, j, k, s] = max(conc[i, j, k, s], 0.0)
                end
            end
        end
    end
end

"""
Vertical advection using a simple upwind scheme in pressure coordinates.
"""
function _advect_z!(conc::AbstractArray{Float64, 4},
                     w::AbstractArray{Float64, 3},
                     dp::AbstractArray{Float64, 3},
                     dt::Float64)
    nlon, nlat, nlevels, nspec = size(conc)

    Threads.@threads for j in 1:nlat
        for i in 1:nlon
            for s in 1:nspec
                # Compute vertical fluxes (upwind)
                flux = zeros(Float64, nlevels + 1)
                for k in 2:nlevels
                    omega = w[i, j, k]  # Pa/s, positive = downward
                    if omega > 0  # Downward: use value from above
                        flux[k] = omega * conc[i, j, k-1, s]
                    else          # Upward: use value from below
                        flux[k] = omega * conc[i, j, k, s]
                    end
                end
                # Zero flux at boundaries
                flux[1] = 0.0
                flux[nlevels + 1] = 0.0

                # Update
                for k in 1:nlevels
                    dp_k = max(dp[i, j, k], 1.0)
                    conc[i, j, k, s] -= dt / dp_k * (flux[k+1] - flux[k])
                    conc[i, j, k, s] = max(conc[i, j, k, s], 0.0)
                end
            end
        end
    end
end

"""
PPM flux computation for periodic domain (zonal direction).
"""
function _ppm_flux_periodic(q::Vector{Float64}, u_face, dx, dt::Float64, n::Int)
    flux = zeros(Float64, n + 1)

    # Compute limited slopes (MC limiter)
    dq = zeros(Float64, n)
    for i in 1:n
        im = mod1(i - 1, n)
        ip = mod1(i + 1, n)
        dq_left  = q[i] - q[im]
        dq_right = q[ip] - q[i]
        dq_cent  = 0.5 * (q[ip] - q[im])
        # MC limiter
        if dq_left * dq_right > 0
            dq[i] = sign(dq_cent) * min(abs(dq_cent), 2 * abs(dq_left), 2 * abs(dq_right))
        else
            dq[i] = 0.0
        end
    end

    # Compute interface values and fluxes
    for i in 1:n
        im = mod1(i - 1, n)
        u_i = u_face[i]
        courant = u_i * dt / dx[i]

        if u_i >= 0  # Flow from left (im → i)
            q_face = q[im] + 0.5 * (1.0 - courant) * dq[im]
        else         # Flow from right (i → im)
            q_face = q[i] - 0.5 * (1.0 + courant) * dq[i]
        end
        flux[i] = u_i * max(q_face, 0.0)
    end
    flux[n + 1] = flux[1]  # Periodic

    return flux
end

"""
PPM flux computation for bounded domain (meridional direction).
"""
function _ppm_flux_bounded(q::Vector{Float64}, v_face, dy, dt::Float64, n::Int)
    flux = zeros(Float64, n + 1)

    # Slopes with MC limiter
    dq = zeros(Float64, n)
    for j in 2:n-1
        dq_left  = q[j] - q[j-1]
        dq_right = q[j+1] - q[j]
        dq_cent  = 0.5 * (q[j+1] - q[j-1])
        if dq_left * dq_right > 0
            dq[j] = sign(dq_cent) * min(abs(dq_cent), 2 * abs(dq_left), 2 * abs(dq_right))
        end
    end

    for j in 2:n
        v_j = v_face[j]
        courant = v_j * dt / dy[j]

        if v_j >= 0
            q_face = q[j-1] + 0.5 * (1.0 - courant) * dq[j-1]
        else
            q_face = q[j] - 0.5 * (1.0 + courant) * dq[j]
        end
        flux[j] = v_j * max(q_face, 0.0)
    end

    # Zero flux at poles
    flux[1] = 0.0
    flux[n + 1] = 0.0

    return flux
end
