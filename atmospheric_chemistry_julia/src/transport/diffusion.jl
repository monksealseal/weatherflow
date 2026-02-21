# ---------------------------------------------------------------------------
# Vertical diffusion — boundary layer turbulent mixing
# ---------------------------------------------------------------------------

"""
    diffuse_vertical!(conc, Kz, vgrid, ps, dt)

Apply vertical turbulent diffusion using an implicit (Crank-Nicolson)
tridiagonal solver.

Arguments:
- `conc`: 4-D concentration array (nlon, nlat, nlevels, nspecies)
- `Kz`: 3-D vertical diffusion coefficient [m² s⁻¹]
- `vgrid`: VerticalGrid
- `ps`: surface pressure [Pa]
- `dt`: timestep [s]

The diffusion equation in pressure coordinates:
  ∂c/∂t = (g²ρ²/p) ∂/∂p [Kz (p/g²ρ²) ∂c/∂p]
"""
function diffuse_vertical!(conc::AbstractArray{Float64, 4},
                            Kz::AbstractArray{Float64, 3},
                            vgrid::VerticalGrid,
                            ps::AbstractMatrix{Float64},
                            dt::Float64)
    nlon, nlat, nlevels, nspec = size(conc)
    dp = layer_thickness_dp(vgrid, ps)

    Threads.@threads for j in 1:nlat
        for i in 1:nlon
            for s in 1:nspec
                # Build tridiagonal system
                a = zeros(Float64, nlevels)  # sub-diagonal
                b = zeros(Float64, nlevels)  # diagonal
                c_tri = zeros(Float64, nlevels)  # super-diagonal
                d = zeros(Float64, nlevels)  # RHS

                for k in 1:nlevels
                    dp_k = max(dp[i, j, k], 1.0)

                    # Diffusion coefficient at interfaces
                    if k > 1
                        Kz_lower = 0.5 * (Kz[i, j, k-1] + Kz[i, j, k])
                        dp_lower = 0.5 * (dp[i, j, k-1] + dp[i, j, k])
                        # Convert Kz [m²/s] to pressure coordinates
                        # Using hydrostatic: Δz ≈ Δp / (ρg), Kz_p ≈ Kz × (ρg)²
                        rho_g2 = (Constants.G * Constants.P_REF / (Constants.R_GAS * Constants.T_REF / Constants.M_AIR))^2
                        α_lower = Kz_lower * rho_g2 * dt / (dp_k * dp_lower)
                    else
                        α_lower = 0.0
                    end

                    if k < nlevels
                        Kz_upper = 0.5 * (Kz[i, j, k] + Kz[i, j, k+1])
                        dp_upper = 0.5 * (dp[i, j, k] + dp[i, j, k+1])
                        rho_g2 = (Constants.G * Constants.P_REF / (Constants.R_GAS * Constants.T_REF / Constants.M_AIR))^2
                        α_upper = Kz_upper * rho_g2 * dt / (dp_k * dp_upper)
                    else
                        α_upper = 0.0
                    end

                    # Crank-Nicolson (θ = 0.5)
                    θ = 0.5
                    a[k] = -θ * α_lower
                    c_tri[k] = -θ * α_upper
                    b[k] = 1.0 + θ * (α_lower + α_upper)

                    # RHS
                    c_old = conc[i, j, k, s]
                    rhs = c_old
                    if k > 1
                        rhs += (1 - θ) * α_lower * (conc[i, j, k-1, s] - c_old)
                    end
                    if k < nlevels
                        rhs += (1 - θ) * α_upper * (conc[i, j, k+1, s] - c_old)
                    end
                    d[k] = rhs
                end

                # Solve tridiagonal system (Thomas algorithm)
                result = _thomas_solve(a, b, c_tri, d, nlevels)

                for k in 1:nlevels
                    conc[i, j, k, s] = max(result[k], 0.0)
                end
            end
        end
    end
end

"""
Thomas algorithm for tridiagonal system Ax = d.
"""
function _thomas_solve(a::Vector{Float64}, b::Vector{Float64},
                       c::Vector{Float64}, d::Vector{Float64}, n::Int)
    # Forward sweep
    c_star = zeros(Float64, n)
    d_star = zeros(Float64, n)

    c_star[1] = c[1] / b[1]
    d_star[1] = d[1] / b[1]

    for i in 2:n
        denom = b[i] - a[i] * c_star[i-1]
        if abs(denom) < 1e-30
            denom = 1e-30
        end
        c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i] * d_star[i-1]) / denom
    end

    # Back substitution
    x = zeros(Float64, n)
    x[n] = d_star[n]
    for i in (n-1):-1:1
        x[i] = d_star[i] - c_star[i] * x[i+1]
    end

    return x
end

"""
    compute_Kz(blh, vgrid, ps)

Compute vertical diffusion coefficients based on boundary layer height.

Uses a simple profile: Kz is large within the PBL and small above.
"""
function compute_Kz(blh::AbstractMatrix{Float64},
                    vgrid::VerticalGrid,
                    ps::AbstractMatrix{Float64};
                    Kz_pbl::Float64=50.0,    # m²/s in PBL
                    Kz_free::Float64=1.0)     # m²/s in free troposphere
    nlon, nlat = size(ps)
    nlevels = vgrid.nlevels
    Kz = fill(Kz_free, nlon, nlat, nlevels)

    for j in 1:nlat, i in 1:nlon
        blh_local = max(blh[i, j], 100.0)  # Minimum BLH = 100 m

        for k in 1:nlevels
            p_k = vgrid.a_full[k] + vgrid.b_full[k] * ps[i, j]
            # Approximate height from pressure
            z_k = -7000.0 * log(p_k / ps[i, j])

            if z_k < blh_local
                # Cubic profile within PBL (maximum at z/h ~ 0.3)
                η = z_k / blh_local
                Kz[i, j, k] = Kz_pbl * η * (1.0 - η)^2 * 6.75 + Kz_free
            end
        end
    end

    return Kz
end
