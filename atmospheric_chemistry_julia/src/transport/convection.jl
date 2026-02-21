# ---------------------------------------------------------------------------
# Convective transport — simple mass-flux parameterisation
# ---------------------------------------------------------------------------

"""
    convective_transport!(conc, T, q, ps, vgrid, dt)

Parameterised convective transport based on a simplified mass-flux approach.

Identifies convectively unstable columns and applies bulk vertical mixing
using an updraft mass flux derived from CAPE and moisture convergence.

This is a simplified scheme inspired by the Zhang-McFarlane (1995) approach:
1. Identify convective columns (positive CAPE)
2. Compute updraft mass flux
3. Apply mass-flux transport to tracers
"""
function convective_transport!(conc::AbstractArray{Float64, 4},
                                T::AbstractArray{Float64, 3},
                                q::AbstractArray{Float64, 3},
                                ps::AbstractMatrix{Float64},
                                vgrid::VerticalGrid,
                                dt::Float64)
    nlon, nlat, nlevels, nspec = size(conc)
    dp = layer_thickness_dp(vgrid, ps)
    p = pressure_at_levels(vgrid, ps)

    Threads.@threads for j in 1:nlat
        for i in 1:nlon
            # Compute CAPE to identify convective columns
            cape = _compute_cape(view(T, i, j, :), view(q, i, j, :),
                                 view(p, i, j, :), vgrid.nlevels)

            if cape < 100.0  # No significant convection
                continue
            end

            # Find cloud base and cloud top
            lcl_k, lfc_k, lnb_k = _find_convective_levels(
                view(T, i, j, :), view(q, i, j, :),
                view(p, i, j, :), vgrid.nlevels)

            if lcl_k <= 0 || lnb_k <= 0 || lnb_k >= lcl_k
                continue
            end

            # Compute updraft mass flux [Pa s⁻¹]
            # Simplified: Mu ∝ sqrt(CAPE) × moisture convergence factor
            Mu_base = 0.01 * sqrt(cape)  # Pa/s
            Mu_base = min(Mu_base, 0.5)  # Cap the mass flux

            # Apply mass-flux transport to each species
            for s in 1:nspec
                _apply_mass_flux!(view(conc, i, j, :, s),
                                  view(dp, i, j, :),
                                  Mu_base, lcl_k, lnb_k,
                                  nlevels, dt)
            end
        end
    end
end

"""
Compute Convective Available Potential Energy (CAPE) [J/kg].
"""
function _compute_cape(T_col::AbstractVector, q_col::AbstractVector,
                       p_col::AbstractVector, nlevels::Int)
    cape = 0.0

    # Start from lowest level
    T_parcel = T_col[nlevels]
    q_parcel = q_col[nlevels]

    for k in (nlevels-1):-1:1
        p_k = p_col[k]
        T_env = T_col[k]

        # Dry adiabatic lapse rate lifting
        T_parcel_dry = T_parcel * (p_k / p_col[k+1])^Constants.KAPPA

        # Saturated: include latent heating (simplified)
        # Use a moist adiabat approximation
        Lv = 2.5e6  # Latent heat of vaporisation [J/kg]
        qs = _sat_mixing_ratio(T_parcel_dry, p_k)

        if q_parcel > qs
            # Parcel is saturated — add latent heating
            dT_latent = Lv * (q_parcel - qs) / Constants.Cp_AIR
            T_parcel = T_parcel_dry + dT_latent * 0.5  # Simplified adjustment
            q_parcel = qs
        else
            T_parcel = T_parcel_dry
        end

        # Buoyancy
        T_v_parcel = T_parcel * (1 + 0.61 * q_parcel)
        T_v_env = T_env * (1 + 0.61 * q_col[k])

        if T_v_parcel > T_v_env
            # Approximate dz from hydrostatic relation
            dz = -(p_col[k] - p_col[k+1]) / (Constants.G *
                  p_k / (Constants.R_GAS / Constants.M_AIR * T_env))
            cape += Constants.G * (T_v_parcel - T_v_env) / T_v_env * abs(dz)
        end
    end

    return cape
end

"""Saturation mixing ratio using Bolton's formula."""
function _sat_mixing_ratio(T::Float64, p::Float64)
    T_C = T - 273.15
    # Saturation vapour pressure [Pa] (Bolton 1980)
    es = 611.2 * exp(17.67 * T_C / (T_C + 243.5))
    # Mixing ratio
    return 0.622 * es / max(p - es, 1.0)
end

"""
Find convective levels: LCL (cloud base), LFC, and LNB (cloud top).
Returns level indices (k increases upward, i.e., k=1 is top, k=nlevels is surface).
"""
function _find_convective_levels(T_col, q_col, p_col, nlevels)
    lcl_k = 0
    lfc_k = 0
    lnb_k = 0

    T_parcel = T_col[nlevels]
    q_parcel = q_col[nlevels]

    for k in (nlevels-1):-1:1
        p_k = p_col[k]
        T_parcel_dry = T_parcel * (p_k / p_col[min(k+1, nlevels)])^Constants.KAPPA

        qs = _sat_mixing_ratio(T_parcel_dry, p_k)

        if lcl_k == 0 && q_parcel >= qs
            lcl_k = k
        end

        if lcl_k > 0
            T_v_parcel = T_parcel_dry * (1 + 0.61 * min(q_parcel, qs))
            T_v_env = T_col[k] * (1 + 0.61 * q_col[k])

            if lfc_k == 0 && T_v_parcel > T_v_env
                lfc_k = k
            end

            if lfc_k > 0 && T_v_parcel < T_v_env
                lnb_k = k + 1
                break
            end
        end

        T_parcel = T_parcel_dry
    end

    # If no LNB found, set to model top
    if lfc_k > 0 && lnb_k == 0
        lnb_k = 1
    end

    return lcl_k, lfc_k, lnb_k
end

"""
Apply mass-flux transport to a single species column.
"""
function _apply_mass_flux!(c_col::AbstractVector, dp_col::AbstractVector,
                           Mu_base::Float64, lcl_k::Int, lnb_k::Int,
                           nlevels::Int, dt::Float64)
    # Mass flux decreases linearly with height from cloud base to cloud top
    for k in min(lcl_k, nlevels):-1:max(lnb_k, 1)
        if lcl_k == lnb_k
            break
        end

        frac = (lcl_k - k) / (lcl_k - lnb_k)
        Mu_k = Mu_base * (1.0 - frac)  # Decreases toward cloud top

        dp_k = max(dp_col[k], 1.0)

        # Entrainment: mix environmental air into updraft
        if k < nlevels
            # Updraft transports air from below
            dc = dt * Mu_k / dp_k * (c_col[min(k+1, nlevels)] - c_col[k])
            # Limit change to avoid instability
            dc = clamp(dc, -0.5 * c_col[k], 0.5 * c_col[min(k+1, nlevels)])
            c_col[k] += dc
            c_col[k] = max(c_col[k], 0.0)
        end
    end
end
