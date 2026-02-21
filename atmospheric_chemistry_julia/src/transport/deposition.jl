# ---------------------------------------------------------------------------
# Dry and wet deposition
# ---------------------------------------------------------------------------

"""
    dry_deposition!(conc, species_list, hgrid, vgrid, ps, u10, v10, blh, dt)

Apply dry deposition to the lowest model level.

Uses a resistance-based approach:
  v_dep = 1 / (Ra + Rb + Rc)
where Ra = aerodynamic resistance, Rb = quasi-laminar sublayer resistance,
Rc = surface resistance (species-dependent).

For simplicity we use pre-tabulated deposition velocities from the Species
definition and scale by wind speed and stability.
"""
function dry_deposition!(conc::AbstractArray{Float64, 4},
                          species_list::Vector{Species},
                          hgrid::HorizontalGrid,
                          vgrid::VerticalGrid,
                          ps::AbstractMatrix{Float64},
                          u10::AbstractMatrix{Float64},
                          v10::AbstractMatrix{Float64},
                          blh::AbstractMatrix{Float64},
                          dt::Float64)
    nlon, nlat, _, nspec = size(conc)
    k_sfc = size(conc, 3)  # Surface level (last index)

    dp = layer_thickness_dp(vgrid, ps)

    Threads.@threads for j in 1:nlat
        for i in 1:nlon
            # Wind speed at 10 m
            ws10 = sqrt(u10[i, j]^2 + v10[i, j]^2)
            ws10 = max(ws10, 0.5)  # Minimum wind speed

            # Aerodynamic resistance
            z_ref = 10.0  # Reference height [m]
            u_star = Constants.VON_KARMAN * ws10 / log(z_ref / 0.1)  # z0 = 0.1 m
            Ra = 1.0 / (Constants.VON_KARMAN * u_star)

            for s in 1:nspec
                sp = species_list[s]
                if !sp.is_deposited || sp.v_dep <= 0.0
                    continue
                end

                # Effective deposition velocity with wind correction
                v_dep_eff = sp.v_dep * (1.0 + 0.3 * (ws10 / 5.0))
                v_dep_eff = min(v_dep_eff, 0.05)  # Cap at 5 cm/s

                # Loss from surface layer
                dp_sfc = max(dp[i, j, k_sfc], 1.0)
                # Convert v_dep [m/s] to loss rate in pressure coords
                # Δc/Δt = -v_dep × c × ρ × g / Δp
                rho_sfc = ps[i, j] / (Constants.R_GAS / Constants.M_AIR * 288.0)
                loss_rate = v_dep_eff * rho_sfc * Constants.G / dp_sfc  # [s⁻¹]

                # Implicit removal
                conc[i, j, k_sfc, s] /= (1.0 + loss_rate * dt)
            end
        end
    end
end

"""
    wet_deposition!(conc, species_list, precip, hgrid, vgrid, ps, dt)

Apply wet deposition (rainout and washout) based on precipitation rates.

Uses Henry's law to determine the fraction of each species scavenged
by rainfall. Applied to all levels below cloud top (simplified: all
levels where precipitation occurs).
"""
function wet_deposition!(conc::AbstractArray{Float64, 4},
                          species_list::Vector{Species},
                          precip::AbstractMatrix{Float64},
                          hgrid::HorizontalGrid,
                          vgrid::VerticalGrid,
                          ps::AbstractMatrix{Float64},
                          T::AbstractArray{Float64, 3},
                          dt::Float64)
    nlon, nlat, nlevels, nspec = size(conc)

    Threads.@threads for j in 1:nlat
        for i in 1:nlon
            # Precipitation rate [mm/hr] → [kg m⁻² s⁻¹]
            # ERA5 provides total precipitation in metres (accumulated)
            # Convert: assume `precip` is already in [mm hr⁻¹]
            P_rate = max(precip[i, j], 0.0) / 3600.0  # mm/hr → mm/s ≈ kg/m²/s

            if P_rate < 1e-8
                continue  # No precipitation
            end

            for s in 1:nspec
                sp = species_list[s]
                if !sp.is_deposited || sp.henry_const <= 0.0
                    continue
                end

                for k in 1:nlevels
                    T_k = T[i, j, k]

                    # Henry's law scavenging
                    # Washout coefficient: Λ = H × R × T × P_rate × f
                    H_eff = sp.henry_const * exp(2000.0 * (1.0/298.15 - 1.0/T_k))

                    # Scavenging rate [s⁻¹]
                    # Λ ≈ 5e-5 × (P_rate [mm/hr])^0.62 for moderate rain (Seinfeld & Pandis)
                    P_rate_mmhr = P_rate * 3600.0  # Convert back for empirical formula
                    lambda = 5e-5 * P_rate_mmhr^0.62

                    # Enhance for very soluble species
                    if H_eff > 1e3
                        lambda *= min(H_eff / 1e3, 10.0)
                    end

                    # Implicit removal
                    conc[i, j, k, s] /= (1.0 + lambda * dt)
                end
            end
        end
    end
end
