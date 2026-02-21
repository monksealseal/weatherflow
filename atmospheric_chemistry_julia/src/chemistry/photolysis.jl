# ---------------------------------------------------------------------------
# Photolysis rate calculations
# ---------------------------------------------------------------------------

"""
    PhotolysisRates

Container for photolysis rate coefficients (j-values) as a function
of solar zenith angle, altitude, and overhead ozone column.

Uses a parameterised lookup table approach similar to the FAST-JX scheme.
"""
struct PhotolysisRates{T <: AbstractFloat}
    j_NO2   :: T    # NO2 photolysis rate [s⁻¹]
    j_O3    :: T    # O3 photolysis rate (to O(¹D)) [s⁻¹]
    j_H2O2  :: T    # H2O2 photolysis rate [s⁻¹]
    j_HCHO  :: T    # HCHO photolysis rate (radical channel) [s⁻¹]
end

"""
    compute_photolysis_rates(sza, altitude_km, overhead_O3_DU, cloud_fraction)

Compute photolysis j-values based on:
- `sza`: Solar zenith angle [degrees]
- `altitude_km`: Altitude above sea level [km]
- `overhead_O3_DU`: Overhead ozone column [Dobson Units]
- `cloud_fraction`: Cloud fraction [0-1] — reduces UV below clouds

Returns a `PhotolysisRates` struct.
"""
function compute_photolysis_rates(sza::Float64, altitude_km::Float64,
                                   overhead_O3_DU::Float64,
                                   cloud_fraction::Float64)
    # No photolysis at night
    if sza >= 90.0
        return PhotolysisRates(0.0, 0.0, 0.0, 0.0)
    end

    cos_sza = cos(deg2rad(sza))
    # Effective path length through atmosphere (Chapman function approximation)
    sec_sza = 1.0 / max(cos_sza, 0.01)

    # Altitude enhancement factor (less absorption above)
    alt_factor = exp(altitude_km / 8.0)  # Scale height ~ 8 km

    # Ozone absorption factor
    # More overhead O3 → less UV reaching lower troposphere
    o3_factor = exp(-0.003 * overhead_O3_DU * sec_sza)

    # Cloud attenuation (simple parameterisation)
    cloud_factor = 1.0 - 0.7 * cloud_fraction

    # Base j-values at surface, clear sky, overhead sun, 300 DU O3
    j_NO2_base  = 8.0e-3   # s⁻¹
    j_O3_base   = 3.0e-5   # s⁻¹ (O(¹D) channel)
    j_H2O2_base = 7.0e-6   # s⁻¹
    j_HCHO_base = 3.0e-5   # s⁻¹ (radical channel)

    # Zenith angle dependence
    zen_factor_vis = cos_sza^0.4    # Visible/near-UV (NO2)
    zen_factor_uv  = cos_sza^1.2    # Short UV (O3 → O(¹D))

    j_NO2  = j_NO2_base * zen_factor_vis * alt_factor * cloud_factor
    j_O3   = j_O3_base * zen_factor_uv * alt_factor * o3_factor * cloud_factor
    j_H2O2 = j_H2O2_base * zen_factor_uv * alt_factor * o3_factor * cloud_factor
    j_HCHO = j_HCHO_base * zen_factor_vis * alt_factor * o3_factor * cloud_factor

    return PhotolysisRates(j_NO2, j_O3, j_H2O2, j_HCHO)
end

"""
    solar_zenith_angle(lat, lon, datetime)

Compute the solar zenith angle [degrees] for a given location and time.
"""
function solar_zenith_angle(lat::Float64, lon::Float64, dt::DateTime)
    # Day of year
    doy = Dates.dayofyear(dt)

    # Solar declination (Spencer formula)
    Γ = 2π * (doy - 1) / 365.0
    δ = 0.006918 - 0.399912 * cos(Γ) + 0.070257 * sin(Γ) -
        0.006758 * cos(2Γ) + 0.000907 * sin(2Γ) -
        0.002697 * cos(3Γ) + 0.00148 * sin(3Γ)

    # Hour angle
    hour_utc = Dates.hour(dt) + Dates.minute(dt) / 60.0
    solar_time = hour_utc + lon / 15.0  # Approximate local solar time
    hour_angle = deg2rad(15.0 * (solar_time - 12.0))

    # Solar zenith angle
    φ = deg2rad(lat)
    cos_sza = sin(φ) * sin(δ) + cos(φ) * cos(δ) * cos(hour_angle)
    sza = rad2deg(acos(clamp(cos_sza, -1.0, 1.0)))

    return sza
end

"""
    compute_photolysis_field!(j_rates, hgrid, vgrid, ps, T, O3_col, cloud_frac, dt)

Compute photolysis rates at every grid cell.
`j_rates` is a 4-D array (nlon, nlat, nlevels, n_photo_rxns).
"""
function compute_photolysis_field!(j_rates::AbstractArray{Float64, 4},
                                    hgrid::HorizontalGrid,
                                    vgrid::VerticalGrid,
                                    ps::AbstractMatrix,
                                    O3_column::AbstractMatrix,
                                    cloud_frac::AbstractMatrix,
                                    dt::DateTime)
    nlon = hgrid.nlon
    nlat = hgrid.nlat
    nlevels = vgrid.nlevels

    Threads.@threads for j in 1:nlat
        for i in 1:nlon
            lat = Float64(hgrid.lat[j])
            lon = Float64(hgrid.lon[i])
            sza = solar_zenith_angle(lat, lon, dt)

            o3_du = Float64(O3_column[i, j])
            cf = Float64(cloud_frac[i, j])

            for k in 1:nlevels
                # Approximate altitude from pressure
                p_k = vgrid.a_full[k] + vgrid.b_full[k] * Float64(ps[i, j])
                altitude_km = -7.0 * log(p_k / Constants.P_REF)  # Scale height approx

                # Overhead ozone decreases with altitude
                overhead_o3 = o3_du * (p_k / Constants.P_REF)

                rates = compute_photolysis_rates(sza, altitude_km,
                                                  overhead_o3, cf)

                j_rates[i, j, k, 1] = rates.j_NO2
                j_rates[i, j, k, 2] = rates.j_O3
                j_rates[i, j, k, 3] = rates.j_H2O2
                j_rates[i, j, k, 4] = rates.j_HCHO
            end
        end
    end
end
