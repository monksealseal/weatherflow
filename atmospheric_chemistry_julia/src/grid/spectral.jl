# ---------------------------------------------------------------------------
# Spectral transform utilities for ERA5 spherical-harmonic data
# ---------------------------------------------------------------------------

"""
    SpectralTransform

Handles conversion between ERA5 spectral coefficients (spherical harmonics)
and the model's gridded representation.

ERA5 stores some fields (e.g. vorticity, divergence, temperature, lnsp) as
truncated spherical harmonic expansions. This module provides routines to
transform them to the 1° regular grid used by the chemistry-transport model.
"""
struct SpectralTransform{T <: AbstractFloat}
    T_trunc :: Int                    # Triangular truncation number
    nlon    :: Int
    nlat    :: Int
    lmax    :: Int                    # Maximum degree
    mmax    :: Int                    # Maximum order
    Pnm     :: Array{T, 3}           # Associated Legendre polynomials (nlat, lmax+1, mmax+1)
end

"""
    SpectralTransform(hgrid::HorizontalGrid; T_trunc=180)

Build the spectral transform for a given horizontal grid.
Default truncation T180 matches ~1° resolution.
"""
function SpectralTransform(hgrid::HorizontalGrid; T_trunc::Int=180)
    nlat = hgrid.nlat
    nlon = hgrid.nlon
    lmax = T_trunc
    mmax = T_trunc

    # Pre-compute associated Legendre polynomials at each latitude
    Pnm = zeros(Float64, nlat, lmax + 1, mmax + 1)
    for j in 1:nlat
        μ = sin(hgrid.φ[j])  # sin(latitude) = cos(colatitude)
        for m in 0:mmax
            for l in m:lmax
                Pnm[j, l+1, m+1] = associated_legendre(l, m, μ)
            end
        end
    end

    return SpectralTransform{Float64}(T_trunc, nlon, nlat, lmax, mmax, Pnm)
end

"""
    associated_legendre(l, m, x)

Compute the normalised associated Legendre polynomial P_l^m(x)
using the standard recurrence relation. Includes the Condon-Shortley phase.
"""
function associated_legendre(l::Int, m::Int, x::Float64)
    if m > l
        return 0.0
    end

    # Compute P_m^m
    pmm = 1.0
    if m > 0
        somx2 = sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in 1:m
            pmm *= -fact * somx2   # Condon-Shortley phase
            fact += 2.0
        end
    end

    if l == m
        return pmm * _norm_factor(l, m)
    end

    # Compute P_{m+1}^m
    pmmp1 = x * (2m + 1) * pmm
    if l == m + 1
        return pmmp1 * _norm_factor(l, m)
    end

    # Recurrence for higher l
    pll = 0.0
    for ll in (m + 2):l
        pll = (x * (2ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    end

    return pll * _norm_factor(l, m)
end

"""Normalisation factor for spherical harmonics."""
function _norm_factor(l::Int, m::Int)
    num = (2l + 1) * factorial(big(l - m))
    den = 4π * factorial(big(l + m))
    return Float64(sqrt(num / den))
end

"""
    spectral_to_grid(coeffs, st::SpectralTransform)

Transform spectral coefficients to grid-point values.

`coeffs` is a vector of complex spherical harmonic coefficients packed
in triangular truncation order (same as ERA5 GRIB).
Returns a (nlon, nlat) real-valued grid.
"""
function spectral_to_grid(coeffs::AbstractVector{Complex{T}},
                          st::SpectralTransform) where T
    field = zeros(T, st.nlon, st.nlat)

    idx = 1
    for m in 0:st.mmax
        for l in m:st.lmax
            if idx > length(coeffs)
                break
            end
            c = coeffs[idx]
            idx += 1

            for j in 1:st.nlat
                plm = st.Pnm[j, l+1, m+1]
                for i in 1:st.nlon
                    # Longitude phase factor
                    λ_i = 2π * (i - 1) / st.nlon
                    if m == 0
                        field[i, j] += real(c) * plm
                    else
                        field[i, j] += 2.0 * plm * real(c * exp(im * m * λ_i))
                    end
                end
            end
        end
    end

    return field
end

"""
    grid_to_spectral(field, st::SpectralTransform)

Transform grid-point values to spectral coefficients.
`field` is (nlon, nlat). Returns a vector of complex coefficients.
"""
function grid_to_spectral(field::AbstractMatrix{T},
                          st::SpectralTransform) where T
    n_coeffs = sum(st.lmax - m + 1 for m in 0:st.mmax)
    coeffs = zeros(Complex{T}, n_coeffs)

    dlon = 2π / st.nlon
    # Simple Gaussian-like weights (trapezoidal in sin(lat))
    wt = zeros(T, st.nlat)
    for j in 2:st.nlat-1
        wt[j] = abs(sin(deg2rad(-90.0 + (j) * 180.0 / (st.nlat-1))) -
                     sin(deg2rad(-90.0 + (j-2) * 180.0 / (st.nlat-1)))) / 2
    end
    wt[1] = wt[2] / 2
    wt[end] = wt[end-1] / 2

    idx = 1
    for m in 0:st.mmax
        for l in m:st.lmax
            c = zero(Complex{T})
            for j in 1:st.nlat
                plm = st.Pnm[j, l+1, m+1]
                for i in 1:st.nlon
                    λ_i = 2π * (i - 1) / st.nlon
                    c += field[i, j] * plm * exp(-im * m * λ_i) * wt[j] * dlon
                end
            end
            coeffs[idx] = c
            idx += 1
        end
    end

    return coeffs
end
