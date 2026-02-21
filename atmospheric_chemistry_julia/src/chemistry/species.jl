# ---------------------------------------------------------------------------
# Chemical species definitions
# ---------------------------------------------------------------------------

"""
    Species(name, mw, is_advected, lifetime_days, initial_ppb)

Defines a chemical species tracked by the model.

Fields:
- `name`: Species name (Symbol)
- `mw`: Molecular weight [kg mol⁻¹]
- `is_advected`: Whether species is transported by winds
- `has_emissions`: Whether EDGAR provides emissions
- `is_deposited`: Whether species undergoes deposition
- `lifetime_days`: Approximate chemical lifetime [days] (for scaling)
- `initial_ppb`: Initial mixing ratio [ppb] for background field
- `henry_const`: Henry's law constant [M atm⁻¹] at 298K (for wet deposition)
- `v_dep`: Dry deposition velocity [m s⁻¹] (surface, reference)
"""
struct Species
    name         :: Symbol
    mw           :: Float64
    is_advected  :: Bool
    has_emissions :: Bool
    is_deposited :: Bool
    lifetime_days :: Float64
    initial_ppb  :: Float64
    henry_const  :: Float64
    v_dep        :: Float64
end

"""
    default_species()

Return the default set of chemical species for the simplified tropospheric
chemistry mechanism.
"""
function default_species()
    return [
        Species(:O3,   48e-3, true,  false, true,  25.0,  30.0,  1.1e-2, 0.005),
        Species(:NO,   30e-3, true,  true,  false,  0.01, 0.05,  1.9e-3, 0.001),
        Species(:NO2,  46e-3, true,  true,  true,   1.0,  0.5,   1.2e-2, 0.003),
        Species(:CO,   28e-3, true,  true,  false, 60.0,  80.0,  9.5e-4, 0.0),
        Species(:SO2,  64e-3, true,  true,  true,   2.0,  0.2,   1.2,    0.008),
        Species(:CH4,  16e-3, true,  true,  false, 3650., 1800., 1.4e-3, 0.0),
        Species(:HCHO, 30e-3, true,  false, true,   0.5,  0.5,   3.2e3,  0.005),
        Species(:HNO3, 63e-3, true,  false, true,   3.0,  0.1,   2.1e5,  0.030),
        Species(:H2O2, 34e-3, true,  false, true,   1.0,  0.5,   8.3e4,  0.010),
        Species(:OH,   17e-3, false, false, false,  0.001, 1e-4, 0.0,    0.0),
        Species(:HO2,  33e-3, false, false, false,  0.01,  1e-3, 0.0,    0.0),
        Species(:ISOP, 68e-3, true,  false, false,  0.05,  0.2,  0.0,    0.0),
        Species(:PAN, 121e-3, true,  false, true,   5.0,  0.05,  2.9,    0.005),
        Species(:PM25,  1e-3, true,  true,  true,   7.0,  5.0,   0.0,    0.002),
    ]
end

"""
    species_index(species_list, name)

Find the index of a species by name.
"""
function species_index(species_list::Vector{Species}, name::Symbol)
    idx = findfirst(s -> s.name == name, species_list)
    if idx === nothing
        error("Species $name not found in mechanism")
    end
    return idx
end

"""
    n_species(species_list)

Number of species.
"""
n_species(species_list::Vector{Species}) = length(species_list)
