# ---------------------------------------------------------------------------
# Physical and mathematical constants used throughout the model
# ---------------------------------------------------------------------------

"""Physical constants for atmospheric chemistry and transport."""
module Constants

# Fundamental
const R_GAS       = 8.314462618        # Universal gas constant [J mol⁻¹ K⁻¹]
const k_B         = 1.380649e-23       # Boltzmann constant [J K⁻¹]
const N_A         = 6.02214076e23      # Avogadro number [mol⁻¹]
const STEFAN_BOLTZ = 5.670374419e-8    # Stefan-Boltzmann constant [W m⁻² K⁻⁴]

# Earth
const R_EARTH     = 6.371e6            # Mean radius [m]
const OMEGA       = 7.2921e-5          # Angular velocity [rad s⁻¹]
const G           = 9.80665            # Gravitational acceleration [m s⁻²]

# Atmosphere
const M_AIR       = 28.97e-3           # Mean molar mass of dry air [kg mol⁻¹]
const M_H2O       = 18.015e-3          # Molar mass of water [kg mol⁻¹]
const Cp_AIR      = 1004.0             # Specific heat at constant pressure [J kg⁻¹ K⁻¹]
const Cv_AIR      = 717.0              # Specific heat at constant volume [J kg⁻¹ K⁻¹]
const P_REF       = 101325.0           # Reference surface pressure [Pa]
const T_REF       = 288.15             # Reference temperature [K]
const KAPPA       = R_GAS / (M_AIR * Cp_AIR)  # Poisson constant ≈ 0.2854

# Dry deposition reference
const VON_KARMAN  = 0.4                # Von Kármán constant

# Molecular weights of key species [kg mol⁻¹]
const MW = Dict{Symbol, Float64}(
    :O3   => 48.0e-3,
    :NO   => 30.0e-3,
    :NO2  => 46.0e-3,
    :CO   => 28.0e-3,
    :SO2  => 64.0e-3,
    :CH4  => 16.0e-3,
    :HCHO => 30.0e-3,
    :HNO3 => 63.0e-3,
    :H2O2 => 34.0e-3,
    :OH   => 17.0e-3,
    :HO2  => 33.0e-3,
    :ISOP => 68.0e-3,   # isoprene
    :PAN  => 121.0e-3,  # peroxyacetyl nitrate
    :PM25 => 1.0e-3,    # dummy for particulate matter
)

end # module Constants
