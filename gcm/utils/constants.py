"""
Physical constants for atmospheric modeling

All values in SI units unless otherwise noted
"""


class PhysicalConstants:
    """Container for physical constants"""

    # Fundamental constants
    g = 9.81  # Gravitational acceleration (m/s^2)
    R_earth = 6.371e6  # Earth radius (m)
    omega = 7.292e-5  # Earth rotation rate (rad/s)

    # Thermodynamic constants
    Rd = 287.0  # Gas constant for dry air (J/kg/K)
    Rv = 461.5  # Gas constant for water vapor (J/kg/K)
    cp = 1004.0  # Specific heat at constant pressure (J/kg/K)
    cv = 717.0  # Specific heat at constant volume (J/kg/K)
    kappa = Rd / cp  # Poisson constant

    # Phase change
    Lv = 2.5e6  # Latent heat of vaporization (J/kg)
    Ls = 2.834e6  # Latent heat of sublimation (J/kg)
    Lf = Ls - Lv  # Latent heat of fusion (J/kg)

    # Reference values
    p0 = 100000.0  # Reference pressure (Pa)
    T0 = 273.15  # Freezing point (K)

    # Radiation
    sigma_sb = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
    solar_constant = 1361.0  # Solar constant (W/m^2)

    # Water
    rho_water = 1000.0  # Density of liquid water (kg/m^3)
    rho_ice = 917.0  # Density of ice (kg/m^3)

    # Atmospheric composition
    epsilon = Rd / Rv  # Ratio of gas constants
    co2_ppmv_default = 400.0  # Default CO2 concentration
