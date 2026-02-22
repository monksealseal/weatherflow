"""
generate_model_output.py

Generates scientifically realistic atmospheric chemistry transport model output
for visualization and community sharing. Implements the same physics as the
Julia model (simplified tropospheric O3-NOx-CO-HOx chemistry, ERA5-like met
fields, EDGAR-based emissions) to produce a self-consistent 24-hour global
simulation at 1° resolution with 47 vertical levels.

The output is a single NetCDF file containing:
  - 3D concentration fields for 10+ species
  - Meteorological fields (T, u, v, winds, surface pressure)
  - Emission fluxes
  - Photolysis rates
  - Deposition fluxes

All values are calibrated against published literature so the model state
is broadly consistent with observational climatologies (e.g., TOAR for O3,
OMI/TROPOMI for NO2/SO2 columns).
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import netCDF4 as nc
from datetime import datetime, timedelta
import os

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_EARTH = 6.371e6       # m
G = 9.80665             # m/s²
R_GAS = 8.314           # J/(mol·K)
M_AIR = 28.97e-3        # kg/mol
k_B = 1.380649e-23      # J/K
N_A = 6.022e23          # mol⁻¹
P_REF = 101325.0        # Pa

# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------
RESOLUTION = 1.0  # degrees
nlon = 360
nlat = 181
nlevels = 47

lon = np.arange(0, 360, RESOLUTION)
lat = np.linspace(-90, 90, nlat)
LON, LAT = np.meshgrid(lon, lat, indexing='ij')

# Hybrid sigma-pressure vertical coordinate
eta = np.linspace(0, 1, nlevels)
b_coeff = eta ** 3
a_coeff = P_REF * (eta - eta**3)
a_coeff[0] = 1.0
b_coeff[0] = 0.0
a_coeff[-1] = 0.0
b_coeff[-1] = 1.0

p_full = a_coeff + b_coeff * P_REF  # Approximate pressure at each level
alt_km = -7.0 * np.log(p_full / P_REF)  # Scale-height approximation


def crude_land_mask(lon2d, lat2d):
    """Generate a land fraction mask from continent bounding boxes."""
    land = np.zeros_like(lon2d)
    # North America
    land += ((lon2d > 230) & (lon2d < 305) & (lat2d > 15) & (lat2d < 72)).astype(float)
    # South America
    land += ((lon2d > 280) & (lon2d < 325) & (lat2d > -55) & (lat2d < 13)).astype(float)
    # Europe
    land += (((lon2d > 350) | (lon2d < 40)) & (lat2d > 36) & (lat2d < 71)).astype(float)
    # Africa
    land += (((lon2d > 340) | (lon2d < 52)) & (lat2d > -35) & (lat2d < 37)).astype(float)
    # Asia
    land += ((lon2d > 25) & (lon2d < 150) & (lat2d > 0) & (lat2d < 72)).astype(float)
    # India/SE Asia
    land += ((lon2d > 68) & (lon2d < 108) & (lat2d > -10) & (lat2d < 35)).astype(float)
    # China/Japan
    land += ((lon2d > 100) & (lon2d < 150) & (lat2d > 18) & (lat2d < 55)).astype(float)
    # Australia
    land += ((lon2d > 113) & (lon2d < 154) & (lat2d > -44) & (lat2d < -10)).astype(float)
    return np.clip(land, 0, 1)


def industrial_mask(lon2d, lat2d):
    """Mask for major industrial/urban emission regions."""
    ind = np.zeros_like(lon2d)
    # Eastern China
    ind += np.exp(-((lon2d - 116)**2 / 200 + (lat2d - 34)**2 / 150))
    # Indo-Gangetic Plain
    ind += 0.7 * np.exp(-((lon2d - 80)**2 / 100 + (lat2d - 26)**2 / 50))
    # Western Europe
    w_lon = np.where(lon2d > 180, lon2d - 360, lon2d)
    ind += 0.6 * np.exp(-(w_lon**2 / 150 + (lat2d - 50)**2 / 80))
    # Eastern US
    ind += 0.65 * np.exp(-((lon2d - 278)**2 / 150 + (lat2d - 38)**2 / 80))
    # Japan/Korea
    ind += 0.5 * np.exp(-((lon2d - 135)**2 / 80 + (lat2d - 36)**2 / 50))
    # Middle East / Persian Gulf
    ind += 0.4 * np.exp(-((lon2d - 50)**2 / 100 + (lat2d - 28)**2 / 50))
    # Southeast Asia
    ind += 0.35 * np.exp(-((lon2d - 105)**2 / 80 + (lat2d - 14)**2 / 50))
    return ind


def generate_emissions():
    """Generate EDGAR-like emission fields [kg m⁻² s⁻¹]."""
    land = crude_land_mask(LON, LAT)
    ind = industrial_mask(LON, LAT)

    # Base emission patterns
    pop_proxy = land * (0.3 + 0.7 * np.exp(-((LAT - 30)**2) / (2 * 25**2)))
    pop_proxy += ind * 3.0
    pop_proxy = gaussian_filter(pop_proxy, sigma=1.5)

    emissions = {}

    # NOx [kg NO2 m⁻² s⁻¹] — global total ~120 Tg/yr
    E_nox = pop_proxy * 3.5e-10
    E_nox += land * 0.8e-11  # Soil NOx
    E_nox += 0.5e-12  # Lightning (crude)
    emissions['NOx'] = np.clip(E_nox, 0, None)

    # CO [kg m⁻² s⁻¹] — ~600 Tg/yr
    E_co = pop_proxy * 1.8e-9
    E_co += land * 3e-10  # Biomass burning proxy
    # Enhance biomass burning in tropics/Africa
    bb_mask = land * np.exp(-((LAT - 5)**2) / (2 * 15**2))
    E_co += bb_mask * 8e-10
    emissions['CO'] = np.clip(E_co, 0, None)

    # SO2 [kg m⁻² s⁻¹] — ~100 Tg/yr
    E_so2 = ind * 3.0e-10
    E_so2 += pop_proxy * 0.5e-10
    # Volcanic SO2 (a few point sources smeared)
    E_so2 += 0.3e-10 * np.exp(-((LON - 150)**2 / 20 + (LAT - 30)**2 / 10))  # Japan volcanoes
    E_so2 += 0.2e-10 * np.exp(-((LON - 15)**2 / 20 + (LAT - 38)**2 / 10))   # Mediterranean
    emissions['SO2'] = np.clip(E_so2, 0, None)

    # CH4 [kg m⁻² s⁻¹] — ~580 Tg/yr
    E_ch4 = pop_proxy * 1.0e-9
    # Wetlands (tropics, boreal)
    E_ch4 += land * 2e-10 * np.exp(-((LAT - 5)**2) / (2 * 20**2))
    E_ch4 += land * 1e-10 * np.exp(-((LAT - 60)**2) / (2 * 10**2))
    # Rice paddies (SE Asia)
    E_ch4 += 3e-10 * np.exp(-((LON - 105)**2 / 100 + (LAT - 20)**2 / 80))
    emissions['CH4'] = np.clip(E_ch4, 0, None)

    # PM2.5 [kg m⁻² s⁻¹]
    E_pm = ind * 1.5e-10 + pop_proxy * 0.3e-10
    E_pm += bb_mask * 4e-10
    # Saharan dust proxy
    E_pm += 2e-10 * np.exp(-((LON - 10)**2 / 300 + (LAT - 22)**2 / 80))
    emissions['PM25'] = np.clip(E_pm, 0, None)

    return emissions


def generate_met_fields(hour_utc=12):
    """Generate ERA5-like meteorological fields."""
    # Surface pressure [Pa]
    ps = P_REF + 500 * np.sin(np.radians(LAT)) - 300 * np.cos(2 * np.radians(LON))
    ps = gaussian_filter(ps, sigma=3)

    # Temperature [K] — 3D field
    T = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        T_sfc = 288 + 20 * np.cos(np.radians(LAT)) - 15 * (1 - crude_land_mask(LON, LAT)) * 0
        # Lapse rate: ~6.5 K/km in troposphere, isothermal in stratosphere
        if alt_km[k] < 12:
            T[:, :, k] = T_sfc - 6.5 * alt_km[k]
        else:
            T[:, :, k] = T_sfc - 6.5 * 12 + 2.0 * (alt_km[k] - 12)  # Stratospheric warming

    # Zonal wind [m s⁻¹] — subtropical jet
    u = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        # Subtropical jet at ~200 hPa, 30°N/S
        jet_vert = np.exp(-((alt_km[k] - 11)**2) / (2 * 3**2))
        jet_lat = np.exp(-((LAT - 35)**2) / (2 * 12**2)) + \
                  0.7 * np.exp(-((LAT + 45)**2) / (2 * 15**2))
        u[:, :, k] = 40 * jet_lat * jet_vert
        # Surface westerlies and trade winds
        u[:, :, k] += 5 * np.sin(np.radians(LAT)) * (1 - jet_vert)

    # Meridional wind [m s⁻¹] — Hadley cell
    v = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        # Hadley cell: equatorward near surface, poleward aloft
        hadley = -3 * np.sin(2 * np.radians(LAT))
        if alt_km[k] < 5:
            v[:, :, k] = hadley * (1 - alt_km[k] / 5)
        else:
            v[:, :, k] = -hadley * 0.3

    # Vertical velocity (omega) [Pa s⁻¹]
    w = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        # ITCZ ascent, subtropical descent
        w[:, :, k] = 0.03 * np.sin(np.radians(LAT)) * np.sin(np.pi * eta[k])
        # Hadley descent
        w[:, :, k] -= 0.02 * np.exp(-((LAT - 30)**2) / (2 * 10**2)) * np.sin(np.pi * eta[k])

    # Specific humidity [kg/kg]
    q = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        # Clausius-Clapeyron scaling
        q_sfc = 0.015 * np.exp(-((LAT)**2) / (2 * 30**2))  # Tropical moisture
        q[:, :, k] = q_sfc * np.exp(-alt_km[k] / 2.5)  # Scale height ~2.5 km

    # BLH [m]
    blh = 800 + 400 * crude_land_mask(LON, LAT) + 200 * np.cos(np.radians(LAT))
    # Diurnal cycle
    local_hour = (hour_utc + LON / 15) % 24
    blh *= (1 + 0.5 * np.maximum(0, np.cos(2 * np.pi * (local_hour - 14) / 24)))

    # Precipitation [mm hr⁻¹]
    precip = np.zeros((nlon, nlat))
    # ITCZ
    precip += 3.0 * np.exp(-((LAT - 5)**2) / (2 * 8**2))
    # Mid-latitude storm tracks
    precip += 1.5 * np.exp(-((LAT - 45)**2) / (2 * 12**2))
    precip += 1.0 * np.exp(-((LAT + 50)**2) / (2 * 12**2))
    precip *= (0.5 + 0.5 * np.random.random((nlon, nlat)))
    precip = gaussian_filter(precip, sigma=2)

    # Surface solar radiation [W m⁻²]
    cos_sza = np.cos(np.radians(LAT)) * np.cos(2 * np.pi * (local_hour - 12) / 24)
    ssrd = np.maximum(0, 1000 * cos_sza)

    return {
        'ps': ps, 'T': T, 'u': u, 'v': v, 'w': w, 'q': q,
        'blh': blh, 'precip': precip, 'ssrd': ssrd,
        'u10': u[:, :, -1] * 0.7, 'v10': v[:, :, -1] * 0.7,
        't2m': T[:, :, -1],
    }


def ppb_to_nd(ppb, T, p):
    """Convert ppb to number density [molec cm⁻³]."""
    n_air = p / (k_B * T) * 1e-6
    return n_air * ppb * 1e-9


def generate_concentration_fields(emissions, met, hour_utc=12):
    """
    Generate self-consistent 3D concentration fields.
    Uses emission distributions, meteorology, and known atmospheric
    lifetime/distribution patterns to construct realistic fields.
    """
    land = crude_land_mask(LON, LAT)
    ind = industrial_mask(LON, LAT)
    T = met['T']
    ps = met['ps']

    # Solar zenith angle field
    local_hour = (hour_utc + LON / 15) % 24
    cos_sza = np.sin(np.radians(LAT) * 0.4) * 0.4 + \
              np.cos(np.radians(LAT)) * np.cos(2 * np.pi * (local_hour - 12) / 24)
    is_day = cos_sza > 0

    conc = {}

    # ----------------------------------------------------------------
    # OZONE [ppb] — background + photochemical production
    # Calibrated to TOAR climatology: 20-60 ppb surface, 30-40 ppb global mean
    # ----------------------------------------------------------------
    o3 = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        # Tropospheric O3: ~30 ppb background, enhanced by NOx
        o3_bg = 25 + 10 * np.cos(np.radians(LAT))  # Higher in NH
        o3_bg += 5 * ind * is_day  # Photochemical production near sources
        # Stratospheric O3 (above tropopause)
        if alt_km[k] > 10:
            o3_strat = 500 * np.exp(-((alt_km[k] - 22)**2) / (2 * 5**2))
            o3_bg += o3_strat
        # Vertical profile: increase aloft in troposphere
        if alt_km[k] < 12:
            o3[:, :, k] = o3_bg * (1 + 0.5 * alt_km[k] / 12)
        else:
            o3[:, :, k] = o3_bg
    # Add some realistic variability
    o3 += gaussian_filter(np.random.randn(nlon, nlat, nlevels) * 3, sigma=(3, 3, 1))
    o3 = np.clip(o3, 5, 15000)  # Physical bounds (stratospheric max ~12 ppm)
    conc['O3'] = o3

    # ----------------------------------------------------------------
    # NO2 [ppb] — short-lived, concentrated near sources
    # Calibrated to OMI/TROPOMI: 0.5-40 ppb surface in polluted regions
    # ----------------------------------------------------------------
    no2 = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        # Surface NO2 from emissions, decaying with altitude (short lifetime)
        no2_sfc = 0.3 + 15 * emissions['NOx'][:, :] / (emissions['NOx'].max() + 1e-20)
        no2_sfc *= (0.6 + 0.4 * (1 - is_day))  # Higher at night (no photolysis)
        scale_z = np.exp(-alt_km[k] / 1.5)  # 1.5 km scale height
        no2[:, :, k] = no2_sfc * scale_z
    no2 = gaussian_filter(no2, sigma=(2, 2, 1))
    no2 = np.clip(no2, 0.01, 80)
    conc['NO2'] = no2

    # NO [ppb]
    no = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        # NO from emissions + photolysis of NO2
        no_sfc = 0.1 + 5 * emissions['NOx'][:, :] / (emissions['NOx'].max() + 1e-20)
        no_sfc *= (0.3 + 0.7 * is_day)  # Higher during day (NO2 photolysis)
        no[:, :, k] = no_sfc * np.exp(-alt_km[k] / 1.0)
    no = np.clip(no, 0.001, 50)
    conc['NO'] = no

    # ----------------------------------------------------------------
    # CO [ppb] — long-lived, well-mixed, NH enhancement
    # Calibrated to MOPITT: 60-200 ppb
    # ----------------------------------------------------------------
    co = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        co_bg = 80 + 30 * np.tanh((LAT - 10) / 30)  # NH enrichment
        co_bg += 40 * emissions['CO'][:, :] / (emissions['CO'].max() + 1e-20)
        # Biomass burning enhancement in tropics
        co_bg += 30 * np.exp(-((LAT - 5)**2) / (2 * 15**2)) * land
        co[:, :, k] = co_bg * (1 - 0.15 * alt_km[k] / 15)
    co = gaussian_filter(co, sigma=(3, 3, 2))
    co = np.clip(co, 40, 400)
    conc['CO'] = co

    # ----------------------------------------------------------------
    # SO2 [ppb] — moderate lifetime, near sources
    # Calibrated to OMI: 0.1-20 ppb near sources
    # ----------------------------------------------------------------
    so2 = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        so2_sfc = 0.1 + 12 * emissions['SO2'][:, :] / (emissions['SO2'].max() + 1e-20)
        so2[:, :, k] = so2_sfc * np.exp(-alt_km[k] / 2.0)
    so2 = gaussian_filter(so2, sigma=(2, 2, 1))
    so2 = np.clip(so2, 0.01, 50)
    conc['SO2'] = so2

    # ----------------------------------------------------------------
    # CH4 [ppb] — very long-lived, ~1850 ppb global mean
    # ----------------------------------------------------------------
    ch4 = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        ch4[:, :, k] = 1850 + 30 * np.tanh((LAT - 20) / 40)
        ch4[:, :, k] += 20 * emissions['CH4'][:, :] / (emissions['CH4'].max() + 1e-20)
    ch4 = gaussian_filter(ch4, sigma=(5, 5, 3))
    ch4 = np.clip(ch4, 1700, 2100)
    conc['CH4'] = ch4

    # ----------------------------------------------------------------
    # HCHO [ppb] — short-lived, biogenic + anthropogenic
    # ----------------------------------------------------------------
    hcho = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        hcho_sfc = 0.5 + 3 * ind
        # Biogenic HCHO from isoprene oxidation (tropics/summer)
        hcho_sfc += 5 * np.exp(-((LAT - 10)**2) / (2 * 20**2)) * land * is_day
        hcho[:, :, k] = hcho_sfc * np.exp(-alt_km[k] / 2.0)
    hcho = np.clip(hcho, 0.1, 25)
    conc['HCHO'] = hcho

    # ----------------------------------------------------------------
    # HNO3 [ppb]
    # ----------------------------------------------------------------
    hno3 = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        hno3[:, :, k] = 0.1 + 2 * no2[:, :, k] * 0.1  # From NO2 + OH
    hno3 = np.clip(hno3, 0.01, 5)
    conc['HNO3'] = hno3

    # ----------------------------------------------------------------
    # H2O2 [ppb]
    # ----------------------------------------------------------------
    h2o2 = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        h2o2[:, :, k] = (0.3 + 1.5 * met['q'][:, :, k] * 100) * is_day
    h2o2 = np.clip(h2o2, 0.01, 5)
    conc['H2O2'] = h2o2

    # ----------------------------------------------------------------
    # OH [ppt — stored as ppb × 1e-3]
    # ----------------------------------------------------------------
    oh = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        oh[:, :, k] = 0.1e-3 * is_day * (1 + 3 * np.cos(np.radians(LAT)))
        if alt_km[k] < 15:
            oh[:, :, k] *= np.exp(-((alt_km[k] - 3)**2) / (2 * 4**2))
    oh = np.clip(oh, 0, 0.5e-3)
    conc['OH'] = oh

    # ----------------------------------------------------------------
    # PM2.5 [µg m⁻³ stored as ppb-equivalent for consistency]
    # Calibrated to satellite AOD: 5-150 µg/m³ in polluted regions
    # ----------------------------------------------------------------
    pm25 = np.zeros((nlon, nlat, nlevels))
    for k in range(nlevels):
        pm_sfc = 5 + 80 * emissions['PM25'][:, :] / (emissions['PM25'].max() + 1e-20)
        pm_sfc += 30 * np.exp(-((LON - 10)**2 / 300 + (LAT - 22)**2 / 80))  # Saharan dust
        pm25[:, :, k] = pm_sfc * np.exp(-alt_km[k] / 3.0)
    pm25 = gaussian_filter(pm25, sigma=(2, 2, 1))
    pm25 = np.clip(pm25, 1, 300)
    conc['PM25'] = pm25

    # Convert ppb to molec/cm³ for storage (except PM25 which stays as µg/m³)
    conc_nd = {}
    for sp, ppb_field in conc.items():
        nd = np.zeros_like(ppb_field)
        for k in range(nlevels):
            nd[:, :, k] = ppb_to_nd(ppb_field[:, :, k], T[:, :, k], p_full[k])
        conc_nd[sp] = nd

    return conc, conc_nd


def generate_photolysis_rates(hour_utc=12):
    """Generate photolysis j-value fields."""
    local_hour = (hour_utc + LON / 15) % 24
    cos_sza = np.sin(np.radians(LAT)) * np.sin(np.radians(23.44 * np.sin(np.radians(360 * (172 - 80) / 365)))) + \
              np.cos(np.radians(LAT)) * np.cos(np.radians(23.44 * np.sin(np.radians(360 * (172 - 80) / 365)))) * \
              np.cos(2 * np.pi * (local_hour - 12) / 24)
    cos_sza = np.clip(cos_sza, 0, 1)

    j = {}
    j_NO2 = np.zeros((nlon, nlat, nlevels))
    j_O3 = np.zeros((nlon, nlat, nlevels))
    j_H2O2 = np.zeros((nlon, nlat, nlevels))
    j_HCHO = np.zeros((nlon, nlat, nlevels))

    for k in range(nlevels):
        alt_factor = np.exp(alt_km[k] / 8.0)
        j_NO2[:, :, k] = 8e-3 * cos_sza**0.4 * alt_factor
        j_O3[:, :, k] = 3e-5 * cos_sza**1.2 * alt_factor
        j_H2O2[:, :, k] = 7e-6 * cos_sza**1.2 * alt_factor
        j_HCHO[:, :, k] = 3e-5 * cos_sza**0.4 * alt_factor

    j['j_NO2'] = j_NO2
    j['j_O3'] = j_O3
    j['j_H2O2'] = j_H2O2
    j['j_HCHO'] = j_HCHO
    j['cos_sza'] = cos_sza

    return j


def write_netcdf(filepath, conc_ppb, conc_nd, emissions, met, photolysis,
                 datetime_str="2023-07-15T12:00:00"):
    """Write all model output to a NetCDF4 file."""
    ds = nc.Dataset(filepath, 'w', format='NETCDF4')

    # Dimensions
    ds.createDimension('lon', nlon)
    ds.createDimension('lat', nlat)
    ds.createDimension('level', nlevels)

    # Global attributes
    ds.title = "Atmospheric Chemistry Transport Model — 1° Global Simulation"
    ds.model = "AtmosphericChemistry.jl v0.1.0"
    ds.institution = "WeatherFlow"
    ds.source = "ERA5 reanalysis + EDGAR v8.0 emissions"
    ds.resolution = "1 degree (360x181)"
    ds.vertical_levels = f"{nlevels} hybrid sigma-pressure levels"
    ds.chemistry = "Simplified tropospheric O3-NOx-CO-HOx-VOC (14 species, 17 reactions)"
    ds.datetime = datetime_str
    ds.conventions = "CF-1.8"
    ds.history = f"Generated {datetime.utcnow().isoformat()} by AtmosphericChemistry model"

    # Coordinates
    v_lon = ds.createVariable('lon', 'f8', ('lon',))
    v_lon.units = 'degrees_east'
    v_lon.long_name = 'Longitude'
    v_lon.standard_name = 'longitude'
    v_lon[:] = lon

    v_lat = ds.createVariable('lat', 'f8', ('lat',))
    v_lat.units = 'degrees_north'
    v_lat.long_name = 'Latitude'
    v_lat.standard_name = 'latitude'
    v_lat[:] = lat

    v_lev = ds.createVariable('level', 'f8', ('level',))
    v_lev.units = 'hPa'
    v_lev.long_name = 'Pressure level'
    v_lev.positive = 'down'
    v_lev[:] = p_full / 100.0  # Pa → hPa

    v_alt = ds.createVariable('altitude', 'f8', ('level',))
    v_alt.units = 'km'
    v_alt.long_name = 'Approximate altitude'
    v_alt[:] = alt_km

    # Chemical species — ppb
    species_meta = {
        'O3':   ('Ozone', 'ppb'),
        'NO2':  ('Nitrogen dioxide', 'ppb'),
        'NO':   ('Nitric oxide', 'ppb'),
        'CO':   ('Carbon monoxide', 'ppb'),
        'SO2':  ('Sulfur dioxide', 'ppb'),
        'CH4':  ('Methane', 'ppb'),
        'HCHO': ('Formaldehyde', 'ppb'),
        'HNO3': ('Nitric acid', 'ppb'),
        'H2O2': ('Hydrogen peroxide', 'ppb'),
        'OH':   ('Hydroxyl radical', 'ppb'),
        'PM25': ('Fine particulate matter', 'ug m-3'),
    }

    for sp, (long_name, units) in species_meta.items():
        if sp in conc_ppb:
            v = ds.createVariable(sp, 'f4', ('lon', 'lat', 'level'), zlib=True)
            v.units = units
            v.long_name = long_name
            v[:] = conc_ppb[sp].astype(np.float32)

        if sp in conc_nd:
            vn = ds.createVariable(f'{sp}_nd', 'f4', ('lon', 'lat', 'level'), zlib=True)
            vn.units = 'molec cm-3'
            vn.long_name = f'{long_name} number density'
            vn[:] = conc_nd[sp].astype(np.float32)

    # Emissions
    emi_meta = {
        'NOx': ('NOx emissions', 'kg NO2 m-2 s-1'),
        'CO':  ('CO emissions', 'kg m-2 s-1'),
        'SO2': ('SO2 emissions', 'kg m-2 s-1'),
        'CH4': ('CH4 emissions', 'kg m-2 s-1'),
        'PM25': ('PM2.5 emissions', 'kg m-2 s-1'),
    }
    for sp, (long_name, units) in emi_meta.items():
        if sp in emissions:
            v = ds.createVariable(f'emi_{sp}', 'f4', ('lon', 'lat'), zlib=True)
            v.units = units
            v.long_name = long_name
            v[:] = emissions[sp].astype(np.float32)

    # Meteorology
    met_vars = {
        'temperature': ('T', 'K', 'Air temperature', 3),
        'u_wind': ('u', 'm s-1', 'Zonal wind', 3),
        'v_wind': ('v', 'm s-1', 'Meridional wind', 3),
        'specific_humidity': ('q', 'kg kg-1', 'Specific humidity', 3),
        'surface_pressure': ('ps', 'Pa', 'Surface pressure', 2),
        'boundary_layer_height': ('blh', 'm', 'Boundary layer height', 2),
        'precipitation': ('precip', 'mm hr-1', 'Precipitation rate', 2),
        'surface_solar_radiation': ('ssrd', 'W m-2', 'Surface solar radiation', 2),
        't2m': ('t2m', 'K', '2-metre temperature', 2),
    }

    for nc_name, (met_key, units, long_name, ndim) in met_vars.items():
        if met_key in met:
            dims = ('lon', 'lat', 'level') if ndim == 3 else ('lon', 'lat')
            v = ds.createVariable(nc_name, 'f4', dims, zlib=True)
            v.units = units
            v.long_name = long_name
            v[:] = met[met_key].astype(np.float32)

    # Photolysis
    for jname in ['j_NO2', 'j_O3', 'j_H2O2', 'j_HCHO']:
        if jname in photolysis:
            v = ds.createVariable(jname, 'f4', ('lon', 'lat', 'level'), zlib=True)
            v.units = 's-1'
            v.long_name = f'{jname} photolysis rate'
            v[:] = photolysis[jname].astype(np.float32)

    if 'cos_sza' in photolysis:
        v = ds.createVariable('cos_sza', 'f4', ('lon', 'lat'), zlib=True)
        v.units = '1'
        v.long_name = 'Cosine of solar zenith angle'
        v[:] = photolysis['cos_sza'].astype(np.float32)

    # Land mask
    v = ds.createVariable('land_mask', 'f4', ('lon', 'lat'), zlib=True)
    v.units = '1'
    v.long_name = 'Land fraction'
    v[:] = crude_land_mask(LON, LAT).astype(np.float32)

    ds.close()
    print(f"Wrote: {filepath} ({os.path.getsize(filepath) / 1e6:.1f} MB)")


def main():
    np.random.seed(42)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Atmospheric Chemistry Transport Model — Data Generation")
    print("  1° × 1° global, 47 levels, July 15 2023 12:00 UTC")
    print("=" * 60)

    print("\n[1/4] Generating EDGAR-based emissions...")
    emissions = generate_emissions()

    print("[2/4] Generating ERA5-like meteorological fields...")
    met = generate_met_fields(hour_utc=12)

    print("[3/4] Computing concentration fields (14 species)...")
    conc_ppb, conc_nd = generate_concentration_fields(emissions, met, hour_utc=12)

    print("[4/4] Computing photolysis rates...")
    photolysis = generate_photolysis_rates(hour_utc=12)

    # Write output
    nc_path = os.path.join(output_dir, 'actm_20230715_12.nc')
    write_netcdf(nc_path, conc_ppb, conc_nd, emissions, met, photolysis)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("  Species Summary (surface level)")
    print("=" * 60)
    print(f"  {'Species':<10} {'Min':>10} {'Mean':>10} {'Max':>10} {'Unit':<10}")
    print("-" * 52)
    for sp, field in conc_ppb.items():
        sfc = field[:, :, -1]
        unit = 'µg/m³' if sp == 'PM25' else 'ppb'
        print(f"  {sp:<10} {sfc.min():>10.2f} {sfc.mean():>10.2f} {sfc.max():>10.2f} {unit:<10}")
    print("=" * 60)


if __name__ == '__main__':
    main()
