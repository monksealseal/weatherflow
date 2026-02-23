"""
Cassandra J. Gaston Research Database

Embedded publication data, measurement parameters, and synthetic datasets
based on published findings. All data is included directly to avoid external
downloads, keeping memory usage minimal for the free Streamlit tier.

References:
    Gaston, C.J. et al. (2010-2026) - See PUBLICATIONS dict for full list.
    Barbados Atmospheric Chemistry Observatory (BACO), Ragged Point, Barbados.
"""

import numpy as np


# =============================================================================
# Publication Database
# =============================================================================

PUBLICATIONS = [
    {
        "year": 2026,
        "authors": "Shrestha, S., Holz, R.E., ..., Ault, A.P., Gaston, C.J.",
        "title": "Transported African Dust in the Lower Marine Atmospheric Boundary Layer is Internally Mixed with Sea Salt",
        "journal": "Atmospheric Chemistry and Physics",
        "doi": "10.5194/acp-26-983-2026",
        "theme": "dust_transport",
        "key_finding": "67% of dust particles internally mixed with sea salt in the MBL, suppressing lidar depolarization.",
    },
    {
        "year": 2025,
        "authors": "Royer, H.M., Sheridan, M.T., ..., Gaston, C.J.",
        "title": "African dust transported to Barbados in the wintertime lacks indicators of chemical aging",
        "journal": "Atmospheric Chemistry and Physics",
        "doi": "10.5194/acp-25-5743-2025",
        "theme": "dust_transport",
        "key_finding": "Despite bulk nitrate/sulfate increases, individual dust particles show limited acid-processing coatings.",
    },
    {
        "year": 2025,
        "authors": "Elliott, H.E., et al., Gaston, C.J.",
        "title": "Composition and Plume Gas Interaction Control Iron Fractional Solubility in Volcanic Ash",
        "journal": "Global Biogeochemical Cycles",
        "doi": "10.1029/2025GB008560",
        "theme": "nutrients",
        "key_finding": "Less than 6% of total Fe in volcanic ash is soluble; plume gas interaction dominates over particle size.",
    },
    {
        "year": 2025,
        "authors": "Christie, J.A., Elliott, H.E., ..., Gaston, C.J.",
        "title": "Halogen Production from Playa Dust Emitted from the Great Salt Lake",
        "journal": "ACS Earth and Space Chemistry",
        "doi": "10.1021/acsearthspacechem.4c00258",
        "theme": "heterogeneous_chemistry",
        "key_finding": "Great Salt Lake playa dust produces ClNO2, Cl2, and BrCl with implications for regional ozone.",
    },
    {
        "year": 2024,
        "authors": "Gaston, C.J., Prospero, J.M., Foley, K., et al.",
        "title": "Diverging trends in aerosol sulfate and nitrate at Barbados attributed to clean air policies and African smoke",
        "journal": "Atmospheric Chemistry and Physics",
        "doi": "10.5194/acp-24-8049-2024",
        "theme": "barbados_trends",
        "key_finding": "21 years of data: nss-sulfate declined (Clean Air Act), nitrate diverged (African biomass burning).",
    },
    {
        "year": 2024,
        "authors": "Lu, L., Li, L., ..., Gaston, C., et al.",
        "title": "Characterizing the atmospheric Mn cycle and its impacts on terrestrial biogeochemistry",
        "journal": "Global Biogeochemical Cycles",
        "doi": "10.1029/2023GB007967",
        "theme": "nutrients",
        "key_finding": "Human activity contributes ~1/3 of global atmospheric Mn; atmospheric deposition shortens soil Mn turnover by 1-2 orders of magnitude.",
    },
    {
        "year": 2024,
        "authors": "Elliott, H.E.G., Popendorf, K.J., ..., Gaston, C.J.",
        "title": "Godzilla mineral dust and La Soufriere volcanic ash immediately stimulate marine microbial phosphate uptake",
        "journal": "Frontiers in Marine Science",
        "doi": "",
        "theme": "nutrients",
        "key_finding": "The 2020 Godzilla dust event and 2021 La Soufriere eruption delivered bioavailable P stimulating microbial uptake.",
    },
    {
        "year": 2023,
        "authors": "Gaston, C.J., et al.",
        "title": "African Smoke Particles Act as Cloud Condensation Nuclei in the Wintertime Tropical North Atlantic",
        "journal": "Atmospheric Chemistry and Physics",
        "doi": "10.5194/acp-23-981-2023",
        "theme": "ccn_clouds",
        "key_finding": "Biomass burning particles from Africa act as effective CCN influencing cloud properties over the Caribbean.",
    },
    {
        "year": 2022,
        "authors": "Barkley, A.E., Pourmand, A., ..., Gaston, C.J.",
        "title": "Interannual Variability in the Source Location of North African Dust Transported to the Amazon",
        "journal": "Geophysical Research Letters",
        "doi": "10.1029/2021GL097344",
        "theme": "dust_transport",
        "key_finding": "Source regions of African dust reaching the Amazon vary interannually; Bodele not the sole dominant source.",
    },
    {
        "year": 2021,
        "authors": "Royer, H.M., Mitroo, D., ..., Gaston, C.J.",
        "title": "The Role of Hydrates, Competing Chemical Constituents, and Surface Composition on ClNO2 Formation",
        "journal": "Environmental Science & Technology",
        "doi": "10.1021/acs.est.0c06067",
        "theme": "heterogeneous_chemistry",
        "key_finding": "Surface composition, not bulk, controls ClNO2 formation; hydrated MgCl2/CaCl2 facilitate production; organics hinder uptake.",
    },
    {
        "year": 2021,
        "authors": "Barkley, A.E., Olson, N.E., ..., Gaston, C.J.",
        "title": "Atmospheric Transport of North African Dust-Bearing Supermicron Freshwater Diatoms to South America",
        "journal": "Geophysical Research Letters",
        "doi": "10.1029/2020GL090476",
        "theme": "dust_transport",
        "key_finding": "Freshwater diatoms contain ~4% Fe, comprise 38% of 10-18um particles, a new vector for Fe transport.",
    },
    {
        "year": 2020,
        "authors": "Gaston, C.J.",
        "title": "Re-examining Dust Chemical Aging and Its Impacts on Earth's Climate",
        "journal": "Accounts of Chemical Research",
        "doi": "10.1021/acs.accounts.0c00102",
        "theme": "dust_transport",
        "key_finding": "Review: dust aging impacts on CCN/INP activity, nutrient solubility, and radiative forcing remain poorly constrained.",
    },
    {
        "year": 2020,
        "authors": "Prospero, J.M., Barkley, A.E., Gaston, C.J., et al.",
        "title": "Characterizing and Quantifying African Dust Transport and Deposition to South America",
        "journal": "Global Biogeochemical Cycles",
        "doi": "10.1029/2020GB006536",
        "theme": "nutrients",
        "key_finding": "15 years of daily measurements from French Guiana quantify African dust P deposition to the Amazon.",
    },
    {
        "year": 2019,
        "authors": "Barkley, A.E., Prospero, J.M., ..., Gaston, C.J.",
        "title": "African biomass burning is a substantial source of phosphorus deposition to the Amazon",
        "journal": "Proceedings of the National Academy of Sciences",
        "doi": "10.1073/pnas.1906091116",
        "theme": "nutrients",
        "key_finding": "African biomass burning supplies up to 50% of P deposited to the Amazon; smoke P is 2x more soluble than dust P.",
    },
    {
        "year": 2019,
        "authors": "Mitroo, D., Gill, T.E., ..., Gaston, C.J.",
        "title": "ClNO2 Production from N2O5 Uptake on Saline Playa Dusts",
        "journal": "Environmental Science & Technology",
        "doi": "10.1021/acs.est.9b01112",
        "theme": "heterogeneous_chemistry",
        "key_finding": "First lab measurements of ClNO2 from playa dusts; gamma(N2O5) = 1e-3 to 0.1; ClNO2 yields >50%.",
    },
    {
        "year": 2018,
        "authors": "Gaston, C.J., Cahill, J.F., ..., Prather, K.A.",
        "title": "The Cloud Nucleating Properties and Mixing State of Marine Aerosols along Southern California",
        "journal": "Atmosphere",
        "doi": "10.3390/atmos9020052",
        "theme": "ccn_clouds",
        "key_finding": "Hygroscopicity kappa ranged 0.1-1.4 (mean 0.22); smaller particles less hygroscopic than larger ones.",
    },
    {
        "year": 2016,
        "authors": "Gaston, C.J. and Thornton, J.A.",
        "title": "Reacto-Diffusive Length of N2O5 in Aqueous Sulfate- and Chloride-Containing Aerosol Particles",
        "journal": "Journal of Physical Chemistry A",
        "doi": "10.1021/acs.jpca.5b11914",
        "theme": "heterogeneous_chemistry",
        "key_finding": "gamma(N2O5) on NaCl is size-independent (surface reaction); on sulfate it is size-dependent (bulk reaction).",
    },
    {
        "year": 2015,
        "authors": "Gaston, C.J., Furutani, H., ..., Prather, K.A.",
        "title": "Direct Night-Time Ejection of Particle-Phase Reduced Biogenic Sulfur Compounds from the Ocean",
        "journal": "Environmental Science & Technology",
        "doi": "10.1021/es505590y",
        "theme": "marine_aerosol",
        "key_finding": "Novel reduced-sulfur sea spray particles detected globally, suggesting a new biogenic sulfur pathway.",
    },
    {
        "year": 2014,
        "authors": "Gaston, C.J., Thornton, J.A., Ng, N.L.",
        "title": "Reactive uptake of N2O5 to internally mixed inorganic and organic particles",
        "journal": "Atmospheric Chemistry and Physics",
        "doi": "10.5194/acp-14-5693-2014",
        "theme": "heterogeneous_chemistry",
        "key_finding": "Organic coatings suppress N2O5 uptake; O:C ratio and phase separations control suppression extent.",
    },
    {
        "year": 2014,
        "authors": "Gaston, C.J., Riedel, T.P., ..., Thornton, J.A.",
        "title": "Reactive Uptake of an Isoprene-Derived Epoxydiol to Submicron Aerosol Particles",
        "journal": "Environmental Science & Technology",
        "doi": "10.1021/es5034266",
        "theme": "soa_formation",
        "key_finding": "gamma(IEPOX) on NH4HSO4 ~0.05, on (NH4)2SO4 <=1e-4; Henry's law coeff = 1.7e8 M/atm.",
    },
    {
        "year": 2010,
        "authors": "Gaston, C.J., Pratt, K.A., Qin, X.Y., Prather, K.A.",
        "title": "Real-Time Detection and Mixing State of Methanesulfonate in Single Particles",
        "journal": "Environmental Science & Technology",
        "doi": "10.1021/es902069d",
        "theme": "marine_aerosol",
        "key_finding": "MSA detected in single particles at inland locations, internally mixed with aged sea salt.",
    },
]

RESEARCH_THEMES = {
    "dust_transport": {
        "name": "African Dust Transport & Chemical Aging",
        "color": "#D4A574",
        "icon": "desert",
        "description": "Transatlantic Saharan dust transport to the Caribbean and Amazon, chemical aging during transport, mixing with sea salt, and impacts on air quality and climate.",
        "key_species": ["Mineral dust", "Sea salt", "HNO3", "H2SO4", "CaCO3"],
    },
    "heterogeneous_chemistry": {
        "name": "Heterogeneous Chemistry & Reactive Halogens",
        "color": "#7B68EE",
        "icon": "flask",
        "description": "Nighttime N2O5 reactive uptake on aerosol surfaces, ClNO2 production, halogen activation from playa dusts, and impacts on tropospheric ozone.",
        "key_species": ["N2O5", "ClNO2", "Cl2", "BrCl", "NO3-", "Cl-"],
    },
    "soa_formation": {
        "name": "Secondary Organic Aerosol Formation",
        "color": "#3CB371",
        "icon": "leaf",
        "description": "IEPOX reactive uptake on acidic sulfate aerosol, acid-catalyzed SOA formation, and effects of organic coatings and aerosol phase state on uptake rates.",
        "key_species": ["IEPOX", "H+", "Organosulfates", "2-methyltetrols", "SOA"],
    },
    "nutrients": {
        "name": "Nutrient Deposition & Biogeochemistry",
        "color": "#FF8C00",
        "icon": "seedling",
        "description": "Phosphorus and iron delivery to the Amazon, ocean fertilization by dust and volcanic ash, biomass burning as a P source, and Mn cycling.",
        "key_species": ["Soluble P", "Soluble Fe", "Mn", "Apatite"],
    },
    "ccn_clouds": {
        "name": "CCN Activity & Cloud Formation",
        "color": "#4682B4",
        "icon": "cloud",
        "description": "Cloud condensation nuclei measurements, hygroscopicity (kappa) characterization, aerosol-cloud interactions, and impacts of different aerosol types on cloud properties.",
        "key_species": ["CCN", "Sea spray", "Biomass burning", "Sulfate"],
    },
    "barbados_trends": {
        "name": "Barbados Long-term Aerosol Trends",
        "color": "#CD5C5C",
        "icon": "chart-line",
        "description": "Decades of aerosol measurements at Ragged Point, Barbados revealing impacts of clean air policies, African biomass burning, and changing anthropogenic emissions.",
        "key_species": ["nss-SO4", "NO3-", "Dust", "African smoke"],
    },
    "marine_aerosol": {
        "name": "Marine Aerosol & Air-Sea Interactions",
        "color": "#20B2AA",
        "icon": "water",
        "description": "Sea spray aerosol composition, biogenic sulfur compounds (DMS, MSA), ocean biology-atmosphere coupling, and marine CCN.",
        "key_species": ["Sea spray", "DMS", "MSA", "Reduced S"],
    },
}


# =============================================================================
# Embedded Measurement Data (based on published findings)
# =============================================================================

def get_barbados_timeseries():
    """
    Generate synthetic Barbados aerosol time series based on Gaston et al. (2024).

    21-year record (1990-2011) showing:
    - nss-sulfate declining due to Clean Air Act
    - Nitrate diverging with African smoke influence
    - Dust seasonal cycle (peak in summer)

    Based on: Gaston et al. (2024) ACP 24, 8049-8066.
    """
    np.random.seed(42)
    years = np.arange(1990, 2012)
    n_years = len(years)

    # Monthly resolution
    months = np.arange(n_years * 12) / 12.0 + 1990

    # nss-Sulfate: declining trend (~40% reduction over 21 years)
    # Based on reported decline from Clean Air Act & EU regulations
    sulfate_trend = 2.5 * np.exp(-0.025 * (months - 1990))
    sulfate_seasonal = 0.4 * np.sin(2 * np.pi * (months - 0.25))  # Peak in summer
    sulfate_noise = 0.3 * np.random.randn(len(months))
    sulfate = np.maximum(0.1, sulfate_trend + sulfate_seasonal + sulfate_noise)

    # Nitrate: diverging trends
    # Summer: increasing (African biomass burning smoke)
    # Winter: decreasing (reduced ship emissions)
    nitrate_base = 0.8
    nitrate_summer_trend = 0.015 * (months - 1990)  # Increasing summer
    nitrate_winter_trend = -0.008 * (months - 1990)  # Decreasing winter
    summer_mask = np.sin(2 * np.pi * (months - 0.25)) > 0
    nitrate_trend = np.where(summer_mask, nitrate_summer_trend, nitrate_winter_trend)
    nitrate_seasonal = 0.3 * np.sin(2 * np.pi * (months - 0.25))
    nitrate_noise = 0.15 * np.random.randn(len(months))
    nitrate = np.maximum(0.05, nitrate_base + nitrate_trend + nitrate_seasonal + nitrate_noise)

    # Dust mass: strong seasonal cycle, peak Jun-Aug (Saharan dust season)
    dust_base = 15.0
    dust_seasonal = 25.0 * np.maximum(0, np.sin(2 * np.pi * (months - 0.33)))
    dust_interannual = 5.0 * np.sin(2 * np.pi * months / 3.7)  # ~4yr variability
    dust_noise = 8.0 * np.random.exponential(0.5, len(months))
    dust = np.maximum(0.5, dust_base + dust_seasonal + dust_interannual + dust_noise)

    return {
        "months": months,
        "years": years,
        "sulfate_ug_m3": sulfate,
        "nitrate_ug_m3": nitrate,
        "dust_ug_m3": dust,
        "description": "Barbados aerosol time series 1990-2011 (based on Gaston et al. 2024)",
    }


def get_dust_transport_data():
    """
    Parameters for the transatlantic dust transport model.

    Based on:
    - Gaston (2020) Accounts of Chemical Research
    - Royer et al. (2025) ACP
    - Prospero et al. (2020) GBC
    - Shrestha et al. (2026) ACP
    """
    return {
        # Transport parameters
        "distance_km": 5000,             # Sahara to Barbados
        "transport_days": 5,             # Typical 5-7 day transport
        "wind_speed_m_s": 12.0,          # Trade wind speed
        "sal_altitude_m": 3000,          # Saharan Air Layer altitude
        "mbl_height_m": 500,             # Marine boundary layer height

        # Initial dust composition (mass fractions)
        "dust_minerals": {
            "illite": 0.35,
            "kaolinite": 0.20,
            "quartz": 0.15,
            "calcite": 0.12,
            "feldspar": 0.08,
            "iron_oxides": 0.05,
            "apatite": 0.03,
            "other": 0.02,
        },

        # Chemical aging rates (from Gaston 2020, Royer et al. 2025)
        "gamma_hno3_dust": 0.1,          # Uptake coeff for HNO3 on dust
        "gamma_h2so4_dust": 0.05,        # Uptake coeff for H2SO4 on dust
        "gamma_hcl_dust": 0.02,          # Uptake coeff for HCl on dust

        # Nutrient content (from Barkley et al. 2019, 2021; Prospero et al. 2020)
        "dust_p_content_ppm": 700,       # P content in dust (ppm)
        "dust_fe_content_pct": 3.5,      # Fe content in dust (%)
        "dust_p_solubility": 0.05,       # Initial P solubility (5%)
        "dust_fe_solubility": 0.02,      # Initial Fe solubility (2%)
        "smoke_p_solubility": 0.10,      # Smoke P solubility (>10%)

        # Mixing state (from Shrestha et al. 2026, Royer et al. 2025)
        "fraction_mixed_sea_salt": 0.67,  # 67% mixed with sea salt in MBL
        "fraction_aged_coatings": 0.26,   # 26% with acid-processed coatings

        # Atmospheric concentrations at Barbados
        "hno3_ppb": 0.5,                 # Typical HNO3 mixing ratio
        "so2_ppb": 0.3,                  # Typical SO2 mixing ratio
    }


def get_nutrient_budget_data():
    """
    Nutrient deposition budget for the Amazon basin.

    Based on:
    - Barkley et al. (2019) PNAS
    - Prospero et al. (2020) GBC
    - Barkley et al. (2021) GRL (diatom Fe)
    - Elliott et al. (2025) GBC (volcanic ash Fe)
    """
    return {
        # Phosphorus budget (Tg P / yr to the Amazon)
        "p_budget": {
            "saharan_dust": 0.022,        # Tg P/yr from Saharan dust
            "biomass_burning": 0.018,      # Tg P/yr from African biomass burning
            "volcanic_ash": 0.002,         # Tg P/yr from volcanic events
            "marine_biogenic": 0.003,      # Tg P/yr from marine sources
            "total_deposition": 0.045,     # Total Tg P/yr
            "amazon_requirement": 0.050,   # Estimated Amazon P requirement
        },

        # Iron budget
        "fe_budget": {
            "saharan_dust_total": 1.4,     # Tg Fe/yr total
            "saharan_dust_soluble": 0.028,  # Tg sol. Fe/yr (2% solubility)
            "diatom_fe": 0.05,             # From freshwater diatoms (4% Fe content)
            "volcanic_soluble": 0.008,     # <6% solubility
            "biomass_burning": 0.015,      # Tg sol. Fe/yr
        },

        # Solubility factors
        "solubility": {
            "dust_p_initial": 0.05,
            "dust_p_aged": 0.12,           # After acid processing
            "smoke_p": 0.10,
            "dust_fe_initial": 0.02,
            "dust_fe_aged": 0.05,
            "volcanic_fe": 0.06,
            "diatom_fe": 0.04,             # Fe in diatom frustules
        },

        # Seasonal variation at Cayenne, French Guiana
        # Peak dust: Feb-May; Peak smoke: Dec-Mar
        "seasonal_months": np.arange(1, 13),
        "dust_deposition_seasonal": np.array(
            [0.8, 1.2, 1.5, 1.3, 0.9, 0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.7]
        ),  # Relative to annual mean
        "smoke_deposition_seasonal": np.array(
            [1.5, 1.3, 1.0, 0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8]
        ),  # Relative to annual mean (peak Dec-Feb)
    }


def get_heterogeneous_kinetics():
    """
    Published heterogeneous reaction kinetics from Gaston's work.

    Returns uptake coefficients and ClNO2 yields for various surfaces.

    Based on:
    - Gaston & Thornton (2016) JPC A
    - Mitroo et al. (2019) ES&T
    - Royer et al. (2021) ES&T
    - Christie et al. (2025) ACS Earth Space Chem.
    """
    return {
        # gamma(N2O5) on different surfaces
        "gamma_n2o5": {
            "NaCl": {"value": 0.025, "range": (0.02, 0.03), "size_dependent": False},
            "NH4HSO4": {"value": 0.020, "range": (0.01, 0.03), "size_dependent": True},
            "(NH4)2SO4": {"value": 0.005, "range": (0.002, 0.01), "size_dependent": True},
            "Sea_salt": {"value": 0.030, "range": (0.02, 0.04), "size_dependent": False},
            "Great_Salt_Lake_playa": {"value": 0.05, "range": (0.001, 0.1), "size_dependent": False},
            "Owens_Lake_playa": {"value": 0.03, "range": (0.001, 0.08), "size_dependent": False},
            "Salton_Sea_playa": {"value": 0.01, "range": (0.001, 0.05), "size_dependent": False},
            "Organic_coated": {"value": 0.002, "range": (0.001, 0.005), "size_dependent": False},
            "Illite_clay": {"value": 0.008, "range": (0.003, 0.015), "size_dependent": False},
        },
        # ClNO2 yield (phi) from N2O5 + Cl-
        "clno2_yield": {
            "NaCl": 0.90,
            "Sea_salt": 0.85,
            "Great_Salt_Lake_playa": 0.65,
            "Owens_Lake_playa": 0.55,
            "Salton_Sea_playa": 0.50,
            "NH4HSO4": 0.0,     # No chloride
            "(NH4)2SO4": 0.0,   # No chloride
        },
    }


def get_iepox_kinetics():
    """
    IEPOX reactive uptake parameters.

    Based on:
    - Gaston et al. (2014) ES&T
    - Zhang et al. (2018) ES&T Letters
    """
    return {
        "gamma_iepox": {
            "NH4HSO4_acidic": 0.05,
            "(NH4)2SO4_neutral": 1e-4,
            "H2SO4_pure": 0.08,
        },
        "henrys_law_coeff_M_atm": 1.7e8,
        "coating_suppression": {
            "threshold_nm": 15,              # Organic coating thickness
            "max_suppression_factor": 0.5,   # Up to 50% reduction
        },
        "products": ["2-methyltetrols", "organosulfates", "C5-alkene triols"],
    }


def get_ccn_parameters():
    """
    CCN activation and hygroscopicity parameters.

    Based on:
    - Gaston et al. (2018) Atmosphere
    - Pohlker et al. (2023) Nature Communications
    - Edwards et al. (2021) Atmospheric Environment
    - Gaston et al. (2023) ACP
    """
    return {
        # Kappa values for different aerosol types
        "kappa": {
            "Sea_salt": {"mean": 1.12, "range": (0.9, 1.4)},
            "Ammonium_sulfate": {"mean": 0.61, "range": (0.55, 0.65)},
            "Ammonium_bisulfate": {"mean": 0.56, "range": (0.50, 0.60)},
            "Biomass_burning": {"mean": 0.12, "range": (0.05, 0.25)},
            "Mineral_dust": {"mean": 0.05, "range": (0.01, 0.10)},
            "Aged_dust_sea_salt": {"mean": 0.35, "range": (0.20, 0.50)},
            "Marine_organic": {"mean": 0.06, "range": (0.02, 0.15)},
            "Continental_SOA": {"mean": 0.12, "range": (0.08, 0.20)},
            "Marine_average_SoCal": {"mean": 0.22, "range": (0.10, 1.40)},
        },
        # Global averages from Pohlker et al. (2023)
        "kappa_org_global": 0.12,
        "kappa_inorg_global": 0.63,
        # Kohler theory parameters
        "sigma_w": 0.072,         # Surface tension of water (N/m)
        "rho_w": 997.0,           # Water density (kg/m3)
        "Mw": 0.018,              # Molar mass of water (kg/mol)
        "R": 8.314,               # Gas constant (J/mol/K)
        "T": 298.15,              # Temperature (K)
    }
