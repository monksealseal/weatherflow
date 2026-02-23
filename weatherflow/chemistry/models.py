"""
Atmospheric Chemistry Simulation Models

Lightweight 0D/1D models for interactive Streamlit simulations of
Cassandra Gaston's research. All models use pure numpy for minimal
memory footprint on the free Streamlit tier.

Models:
    DustTransportModel - 1D Lagrangian transatlantic dust transport
    N2O5BoxModel - Nighttime heterogeneous chemistry box model
    IEPOXModel - IEPOX reactive uptake and SOA formation
    NutrientDepositionModel - P/Fe deposition budget calculator
    CCNActivationModel - Kappa-Kohler CCN activation
    BarbadosTrendsModel - Long-term aerosol trend analysis
    GreatSaltLakeModel - Halogen production from playa dust
"""

import numpy as np


class DustTransportModel:
    """
    1D Lagrangian model for transatlantic Saharan dust transport.

    Simulates a dust parcel trajectory from the Sahara to Barbados through
    the Saharan Air Layer (SAL) with:
    - Gravitational settling (size-dependent)
    - Chemical aging (HNO3, H2SO4 uptake on dust surfaces)
    - Nutrient solubilization during transport
    - Mixing with sea salt upon descent into the MBL

    Based on: Gaston (2020) Acc. Chem. Res., Royer et al. (2025) ACP,
    Shrestha et al. (2026) ACP.
    """

    def __init__(
        self,
        distance_km=5000,
        transport_days=5,
        wind_speed=12.0,
        initial_dust_conc=200.0,
        hno3_ppb=0.5,
        h2so4_ppb=0.1,
        particle_diameter_um=3.0,
    ):
        self.distance_km = distance_km
        self.transport_days = transport_days
        self.wind_speed = wind_speed
        self.initial_dust_conc = initial_dust_conc
        self.hno3_ppb = hno3_ppb
        self.h2so4_ppb = h2so4_ppb
        self.particle_diameter_um = particle_diameter_um

        # Physical constants
        self.g = 9.81
        self.rho_dust = 2650.0       # kg/m3
        self.rho_air = 1.1           # kg/m3
        self.mu_air = 1.8e-5         # Pa s (dynamic viscosity)

        # Chemistry parameters (from Gaston's published values)
        self.gamma_hno3 = 0.1        # Uptake coefficient
        self.gamma_h2so4 = 0.05
        self.k_settling = self._compute_settling_velocity()

    def _compute_settling_velocity(self):
        """Stokes settling velocity (m/s)."""
        d = self.particle_diameter_um * 1e-6
        v_s = (self.rho_dust * d**2 * self.g) / (18 * self.mu_air)
        # Cunningham slip correction for particles < 10 um
        mfp = 0.065e-6  # Mean free path of air (m)
        Cc = 1 + (2 * mfp / d) * (1.257 + 0.4 * np.exp(-1.1 * d / (2 * mfp)))
        return v_s * Cc

    def run(self, dt_hours=1.0):
        """
        Run the transport simulation.

        Returns dict with time series of all tracked quantities.
        """
        dt = dt_hours * 3600.0
        n_steps = int(self.transport_days * 24 / dt_hours)
        t = np.arange(n_steps) * dt_hours

        # State variables
        dust_conc = np.zeros(n_steps)       # ug/m3
        nitrate_coating = np.zeros(n_steps)  # Fractional coating
        sulfate_coating = np.zeros(n_steps)  # Fractional coating
        p_solubility = np.zeros(n_steps)     # Phosphorus solubility
        fe_solubility = np.zeros(n_steps)    # Iron solubility
        altitude = np.zeros(n_steps)         # Altitude (m)
        distance = np.zeros(n_steps)         # Distance traveled (km)
        sea_salt_mixing = np.zeros(n_steps)  # Fraction mixed with sea salt
        depol_ratio = np.zeros(n_steps)      # Lidar depolarization ratio

        # Initial conditions
        dust_conc[0] = self.initial_dust_conc
        p_solubility[0] = 0.05              # 5% initial (Prospero et al. 2020)
        fe_solubility[0] = 0.02             # 2% initial
        altitude[0] = 3000.0                # SAL at ~3 km
        depol_ratio[0] = 0.30               # Pure dust depol (Shrestha et al. 2026)

        for i in range(1, n_steps):
            # Transport
            distance[i] = min(self.distance_km, self.wind_speed * t[i] * 3.6)
            progress = distance[i] / self.distance_km

            # Altitude: gradual descent from SAL to MBL
            # Dust descends more rapidly in the last third of transport
            if progress < 0.7:
                altitude[i] = altitude[0] * (1 - 0.3 * progress / 0.7)
            else:
                remaining = (progress - 0.7) / 0.3
                altitude[i] = altitude[0] * 0.7 * (1 - remaining * 0.85)
                altitude[i] = max(altitude[i], 100)

            # Gravitational settling loss
            settling_loss = self.k_settling * dt / altitude[max(0, i - 1)]
            dust_conc[i] = dust_conc[i - 1] * (1 - min(0.1, settling_loss))

            # Chemical aging - HNO3 and H2SO4 uptake on dust surfaces
            # Timescale: ~2-3 days for significant coating (Gaston 2020)
            # Royer et al. (2025): ~26% of particles show acid-processed coatings
            # Calibrate so 5-day transport gives ~25-30% coating at typical conditions
            aging_rate_scale = 1.5 * dt_hours / 24  # per-day-equivalent rate factor
            hno3_uptake_rate = self.gamma_hno3 * self.hno3_ppb * aging_rate_scale
            h2so4_uptake_rate = self.gamma_h2so4 * self.h2so4_ppb * aging_rate_scale

            # Coatings accumulate but saturate
            max_coating = 1.0
            nitrate_coating[i] = min(
                max_coating,
                nitrate_coating[i - 1] + hno3_uptake_rate * (1 - nitrate_coating[i - 1]),
            )
            sulfate_coating[i] = min(
                max_coating,
                sulfate_coating[i - 1]
                + h2so4_uptake_rate * (1 - sulfate_coating[i - 1]),
            )

            total_aging = nitrate_coating[i] + sulfate_coating[i]

            # Nutrient solubilization increases with acid processing
            # P solubility: 5% -> up to 12% with aging (Gaston 2020)
            p_solubility[i] = 0.05 + 0.07 * min(1.0, total_aging)
            # Fe solubility: 2% -> up to 5% (slower)
            fe_solubility[i] = 0.02 + 0.03 * min(1.0, total_aging * 0.8)

            # Sea salt mixing (upon descent into MBL)
            # Shrestha et al. (2026): ~67% internally mixed in MBL
            in_mbl = altitude[i] < 600
            if in_mbl:
                # Mixing rate increases as dust spends more time in MBL
                mbl_first = np.argmax(altitude[:i+1] < 600) if np.any(altitude[:i+1] < 600) else i
                mbl_hours = t[i] - t[mbl_first]
                sea_salt_mixing[i] = 0.67 * (1 - np.exp(-mbl_hours / 6.0))
            else:
                sea_salt_mixing[i] = sea_salt_mixing[i - 1]

            # Lidar depolarization ratio
            # Pure dust ~0.30, mixed with sea salt <0.10
            depol_ratio[i] = 0.30 * (1 - sea_salt_mixing[i]) + 0.05 * sea_salt_mixing[i]

        return {
            "time_hours": t,
            "distance_km": distance,
            "altitude_m": altitude,
            "dust_conc_ug_m3": dust_conc,
            "nitrate_coating": nitrate_coating,
            "sulfate_coating": sulfate_coating,
            "p_solubility": p_solubility,
            "fe_solubility": fe_solubility,
            "sea_salt_mixing": sea_salt_mixing,
            "depol_ratio": depol_ratio,
        }


class N2O5BoxModel:
    """
    0D box model for nighttime N2O5 heterogeneous chemistry.

    Simulates:
    - NO3 + NO2 <-> N2O5 equilibrium
    - N2O5 uptake on aerosol surfaces
    - ClNO2 production from N2O5 + Cl-
    - HNO3 production from N2O5 hydrolysis
    - Lifetime of N2O5 on different aerosol types

    Based on: Gaston & Thornton (2016) JPC A, Gaston et al. (2014) ACP,
    Mitroo et al. (2019) ES&T, Royer et al. (2021) ES&T.
    """

    def __init__(
        self,
        T=298.0,
        P=101325.0,
        RH=50.0,
        aerosol_surface_area=200e-6,
        aerosol_type="NaCl",
    ):
        self.T = T
        self.P = P
        self.RH = RH
        self.Sa = aerosol_surface_area      # cm2/cm3
        self.aerosol_type = aerosol_type

        # Kinetics parameters
        from weatherflow.chemistry.gaston_research import get_heterogeneous_kinetics
        kinetics = get_heterogeneous_kinetics()

        surface = kinetics["gamma_n2o5"].get(
            aerosol_type, {"value": 0.01, "range": (0.005, 0.02), "size_dependent": False}
        )
        self.gamma = surface["value"]
        self.phi_clno2 = kinetics["clno2_yield"].get(aerosol_type, 0.0)

        # Mean molecular speed of N2O5 (cm/s)
        M_n2o5 = 0.108  # kg/mol
        self.c_bar = np.sqrt(8 * 8.314 * T / (np.pi * M_n2o5)) * 100

        # NO3 + NO2 -> N2O5 equilibrium constant
        self.Keq = 2.7e-27 * np.exp(10930 / T)  # cm3 molecule-1

        # Organic coating suppression
        self.organic_suppression = 1.0
        if "Organic" in aerosol_type:
            self.organic_suppression = 0.2  # 80% reduction (Gaston et al. 2014)

    def compute_n2o5_lifetime(self):
        """Compute N2O5 lifetime (seconds) due to heterogeneous uptake."""
        k_het = 0.25 * self.c_bar * self.Sa * self.gamma * self.organic_suppression
        if k_het > 0:
            return 1.0 / k_het
        return np.inf

    def run(self, duration_hours=12.0, dt_seconds=10.0, no2_ppb=5.0, o3_ppb=30.0):
        """
        Run the nighttime chemistry simulation.

        Args:
            duration_hours: Simulation length (hours)
            dt_seconds: Time step (seconds)
            no2_ppb: Initial NO2 mixing ratio
            o3_ppb: Initial O3 mixing ratio

        Returns:
            Dict with time series of all species.
        """
        n_steps = int(duration_hours * 3600 / dt_seconds)
        t = np.arange(n_steps) * dt_seconds / 3600.0  # hours

        # Convert ppb to molecules/cm3
        M = self.P / (1.38e-23 * self.T) * 1e-6  # air number density (cm-3)
        ppb_to_molec = M * 1e-9

        # State (molecules/cm3)
        NO2 = np.zeros(n_steps)
        O3 = np.zeros(n_steps)
        NO3 = np.zeros(n_steps)
        N2O5 = np.zeros(n_steps)
        ClNO2 = np.zeros(n_steps)
        HNO3 = np.zeros(n_steps)

        NO2[0] = no2_ppb * ppb_to_molec
        O3[0] = o3_ppb * ppb_to_molec

        # Rate constants
        k1 = 1.8e-12 * np.exp(-1370 / self.T)  # O3 + NO2 -> NO3 + O2
        kf = 1.9e-12 * (self.T / 300) ** 0.2    # NO3 + NO2 -> N2O5
        kr = kf / self.Keq                       # N2O5 -> NO3 + NO2

        # Heterogeneous loss rate
        k_het = 0.25 * self.c_bar * self.Sa * self.gamma * self.organic_suppression

        for i in range(1, n_steps):
            dt = dt_seconds

            # Sub-step for numerical stability with stiff chemistry
            n_sub = max(1, int(dt / 2.0))
            dt_sub = dt / n_sub

            o3_cur = O3[i - 1]
            no2_cur = NO2[i - 1]
            no3_cur = NO3[i - 1]
            n2o5_cur = N2O5[i - 1]
            clno2_cur = ClNO2[i - 1]
            hno3_cur = HNO3[i - 1]

            for _ in range(n_sub):
                # O3 + NO2 -> NO3
                r1 = k1 * o3_cur * no2_cur

                # NO3 + NO2 <-> N2O5
                r_fwd = kf * no3_cur * no2_cur
                r_rev = kr * n2o5_cur

                # N2O5 heterogeneous loss
                r_het = k_het * n2o5_cur

                # Rate-limit: no species can go below zero
                max_r1 = min(o3_cur, no2_cur) / max(dt_sub, 1e-30) * 0.5
                r1 = min(r1, max_r1)
                max_fwd = min(no3_cur, no2_cur) / max(dt_sub, 1e-30) * 0.5
                r_fwd = min(r_fwd, max_fwd)
                r_rev = min(r_rev, n2o5_cur / max(dt_sub, 1e-30) * 0.5)
                r_het = min(r_het, n2o5_cur / max(dt_sub, 1e-30) * 0.5)

                o3_cur = max(0, o3_cur - r1 * dt_sub)
                no3_cur = max(0, no3_cur + (r1 - r_fwd + r_rev) * dt_sub)
                no2_cur = max(0, no2_cur - (r1 + r_fwd - r_rev) * dt_sub)
                n2o5_cur = max(0, n2o5_cur + (r_fwd - r_rev - r_het) * dt_sub)
                clno2_cur += self.phi_clno2 * r_het * dt_sub
                hno3_cur += (2 - self.phi_clno2) * r_het * dt_sub

            O3[i] = o3_cur
            NO2[i] = no2_cur
            NO3[i] = no3_cur
            N2O5[i] = n2o5_cur
            ClNO2[i] = clno2_cur
            HNO3[i] = hno3_cur

        # Convert back to ppb
        to_ppb = 1.0 / ppb_to_molec

        return {
            "time_hours": t,
            "NO2_ppb": NO2 * to_ppb,
            "O3_ppb": O3 * to_ppb,
            "NO3_ppt": NO3 * to_ppb * 1000,
            "N2O5_ppt": N2O5 * to_ppb * 1000,
            "ClNO2_ppt": ClNO2 * to_ppb * 1000,
            "HNO3_ppb": HNO3 * to_ppb,
            "N2O5_lifetime_min": self.compute_n2o5_lifetime() / 60,
        }


class IEPOXModel:
    """
    Reactive uptake model for IEPOX (isoprene-derived epoxydiols) on
    acidic sulfate aerosol.

    Computes:
    - Uptake coefficient as function of aerosol acidity
    - SOA mass production rate
    - Effect of organic coatings on uptake
    - Product distribution (organosulfates, 2-methyltetrols)

    Based on: Gaston et al. (2014) ES&T, Zhang et al. (2018) ES&T Letters.
    """

    def __init__(self, T=298.0, RH=50.0):
        self.T = T
        self.RH = RH
        from weatherflow.chemistry.gaston_research import get_iepox_kinetics
        self.kinetics = get_iepox_kinetics()

    def compute_gamma(self, pH, organic_coating_nm=0.0):
        """
        Compute IEPOX reactive uptake coefficient.

        Args:
            pH: Aerosol pH (0-7)
            organic_coating_nm: Organic coating thickness (nm)

        Returns:
            gamma: Uptake coefficient
        """
        # pH dependence: gamma increases dramatically with acidity
        # Based on Gaston et al. (2014): NH4HSO4 (pH~1) = 0.05, (NH4)2SO4 (pH~5) = 1e-4
        gamma_base = 0.08 * np.exp(-0.8 * pH)
        gamma_base = np.clip(gamma_base, 1e-6, 0.1)

        # Organic coating suppression (Zhang et al. 2018)
        # Coatings >15 nm reduce gamma by up to 50%
        suppression = 1.0
        if organic_coating_nm > 0:
            threshold = self.kinetics["coating_suppression"]["threshold_nm"]
            max_supp = self.kinetics["coating_suppression"]["max_suppression_factor"]
            if organic_coating_nm > threshold:
                suppression = 1.0 - max_supp * min(
                    1.0, (organic_coating_nm - threshold) / threshold
                )

        return gamma_base * suppression

    def run_uptake(
        self,
        iepox_ppb=1.0,
        aerosol_surface_area=300e-6,
        pH=1.0,
        organic_coating_nm=0.0,
        duration_hours=6.0,
        dt_seconds=60.0,
    ):
        """
        Simulate IEPOX reactive uptake over time.

        Returns dict with time series of IEPOX decay and SOA production.
        """
        n_steps = int(duration_hours * 3600 / dt_seconds)
        t = np.arange(n_steps) * dt_seconds / 3600.0

        gamma = self.compute_gamma(pH, organic_coating_nm)

        # Mean molecular speed of IEPOX (cm/s)
        M_iepox = 0.118  # kg/mol
        c_bar = np.sqrt(8 * 8.314 * self.T / (np.pi * M_iepox)) * 100

        # First-order uptake rate
        k_uptake = 0.25 * c_bar * aerosol_surface_area * gamma

        # IEPOX decay (ppb)
        iepox = iepox_ppb * np.exp(-k_uptake * t * 3600)

        # SOA production (ug/m3)
        # 1 ppb of IEPOX (MW=118) at STP ~ 4.83 ug/m3
        # Assume ~30% mass yield to particulate SOA
        soa_yield = 0.3
        ppb_to_ugm3 = 118.0 / 24.45  # MW / molar volume at 298K (L/mol)
        iepox_consumed = iepox_ppb - iepox  # ppb consumed
        soa_mass = iepox_consumed * ppb_to_ugm3 * soa_yield

        # Product speciation
        organosulfate_fraction = 0.4 * min(1.0, 10 ** (1 - pH))
        methyltetrol_fraction = 0.35
        other_fraction = 1.0 - organosulfate_fraction - methyltetrol_fraction

        return {
            "time_hours": t,
            "iepox_ppb": iepox,
            "soa_ug_m3": soa_mass,
            "gamma": gamma,
            "lifetime_hours": (1.0 / k_uptake / 3600) if k_uptake > 0 else np.inf,
            "organosulfate_ug_m3": soa_mass * organosulfate_fraction,
            "methyltetrol_ug_m3": soa_mass * methyltetrol_fraction,
            "other_soa_ug_m3": soa_mass * other_fraction,
        }

    def ph_sensitivity(self, pH_range=None, organic_coating_nm=0.0):
        """Compute gamma as function of pH."""
        if pH_range is None:
            pH_range = np.linspace(0, 7, 100)
        gammas = np.array(
            [self.compute_gamma(pH, organic_coating_nm) for pH in pH_range]
        )
        return pH_range, gammas


class NutrientDepositionModel:
    """
    Nutrient deposition budget model for the Amazon basin.

    Computes annual P and Fe deposition from:
    - Saharan mineral dust
    - African biomass burning smoke
    - Volcanic ash events
    - Freshwater diatoms (Fe vector)

    Based on: Barkley et al. (2019) PNAS, Prospero et al. (2020) GBC,
    Barkley et al. (2021) GRL, Elliott et al. (2024, 2025).
    """

    def __init__(self):
        from weatherflow.chemistry.gaston_research import get_nutrient_budget_data
        self.budget = get_nutrient_budget_data()

    def compute_annual_budget(
        self,
        dust_flux_tg_yr=180,
        smoke_flux_tg_yr=20,
        volcanic_events_per_year=0.1,
        volcanic_ash_tg_event=5.0,
    ):
        """
        Compute annual P and Fe deposition budget.

        Args:
            dust_flux_tg_yr: Saharan dust flux to Amazon (Tg/yr)
            smoke_flux_tg_yr: African smoke flux to Amazon (Tg/yr)
            volcanic_events_per_year: Average volcanic events per year
            volcanic_ash_tg_event: Ash per event (Tg)
        """
        sol = self.budget["solubility"]

        # Phosphorus
        dust_p_total = dust_flux_tg_yr * 700e-6  # 700 ppm P content
        dust_p_soluble = dust_p_total * sol["dust_p_initial"]
        dust_p_aged = dust_p_total * sol["dust_p_aged"]

        smoke_p_total = smoke_flux_tg_yr * 500e-6  # ~500 ppm P in smoke
        smoke_p_soluble = smoke_p_total * sol["smoke_p"]

        volcanic_p = (
            volcanic_events_per_year * volcanic_ash_tg_event * 400e-6 * 0.08
        )

        # Iron
        dust_fe_total = dust_flux_tg_yr * 0.035  # 3.5% Fe
        dust_fe_soluble = dust_fe_total * sol["dust_fe_initial"]
        dust_fe_aged = dust_fe_total * sol["dust_fe_aged"]

        diatom_fe = dust_flux_tg_yr * 0.001 * sol["diatom_fe"]  # Small fraction

        smoke_fe_soluble = smoke_flux_tg_yr * 0.005 * 0.10  # 0.5% Fe, 10% sol

        return {
            "p_budget": {
                "dust_total_Tg": dust_p_total,
                "dust_soluble_Tg": dust_p_soluble,
                "dust_aged_Tg": dust_p_aged,
                "smoke_total_Tg": smoke_p_total,
                "smoke_soluble_Tg": smoke_p_soluble,
                "volcanic_Tg": volcanic_p,
                "total_soluble_Tg": dust_p_soluble + smoke_p_soluble + volcanic_p,
                "total_with_aging_Tg": dust_p_aged + smoke_p_soluble + volcanic_p,
            },
            "fe_budget": {
                "dust_total_Tg": dust_fe_total,
                "dust_soluble_Tg": dust_fe_soluble,
                "dust_aged_Tg": dust_fe_aged,
                "diatom_Tg": diatom_fe,
                "smoke_soluble_Tg": smoke_fe_soluble,
                "total_soluble_Tg": dust_fe_soluble + diatom_fe + smoke_fe_soluble,
            },
            "smoke_p_fraction": smoke_p_soluble
            / max(1e-10, dust_p_soluble + smoke_p_soluble + volcanic_p),
        }

    def seasonal_deposition(self, months=None):
        """Compute monthly deposition pattern."""
        if months is None:
            months = np.arange(1, 13)
        dust_dep = self.budget["dust_deposition_seasonal"]
        smoke_dep = self.budget["smoke_deposition_seasonal"]

        return {
            "months": months,
            "dust_relative": dust_dep,
            "smoke_relative": smoke_dep,
            "total_relative": 0.6 * dust_dep + 0.4 * smoke_dep,
        }


class CCNActivationModel:
    """
    Kappa-Kohler CCN activation model.

    Computes critical supersaturation for aerosol particles of different
    compositions and sizes using single-parameter kappa-Kohler theory.

    Based on: Gaston et al. (2018) Atmosphere, Pohlker et al. (2023)
    Nat. Comm., Edwards et al. (2021) Atmos. Environ.
    """

    def __init__(self, T=298.15):
        self.T = T
        from weatherflow.chemistry.gaston_research import get_ccn_parameters
        params = get_ccn_parameters()
        self.kappa_db = params["kappa"]
        self.sigma_w = params["sigma_w"]
        self.rho_w = params["rho_w"]
        self.Mw = params["Mw"]
        self.R = params["R"]

    def critical_supersaturation(self, dry_diameter_nm, kappa):
        """
        Compute critical supersaturation (%) for a particle.

        Uses kappa-Kohler theory:
        Sc = (4 * A^3 / (27 * kappa * Dd^3))^0.5
        where A = 4 * sigma * Mw / (R * T * rho_w)
        """
        Dd = dry_diameter_nm * 1e-9  # Convert to meters
        A = 4 * self.sigma_w * self.Mw / (self.R * self.T * self.rho_w)

        if kappa <= 0 or Dd <= 0:
            return np.inf

        Sc = np.sqrt(4 * A**3 / (27 * kappa * Dd**3))
        return Sc * 100  # Convert to percent

    def activation_spectrum(self, kappa, diameter_range_nm=None):
        """
        Compute Sc vs diameter for a given kappa.

        Returns arrays of diameters and critical supersaturations.
        """
        if diameter_range_nm is None:
            diameter_range_nm = np.logspace(1, 3, 200)  # 10 nm to 1000 nm

        Sc = np.array(
            [self.critical_supersaturation(d, kappa) for d in diameter_range_nm]
        )

        return diameter_range_nm, Sc

    def compare_aerosol_types(self, diameter_nm=100):
        """Compare Sc for different aerosol types at a given size."""
        results = {}
        for name, params in self.kappa_db.items():
            kappa = params["mean"]
            Sc = self.critical_supersaturation(diameter_nm, kappa)
            results[name] = {
                "kappa": kappa,
                "Sc_percent": Sc,
                "activates_at_0.3": Sc < 0.3,
                "activates_at_0.1": Sc < 0.1,
            }
        return results

    def kohler_curve(self, dry_diameter_nm, kappa, D_ratio_range=None):
        """
        Compute full Kohler curve (S vs wet diameter ratio).

        Returns growth factor and corresponding saturation ratio.
        """
        Dd = dry_diameter_nm * 1e-9
        A = 4 * self.sigma_w * self.Mw / (self.R * self.T * self.rho_w)

        if D_ratio_range is None:
            D_ratio_range = np.linspace(1.0, 5.0, 500)

        # Saturation ratio S = exp(A/Dw) * (Dw^3 - Dd^3) / (Dw^3 - Dd^3*(1-kappa))
        Dw = Dd * D_ratio_range
        kelvin = np.exp(A / Dw)
        raoult = (Dw**3 - Dd**3) / (Dw**3 - Dd**3 * (1 - kappa))
        S = kelvin * raoult

        # Supersaturation in percent
        ss = (S - 1) * 100

        return D_ratio_range, ss


class BarbadosTrendsModel:
    """
    Analyzes long-term aerosol trends at Barbados.

    Computes:
    - Decadal trends in sulfate, nitrate, dust
    - Seasonal decomposition
    - Attribution to clean air policies vs. biomass burning

    Based on: Gaston et al. (2024) ACP.
    """

    def __init__(self):
        from weatherflow.chemistry.gaston_research import get_barbados_timeseries
        self.data = get_barbados_timeseries()

    def compute_trends(self):
        """Compute linear trends in sulfate and nitrate."""
        months = self.data["months"]
        sulfate = self.data["sulfate_ug_m3"]
        nitrate = self.data["nitrate_ug_m3"]
        dust = self.data["dust_ug_m3"]

        # Linear regression
        def linear_trend(x, y):
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0], coeffs[1]  # slope, intercept

        so4_slope, so4_intercept = linear_trend(months, sulfate)
        no3_slope, no3_intercept = linear_trend(months, nitrate)
        dust_slope, dust_intercept = linear_trend(months, dust)

        return {
            "sulfate_trend_per_year": so4_slope * 12,
            "nitrate_trend_per_year": no3_slope * 12,
            "dust_trend_per_year": dust_slope * 12,
            "sulfate_pct_change": so4_slope * 12 * 21 / (so4_intercept + so4_slope * 10.5 * 12) * 100,
            "nitrate_pct_change": no3_slope * 12 * 21 / (no3_intercept + no3_slope * 10.5 * 12) * 100,
        }

    def seasonal_analysis(self):
        """Compute monthly climatology."""
        months = self.data["months"]
        month_of_year = ((months - np.floor(months)) * 12).astype(int) + 1
        month_of_year = np.clip(month_of_year, 1, 12)

        result = {"month": np.arange(1, 13)}
        for var_name in ["sulfate_ug_m3", "nitrate_ug_m3", "dust_ug_m3"]:
            values = self.data[var_name]
            monthly_means = np.array(
                [np.mean(values[month_of_year == m]) for m in range(1, 13)]
            )
            result[var_name.replace("_ug_m3", "_monthly_mean")] = monthly_means

        return result


class GreatSaltLakeModel:
    """
    Halogen production model for Great Salt Lake playa dust.

    Simulates ClNO2, Cl2, and BrCl production from N2O5 uptake on
    saline playa dust emitted from the shrinking Great Salt Lake.

    Based on: Christie et al. (2025) ACS Earth Space Chem.,
    Mitroo et al. (2019) ES&T.
    """

    def __init__(
        self,
        T=298.0,
        RH=50.0,
        dust_loading_ug_m3=50.0,
        lake_shrinkage_pct=50.0,
    ):
        self.T = T
        self.RH = RH
        self.dust_loading = dust_loading_ug_m3
        self.lake_shrinkage_pct = lake_shrinkage_pct

        # Salt content depends on lake shrinkage
        # As the lake shrinks, more saline playa is exposed
        self.nacl_fraction = 0.15 + 0.35 * (lake_shrinkage_pct / 100)
        self.mgcl2_fraction = 0.05 + 0.10 * (lake_shrinkage_pct / 100)

        # Compute effective gamma
        self.gamma_n2o5 = 0.02 + 0.08 * self.nacl_fraction
        self.clno2_yield = 0.5 + 0.3 * self.nacl_fraction
        self.cl2_yield = 0.05 + 0.15 * self.mgcl2_fraction
        self.brcl_yield = 0.02 + 0.05 * self.nacl_fraction

    def run(self, duration_hours=12.0, dt_seconds=30.0, no2_ppb=10.0, o3_ppb=40.0):
        """
        Run nighttime halogen production simulation.
        """
        # Convert dust loading to surface area
        # Assume mean diameter ~5 um, density 2600 kg/m3
        d_p = 5e-6  # m
        rho_p = 2600  # kg/m3
        mass_conc = self.dust_loading * 1e-9  # kg/m3
        N_p = mass_conc / (np.pi / 6 * d_p**3 * rho_p)  # particles/m3
        Sa = N_p * np.pi * d_p**2 * 1e-2  # cm2/cm3 (m2/m3 * 1e-2)

        # Use N2O5 box model
        model = N2O5BoxModel(
            T=self.T,
            P=101325.0,
            RH=self.RH,
            aerosol_surface_area=Sa,
            aerosol_type="Great_Salt_Lake_playa",
        )

        result = model.run(
            duration_hours=duration_hours,
            dt_seconds=dt_seconds,
            no2_ppb=no2_ppb,
            o3_ppb=o3_ppb,
        )

        # Add halogen-specific products
        clno2_total = result["ClNO2_ppt"]
        # Cl2 and BrCl are secondary products
        result["Cl2_ppt"] = clno2_total * self.cl2_yield / max(0.01, self.clno2_yield)
        result["BrCl_ppt"] = clno2_total * self.brcl_yield / max(0.01, self.clno2_yield)

        # Dawn Cl radical production (photolysis of ClNO2 at sunrise)
        result["Cl_radical_potential_ppt"] = result["ClNO2_ppt"] + 2 * result["Cl2_ppt"]

        # Ozone production potential (each Cl radical can produce ~2 O3)
        result["O3_production_potential_ppb"] = result["Cl_radical_potential_ppt"] * 2e-3

        result["lake_shrinkage_pct"] = self.lake_shrinkage_pct
        result["nacl_fraction"] = self.nacl_fraction
        result["gamma_n2o5"] = self.gamma_n2o5
        result["clno2_yield"] = self.clno2_yield

        return result
