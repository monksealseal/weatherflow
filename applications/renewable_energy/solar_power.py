"""
Solar Power Conversion Module

Convert solar irradiance and temperature forecasts to PV power output.
"""

import numpy as np
from typing import Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PVSystemSpec:
    """Specification for a PV system."""
    name: str
    module_type: str
    capacity_dc: float  # MW
    module_efficiency: float  # 0-1
    temperature_coefficient: float  # %/°C
    tilt_angle: float  # degrees
    azimuth: float  # degrees, 180=south
    inverter_efficiency: float = 0.96
    system_losses: float = 0.14  # cable, soiling, etc.


# Standard PV system specifications
PV_LIBRARY = {
    'mono-Si-standard': PVSystemSpec(
        name='Monocrystalline Silicon Standard',
        module_type='mono-Si',
        capacity_dc=1.0,
        module_efficiency=0.19,
        temperature_coefficient=-0.45,  # %/°C
        tilt_angle=30.0,
        azimuth=180.0
    ),
    'poly-Si-standard': PVSystemSpec(
        name='Polycrystalline Silicon Standard',
        module_type='poly-Si',
        capacity_dc=1.0,
        module_efficiency=0.17,
        temperature_coefficient=-0.40,
        tilt_angle=30.0,
        azimuth=180.0
    ),
    'thin-film': PVSystemSpec(
        name='Thin Film CdTe',
        module_type='CdTe',
        capacity_dc=1.0,
        module_efficiency=0.14,
        temperature_coefficient=-0.25,
        tilt_angle=30.0,
        azimuth=180.0
    )
}


class SolarPowerConverter:
    """
    Convert solar irradiance and temperature forecasts to PV power output.

    Examples:
        >>> converter = SolarPowerConverter(capacity=100.0, tilt_angle=30)
        >>> ghi = np.array([0, 200, 500, 800, 1000])  # W/m²
        >>> temp = np.array([15, 20, 25, 30, 35]) + 273.15  # K
        >>> power = converter.irradiance_to_power(ghi, temp)
        >>> print(f"Peak power: {power.max():.1f} MW")
    """

    def __init__(
        self,
        panel_type: str = 'mono-Si-standard',
        capacity: float = 1.0,
        tilt_angle: Optional[float] = None,
        azimuth: Optional[float] = None,
        farm_location: Optional[Dict[str, float]] = None,
        tracking: bool = False
    ):
        """
        Initialize solar power converter.

        Args:
            panel_type: PV module type from PV_LIBRARY
            capacity: System capacity (MW DC)
            tilt_angle: Panel tilt angle (degrees from horizontal)
            azimuth: Panel azimuth (degrees, 180=south)
            farm_location: {'lat': latitude, 'lon': longitude}
            tracking: Whether system uses single-axis tracking
        """
        if panel_type not in PV_LIBRARY:
            raise ValueError(f"Unknown panel type: {panel_type}")

        self.system = PV_LIBRARY[panel_type]
        self.system.capacity_dc = capacity

        if tilt_angle is not None:
            self.system.tilt_angle = tilt_angle
        if azimuth is not None:
            self.system.azimuth = azimuth

        self.farm_location = farm_location or {'lat': 35.0, 'lon': -110.0}
        self.tracking = tracking

    def irradiance_to_power(
        self,
        ghi: np.ndarray,
        temperature: np.ndarray,
        wind_speed: Optional[np.ndarray] = None,
        solar_zenith: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert global horizontal irradiance to power output.

        Args:
            ghi: Global horizontal irradiance (W/m²)
            temperature: Ambient temperature (K)
            wind_speed: Wind speed (m/s) for cell temperature calculation
            solar_zenith: Solar zenith angle (degrees)

        Returns:
            AC power output (MW)
        """
        ghi = np.atleast_1d(ghi)
        temperature = np.atleast_1d(temperature) - 273.15  # Convert to Celsius

        # Calculate plane-of-array (POA) irradiance
        if solar_zenith is not None and not self.tracking:
            poa = self._poa_irradiance(ghi, solar_zenith)
        else:
            # Simplified assumption: POA ≈ GHI for optimal tilt
            poa = ghi * np.cos(np.radians(np.abs(self.system.tilt_angle -
                                                  self.farm_location['lat'])))

        # Calculate cell temperature
        cell_temp = self._cell_temperature(temperature, poa, wind_speed)

        # Calculate DC power using PVWatts-like model
        # Standard test conditions: 1000 W/m², 25°C
        stc_irradiance = 1000.0
        stc_temperature = 25.0

        # Irradiance effect (linear)
        irradiance_factor = poa / stc_irradiance

        # Temperature effect
        temp_diff = cell_temp - stc_temperature
        temp_factor = 1 + (self.system.temperature_coefficient / 100) * temp_diff

        # DC power
        dc_power = (
            self.system.capacity_dc *
            self.system.module_efficiency *
            irradiance_factor *
            temp_factor
        )

        # Ensure non-negative
        dc_power = np.maximum(dc_power, 0.0)

        # Convert to AC power
        ac_power = self._dc_to_ac(dc_power)

        return ac_power

    def _poa_irradiance(
        self,
        ghi: np.ndarray,
        solar_zenith: np.ndarray
    ) -> np.ndarray:
        """
        Calculate plane-of-array irradiance from GHI.

        Simplified model using Perez diffuse model concepts.

        Args:
            ghi: Global horizontal irradiance (W/m²)
            solar_zenith: Solar zenith angle (degrees)

        Returns:
            Plane-of-array irradiance (W/m²)
        """
        # Simplified model: assume diffuse fraction
        diffuse_fraction = 0.15  # typical clear-sky value

        # Direct normal irradiance (DNI) approximation
        dni = ghi * (1 - diffuse_fraction) / np.cos(np.radians(solar_zenith))
        dni = np.clip(dni, 0, 1361)  # Solar constant upper limit

        # Diffuse horizontal irradiance
        dhi = ghi * diffuse_fraction

        # Angle of incidence on tilted surface
        # Simplified: assume solar noon conditions
        aoi = np.abs(solar_zenith - self.system.tilt_angle)

        # POA components
        poa_direct = dni * np.cos(np.radians(aoi))
        poa_diffuse = dhi * (1 + np.cos(np.radians(self.system.tilt_angle))) / 2
        poa_reflected = ghi * 0.2 * (1 - np.cos(np.radians(self.system.tilt_angle))) / 2

        poa = poa_direct + poa_diffuse + poa_reflected
        return np.maximum(poa, 0.0)

    def _cell_temperature(
        self,
        ambient_temp: np.ndarray,
        poa: np.ndarray,
        wind_speed: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate PV cell temperature using NOCT model.

        Args:
            ambient_temp: Ambient air temperature (°C)
            poa: Plane-of-array irradiance (W/m²)
            wind_speed: Wind speed (m/s)

        Returns:
            Cell temperature (°C)
        """
        # NOCT (Nominal Operating Cell Temperature) model
        # Typical NOCT = 45°C for ambient 20°C, irradiance 800 W/m², wind 1 m/s
        noct = 45.0
        noct_irradiance = 800.0
        noct_ambient = 20.0

        # Temperature rise above ambient
        temp_rise = (noct - noct_ambient) * (poa / noct_irradiance)

        # Wind cooling effect (simplified)
        if wind_speed is not None:
            wind_factor = 1.0 / (1.0 + 0.05 * wind_speed)
            temp_rise *= wind_factor

        cell_temp = ambient_temp + temp_rise
        return cell_temp

    def _dc_to_ac(self, dc_power: np.ndarray) -> np.ndarray:
        """
        Convert DC to AC power through inverter.

        Args:
            dc_power: DC power (MW)

        Returns:
            AC power (MW)
        """
        # Apply system losses
        dc_power_adjusted = dc_power * (1 - self.system.system_losses)

        # Inverter efficiency (simplified constant efficiency)
        ac_power = dc_power_adjusted * self.system.inverter_efficiency

        # Inverter clipping at rated capacity
        inverter_capacity = self.system.capacity_dc * 0.95  # typical AC/DC ratio
        ac_power = np.minimum(ac_power, inverter_capacity)

        return ac_power

    def daily_energy(
        self,
        hourly_power: np.ndarray
    ) -> float:
        """
        Calculate daily energy production.

        Args:
            hourly_power: Hourly power values (MW)

        Returns:
            Daily energy (MWh)
        """
        # Integrate power over time (assuming hourly intervals)
        return np.sum(hourly_power)

    def capacity_factor(
        self,
        power_output: np.ndarray
    ) -> float:
        """
        Calculate capacity factor over a time period.

        Args:
            power_output: Time series of power output (MW)

        Returns:
            Capacity factor (0-1)
        """
        return np.mean(power_output) / self.system.capacity_dc

    def convert_forecast(
        self,
        weather_forecast: Dict[str, np.ndarray],
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Convert weather forecast to power forecast.

        Args:
            weather_forecast: Dictionary containing:
                - 'ssrd': Surface solar radiation downwards (J/m²) or
                - 'ghi': Global horizontal irradiance (W/m²)
                - 't2m': 2-meter temperature (K)
                - Optional: 'u10', 'v10': 10-meter wind components (m/s)
            timestamps: Array of timestamps for solar position calculation

        Returns:
            Dictionary with power forecasts and statistics
        """
        # Extract irradiance (convert from J/m² if needed)
        if 'ssrd' in weather_forecast:
            # ERA5 provides accumulated radiation in J/m²
            # Convert to W/m² (assuming hourly data)
            ghi = weather_forecast['ssrd'] / 3600.0
        elif 'ghi' in weather_forecast:
            ghi = weather_forecast['ghi']
        else:
            raise ValueError("No irradiance data found in forecast")

        # Temperature
        temperature = weather_forecast['t2m']

        # Wind speed (optional)
        wind_speed = None
        if 'u10' in weather_forecast and 'v10' in weather_forecast:
            u = weather_forecast['u10']
            v = weather_forecast['v10']
            wind_speed = np.sqrt(u**2 + v**2)

        # Solar position (simplified)
        solar_zenith = None
        if timestamps is not None:
            solar_zenith = self._calculate_solar_zenith(timestamps)

        # Convert to power
        power = self.irradiance_to_power(ghi, temperature, wind_speed, solar_zenith)

        return {
            'power_mw': power,
            'ghi': ghi,
            'cell_temperature': self._cell_temperature(
                temperature - 273.15, ghi, wind_speed
            ),
            'capacity_factor': self.capacity_factor(power),
            'mean_power': np.mean(power),
            'std_power': np.std(power),
            'max_power': np.max(power),
            'daily_energy_mwh': self.daily_energy(power) if len(power) >= 24 else None
        }

    def _calculate_solar_zenith(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Calculate solar zenith angle (simplified).

        Args:
            timestamps: Array of datetime objects or hours

        Returns:
            Solar zenith angle (degrees)
        """
        # Simplified solar position calculation
        # For production use, consider using pvlib.solarposition

        # Assume timestamps are hours of day (0-23)
        if isinstance(timestamps[0], (int, float)):
            hour = timestamps
        else:
            hour = np.array([ts.hour for ts in timestamps])

        # Solar noon at 12:00
        hour_angle = 15 * (hour - 12)  # degrees

        # Simplified zenith angle (ignoring season, using average declination)
        lat = self.farm_location['lat']
        declination = 0  # average

        zenith = np.degrees(np.arccos(
            np.sin(np.radians(lat)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(lat)) * np.cos(np.radians(declination)) *
            np.cos(np.radians(hour_angle))
        ))

        return np.clip(zenith, 0, 90)


def main():
    """Example usage of SolarPowerConverter."""
    # Create converter for a 100 MW solar farm
    converter = SolarPowerConverter(
        panel_type='mono-Si-standard',
        capacity=100.0,
        tilt_angle=30.0,
        farm_location={'lat': 35.0, 'lon': -110.0}
    )

    # Example irradiance values through a day (W/m²)
    hours = np.arange(0, 24)
    # Sinusoidal pattern for daytime
    ghi = np.maximum(0, 1000 * np.sin(np.pi * (hours - 6) / 12))

    # Temperature pattern (K)
    temperature = 273.15 + 15 + 10 * np.sin(np.pi * (hours - 6) / 12)

    # Convert to power
    power = converter.irradiance_to_power(ghi, temperature)

    print("Solar Power Output Over 24 Hours:")
    print("=" * 50)
    for h, g, p in zip(hours, ghi, power):
        print(f"Hour {h:02d}:00  GHI: {g:6.1f} W/m²  →  Power: {p:6.2f} MW")

    print(f"\nDaily Statistics:")
    print(f"Peak Power: {power.max():.1f} MW")
    print(f"Daily Energy: {converter.daily_energy(power):.1f} MWh")
    print(f"Capacity Factor: {converter.capacity_factor(power):.2%}")


if __name__ == '__main__':
    main()
