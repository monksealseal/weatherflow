"""
Wind Power Conversion Module

Convert wind speed forecasts to power output using turbine power curves.
"""

import numpy as np
import torch
from typing import Dict, Optional, List, Union
from dataclasses import dataclass


@dataclass
class TurbineSpec:
    """Specification for a wind turbine."""
    name: str
    rated_power: float  # MW
    cut_in_speed: float  # m/s
    rated_speed: float  # m/s
    cut_out_speed: float  # m/s
    hub_height: float  # meters
    rotor_diameter: float  # meters
    power_curve: Optional[np.ndarray] = None  # (wind_speed, power) pairs


# Standard turbine specifications
TURBINE_LIBRARY = {
    'IEA-3.4MW': TurbineSpec(
        name='IEA 3.4 MW',
        rated_power=3.4,
        cut_in_speed=3.0,
        rated_speed=13.0,
        cut_out_speed=25.0,
        hub_height=110.0,
        rotor_diameter=130.0
    ),
    'NREL-5MW': TurbineSpec(
        name='NREL 5 MW Reference',
        rated_power=5.0,
        cut_in_speed=3.0,
        rated_speed=11.4,
        cut_out_speed=25.0,
        hub_height=90.0,
        rotor_diameter=126.0
    ),
    'Vestas-V90': TurbineSpec(
        name='Vestas V90 2.0 MW',
        rated_power=2.0,
        cut_in_speed=4.0,
        rated_speed=15.0,
        cut_out_speed=25.0,
        hub_height=80.0,
        rotor_diameter=90.0
    )
}


class WindPowerConverter:
    """
    Convert wind forecasts to power output.

    Examples:
        >>> converter = WindPowerConverter(turbine_type='IEA-3.4MW', num_turbines=50)
        >>> wind_speed = np.array([5.0, 10.0, 15.0, 20.0])  # m/s
        >>> power = converter.wind_speed_to_power(wind_speed)
        >>> print(f"Total farm power: {power.sum():.1f} MW")
    """

    def __init__(
        self,
        turbine_type: str = 'IEA-3.4MW',
        num_turbines: int = 1,
        hub_height: Optional[float] = None,
        farm_location: Optional[Dict[str, float]] = None,
        array_efficiency: float = 0.95,
        availability: float = 0.97
    ):
        """
        Initialize wind power converter.

        Args:
            turbine_type: Turbine model from TURBINE_LIBRARY
            num_turbines: Number of turbines in the farm
            hub_height: Override hub height (meters)
            farm_location: {'lat': latitude, 'lon': longitude}
            array_efficiency: Array losses factor (0-1)
            availability: Turbine availability factor (0-1)
        """
        if turbine_type not in TURBINE_LIBRARY:
            raise ValueError(f"Unknown turbine type: {turbine_type}")

        self.turbine = TURBINE_LIBRARY[turbine_type]
        self.num_turbines = num_turbines
        self.array_efficiency = array_efficiency
        self.availability = availability

        if hub_height is not None:
            self.turbine.hub_height = hub_height

        self.farm_location = farm_location or {'lat': 45.0, 'lon': -95.0}

        # Generate power curve if not provided
        if self.turbine.power_curve is None:
            self.turbine.power_curve = self._generate_power_curve()

    def _generate_power_curve(self) -> np.ndarray:
        """Generate a generic power curve for the turbine."""
        # Wind speeds from 0 to cut-out
        wind_speeds = np.linspace(0, self.turbine.cut_out_speed, 100)
        powers = np.zeros_like(wind_speeds)

        for i, ws in enumerate(wind_speeds):
            if ws < self.turbine.cut_in_speed:
                powers[i] = 0.0
            elif ws >= self.turbine.cut_out_speed:
                powers[i] = 0.0
            elif ws >= self.turbine.rated_speed:
                powers[i] = self.turbine.rated_power
            else:
                # Cubic interpolation between cut-in and rated
                normalized = (ws - self.turbine.cut_in_speed) / \
                           (self.turbine.rated_speed - self.turbine.cut_in_speed)
                powers[i] = self.turbine.rated_power * normalized ** 3

        return np.column_stack([wind_speeds, powers])

    def wind_speed_to_power(
        self,
        wind_speed: Union[np.ndarray, float],
        temperature: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert wind speed to power output.

        Args:
            wind_speed: Wind speed (m/s), scalar or array
            temperature: Air temperature (K) for density correction
            pressure: Surface pressure (Pa) for density correction

        Returns:
            Power output (MW) per turbine
        """
        wind_speed = np.atleast_1d(wind_speed)

        # Apply density correction if temperature and pressure provided
        if temperature is not None and pressure is not None:
            density_ratio = self._density_correction(temperature, pressure)
        else:
            density_ratio = 1.0

        # Interpolate power curve
        curve_ws = self.turbine.power_curve[:, 0]
        curve_power = self.turbine.power_curve[:, 1]

        power = np.interp(wind_speed, curve_ws, curve_power)

        # Apply density correction (power scales with density)
        power = power * density_ratio

        return power

    def _density_correction(
        self,
        temperature: np.ndarray,
        pressure: np.ndarray
    ) -> np.ndarray:
        """
        Calculate air density correction factor.

        Args:
            temperature: Air temperature (K)
            pressure: Surface pressure (Pa)

        Returns:
            Density ratio relative to standard conditions
        """
        # Standard conditions: 15°C, 101325 Pa
        R_SPECIFIC = 287.05  # J/(kg·K)
        rho_standard = 101325 / (R_SPECIFIC * 288.15)
        rho = pressure / (R_SPECIFIC * temperature)
        return rho / rho_standard

    def adjust_height(
        self,
        wind_speed: np.ndarray,
        measurement_height: float,
        terrain_roughness: float = 0.03
    ) -> np.ndarray:
        """
        Adjust wind speed to hub height using power law.

        Args:
            wind_speed: Wind speed at measurement height (m/s)
            measurement_height: Height of wind measurement (m)
            terrain_roughness: Roughness length (m)

        Returns:
            Wind speed at hub height (m/s)
        """
        # Power law exponent
        alpha = 0.143 * (terrain_roughness ** 0.25)

        # Extrapolate to hub height
        ws_hub = wind_speed * (self.turbine.hub_height / measurement_height) ** alpha

        return ws_hub

    def farm_power(
        self,
        wind_speed: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate total farm power output.

        Args:
            wind_speed: Wind speed (m/s)
            **kwargs: Additional arguments for wind_speed_to_power

        Returns:
            Total farm power (MW)
        """
        # Single turbine power
        single_power = self.wind_speed_to_power(wind_speed, **kwargs)

        # Farm power with losses
        farm_power = (
            single_power *
            self.num_turbines *
            self.array_efficiency *
            self.availability
        )

        return farm_power

    def capacity_factor(
        self,
        wind_speed: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate capacity factor over a time period.

        Args:
            wind_speed: Time series of wind speed (m/s)
            **kwargs: Additional arguments for farm_power

        Returns:
            Capacity factor (0-1)
        """
        power_output = self.farm_power(wind_speed, **kwargs)
        max_capacity = self.turbine.rated_power * self.num_turbines
        return np.mean(power_output) / max_capacity

    def convert_forecast(
        self,
        weather_forecast: Dict[str, np.ndarray],
        measurement_height: float = 10.0
    ) -> Dict[str, np.ndarray]:
        """
        Convert weather forecast to power forecast.

        Args:
            weather_forecast: Dictionary containing:
                - 'u': U-component of wind (m/s)
                - 'v': V-component of wind (m/s)
                - Optional: 't': Temperature (K)
                - Optional: 'sp': Surface pressure (Pa)
            measurement_height: Height of wind measurements (m)

        Returns:
            Dictionary with power forecasts and statistics
        """
        # Calculate wind speed from components
        u = weather_forecast['u']
        v = weather_forecast['v']
        wind_speed = np.sqrt(u**2 + v**2)

        # Adjust to hub height
        wind_speed_hub = self.adjust_height(wind_speed, measurement_height)

        # Convert to power
        temp = weather_forecast.get('t')
        pressure = weather_forecast.get('sp')

        power = self.farm_power(wind_speed_hub, temperature=temp, pressure=pressure)

        # Calculate wind direction
        wind_direction = np.degrees(np.arctan2(u, v)) % 360

        return {
            'power_mw': power,
            'wind_speed_hub': wind_speed_hub,
            'wind_direction': wind_direction,
            'capacity_factor': power / (self.turbine.rated_power * self.num_turbines),
            'mean_power': np.mean(power),
            'std_power': np.std(power),
            'max_power': np.max(power),
        }


def main():
    """Example usage of WindPowerConverter."""
    # Create converter for a 50-turbine wind farm
    converter = WindPowerConverter(
        turbine_type='IEA-3.4MW',
        num_turbines=50,
        array_efficiency=0.95
    )

    # Example wind speeds (m/s)
    wind_speeds = np.array([0, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30])

    # Convert to power
    power = converter.wind_speed_to_power(wind_speeds)

    print("Wind Power Curve:")
    print("=" * 40)
    for ws, p in zip(wind_speeds, power):
        print(f"Wind Speed: {ws:5.1f} m/s  →  Power: {p:6.2f} MW")

    # Total farm power
    farm_power = converter.farm_power(wind_speeds)
    print(f"\nTotal Farm Power (50 turbines):")
    print(f"Peak: {farm_power.max():.1f} MW")
    print(f"Capacity: {converter.turbine.rated_power * converter.num_turbines:.1f} MW")


if __name__ == '__main__':
    main()
