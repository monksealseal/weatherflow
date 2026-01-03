"""
Extreme Event Detection Algorithms

Detect and characterize extreme weather events in forecasts.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ExtremeEvent:
    """Container for an extreme weather event."""
    event_type: str
    start_time: datetime
    end_time: datetime
    duration_hours: float
    peak_value: float
    mean_value: float
    affected_area_km2: float
    center_lat: float
    center_lon: float
    metadata: Dict


class HeatwaveDetector:
    """
    Detect heatwave events in temperature forecasts.

    A heatwave is defined as a period when temperature exceeds a threshold
    for a minimum duration over a minimum spatial extent.

    Examples:
        >>> detector = HeatwaveDetector(temperature_threshold=35.0, duration_days=3)
        >>> events = detector.detect(temperature_forecast)
        >>> print(f"Found {len(events)} heatwaves")
    """

    def __init__(
        self,
        temperature_threshold: Optional[float] = None,
        percentile_threshold: Optional[float] = None,
        duration_days: int = 3,
        spatial_extent: float = 0.1,
        climatology: Optional[np.ndarray] = None
    ):
        """
        Initialize heatwave detector.

        Args:
            temperature_threshold: Absolute temperature threshold (°C)
            percentile_threshold: Percentile-based threshold (0-100)
            duration_days: Minimum duration in days
            spatial_extent: Minimum fraction of domain affected (0-1)
            climatology: Climatological temperature for percentile calculation
        """
        if temperature_threshold is None and percentile_threshold is None:
            raise ValueError("Must specify either temperature or percentile threshold")

        self.temperature_threshold = temperature_threshold
        self.percentile_threshold = percentile_threshold
        self.duration_days = duration_days
        self.spatial_extent = spatial_extent
        self.climatology = climatology

    def detect(
        self,
        temperature: np.ndarray,
        times: Optional[np.ndarray] = None,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None
    ) -> List[ExtremeEvent]:
        """
        Detect heatwave events.

        Args:
            temperature: Temperature array, shape (time, lat, lon) in Kelvin or Celsius
            times: Array of datetime objects
            lats: Latitude array
            lons: Longitude array

        Returns:
            List of detected heatwave events
        """
        # Convert to Celsius if needed (assume K if values > 200)
        if temperature.mean() > 200:
            temperature = temperature - 273.15

        # Determine threshold
        if self.percentile_threshold is not None:
            if self.climatology is None:
                # Use data itself as climatology
                threshold = np.percentile(temperature, self.percentile_threshold)
            else:
                threshold = np.percentile(self.climatology, self.percentile_threshold)
        else:
            threshold = self.temperature_threshold

        # Identify exceedances
        exceeds = temperature > threshold

        # Calculate affected area at each time
        affected_fraction = exceeds.mean(axis=(1, 2))

        # Find periods where spatial extent criterion is met
        spatial_criterion = affected_fraction > self.spatial_extent

        # Find periods where duration criterion is met
        min_duration_steps = int(self.duration_days * 24 / 6)  # Assuming 6-hourly data
        events = self._find_persistent_events(
            spatial_criterion,
            min_duration_steps,
            temperature,
            exceeds,
            times,
            lats,
            lons
        )

        return events

    def _find_persistent_events(
        self,
        criterion: np.ndarray,
        min_duration: int,
        temperature: np.ndarray,
        exceeds: np.ndarray,
        times: Optional[np.ndarray],
        lats: Optional[np.ndarray],
        lons: Optional[np.ndarray]
    ) -> List[ExtremeEvent]:
        """Find events that persist for minimum duration."""
        events = []
        in_event = False
        event_start = 0

        for i in range(len(criterion)):
            if criterion[i] and not in_event:
                # Event starts
                in_event = True
                event_start = i
            elif not criterion[i] and in_event:
                # Event ends
                event_duration = i - event_start
                if event_duration >= min_duration:
                    event = self._create_event(
                        event_start, i,
                        temperature[event_start:i],
                        exceeds[event_start:i],
                        times, lats, lons
                    )
                    events.append(event)
                in_event = False

        # Check if event extends to end of period
        if in_event and (len(criterion) - event_start) >= min_duration:
            event = self._create_event(
                event_start, len(criterion),
                temperature[event_start:],
                exceeds[event_start:],
                times, lats, lons
            )
            events.append(event)

        return events

    def _create_event(
        self,
        start_idx: int,
        end_idx: int,
        temperature: np.ndarray,
        exceeds: np.ndarray,
        times: Optional[np.ndarray],
        lats: Optional[np.ndarray],
        lons: Optional[np.ndarray]
    ) -> ExtremeEvent:
        """Create an ExtremeEvent object."""
        # Times
        if times is not None:
            start_time = times[start_idx]
            end_time = times[end_idx - 1]
        else:
            start_time = datetime(2000, 1, 1) + timedelta(hours=start_idx * 6)
            end_time = datetime(2000, 1, 1) + timedelta(hours=(end_idx - 1) * 6)

        duration_hours = (end_idx - start_idx) * 6  # Assuming 6-hourly

        # Temperature statistics
        peak_temp = temperature.max()
        mean_temp = temperature[exceeds].mean() if exceeds.any() else temperature.mean()

        # Spatial extent
        affected_area = self._calculate_area(exceeds.mean(axis=0), lats, lons)

        # Center of event
        if lats is not None and lons is not None:
            event_mask = exceeds.mean(axis=0) > 0.5
            if event_mask.any():
                lat_idx, lon_idx = np.where(event_mask)
                center_lat = lats[lat_idx].mean()
                center_lon = lons[lon_idx].mean()
            else:
                center_lat, center_lon = lats.mean(), lons.mean()
        else:
            center_lat, center_lon = 0.0, 0.0

        return ExtremeEvent(
            event_type='heatwave',
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            peak_value=peak_temp,
            mean_value=mean_temp,
            affected_area_km2=affected_area,
            center_lat=center_lat,
            center_lon=center_lon,
            metadata={
                'threshold': self.temperature_threshold or self.percentile_threshold,
                'threshold_type': 'absolute' if self.temperature_threshold else 'percentile'
            }
        )

    def _calculate_area(
        self,
        mask: np.ndarray,
        lats: Optional[np.ndarray],
        lons: Optional[np.ndarray]
    ) -> float:
        """Calculate area covered by event."""
        if lats is None or lons is None:
            # Use grid cell count as proxy
            return float(mask.sum())

        # Calculate area considering Earth's spherical geometry
        R_EARTH = 6371.0  # km
        dlat = np.abs(np.diff(lats).mean() if len(lats) > 1 else 1.0)
        dlon = np.abs(np.diff(lons).mean() if len(lons) > 1 else 1.0)

        # Area of each grid cell (km²)
        lat_rad = np.radians(lats)
        cell_area = R_EARTH**2 * np.radians(dlat) * np.radians(dlon) * np.cos(lat_rad[:, None])

        # Total affected area
        total_area = (mask * cell_area).sum()
        return float(total_area)


class AtmosphericRiverDetector:
    """
    Detect atmospheric river events.

    ARs are identified by high values of integrated vapor transport (IVT)
    organized in long, narrow corridors.

    Examples:
        >>> detector = AtmosphericRiverDetector(ivt_threshold=250)
        >>> events = detector.detect(water_vapor_flux)
        >>> for ar in events:
        ...     print(f"AR: Max IVT = {ar.peak_value:.0f} kg/m/s")
    """

    def __init__(
        self,
        ivt_threshold: float = 250.0,  # kg/m/s
        length_threshold: float = 2000.0,  # km
        width_threshold: float = 1000.0,  # km
        min_duration_hours: int = 6
    ):
        """
        Initialize AR detector.

        Args:
            ivt_threshold: Minimum IVT value (kg/m/s)
            length_threshold: Minimum length (km)
            width_threshold: Maximum width (km)
            min_duration_hours: Minimum duration
        """
        self.ivt_threshold = ivt_threshold
        self.length_threshold = length_threshold
        self.width_threshold = width_threshold
        self.min_duration_hours = min_duration_hours

    def detect(
        self,
        ivt: Optional[np.ndarray] = None,
        u_flux: Optional[np.ndarray] = None,
        v_flux: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None
    ) -> List[ExtremeEvent]:
        """
        Detect atmospheric rivers.

        Args:
            ivt: Integrated vapor transport (time, lat, lon) in kg/m/s
                 OR provide u_flux and v_flux separately
            u_flux: Eastward water vapor flux
            v_flux: Northward water vapor flux
            times: Array of datetime objects
            lats: Latitude array
            lons: Longitude array

        Returns:
            List of detected AR events
        """
        # Calculate IVT if components provided
        if ivt is None:
            if u_flux is None or v_flux is None:
                raise ValueError("Must provide either ivt or (u_flux, v_flux)")
            ivt = np.sqrt(u_flux**2 + v_flux**2)

        events = []

        # Detect ARs at each time step
        for t in range(ivt.shape[0]):
            ivt_snapshot = ivt[t]

            # Find regions exceeding threshold
            ar_mask = ivt_snapshot > self.ivt_threshold

            if not ar_mask.any():
                continue

            # Identify connected components (individual ARs)
            from scipy import ndimage
            labeled, num_features = ndimage.label(ar_mask)

            for feature_id in range(1, num_features + 1):
                feature_mask = labeled == feature_id

                # Calculate geometry
                geometry = self._calculate_geometry(feature_mask, lats, lons)

                # Check AR criteria
                if (geometry['length_km'] >= self.length_threshold and
                    geometry['width_km'] <= self.width_threshold):

                    # Calculate IVT statistics
                    peak_ivt = ivt_snapshot[feature_mask].max()
                    mean_ivt = ivt_snapshot[feature_mask].mean()

                    # Create event
                    time = times[t] if times is not None else datetime(2000, 1, 1)

                    event = ExtremeEvent(
                        event_type='atmospheric_river',
                        start_time=time,
                        end_time=time + timedelta(hours=6),
                        duration_hours=6.0,
                        peak_value=peak_ivt,
                        mean_value=mean_ivt,
                        affected_area_km2=geometry['area_km2'],
                        center_lat=geometry['center_lat'],
                        center_lon=geometry['center_lon'],
                        metadata={
                            'length_km': geometry['length_km'],
                            'width_km': geometry['width_km'],
                            'orientation': geometry['orientation']
                        }
                    )
                    events.append(event)

        return events

    def _calculate_geometry(
        self,
        mask: np.ndarray,
        lats: Optional[np.ndarray],
        lons: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate geometric properties of AR."""
        if lats is None or lons is None:
            # Simplified calculation in grid coordinates
            lat_extent = mask.sum(axis=0).max()
            lon_extent = mask.sum(axis=1).max()
            return {
                'length_km': float(max(lat_extent, lon_extent) * 100),
                'width_km': float(min(lat_extent, lon_extent) * 100),
                'area_km2': float(mask.sum() * 10000),
                'center_lat': 0.0,
                'center_lon': 0.0,
                'orientation': 0.0
            }

        # Find indices of AR points
        lat_idx, lon_idx = np.where(mask)

        if len(lat_idx) == 0:
            return {
                'length_km': 0.0,
                'width_km': 0.0,
                'area_km2': 0.0,
                'center_lat': 0.0,
                'center_lon': 0.0,
                'orientation': 0.0
            }

        # Center of AR
        center_lat = lats[lat_idx].mean()
        center_lon = lons[lon_idx].mean()

        # Approximate length and width using principal component analysis
        # Simplified: use lat/lon extent
        lat_extent_deg = lats[lat_idx].max() - lats[lat_idx].min()
        lon_extent_deg = lons[lon_idx].max() - lons[lon_idx].min()

        R_EARTH = 6371.0  # km
        lat_extent_km = lat_extent_deg * 111.0  # degrees to km
        lon_extent_km = lon_extent_deg * 111.0 * np.cos(np.radians(center_lat))

        length_km = max(lat_extent_km, lon_extent_km)
        width_km = min(lat_extent_km, lon_extent_km)

        # Orientation (simplified)
        orientation = np.degrees(np.arctan2(lat_extent_deg, lon_extent_deg))

        # Area
        dlat = np.abs(np.diff(lats).mean() if len(lats) > 1 else 1.0)
        dlon = np.abs(np.diff(lons).mean() if len(lons) > 1 else 1.0)
        lat_rad = np.radians(lats)
        cell_area = R_EARTH**2 * np.radians(dlat) * np.radians(dlon) * np.cos(lat_rad[:, None])
        area_km2 = (mask * cell_area).sum()

        return {
            'length_km': float(length_km),
            'width_km': float(width_km),
            'area_km2': float(area_km2),
            'center_lat': float(center_lat),
            'center_lon': float(center_lon),
            'orientation': float(orientation)
        }


class ExtremePrecipitationDetector:
    """
    Detect extreme precipitation events.

    Examples:
        >>> detector = ExtremePrecipitationDetector(threshold_mm=50.0)
        >>> events = detector.detect(daily_precipitation)
    """

    def __init__(
        self,
        threshold_mm: Optional[float] = None,
        percentile_threshold: Optional[float] = 99.0,
        min_area_km2: float = 10000.0
    ):
        """
        Initialize extreme precipitation detector.

        Args:
            threshold_mm: Absolute threshold (mm/day)
            percentile_threshold: Percentile-based threshold
            min_area_km2: Minimum affected area
        """
        self.threshold_mm = threshold_mm
        self.percentile_threshold = percentile_threshold
        self.min_area_km2 = min_area_km2

    def detect(
        self,
        precipitation: np.ndarray,
        times: Optional[np.ndarray] = None,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None
    ) -> List[ExtremeEvent]:
        """
        Detect extreme precipitation events.

        Args:
            precipitation: Precipitation array (time, lat, lon) in mm
            times: Array of datetime objects
            lats: Latitude array
            lons: Longitude array

        Returns:
            List of extreme precipitation events
        """
        # Determine threshold
        if self.threshold_mm is not None:
            threshold = self.threshold_mm
        else:
            threshold = np.percentile(precipitation, self.percentile_threshold)

        events = []

        # Detect events at each time
        for t in range(precipitation.shape[0]):
            precip_snapshot = precipitation[t]
            exceeds = precip_snapshot > threshold

            if not exceeds.any():
                continue

            # Calculate statistics
            peak_precip = precip_snapshot[exceeds].max()
            mean_precip = precip_snapshot[exceeds].mean()

            # Calculate area
            if lats is not None and lons is not None:
                R_EARTH = 6371.0
                dlat = np.abs(np.diff(lats).mean() if len(lats) > 1 else 1.0)
                dlon = np.abs(np.diff(lons).mean() if len(lons) > 1 else 1.0)
                lat_rad = np.radians(lats)
                cell_area = R_EARTH**2 * np.radians(dlat) * np.radians(dlon) * np.cos(lat_rad[:, None])
                area_km2 = (exceeds * cell_area).sum()
            else:
                area_km2 = exceeds.sum() * 100  # arbitrary units

            if area_km2 < self.min_area_km2:
                continue

            # Center
            if lats is not None and lons is not None:
                lat_idx, lon_idx = np.where(exceeds)
                center_lat = lats[lat_idx].mean() if len(lat_idx) > 0 else 0.0
                center_lon = lons[lon_idx].mean() if len(lon_idx) > 0 else 0.0
            else:
                center_lat, center_lon = 0.0, 0.0

            time = times[t] if times is not None else datetime(2000, 1, 1)

            event = ExtremeEvent(
                event_type='extreme_precipitation',
                start_time=time,
                end_time=time + timedelta(hours=24),
                duration_hours=24.0,
                peak_value=peak_precip,
                mean_value=mean_precip,
                affected_area_km2=float(area_km2),
                center_lat=float(center_lat),
                center_lon=float(center_lon),
                metadata={'threshold_mm': threshold}
            )
            events.append(event)

        return events


__all__ = [
    'ExtremeEvent',
    'HeatwaveDetector',
    'AtmosphericRiverDetector',
    'ExtremePrecipitationDetector',
]
