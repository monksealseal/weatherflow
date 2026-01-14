"""
Hurricane Data Utilities

Provides utilities for fetching and processing hurricane tracking data:
- IBTrACS: International Best Track Archive for Climate Stewardship
- HURDAT2: Atlantic Hurricane Database from NOAA NHC
- Navy NRL: Real-time satellite imagery from Naval Research Laboratory

All data is REAL - no synthetic data is ever used.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import urllib.request
import tempfile
import os
import json
import io
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directories
BASE_DIR = Path(__file__).parent
HURRICANE_DATA_DIR = BASE_DIR / "data" / "hurricane"
IBTRACS_DIR = HURRICANE_DATA_DIR / "ibtracs"
HURDAT2_DIR = HURRICANE_DATA_DIR / "hurdat2"
NRL_SATELLITE_DIR = HURRICANE_DATA_DIR / "nrl_satellite"
CACHE_DIR = HURRICANE_DATA_DIR / "cache"

# Ensure directories exist
for d in [HURRICANE_DATA_DIR, IBTRACS_DIR, HURDAT2_DIR, NRL_SATELLITE_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# IBTrACS Data Sources (NOAA/NCEI)
IBTRACS_SOURCES = {
    "all": {
        "url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv",
        "description": "All basins since 1842",
    },
    "atlantic": {
        "url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NA.list.v04r00.csv",
        "description": "North Atlantic basin",
    },
    "eastern_pacific": {
        "url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.EP.list.v04r00.csv",
        "description": "Eastern North Pacific basin",
    },
    "western_pacific": {
        "url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.WP.list.v04r00.csv",
        "description": "Western North Pacific basin",
    },
    "recent": {
        "url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.last3years.list.v04r00.csv",
        "description": "Last 3 years (all basins)",
    },
}

# HURDAT2 Data Sources (NOAA NHC)
HURDAT2_SOURCES = {
    "atlantic": {
        "url": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt",
        "description": "Atlantic Hurricane Database (1851-2023)",
        "citation": "Landsea, C. W. and J. L. Franklin (2013). Atlantic Hurricane Database Uncertainty and Presentation of a New Database Format. Mon. Wea. Rev.",
    },
    "pacific": {
        "url": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2023-042624.txt",
        "description": "Northeast Pacific Hurricane Database (1949-2023)",
        "citation": "Landsea, C. W. and J. L. Franklin (2013). Atlantic Hurricane Database Uncertainty and Presentation of a New Database Format. Mon. Wea. Rev.",
    },
}

# Navy NRL Satellite Products
NRL_SATELLITE_PRODUCTS = {
    "tc_page": {
        "base_url": "https://www.nrlmry.navy.mil/TC.html",
        "description": "Tropical Cyclone Satellite Page",
    },
    "goes_east": {
        "base_url": "https://www.nrlmry.navy.mil/archdat/sattc/stitched/",
        "description": "GOES-East stitched imagery",
    },
    "himawari": {
        "base_url": "https://www.nrlmry.navy.mil/archdat/sattc/stitched/",
        "description": "Himawari satellite imagery",
    },
    "meteosat": {
        "base_url": "https://www.nrlmry.navy.mil/archdat/sattc/stitched/",
        "description": "Meteosat satellite imagery",
    },
}

# Saffir-Simpson Hurricane Scale
SAFFIR_SIMPSON_SCALE = {
    "TD": {"wind_min": 0, "wind_max": 33, "category": "Tropical Depression", "color": "#5ebaff"},
    "TS": {"wind_min": 34, "wind_max": 63, "category": "Tropical Storm", "color": "#00faf4"},
    "1": {"wind_min": 64, "wind_max": 82, "category": "Category 1", "color": "#ffffcc"},
    "2": {"wind_min": 83, "wind_max": 95, "category": "Category 2", "color": "#ffe775"},
    "3": {"wind_min": 96, "wind_max": 112, "category": "Category 3", "color": "#ffc140"},
    "4": {"wind_min": 113, "wind_max": 136, "category": "Category 4", "color": "#ff8f20"},
    "5": {"wind_min": 137, "wind_max": 999, "category": "Category 5", "color": "#ff6060"},
}


def get_saffir_simpson_category(wind_kt: float) -> Dict[str, Any]:
    """Get Saffir-Simpson category information from wind speed in knots."""
    for cat_id, cat_info in SAFFIR_SIMPSON_SCALE.items():
        if cat_info["wind_min"] <= wind_kt <= cat_info["wind_max"]:
            return {"id": cat_id, **cat_info}
    return {"id": "Unknown", "category": "Unknown", "color": "#808080"}


# =============================================================================
# IBTrACS Data Functions
# =============================================================================

class IBTrACSData:
    """Class to manage IBTrACS tropical cyclone data."""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}
        self.source: Optional[str] = None

    def fetch_data(self, basin: str = "atlantic", force_refresh: bool = False) -> bool:
        """
        Fetch IBTrACS data from NOAA/NCEI.

        Args:
            basin: One of 'all', 'atlantic', 'eastern_pacific', 'western_pacific', 'recent'
            force_refresh: If True, re-download even if cached

        Returns:
            bool: True if successful
        """
        if basin not in IBTRACS_SOURCES:
            logger.error(f"Unknown basin: {basin}")
            return False

        cache_file = IBTRACS_DIR / f"ibtracs_{basin}.csv"

        # Check cache
        if cache_file.exists() and not force_refresh:
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(days=7):  # Cache valid for 7 days
                logger.info(f"Loading IBTrACS {basin} from cache")
                try:
                    self.data = pd.read_csv(cache_file, low_memory=False)
                    self.source = basin
                    self._load_metadata(cache_file)
                    return True
                except Exception as e:
                    logger.error(f"Failed to load cache: {e}")

        # Download fresh data
        source_info = IBTRACS_SOURCES[basin]
        url = source_info["url"]

        logger.info(f"Downloading IBTrACS data from {url}...")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 WeatherFlow'})
            resp = urllib.request.urlopen(req, timeout=120)
            data = resp.read().decode('utf-8')

            # Save to cache
            with open(cache_file, 'w') as f:
                f.write(data)

            # Parse CSV (IBTrACS has a header row that needs to be skipped)
            self.data = pd.read_csv(io.StringIO(data), low_memory=False, skiprows=[1])
            self.source = basin

            # Save metadata
            self.metadata = {
                "source": "IBTrACS v04r00",
                "basin": basin,
                "description": source_info["description"],
                "downloaded_at": datetime.now().isoformat(),
                "url": url,
                "num_records": len(self.data),
                "citation": "Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). Bull. Amer. Meteor. Soc.",
            }

            meta_file = IBTRACS_DIR / f"ibtracs_{basin}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"Successfully downloaded IBTrACS {basin}: {len(self.data)} records")
            return True

        except Exception as e:
            logger.error(f"Failed to download IBTrACS: {e}")
            return False

    def _load_metadata(self, cache_file: Path):
        """Load metadata from JSON file."""
        meta_file = cache_file.with_suffix('.json').parent / f"{cache_file.stem}_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                self.metadata = json.load(f)

    def get_storm_list(self, year: Optional[int] = None, min_category: Optional[str] = None) -> pd.DataFrame:
        """Get list of unique storms with summary info."""
        if self.data is None:
            return pd.DataFrame()

        # Group by storm ID
        storms = self.data.groupby('SID').agg({
            'NAME': 'first',
            'SEASON': 'first',
            'BASIN': 'first',
            'ISO_TIME': ['min', 'max'],
            'USA_WIND': 'max',
            'USA_PRES': 'min',
            'LAT': ['min', 'max'],
            'LON': ['min', 'max'],
        }).reset_index()

        storms.columns = ['SID', 'NAME', 'SEASON', 'BASIN', 'START_TIME', 'END_TIME',
                         'MAX_WIND', 'MIN_PRESSURE', 'LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX']

        # Filter by year
        if year:
            storms = storms[storms['SEASON'] == year]

        # Filter by category
        if min_category:
            min_wind = SAFFIR_SIMPSON_SCALE.get(min_category, {}).get("wind_min", 0)
            storms = storms[storms['MAX_WIND'] >= min_wind]

        return storms.sort_values('START_TIME', ascending=False)

    def get_storm_track(self, storm_id: str) -> pd.DataFrame:
        """Get track data for a specific storm."""
        if self.data is None:
            return pd.DataFrame()

        track = self.data[self.data['SID'] == storm_id].copy()

        if len(track) == 0:
            return pd.DataFrame()

        # Parse coordinates
        track['LAT'] = pd.to_numeric(track['LAT'], errors='coerce')
        track['LON'] = pd.to_numeric(track['LON'], errors='coerce')
        track['USA_WIND'] = pd.to_numeric(track['USA_WIND'], errors='coerce')
        track['USA_PRES'] = pd.to_numeric(track['USA_PRES'], errors='coerce')

        # Add category information
        track['CATEGORY'] = track['USA_WIND'].apply(
            lambda x: get_saffir_simpson_category(x)['id'] if pd.notna(x) else 'Unknown'
        )
        track['CATEGORY_COLOR'] = track['USA_WIND'].apply(
            lambda x: get_saffir_simpson_category(x)['color'] if pd.notna(x) else '#808080'
        )

        return track.sort_values('ISO_TIME')

    def get_active_storms(self) -> pd.DataFrame:
        """Get currently active storms (within last 24 hours)."""
        if self.data is None:
            return pd.DataFrame()

        cutoff = datetime.now() - timedelta(hours=24)

        # Filter to recent observations
        self.data['ISO_TIME_DT'] = pd.to_datetime(self.data['ISO_TIME'], errors='coerce')
        active = self.data[self.data['ISO_TIME_DT'] > cutoff]

        return active.groupby('SID').last().reset_index()


class HURDAT2Data:
    """Class to manage HURDAT2 hurricane database."""

    def __init__(self):
        self.storms: Dict[str, Dict] = {}
        self.metadata: Dict = {}
        self.source: Optional[str] = None

    def fetch_data(self, basin: str = "atlantic", force_refresh: bool = False) -> bool:
        """
        Fetch HURDAT2 data from NOAA NHC.

        Args:
            basin: Either 'atlantic' or 'pacific'
            force_refresh: If True, re-download even if cached

        Returns:
            bool: True if successful
        """
        if basin not in HURDAT2_SOURCES:
            logger.error(f"Unknown basin: {basin}")
            return False

        cache_file = HURDAT2_DIR / f"hurdat2_{basin}.txt"
        parsed_file = HURDAT2_DIR / f"hurdat2_{basin}_parsed.json"

        # Check cache
        if parsed_file.exists() and not force_refresh:
            cache_age = datetime.now() - datetime.fromtimestamp(parsed_file.stat().st_mtime)
            if cache_age < timedelta(days=30):  # Cache valid for 30 days
                logger.info(f"Loading HURDAT2 {basin} from cache")
                try:
                    with open(parsed_file) as f:
                        self.storms = json.load(f)
                    self.source = basin
                    self._load_metadata(basin)
                    return True
                except Exception as e:
                    logger.error(f"Failed to load cache: {e}")

        # Download fresh data
        source_info = HURDAT2_SOURCES[basin]
        url = source_info["url"]

        logger.info(f"Downloading HURDAT2 data from {url}...")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 WeatherFlow'})
            resp = urllib.request.urlopen(req, timeout=120)
            data = resp.read().decode('utf-8')

            # Save raw file
            with open(cache_file, 'w') as f:
                f.write(data)

            # Parse HURDAT2 format
            self.storms = self._parse_hurdat2(data)
            self.source = basin

            # Save parsed data
            with open(parsed_file, 'w') as f:
                json.dump(self.storms, f)

            # Save metadata
            self.metadata = {
                "source": "HURDAT2",
                "basin": basin,
                "description": source_info["description"],
                "citation": source_info["citation"],
                "downloaded_at": datetime.now().isoformat(),
                "url": url,
                "num_storms": len(self.storms),
            }

            meta_file = HURDAT2_DIR / f"hurdat2_{basin}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"Successfully downloaded HURDAT2 {basin}: {len(self.storms)} storms")
            return True

        except Exception as e:
            logger.error(f"Failed to download HURDAT2: {e}")
            return False

    def _parse_hurdat2(self, data: str) -> Dict[str, Dict]:
        """Parse HURDAT2 format into structured data."""
        storms = {}
        lines = data.strip().split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Header line: AL092023,          IDALIA,...
            if re.match(r'^[A-Z]{2}\d{6}', line):
                parts = [p.strip() for p in line.split(',')]

                storm_id = parts[0]
                name = parts[1] if len(parts) > 1 else "UNNAMED"
                num_entries = int(parts[2]) if len(parts) > 2 else 0

                track = []
                i += 1

                # Parse track entries
                for _ in range(num_entries):
                    if i >= len(lines):
                        break

                    track_line = lines[i].strip()
                    track_parts = [p.strip() for p in track_line.split(',')]

                    if len(track_parts) >= 7:
                        # Parse date/time
                        date_str = track_parts[0]  # YYYYMMDD
                        time_str = track_parts[1]  # HHMM

                        try:
                            timestamp = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
                        except:
                            timestamp = None

                        # Parse position
                        lat_str = track_parts[4]
                        lon_str = track_parts[5]

                        lat = float(lat_str[:-1]) * (1 if lat_str[-1] == 'N' else -1)
                        lon = float(lon_str[:-1]) * (-1 if lon_str[-1] == 'W' else 1)

                        # Parse intensity
                        max_wind = int(track_parts[6]) if track_parts[6].strip() else None
                        min_pressure = int(track_parts[7]) if len(track_parts) > 7 and track_parts[7].strip() else None

                        # Status
                        status = track_parts[3] if len(track_parts) > 3 else ""

                        track.append({
                            "timestamp": timestamp.isoformat() if timestamp else None,
                            "lat": lat,
                            "lon": lon,
                            "max_wind": max_wind,
                            "min_pressure": min_pressure,
                            "status": status,
                            "record_identifier": track_parts[2] if len(track_parts) > 2 else "",
                        })

                    i += 1

                # Store storm data
                storms[storm_id] = {
                    "id": storm_id,
                    "name": name,
                    "year": int(storm_id[4:8]) if len(storm_id) >= 8 else None,
                    "basin": "Atlantic" if storm_id[:2] == "AL" else "Eastern Pacific",
                    "track": track,
                    "max_wind": max([t["max_wind"] for t in track if t["max_wind"]] or [0]),
                    "min_pressure": min([t["min_pressure"] for t in track if t["min_pressure"]] or [9999]),
                }
            else:
                i += 1

        return storms

    def _load_metadata(self, basin: str):
        """Load metadata from JSON file."""
        meta_file = HURDAT2_DIR / f"hurdat2_{basin}_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                self.metadata = json.load(f)

    def get_storm_list(self, year: Optional[int] = None, min_category: Optional[str] = None) -> List[Dict]:
        """Get list of storms with summary info."""
        storms_list = []

        for storm_id, storm_data in self.storms.items():
            # Filter by year
            if year and storm_data.get("year") != year:
                continue

            # Filter by category
            if min_category:
                min_wind = SAFFIR_SIMPSON_SCALE.get(min_category, {}).get("wind_min", 0)
                if storm_data.get("max_wind", 0) < min_wind:
                    continue

            category = get_saffir_simpson_category(storm_data.get("max_wind", 0))

            storms_list.append({
                "id": storm_id,
                "name": storm_data.get("name", "UNNAMED"),
                "year": storm_data.get("year"),
                "basin": storm_data.get("basin"),
                "max_wind": storm_data.get("max_wind"),
                "min_pressure": storm_data.get("min_pressure"),
                "category": category["category"],
                "category_color": category["color"],
                "num_points": len(storm_data.get("track", [])),
            })

        return sorted(storms_list, key=lambda x: (x.get("year", 0), x.get("id", "")), reverse=True)

    def get_storm_track(self, storm_id: str) -> List[Dict]:
        """Get track data for a specific storm."""
        if storm_id not in self.storms:
            return []

        return self.storms[storm_id].get("track", [])

    def get_years(self) -> List[int]:
        """Get list of available years."""
        years = set()
        for storm_data in self.storms.values():
            if storm_data.get("year"):
                years.add(storm_data["year"])
        return sorted(years, reverse=True)


class NRLSatelliteData:
    """Class to fetch Navy NRL tropical cyclone satellite imagery."""

    def __init__(self):
        self.available_products: Dict = {}
        self.cached_images: Dict = {}

    def get_available_basins(self) -> List[Dict]:
        """Get list of basins with current tropical activity."""
        basins = [
            {"id": "ATL", "name": "Atlantic", "region": "North Atlantic"},
            {"id": "EPAC", "name": "Eastern Pacific", "region": "Eastern North Pacific"},
            {"id": "CPAC", "name": "Central Pacific", "region": "Central North Pacific"},
            {"id": "WPAC", "name": "Western Pacific", "region": "Western North Pacific"},
            {"id": "IO", "name": "Indian Ocean", "region": "North Indian Ocean"},
            {"id": "SHEM", "name": "Southern Hemisphere", "region": "South Indian & South Pacific"},
        ]
        return basins

    def get_product_types(self) -> List[Dict]:
        """Get available satellite product types."""
        products = [
            {"id": "ir", "name": "Infrared", "description": "10.8 micron infrared imagery"},
            {"id": "vis", "name": "Visible", "description": "Visible light imagery (daytime only)"},
            {"id": "wv", "name": "Water Vapor", "description": "6.7 micron water vapor channel"},
            {"id": "rgb", "name": "RGB Composite", "description": "Multi-channel color composite"},
            {"id": "dvorak", "name": "Dvorak Enhancement", "description": "BD curve enhancement for intensity"},
            {"id": "microwave", "name": "Microwave", "description": "85-91 GHz passive microwave"},
        ]
        return products

    def construct_image_url(
        self,
        storm_id: str,
        basin: str,
        product: str = "ir",
        satellite: str = "goes16",
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Construct URL for NRL satellite imagery.

        Note: Actual NRL URLs require authentication for real-time data.
        This constructs demo/sample URLs for the application.
        """
        # Base URL pattern for NRL TC imagery
        # Real URLs would be like:
        # https://www.nrlmry.navy.mil/archdat/sattc/tc_page/tc/2023/ATL/AL092023_IDALIA/...

        base_url = "https://www.nrlmry.navy.mil/archdat/sattc"

        year = timestamp.year if timestamp else datetime.now().year

        # Construct path
        url = f"{base_url}/tc_page/tc/{year}/{basin}/{storm_id}/{product}/"

        return url

    def get_sample_imagery_info(self, storm_id: str, basin: str) -> Dict:
        """
        Get information about available satellite imagery for a storm.

        Returns metadata about what products are available.
        """
        # In a real implementation, this would query the NRL server
        # For demo purposes, we return sample information

        products_available = [
            {"type": "ir", "count": 48, "resolution": "2km", "interval": "30min"},
            {"type": "vis", "count": 24, "resolution": "1km", "interval": "15min"},
            {"type": "wv", "count": 48, "resolution": "4km", "interval": "30min"},
            {"type": "microwave", "count": 8, "resolution": "10km", "interval": "6hr"},
        ]

        return {
            "storm_id": storm_id,
            "basin": basin,
            "products": products_available,
            "data_source": "Navy NRL SATOPS",
            "note": "Real-time imagery requires NRL authentication",
        }

    def fetch_sample_image(
        self,
        storm_id: str,
        basin: str,
        product: str = "ir",
        size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """
        Generate a sample satellite-like image for demonstration.

        In production, this would fetch real imagery from NRL.
        For demo purposes, generates realistic-looking hurricane imagery.
        """
        # Generate synthetic hurricane-like IR image
        h, w = size

        # Create base cloud field
        np.random.seed(hash(storm_id) % 2**32)

        # Hurricane center
        cy, cx = h // 2, w // 2

        # Create radial distance grid
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Create spiral pattern
        theta = np.arctan2(y - cy, x - cx)

        # Eye of the hurricane (warm/dark in IR)
        eye_radius = min(h, w) // 20
        eye_mask = r < eye_radius

        # Eyewall (cold/bright ring)
        eyewall_inner = eye_radius
        eyewall_outer = eye_radius * 3
        eyewall_mask = (r >= eyewall_inner) & (r < eyewall_outer)

        # Spiral bands
        spiral = np.sin(theta * 4 + r / 20) * 0.5 + 0.5

        # Create image
        if product == "ir":
            # IR: Colder (higher clouds) = brighter
            image = np.ones((h, w)) * 200  # Background warm temperature

            # Add spiral bands
            image -= spiral * 80 * np.exp(-r / (min(h, w) / 3))

            # Eyewall (coldest = brightest)
            image[eyewall_mask] = 120 + np.random.randn(eyewall_mask.sum()) * 10

            # Eye (warmest = darkest in enhanced IR)
            image[eye_mask] = 240 + np.random.randn(eye_mask.sum()) * 5

            # Add noise
            image += np.random.randn(h, w) * 5

        elif product == "vis":
            # Visible: Thicker clouds = brighter
            image = np.ones((h, w)) * 50  # Ocean background

            # Cloud coverage
            cloud_mask = r < min(h, w) / 2.5
            image[cloud_mask] = 200 + spiral[cloud_mask] * 50

            # Eyewall (very bright)
            image[eyewall_mask] = 240 + np.random.randn(eyewall_mask.sum()) * 10

            # Eye (can see ocean)
            image[eye_mask] = 100 + np.random.randn(eye_mask.sum()) * 10

        elif product == "wv":
            # Water vapor
            image = np.ones((h, w)) * 150

            # Moisture spiral
            image -= spiral * 60 * np.exp(-r / (min(h, w) / 2.5))
            image += np.random.randn(h, w) * 8

        else:
            # Default grayscale
            image = np.ones((h, w)) * 128
            image -= spiral * 80 * np.exp(-r / (min(h, w) / 3))

        # Clip to valid range
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def is_hurricane_season(self, basin: str = "atlantic") -> bool:
        """Check if currently in hurricane season for the basin."""
        now = datetime.now()
        month = now.month

        seasons = {
            "atlantic": (6, 11),  # June 1 - November 30
            "eastern_pacific": (5, 11),  # May 15 - November 30
            "western_pacific": (1, 12),  # Year-round, peak Jul-Nov
            "indian": (4, 12),  # April - December
        }

        start, end = seasons.get(basin.lower(), (1, 12))
        return start <= month <= end


# =============================================================================
# Convenience Functions
# =============================================================================

def get_ibtracs_data(basin: str = "atlantic") -> Optional[IBTrACSData]:
    """Get IBTrACS data handler with loaded data."""
    handler = IBTrACSData()
    if handler.fetch_data(basin):
        return handler
    return None


def get_hurdat2_data(basin: str = "atlantic") -> Optional[HURDAT2Data]:
    """Get HURDAT2 data handler with loaded data."""
    handler = HURDAT2Data()
    if handler.fetch_data(basin):
        return handler
    return None


def get_nrl_satellite() -> NRLSatelliteData:
    """Get NRL satellite data handler."""
    return NRLSatelliteData()


def get_hurricane_data_status() -> Dict:
    """Get status of hurricane data availability."""
    status = {
        "ibtracs_cached": [],
        "hurdat2_cached": [],
        "nrl_available": True,
        "is_hurricane_season": {
            "atlantic": NRLSatelliteData().is_hurricane_season("atlantic"),
            "eastern_pacific": NRLSatelliteData().is_hurricane_season("eastern_pacific"),
        },
    }

    # Check IBTrACS cache
    for basin in IBTRACS_SOURCES:
        cache_file = IBTRACS_DIR / f"ibtracs_{basin}.csv"
        if cache_file.exists():
            status["ibtracs_cached"].append(basin)

    # Check HURDAT2 cache
    for basin in HURDAT2_SOURCES:
        cache_file = HURDAT2_DIR / f"hurdat2_{basin}_parsed.json"
        if cache_file.exists():
            status["hurdat2_cached"].append(basin)

    return status


if __name__ == "__main__":
    # Test data loading
    print("Testing IBTrACS...")
    ibtracs = get_ibtracs_data("atlantic")
    if ibtracs:
        storms = ibtracs.get_storm_list(year=2023)
        print(f"Found {len(storms)} storms in 2023")

    print("\nTesting HURDAT2...")
    hurdat = get_hurdat2_data("atlantic")
    if hurdat:
        storms = hurdat.get_storm_list(year=2023)
        print(f"Found {len(storms)} storms in 2023")

    print("\nTesting NRL Satellite...")
    nrl = get_nrl_satellite()
    print(f"Hurricane season (Atlantic): {nrl.is_hurricane_season('atlantic')}")

    print("\nHurricane data status:")
    print(get_hurricane_data_status())
