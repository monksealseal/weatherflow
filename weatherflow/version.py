# weatherflow/version.py
"""Version information for the WeatherFlow package."""
import re

__version__ = "0.4.2"

# Function to get version
def get_version():
    """Return the package version as a string."""
    return __version__

# Version components - handle pre-release versions (e.g., "0.4.2rc1", "0.4.2.post1")
def _parse_version_component(component: str) -> int:
    """Extract numeric part from version component, ignoring pre-release suffixes."""
    match = re.match(r'(\d+)', component)
    return int(match.group(1)) if match else 0

_version_parts = __version__.split('.')
VERSION_MAJOR = _parse_version_component(_version_parts[0]) if len(_version_parts) > 0 else 0
VERSION_MINOR = _parse_version_component(_version_parts[1]) if len(_version_parts) > 1 else 0
VERSION_PATCH = _parse_version_component(_version_parts[2]) if len(_version_parts) > 2 else 0

# Version string for debugging
VERSION_STRING = f"WeatherFlow v{__version__}"

# Version as tuple
VERSION_TUPLE = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)
