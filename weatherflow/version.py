# weatherflow/version.py
"""Version information for the WeatherFlow package."""

__version__ = "0.3.3"

# Function to get version
def get_version():
    """Return the package version as a string."""
    return __version__

# Version components
VERSION_MAJOR = int(__version__.split('.')[0])
VERSION_MINOR = int(__version__.split('.')[1])
VERSION_PATCH = int(__version__.split('.')[2])

# Version string for debugging
VERSION_STRING = f"WeatherFlow v{__version__}"

# Version as tuple
VERSION_TUPLE = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)