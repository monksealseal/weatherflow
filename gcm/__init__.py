"""
Sophisticated General Circulation Model with Advanced Physics

A comprehensive atmospheric GCM with state-of-the-art parameterizations
for radiation, clouds, convection, turbulence, and surface processes.
"""

from .core.model import GCM
from .core.state import ModelState
from .grid.spherical import SphericalGrid

__version__ = "1.0.0"
__all__ = ["GCM", "ModelState", "SphericalGrid"]
