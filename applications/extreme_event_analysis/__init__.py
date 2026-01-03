"""
Extreme Event Analysis Applications

Tools for detecting and analyzing extreme weather events.
"""

from .detectors import (
    ExtremeEvent,
    HeatwaveDetector,
    AtmosphericRiverDetector,
    ExtremePrecipitationDetector,
)

__all__ = [
    'ExtremeEvent',
    'HeatwaveDetector',
    'AtmosphericRiverDetector',
    'ExtremePrecipitationDetector',
]
