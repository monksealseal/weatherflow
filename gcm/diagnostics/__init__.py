"""
GCM Diagnostics Module

Comprehensive diagnostics for validating GCM physics including:
- Energy spectra (kinetic and available potential energy)
- Mean zonal circulation (jets, Hadley cell)
- Eddy statistics (stationary and transient)
- Angular momentum budget
- Heat transport diagnostics
"""

from .spectral import SpectralDiagnostics
from .zonal_mean import ZonalMeanDiagnostics
from .eddy_diagnostics import EddyDiagnostics
from .circulation import CirculationDiagnostics
from .comprehensive import ComprehensiveGCMDiagnostics

__all__ = [
    'SpectralDiagnostics',
    'ZonalMeanDiagnostics',
    'EddyDiagnostics',
    'CirculationDiagnostics',
    'ComprehensiveGCMDiagnostics'
]
