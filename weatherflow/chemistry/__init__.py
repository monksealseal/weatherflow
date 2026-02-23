"""
Atmospheric Chemistry Module - Inspired by the research of Cassandra J. Gaston

Lightweight, memory-efficient atmospheric chemistry models for interactive
Streamlit simulations. Includes:
- Transatlantic dust transport and chemical aging
- N2O5 heterogeneous chemistry and ClNO2 production
- IEPOX-SOA reactive uptake
- Nutrient (P, Fe) deposition budgets
- CCN activation (kappa-Kohler theory)
- Long-term aerosol trend analysis

All models are 0D/1D box models optimized for the free Streamlit tier (~1GB RAM).
Data is embedded directly - no external downloads required.
"""

from weatherflow.chemistry.models import (
    DustTransportModel,
    N2O5BoxModel,
    IEPOXModel,
    NutrientDepositionModel,
    CCNActivationModel,
    BarbadosTrendsModel,
    GreatSaltLakeModel,
)

from weatherflow.chemistry.gaston_research import (
    PUBLICATIONS,
    RESEARCH_THEMES,
    get_barbados_timeseries,
    get_dust_transport_data,
    get_nutrient_budget_data,
)

__all__ = [
    "DustTransportModel",
    "N2O5BoxModel",
    "IEPOXModel",
    "NutrientDepositionModel",
    "CCNActivationModel",
    "BarbadosTrendsModel",
    "GreatSaltLakeModel",
    "PUBLICATIONS",
    "RESEARCH_THEMES",
    "get_barbados_timeseries",
    "get_dust_transport_data",
    "get_nutrient_budget_data",
]
