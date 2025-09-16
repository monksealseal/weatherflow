"""Utility exports for the :mod:`weatherflow.utils` namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .flow_visualization import FlowVisualizer
from .evaluation import WeatherEvaluator, WeatherMetrics

try:  # pragma: no cover - exercised indirectly via tests
    from .visualization import WeatherVisualizer as _WeatherVisualizer
except ImportError as exc:  # pragma: no cover - depends on optional deps
    _VIS_IMPORT_ERROR = exc

    class WeatherVisualizer:  # type: ignore[override]
        """Placeholder that explains missing optional visualisation extras."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            raise ImportError(
                "WeatherVisualizer requires optional dependencies such as "
                "cartopy and Pillow. Install weatherflow with the 'viz' extra "
                "(e.g. `pip install weatherflow[viz]`) to enable these features."
            ) from _VIS_IMPORT_ERROR
else:
    WeatherVisualizer = _WeatherVisualizer

if TYPE_CHECKING:  # pragma: no cover
    from .visualization import WeatherVisualizer as WeatherVisualizerType

    WeatherVisualizer = WeatherVisualizerType

__all__ = ["FlowVisualizer", "WeatherVisualizer", "WeatherMetrics", "WeatherEvaluator"]
