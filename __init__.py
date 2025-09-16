"""Compatibility wrapper to expose the actual ``weatherflow`` package."""

from __future__ import annotations

import importlib
import sys

from .weatherflow import *  # noqa: F401,F403

_CORE_PACKAGE = "weatherflow.weatherflow"

for _submodule in ("data", "models", "path", "solvers", "manifolds", "utils", "training"):
    _module = importlib.import_module(f".{_submodule}", _CORE_PACKAGE)
    sys.modules[f"{__name__}.{_submodule}"] = _module

del importlib, sys, _CORE_PACKAGE, _module, _submodule