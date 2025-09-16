"""Pytest configuration shared by all tests."""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg", force=True)
