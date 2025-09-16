"""Test package initialization utilities."""

import os
import sys

# Ensure project root is available on sys.path so tests can import the
# weatherflow package regardless of the current working directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

__all__ = []
