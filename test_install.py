"""
# test_install.py
"""Lightweight install test for CI / local checks.

This test verifies that the package imports and exposes version metadata without
requiring heavy optional dependencies.
"""

from weatherflow import __version__, get_version

print("weatherflow imported successfully!")
print(f"__version__ = {__version__}")
try:
    print("get_version() ->", get_version())
except Exception as e:
    print("get_version() raised an error (non-fatal):", e)
