# Vercel Serverless Function - FastAPI Adapter
# Vercel auto-discovers this file and serves it as /api/*
"""Entry point for Vercel serverless deployment of the WeatherFlow API."""

import sys
from pathlib import Path

# Ensure the project root is on the path so weatherflow package is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weatherflow.server.app import app  # noqa: E402

# Vercel expects the ASGI app to be named 'app' or 'handler'
# FastAPI is ASGI-compatible, so this just works.
handler = app
