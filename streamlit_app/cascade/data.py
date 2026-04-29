"""Synthetic data for the cascade explorer.

Three levels:
  - World: data centers (geocoded)
  - Data center: servers (NOT geocoded — they live inside a building)
  - Server: time-series + recent alerts (also not geocoded)

The whole graph is deterministic for a given seed, so demos are
reproducible and a user clicking around always sees the same world.
Replace these functions with SQL queries to point the explorer at a
real Postgres / warehouse source — the return shapes are the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

SEED = 7

DATACENTERS = [
    # id, name, city, country, lat, lon, region
    ("dc-iad", "US-East (IAD)", "Ashburn", "USA", 39.04, -77.49, "us-east"),
    ("dc-sjc", "US-West (SJC)", "San Jose", "USA", 37.34, -121.89, "us-west"),
    ("dc-dfw", "US-Central (DFW)", "Dallas", "USA", 32.90, -97.04, "us-central"),
    ("dc-fra", "EU-Central (FRA)", "Frankfurt", "Germany", 50.11, 8.68, "eu-central"),
    ("dc-lhr", "EU-West (LHR)", "London", "UK", 51.47, -0.45, "eu-west"),
    ("dc-sin", "AP-Southeast (SIN)", "Singapore", "Singapore", 1.36, 103.99, "ap-southeast"),
    ("dc-nrt", "AP-Northeast (NRT)", "Tokyo", "Japan", 35.55, 139.78, "ap-northeast"),
    ("dc-syd", "AP-South (SYD)", "Sydney", "Australia", -33.94, 151.18, "ap-south"),
]


@dataclass(frozen=True)
class HealthBands:
    """Thresholds for server health classification (% of capacity used)."""

    warn: float = 70.0
    crit: float = 90.0

    def classify(self, cpu: float, mem: float) -> str:
        worst = max(cpu, mem)
        if worst >= self.crit:
            return "critical"
        if worst >= self.warn:
            return "warning"
        return "healthy"


HEALTH = HealthBands()
HEALTH_COLOR = {
    "healthy": "#2ecc71",
    "warning": "#f39c12",
    "critical": "#e74c3c",
}


def _rng(*parts) -> np.random.Generator:
    """Deterministic RNG keyed by a tuple — same key, same numbers, every call."""
    h = abs(hash((SEED, *parts))) % (2**32)
    return np.random.default_rng(h)


@lru_cache(maxsize=1)
def load_world() -> pd.DataFrame:
    """One row per data center with a roll-up of its servers."""
    rows = []
    for dc_id, name, city, country, lat, lon, region in DATACENTERS:
        servers = load_datacenter(dc_id)
        n = len(servers)
        crit = int((servers["status"] == "critical").sum())
        warn = int((servers["status"] == "warning").sum())
        healthy = n - crit - warn
        rows.append(
            {
                "dc_id": dc_id,
                "name": name,
                "city": city,
                "country": country,
                "region": region,
                "lat": lat,
                "lon": lon,
                "servers": n,
                "critical": crit,
                "warning": warn,
                "healthy": healthy,
                "avg_cpu": round(float(servers["cpu_pct"].mean()), 1),
                "avg_mem": round(float(servers["mem_pct"].mean()), 1),
                # health score: 0 (all critical) -> 100 (all healthy)
                "health_score": round(
                    100 * (healthy + 0.5 * warn) / n, 1
                ),
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=16)
def load_datacenter(dc_id: str) -> pd.DataFrame:
    """One row per server in this data center.

    Servers have rack/slot coordinates (not geographic) — they're the
    bridge from "place on a map" to "thing in a building."
    """
    rng = _rng("dc", dc_id)
    n_servers = int(rng.integers(28, 64))
    n_racks = max(4, n_servers // 8)

    roles = ["web", "db", "cache", "batch", "ml-train"]
    role_weights = [0.35, 0.20, 0.15, 0.20, 0.10]

    rows = []
    for i in range(n_servers):
        role = rng.choice(roles, p=role_weights)
        rack = int(rng.integers(0, n_racks))
        slot = int(rng.integers(0, 8))
        # role-conditioned baseline load
        base_cpu = {"web": 45, "db": 60, "cache": 30, "batch": 75, "ml-train": 85}[role]
        base_mem = {"web": 50, "db": 75, "cache": 70, "batch": 55, "ml-train": 80}[role]
        cpu = float(np.clip(base_cpu + rng.normal(0, 12), 2, 99.5))
        mem = float(np.clip(base_mem + rng.normal(0, 10), 5, 99.5))
        rows.append(
            {
                "server_id": f"{dc_id}-{role}-{i:03d}",
                "role": role,
                "rack": rack,
                "slot": slot,
                "cpu_pct": round(cpu, 1),
                "mem_pct": round(mem, 1),
                "net_mbps": round(float(rng.uniform(50, 950)), 0),
                "uptime_days": int(rng.integers(1, 720)),
                "status": HEALTH.classify(cpu, mem),
            }
        )
    df = pd.DataFrame(rows)
    # Sort with worst first so the table is decision-ordered.
    status_rank = {"critical": 0, "warning": 1, "healthy": 2}
    df = df.sort_values(
        by=["status", "cpu_pct"],
        key=lambda c: c.map(status_rank) if c.name == "status" else -c,
    ).reset_index(drop=True)
    return df


@lru_cache(maxsize=64)
def load_server(server_id: str) -> dict:
    """24h time series + recent alerts for a single server."""
    rng = _rng("server", server_id)
    now = datetime(2026, 4, 29, 12, 0)
    hours = 24
    points_per_hour = 12  # 5-minute samples
    n = hours * points_per_hour
    times = [now - timedelta(minutes=5 * i) for i in range(n)][::-1]

    # AR(1) walks for realism
    def walk(start: float, vol: float, lo: float, hi: float) -> np.ndarray:
        x = np.empty(n)
        x[0] = start
        for k in range(1, n):
            x[k] = np.clip(0.92 * x[k - 1] + 0.08 * start + rng.normal(0, vol), lo, hi)
        return x

    cpu = walk(rng.uniform(20, 80), 4.0, 1, 100)
    mem = walk(rng.uniform(30, 85), 2.5, 5, 100)
    net = walk(rng.uniform(100, 700), 30.0, 10, 1000)

    series = pd.DataFrame({"time": times, "cpu_pct": cpu, "mem_pct": mem, "net_mbps": net})

    alert_pool = [
        ("warn", "CPU > 80% sustained 5m"),
        ("warn", "memory pressure"),
        ("info", "kernel updated, reboot pending"),
        ("crit", "disk I/O wait spike"),
        ("info", "package security patches available"),
        ("warn", "swap usage rising"),
        ("crit", "OOM killer invoked"),
    ]
    n_alerts = int(rng.integers(2, 7))
    chosen = rng.choice(len(alert_pool), size=n_alerts, replace=False)
    alerts = pd.DataFrame(
        [
            {
                "time": now - timedelta(minutes=int(rng.integers(1, 60 * 24))),
                "severity": alert_pool[i][0],
                "message": alert_pool[i][1],
            }
            for i in chosen
        ]
    ).sort_values("time", ascending=False).reset_index(drop=True)

    return {"series": series, "alerts": alerts}
