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


# =============================================================================
# Control room — IT Director's domain
# =============================================================================
#
# An IT Director with a 60-person team across 5 countries lives at the
# intersection of: infrastructure, security, identity, endpoints, help desk,
# network, vendors, spend, projects, headcount, compliance, on-call.
#
# The control room is the *top* level — every other level is reached by
# drilling from one of its domain cards. Some drill paths (deepest first):
#
#   Infra:    Control Room → Infrastructure → Site → Server → Incident →
#             Service → Team → Person → Project → Ticket          (10 levels)
#   Security: Control Room → Security → Vuln class → Affected hosts → Server
#   Spend:    Control Room → Spend → Vendor → Contract → Renewal
#   Help:     Control Room → Help desk → Country queue → Agent → Ticket
#   People:   Control Room → People → Country office → Team → Person
#
# Everything is deterministic — same key, same world, every visit.

COUNTRIES = [
    ("USA",       "🇺🇸"),
    ("UK",        "🇬🇧"),
    ("Germany",   "🇩🇪"),
    ("Singapore", "🇸🇬"),
    ("Australia", "🇦🇺"),
]

SERVICES = [
    # id, label, tier, team_id
    ("svc-pay",    "Payments API",        "tier-0", "team-platform"),
    ("svc-search", "Search Backend",      "tier-1", "team-discovery"),
    ("svc-ml",     "ML Training",         "tier-2", "team-mlops"),
    ("svc-portal", "Customer Portal",     "tier-1", "team-web"),
    ("svc-id",     "Identity / SSO",      "tier-0", "team-platform"),
    ("svc-notif",  "Notifications",       "tier-2", "team-platform"),
    ("svc-data",   "Data Pipeline",       "tier-1", "team-data"),
    ("svc-int",    "Internal IT Tools",   "tier-3", "team-itops"),
]

TEAMS = [
    # id, name, charter
    ("team-platform",  "Platform",          "core APIs, identity, on-call backbone"),
    ("team-discovery", "Discovery",         "search, recommendations, ranking"),
    ("team-mlops",     "ML Platform",       "training infra, model registry, GPUs"),
    ("team-web",       "Web Experience",    "customer-facing portal and apps"),
    ("team-data",      "Data Engineering",  "warehouse, pipelines, analytics"),
    ("team-itops",     "IT Operations",     "endpoints, help desk, internal SaaS"),
    ("team-secops",    "Security Ops",      "SOC, vuln mgmt, audit, IAM"),
    ("team-network",   "Network",           "WAN, SD-WAN, VPN, ISP relationships"),
]

# 60 synthetic people across 5 countries, deterministic.
FIRST_NAMES = [
    "Alex", "Jordan", "Sam", "Priya", "Yuki", "Mateo", "Aisha", "Liam",
    "Chloe", "Noor", "Hiro", "Sofia", "Ethan", "Aanya", "Kai", "Nora",
    "Wei", "Adwoa", "Lukas", "Mei", "Ravi", "Zara", "Diego", "Ines",
    "Olu", "Park", "Tova", "Luca", "Iris", "Felix",
]
LAST_NAMES = [
    "Chen", "Patel", "Kim", "Garcia", "Schmidt", "Tan", "Singh", "Müller",
    "Cohen", "Nakamura", "Okafor", "Rossi", "Park", "Andersen", "Khan",
    "Silva", "Lopez", "Wong", "Petrov", "O'Brien",
]
ROLES = [
    ("Engineer",         0.45),
    ("Senior Engineer",  0.20),
    ("Staff Engineer",   0.05),
    ("Manager",          0.10),
    ("IT Support",       0.10),
    ("Security Eng",     0.05),
    ("Network Eng",      0.05),
]


@lru_cache(maxsize=1)
def load_people() -> pd.DataFrame:
    """60 deterministic people across 5 countries and 8 teams."""
    rng = _rng("people")
    rows = []
    role_labels = [r[0] for r in ROLES]
    role_p = [r[1] for r in ROLES]
    for i in range(60):
        first = FIRST_NAMES[i % len(FIRST_NAMES)]
        last = LAST_NAMES[(i * 7 + 3) % len(LAST_NAMES)]
        country, flag = COUNTRIES[i % len(COUNTRIES)]
        role = rng.choice(role_labels, p=role_p)
        team = TEAMS[i % len(TEAMS)][0]
        # managers are deterministic-but-not-self
        manager_idx = (i + 7) % 60 if "Manager" not in role else None
        rows.append({
            "id": f"p-{i:03d}",
            "name": f"{first} {last}",
            "first": first,
            "last": last,
            "role": role,
            "country": country,
            "flag": flag,
            "team_id": team,
            "manager_id": f"p-{manager_idx:03d}" if manager_idx is not None else None,
            "on_call_today": bool(rng.random() < 0.10),
            "tickets_open": int(rng.integers(0, 9)),
            "projects_active": int(rng.integers(1, 4)),
            "joined_year": 2018 + int(rng.integers(0, 8)),
        })
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_team_rollup() -> pd.DataFrame:
    """One row per team, with member counts by country and current load."""
    people = load_people()
    rows = []
    for tid, name, charter in TEAMS:
        members = people[people["team_id"] == tid]
        managers = members[members["role"].str.contains("Manager")]
        on_call = members[members["on_call_today"]]
        rows.append({
            "id": tid,
            "name": name,
            "charter": charter,
            "headcount": len(members),
            "manager_id": managers.iloc[0]["id"] if len(managers) else None,
            "manager_name": managers.iloc[0]["name"] if len(managers) else "—",
            "countries": members["country"].nunique(),
            "open_tickets": int(members["tickets_open"].sum()),
            "on_call_count": int(len(on_call)),
        })
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def load_service(service_id: str) -> dict:
    """Business-service detail: SLO posture, dependencies, recent incidents."""
    rng = _rng("service", service_id)
    meta = next((s for s in SERVICES if s[0] == service_id), None)
    if meta is None:
        return {}
    sid, label, tier, team_id = meta
    slo_target = {"tier-0": 99.99, "tier-1": 99.95, "tier-2": 99.9, "tier-3": 99.5}[tier]
    slo_actual = round(slo_target - abs(rng.normal(0, 0.05)), 3)
    burn_rate = round(max(0.1, rng.normal(2.0 if slo_actual < slo_target else 0.5, 1.0)), 2)
    deps = [s[0] for s in SERVICES if s[0] != sid][:3 + int(rng.integers(0, 3))]
    return {
        "id": sid,
        "label": label,
        "tier": tier,
        "team_id": team_id,
        "slo_target": slo_target,
        "slo_actual": slo_actual,
        "error_budget_burn": burn_rate,
        "dependencies": deps,
        "deployed_30d": int(rng.integers(8, 35)),
        "incidents_30d": int(rng.integers(0, 7)),
    }


def _all_incidents() -> pd.DataFrame:
    """Deterministic flat list of all current/recent incidents."""
    if hasattr(_all_incidents, "_cache"):
        return _all_incidents._cache  # type: ignore
    rng = _rng("incidents")
    titles = [
        ("Elevated 5xx on payments-api",        "svc-pay",    "sev1"),
        ("Search relevance regression",          "svc-search", "sev2"),
        ("ML training queue stuck",              "svc-ml",     "sev3"),
        ("Portal slow for EU users",             "svc-portal", "sev2"),
        ("SSO intermittent failures",            "svc-id",     "sev1"),
        ("Email notifications delayed",          "svc-notif",  "sev3"),
        ("Data pipeline backfill behind",        "svc-data",   "sev2"),
        ("Internal Jira down",                   "svc-int",    "sev3"),
        ("DC SIN cooling alert",                 "svc-pay",    "sev2"),
        ("Certificate expiry approaching",       "svc-id",     "sev2"),
        ("Customer reports stuck transactions",  "svc-pay",    "sev1"),
        ("Region failover drill",                "svc-portal", "sev3"),
    ]
    now = datetime(2026, 4, 29, 12, 0)
    rows = []
    for i, (title, svc, sev) in enumerate(titles):
        started = now - timedelta(minutes=int(rng.integers(15, 60 * 36)))
        active = rng.random() < 0.4
        rows.append({
            "id": f"INC-2026-{i:04d}",
            "title": title,
            "service_id": svc,
            "severity": sev,
            "status": "active" if active else rng.choice(["monitoring", "resolved"]),
            "started_at": started,
            "resolved_at": None if active else started + timedelta(minutes=int(rng.integers(20, 240))),
            "lead_id": f"p-{int(rng.integers(0, 60)):03d}",
            "customers_affected": int(rng.integers(1, 240)),
            "arr_at_risk_usd": int(rng.integers(20_000, 4_500_000)),
            "server_id": None,  # filled in by load_incidents_for_server
        })
    df = pd.DataFrame(rows)
    _all_incidents._cache = df  # type: ignore
    return df


@lru_cache(maxsize=64)
def load_incidents_for_server(server_id: str) -> pd.DataFrame:
    """Incidents associated with a single server (deterministic subset)."""
    rng = _rng("server-incidents", server_id)
    df = _all_incidents().copy()
    n = int(rng.integers(0, 4))
    if n == 0:
        return df.iloc[0:0]
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    out = df.iloc[idx].copy()
    out["server_id"] = server_id
    return out.reset_index(drop=True)


@lru_cache(maxsize=32)
def load_incident(incident_id: str) -> dict:
    """Detailed incident view: meta + timeline + affected customers."""
    df = _all_incidents()
    row = df[df["id"] == incident_id]
    if row.empty:
        return {}
    meta = row.iloc[0].to_dict()
    rng = _rng("incident-detail", incident_id)
    started = meta["started_at"]
    timeline_pool = [
        ("page received from monitoring",   "system"),
        ("on-call ack",                      "lead"),
        ("incident channel opened",          "system"),
        ("communications draft posted",      "lead"),
        ("scope confirmed: 1 region",        "lead"),
        ("hotfix deployed to canary",        "engineer"),
        ("metrics recovering, watching",     "lead"),
        ("status page updated",              "comms"),
        ("postmortem scheduled",             "lead"),
    ]
    n_events = 4 + int(rng.integers(0, 5))
    events = []
    t = started
    for ev, who in timeline_pool[:n_events]:
        t = t + timedelta(minutes=int(rng.integers(2, 25)))
        events.append({"time": t, "event": ev, "actor": who})
    timeline = pd.DataFrame(events)

    # affected customers — drawn from a deterministic pool
    customers = load_customers()
    n_cust = min(meta["customers_affected"], 12)
    cust_idx = rng.choice(len(customers), size=n_cust, replace=False)
    affected = customers.iloc[cust_idx].copy()
    affected["impact_minutes"] = [int(rng.integers(2, 95)) for _ in range(n_cust)]
    affected = affected.sort_values("arr_usd", ascending=False).reset_index(drop=True)

    return {"meta": meta, "timeline": timeline, "customers": affected}


@lru_cache(maxsize=1)
def load_customers() -> pd.DataFrame:
    """30 fictional customers with ARR, region, tier."""
    rng = _rng("customers")
    cos = [
        "Northwind", "Acme", "Globex", "Initech", "Hooli", "Soylent",
        "Stark Industries", "Wayne Enterprises", "Pied Piper", "Dunder Mifflin",
        "Cyberdyne", "Tyrell", "Wonka", "Vandelay", "Massive Dynamic",
        "Aperture", "Black Mesa", "Oscorp", "Umbrella", "Weyland",
        "Krusty Brand", "Bluth", "Costanza Imports", "Pearson Hardman",
        "Sterling Cooper", "Ollivander", "Gringotts", "Wuxia", "Genco", "Prestige",
    ]
    rows = []
    for i, name in enumerate(cos):
        country, flag = COUNTRIES[i % len(COUNTRIES)]
        tier = ["enterprise", "business", "standard"][i % 3]
        rows.append({
            "id": f"cust-{i:03d}",
            "name": name,
            "country": country,
            "flag": flag,
            "tier": tier,
            "arr_usd": {"enterprise": 1_500_000, "business": 250_000, "standard": 40_000}[tier]
                       + int(rng.integers(-15_000, 50_000)),
        })
    return pd.DataFrame(rows)


@lru_cache(maxsize=64)
def load_person(person_id: str) -> dict:
    """Single human profile: location, manager, projects, on-call schedule."""
    people = load_people()
    row = people[people["id"] == person_id]
    if row.empty:
        return {}
    meta = row.iloc[0].to_dict()
    rng = _rng("person", person_id)
    # synthetic week of on-call shifts
    on_call_week = pd.DataFrame({
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "shift": [bool(rng.random() < (0.4 if meta["on_call_today"] else 0.15)) for _ in range(7)],
    })
    # 1-3 active projects
    n_proj = int(meta["projects_active"])
    projects = pd.DataFrame([
        {
            "id": f"proj-{person_id}-{j}",
            "name": _project_name(rng),
            "status": rng.choice(["green", "yellow", "red"], p=[0.55, 0.30, 0.15]),
            "due_in_days": int(rng.integers(-5, 60)),
        } for j in range(n_proj)
    ])
    return {"meta": meta, "on_call_week": on_call_week, "projects": projects}


def _project_name(rng) -> str:
    verbs = ["migrate", "harden", "consolidate", "replatform", "automate", "decommission",
             "modernize", "audit", "rotate", "rearchitect"]
    nouns = ["IAM", "SSO", "endpoint fleet", "WAN links", "VPN gateway", "SIEM",
             "backup vault", "patch pipeline", "MDM policy", "Jira instance",
             "audit logging", "PAM tooling"]
    return f"{rng.choice(verbs)} {rng.choice(nouns)}".capitalize()


@lru_cache(maxsize=128)
def load_project(project_id: str) -> dict:
    """Project status: sprint, blockers, recent updates, linked tickets."""
    rng = _rng("project", project_id)
    sprint = int(rng.integers(12, 60))
    progress = int(rng.integers(10, 95))
    status = "green" if progress > 60 else ("yellow" if progress > 30 else "red")
    blockers = []
    if status != "green":
        pool = [
            "vendor renewal stalled in legal",
            "approval pending from CFO",
            "dependency on team-network's WAN cutover",
            "compliance review (SOC2) blocking deploy",
            "budget reduction mid-sprint",
        ]
        n = int(rng.integers(1, 4))
        idx = rng.choice(len(pool), size=n, replace=False)
        blockers = [pool[i] for i in idx]
    n_tickets = 5 + int(rng.integers(0, 11))
    ticket_titles = [
        "design doc review", "integrate with SSO", "migrate Singapore office",
        "decommission legacy hosts", "patch staging cluster", "rotate API keys",
        "draft runbook", "update on-call schedule", "review vendor SOW",
        "prepare audit evidence", "harden firewall ruleset", "monthly DR drill",
        "renew TLS certificates", "review pen-test findings",
    ]
    tickets = pd.DataFrame([
        {
            "id": f"{project_id}-T{i:03d}",
            "title": ticket_titles[(i + sprint) % len(ticket_titles)],
            "status": rng.choice(["open", "in-progress", "review", "done"], p=[0.25, 0.40, 0.20, 0.15]),
            "assignee": f"p-{int(rng.integers(0, 60)):03d}",
            "priority": rng.choice(["low", "med", "high"], p=[0.40, 0.45, 0.15]),
        } for i in range(n_tickets)
    ])
    return {
        "id": project_id,
        "name": project_id.replace("proj-", "").replace("-", " ").title(),
        "sprint": sprint,
        "progress": progress,
        "status": status,
        "blockers": blockers,
        "tickets": tickets,
    }


@lru_cache(maxsize=256)
def load_ticket(ticket_id: str) -> dict:
    """Single ticket: full thread, linked PRs, related incidents."""
    rng = _rng("ticket", ticket_id)
    comments = []
    actors = ["reporter", "assignee", "manager", "reviewer"]
    n_comments = 2 + int(rng.integers(0, 6))
    base = datetime(2026, 4, 27, 9, 0)
    for i in range(n_comments):
        base = base + timedelta(hours=int(rng.integers(1, 18)))
        comments.append({
            "time": base,
            "actor": rng.choice(actors),
            "body": rng.choice([
                "Looking into this now.",
                "Reproduced — patch in flight.",
                "Need approval from Security before merging.",
                "Pushed to canary, watching metrics for 30 min.",
                "Resolved — closing.",
                "Linking related PR.",
                "Bumping priority — customer escalation.",
            ]),
        })
    return {
        "id": ticket_id,
        "comments": pd.DataFrame(comments),
        "linked_prs": [f"#{int(rng.integers(1000, 9999))}" for _ in range(int(rng.integers(0, 4)))],
    }


# ── Control room rollup ──────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_control_room() -> dict:
    """The IT Director's vital signs across every domain at once."""
    world = load_world()
    people = load_people()
    incidents = _all_incidents()
    rng = _rng("control-room")

    domains = {
        "infrastructure": {
            "servers": int(world["servers"].sum()),
            "critical": int(world["critical"].sum()),
            "warning": int(world["warning"].sum()),
            "sites": len(world),
        },
        "incidents": {
            "active": int((incidents["status"] == "active").sum()),
            "sev1": int(((incidents["severity"] == "sev1") & (incidents["status"] == "active")).sum()),
            "customers_affected": int(incidents.loc[incidents["status"] == "active", "customers_affected"].sum()),
            "arr_at_risk_m": round(incidents.loc[incidents["status"] == "active", "arr_at_risk_usd"].sum() / 1e6, 1),
        },
        "security": {
            "open_vulns": 17 + int(rng.integers(0, 30)),
            "critical_vulns": int(rng.integers(0, 4)),
            "audits_due_30d": int(rng.integers(1, 4)),
            "compliance_score": round(85 + rng.normal(0, 4), 1),
        },
        "identity": {
            "joiners_this_week": int(rng.integers(2, 9)),
            "leavers_this_week": int(rng.integers(0, 5)),
            "stale_accounts": int(rng.integers(3, 22)),
            "mfa_coverage_pct": round(94 + rng.normal(0, 2), 1),
        },
        "endpoints": {
            "managed": 412 + int(rng.integers(-10, 20)),
            "unpatched_critical": int(rng.integers(0, 14)),
            "encryption_pct": round(99 + rng.normal(0, 0.5), 1),
            "lost_or_stolen_30d": int(rng.integers(0, 4)),
        },
        "help_desk": {
            "open_tickets": int(people["tickets_open"].sum()),
            "breached_sla": int(rng.integers(0, 12)),
            "csat_30d": round(4.2 + rng.normal(0, 0.15), 2),
            "first_response_min": round(8 + rng.normal(0, 3), 0),
        },
        "network": {
            "wan_sites_up": 5,
            "wan_sites_total": 5,
            "isp_incidents_30d": int(rng.integers(0, 5)),
            "mean_latency_ms": round(38 + rng.normal(0, 5), 0),
        },
        "saas": {
            "apps_tracked": 42,
            "apps_degraded": int(rng.integers(0, 4)),
            "shadow_it_findings": int(rng.integers(0, 9)),
            "renewals_90d": int(rng.integers(2, 8)),
        },
        "spend": {
            "monthly_run_rate_m": round(2.4 + rng.normal(0, 0.1), 2),
            "vs_budget_pct": round(98 + rng.normal(0, 3), 1),
            "top_mover": rng.choice(["AWS", "Datadog", "GitHub", "Atlassian", "Okta"]),
            "renewals_60d": int(rng.integers(2, 8)),
        },
        "people": {
            "headcount": len(people),
            "open_reqs": int(rng.integers(2, 7)),
            "on_call_now": int(people["on_call_today"].sum()),
            "countries": int(people["country"].nunique()),
        },
        "compliance": {
            "frameworks": ["SOC2", "ISO27001", "GDPR"],
            "evidence_overdue": int(rng.integers(0, 6)),
            "audit_in_flight": rng.choice(["SOC2 Type II", "ISO27001 surveillance", "—"]),
        },
        "dr_bcp": {
            "last_drill_days": int(rng.integers(8, 90)),
            "rto_hours": 4,
            "rpo_minutes": 15,
            "backups_failed_24h": int(rng.integers(0, 3)),
        },
    }
    # narrative — what does the control room itself say?
    narrative = _narrate_top(domains, world, incidents)
    return {"domains": domains, "narrative": narrative}


def _narrate_top(domains, world, incidents) -> str:
    """One paragraph in the system's voice — what is worth your attention right now."""
    worst = world.sort_values("health_score").iloc[0]
    sev1_active = int(((incidents["severity"] == "sev1") & (incidents["status"] == "active")).sum())
    arr = domains["incidents"]["arr_at_risk_m"]
    parts = []
    if sev1_active > 0:
        parts.append(
            f"{sev1_active} sev-1 incident{'s' if sev1_active > 1 else ''} active right now "
            f"— roughly ${arr:.1f}M of ARR is exposed."
        )
    parts.append(
        f"The site demanding attention is **{worst['name']}** "
        f"({worst['critical']} critical, {worst['warning']} warning servers)."
    )
    if domains["help_desk"]["breached_sla"] > 5:
        parts.append(f"Help desk is hot — {domains['help_desk']['breached_sla']} tickets past SLA.")
    if domains["security"]["critical_vulns"] > 0:
        parts.append(f"{domains['security']['critical_vulns']} critical vulnerabilities still open.")
    parts.append("Everywhere else is quiet. Click any tile to step inside.")
    return "  ".join(parts)

