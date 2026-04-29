"""Render functions for each cascade level.

Design principles:
  - Every level opens with the *one question* it helps answer.
  - Map and table are two views of the same rows; clicking either drills down.
  - Worst-first ordering everywhere, so the user's eye lands on the decision.
  - Drill-down is a single click (or a single keystroke equivalent).
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .data import (
    HEALTH_COLOR,
    SERVICES,
    TEAMS,
    load_control_room,
    load_customers,
    load_datacenter,
    load_incident,
    load_incidents_for_server,
    load_people,
    load_person,
    load_project,
    load_server,
    load_service,
    load_team_rollup,
    load_ticket,
    load_world,
)


# ---------- shared helpers ----------------------------------------------------


# CSS injected once per session — animations, gradients, level accent colors.
_CSS = """
<style>
@keyframes wf-pulse-soft {
  0%   { box-shadow: 0 0 0 0 rgba(46,204,113,0.55); }
  70%  { box-shadow: 0 0 0 10px rgba(46,204,113,0); }
  100% { box-shadow: 0 0 0 0 rgba(46,204,113,0); }
}
@keyframes wf-pulse-warn {
  0%   { box-shadow: 0 0 0 0 rgba(243,156,18,0.6); }
  70%  { box-shadow: 0 0 0 10px rgba(243,156,18,0); }
  100% { box-shadow: 0 0 0 0 rgba(243,156,18,0); }
}
@keyframes wf-pulse-crit {
  0%   { box-shadow: 0 0 0 0 rgba(231,76,60,0.85); transform: scale(1.0); }
  50%  { box-shadow: 0 0 0 14px rgba(231,76,60,0); transform: scale(1.15); }
  100% { box-shadow: 0 0 0 0 rgba(231,76,60,0); transform: scale(1.0); }
}
@keyframes wf-breathe {
  0%, 100% { box-shadow: inset 0 0 0 1px rgba(78,161,255,0.20); }
  50%      { box-shadow: inset 0 0 0 1px rgba(78,161,255,0.55); }
}
.wf-pulse {
  display:inline-block; width:10px; height:10px; border-radius:50%;
  margin-right:8px; vertical-align:middle;
}
.wf-pulse--healthy  { background:#2ecc71; animation: wf-pulse-soft 2.4s infinite; }
.wf-pulse--warning  { background:#f39c12; animation: wf-pulse-warn 1.6s infinite; }
.wf-pulse--critical { background:#e74c3c; animation: wf-pulse-crit 1.0s infinite; }
.wf-pulse--idle     { background:#5b6473; }

.wf-narrate {
  font-size: 0.95rem; color: #c4c8cf; line-height: 1.55;
  background: linear-gradient(135deg, rgba(78,161,255,0.08), rgba(167,139,250,0.06));
  border-left: 3px solid #4ea1ff;
  padding: 10px 14px; margin: 4px 0 14px 0; border-radius: 4px;
  animation: wf-breathe 3.5s infinite;
}
.wf-narrate strong { color: #f4f6fa; }

.wf-tile {
  background: linear-gradient(180deg, #131a25, #0e1117);
  border: 1px solid #2a313c; border-radius: 10px;
  padding: 14px 16px; height: 100%;
  transition: transform 120ms ease, border-color 120ms ease;
}
.wf-tile:hover { transform: translateY(-2px); border-color: #4ea1ff; }
.wf-tile__head { display:flex; justify-content:space-between; align-items:center;
                 font-size: 11px; color:#9aa0a6; text-transform: uppercase; letter-spacing: .08em; }
.wf-tile__icon { font-size: 18px; }
.wf-tile__metric { font-size: 28px; font-weight: 700; color:#f4f6fa; margin-top: 6px; }
.wf-tile__sub { font-size: 12px; color:#9aa0a6; margin-top: 2px; }

.wf-level {
  display:flex; align-items:center; gap:10px; padding: 6px 10px;
  border-radius: 6px; font-size: 11px; letter-spacing: 0.08em;
  text-transform: uppercase; color:#9aa0a6;
  background: rgba(78,161,255,0.08); border-left: 3px solid #4ea1ff;
  width: fit-content; margin-bottom: 4px;
}
.wf-arrived {
  font-size: 12px; color: #4ea1ff; font-style: italic; margin-bottom: 4px;
}
.wf-sev1 { color:#e74c3c; font-weight:700; }
.wf-sev2 { color:#f39c12; font-weight:600; }
.wf-sev3 { color:#4ea1ff; }
.wf-sev4 { color:#9aa0a6; }
</style>
"""


def _inject_css() -> None:
    if not st.session_state.get("_cascade_css", False):
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state._cascade_css = True


def _pulse(status: str) -> str:
    """Return HTML for a status dot that pulses based on severity."""
    cls = {
        "healthy": "wf-pulse--healthy",
        "warning": "wf-pulse--warning",
        "critical": "wf-pulse--critical",
        "active": "wf-pulse--critical",
        "monitoring": "wf-pulse--warning",
        "resolved": "wf-pulse--healthy",
        "green": "wf-pulse--healthy",
        "yellow": "wf-pulse--warning",
        "red": "wf-pulse--critical",
    }.get(status, "wf-pulse--idle")
    return f"<span class='wf-pulse {cls}'></span>"


def _narrate(html: str) -> None:
    """One paragraph in the system's voice — what it sees and what just changed."""
    st.markdown(f"<div class='wf-narrate'>{html}</div>", unsafe_allow_html=True)


def _level_chip(label: str) -> None:
    st.markdown(f"<div class='wf-level'>{label}</div>", unsafe_allow_html=True)


def _arrived(text: str) -> None:
    st.markdown(f"<div class='wf-arrived'>↳  {text}</div>", unsafe_allow_html=True)


def _question(text: str) -> None:
    """Big, plain-language prompt at the top of each level."""
    st.markdown(
        f"<div style='font-size:1.05rem;color:#9aa0a6;margin:-0.5rem 0 0.75rem 0;'>"
        f"<em>{text}</em></div>",
        unsafe_allow_html=True,
    )


def _drill(level: str, **params) -> None:
    """Push a new level onto the navigation stack and rerun."""
    stack = st.session_state.setdefault(
        "cascade_stack", [{"level": "control_room", "params": {}}]
    )
    stack.append({"level": level, "params": params})
    st.rerun()


# ---------- level 1: world ---------------------------------------------------


def render_world() -> None:
    st.subheader("Global fleet")
    _question("Where should I focus my attention right now?")

    df = load_world().sort_values("health_score").reset_index(drop=True)

    map_col, table_col = st.columns([3, 2], gap="large")

    with map_col:
        fig = go.Figure(
            go.Scattergeo(
                lon=df["lon"],
                lat=df["lat"],
                text=df.apply(
                    lambda r: (
                        f"<b>{r['name']}</b><br>"
                        f"{r['city']}, {r['country']}<br>"
                        f"servers: {r['servers']}  "
                        f"<span style='color:#e74c3c'>crit {r['critical']}</span>  "
                        f"<span style='color:#f39c12'>warn {r['warning']}</span><br>"
                        f"health score: {r['health_score']}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                marker=dict(
                    size=df["servers"] * 0.9 + 14,
                    color=df["health_score"],
                    colorscale=[[0, "#e74c3c"], [0.5, "#f39c12"], [1.0, "#2ecc71"]],
                    cmin=0,
                    cmax=100,
                    line=dict(width=1, color="#222"),
                    showscale=True,
                    colorbar=dict(title="health", thickness=10, len=0.5),
                ),
                customdata=df[["dc_id"]].values,
            )
        )
        fig.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor="#1f2630",
            showocean=True,
            oceancolor="#0e1117",
            showcountries=True,
            countrycolor="#2a313c",
            showframe=False,
        )
        fig.update_layout(
            height=460,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#0e1117",
            geo_bgcolor="#0e1117",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bubble size = server count.  Color = health score (red worst).")

    with table_col:
        st.markdown("**Worst-first**")
        st.dataframe(
            df[
                [
                    "name",
                    "city",
                    "servers",
                    "critical",
                    "warning",
                    "health_score",
                    "avg_cpu",
                ]
            ].rename(columns={"avg_cpu": "avg CPU %"}),
            use_container_width=True,
            hide_index=True,
            height=380,
        )

        st.markdown("**Drill in →**")
        # Buttons in worst-first order — same visual ranking as the table.
        for _, row in df.iterrows():
            badge = (
                "🔴" if row["critical"] > 0 else ("🟠" if row["warning"] > 0 else "🟢")
            )
            label = f"{badge}  {row['name']}  ·  {int(row['servers'])} servers"
            if st.button(label, key=f"open-{row['dc_id']}", use_container_width=True):
                _drill("datacenter", dc_id=row["dc_id"])


# ---------- level 2: data center ---------------------------------------------


def render_datacenter(dc_id: str) -> None:
    world = load_world().set_index("dc_id")
    if dc_id not in world.index:
        st.error(f"Unknown data center {dc_id!r}")
        return
    meta = world.loc[dc_id]
    df = load_datacenter(dc_id)

    st.subheader(f"{meta['name']} — {meta['city']}, {meta['country']}")
    _question("Which servers in this building need a human in the loop?")

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Servers", int(meta["servers"]))
    k2.metric("Critical", int(meta["critical"]))
    k3.metric("Warning", int(meta["warning"]))
    k4.metric("Avg CPU", f"{meta['avg_cpu']}%")

    grid_col, table_col = st.columns([3, 2], gap="large")

    with grid_col:
        # Rack/slot heatmap — physical layout of the room.
        # Each cell is a server; color = current status.
        n_racks = int(df["rack"].max()) + 1
        status_to_z = {"healthy": 0, "warning": 1, "critical": 2}
        # Build a 2D grid (slot, rack) with NaN for empty positions.
        z = [[None] * n_racks for _ in range(8)]
        text = [[""] * n_racks for _ in range(8)]
        ids = [[None] * n_racks for _ in range(8)]
        for _, r in df.iterrows():
            z[r["slot"]][r["rack"]] = status_to_z[r["status"]]
            text[r["slot"]][r["rack"]] = (
                f"{r['server_id']}<br>{r['role']}<br>"
                f"CPU {r['cpu_pct']}%  MEM {r['mem_pct']}%"
            )
            ids[r["slot"]][r["rack"]] = r["server_id"]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                text=text,
                hoverinfo="text",
                colorscale=[
                    [0.0, HEALTH_COLOR["healthy"]],
                    [0.5, HEALTH_COLOR["warning"]],
                    [1.0, HEALTH_COLOR["critical"]],
                ],
                zmin=0,
                zmax=2,
                showscale=False,
                xgap=3,
                ygap=3,
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            xaxis=dict(
                title="rack",
                tickmode="array",
                tickvals=list(range(n_racks)),
                color="#9aa0a6",
            ),
            yaxis=dict(title="slot", color="#9aa0a6", autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Physical layout — each cell is one rack-mounted server. "
            "Color = current status."
        )

    with table_col:
        st.markdown("**Worst-first**")
        st.dataframe(
            df[
                [
                    "server_id",
                    "role",
                    "cpu_pct",
                    "mem_pct",
                    "net_mbps",
                    "status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        st.markdown("**Drill in →**")
        # Show top 10 worst — beyond that, use the search box.
        top = df.head(10)
        search = st.text_input(
            "search server id",
            key="dc-search",
            placeholder="e.g. web-007",
            label_visibility="collapsed",
        )
        if search:
            top = df[df["server_id"].str.contains(search, case=False)].head(10)
        for _, r in top.iterrows():
            badge = {"healthy": "🟢", "warning": "🟠", "critical": "🔴"}[r["status"]]
            label = f"{badge}  {r['server_id']}  ·  CPU {r['cpu_pct']}%"
            if st.button(label, key=f"open-{r['server_id']}", use_container_width=True):
                _drill("server", dc_id=dc_id, server_id=r["server_id"])


# ---------- level 3: server --------------------------------------------------


def render_server(dc_id: str, server_id: str) -> None:
    df = load_datacenter(dc_id)
    row = df[df["server_id"] == server_id]
    if row.empty:
        st.error(f"Unknown server {server_id!r} in {dc_id}")
        return
    s = row.iloc[0]
    bundle = load_server(server_id)
    series = bundle["series"]
    alerts = bundle["alerts"]

    st.subheader(f"{server_id}")
    _question("What is happening on this machine, and is it getting worse?")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Role", s["role"])
    k2.metric("CPU now", f"{s['cpu_pct']}%")
    k3.metric("Memory now", f"{s['mem_pct']}%")
    k4.metric("Uptime", f"{s['uptime_days']} d")

    # Three stacked time series — same x-axis, easy to scan vertically for
    # correlated incidents.
    fig = px.line(
        series.melt("time", var_name="metric", value_name="value"),
        x="time",
        y="value",
        facet_row="metric",
        height=480,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        showlegend=False,
        font_color="#d0d4da",
    )
    fig.for_each_yaxis(lambda a: a.update(matches=None, gridcolor="#2a313c"))
    fig.for_each_xaxis(lambda a: a.update(gridcolor="#2a313c"))
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1], font_color="#9aa0a6")
    )
    fig.update_traces(line=dict(color="#4ea1ff", width=1.6))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Recent alerts**")
    if alerts.empty:
        st.info("No alerts in the last 24 hours.")
    else:
        sev_emoji = {"info": "ℹ️", "warn": "🟠", "crit": "🔴"}
        view = alerts.copy()
        view["severity"] = view["severity"].map(lambda s: f"{sev_emoji.get(s,'')} {s}")
        view["time"] = view["time"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(view, use_container_width=True, hide_index=True)

    # Open incidents on this server are the natural next step for a Director.
    inc = load_incidents_for_server(server_id)
    st.markdown("**Linked incidents →**")
    if inc.empty:
        st.caption("No open or recent incidents tied to this host.")
    else:
        for _, r in inc.iterrows():
            sev_class = f"wf-{r['severity']}"
            label = (
                f"{_pulse(r['status'])}  <span class='{sev_class}'>{r['severity'].upper()}</span>  "
                f"{r['title']}  ·  {r['status']}"
            )
            cols = st.columns([8, 2])
            cols[0].markdown(label, unsafe_allow_html=True)
            if cols[1].button("Open incident →", key=f"open-inc-{r['id']}", use_container_width=True):
                _drill("incident", incident_id=r["id"])


# ---------- level: control room (top) ----------------------------------------


# Each domain card declares: icon, title, the metric+sub line, the "click" target.
_DOMAIN_DEFS = [
    ("infrastructure", "🌐", "Infrastructure",
        lambda d: (f"{d['servers']} servers", f"{d['critical']} critical · {d['warning']} warning across {d['sites']} sites"),
        ("infrastructure", {})),
    ("incidents", "🚨", "Incidents",
        lambda d: (f"{d['active']} active", f"{d['sev1']} sev-1 · ~${d['arr_at_risk_m']}M ARR exposed"),
        ("incidents", {})),
    ("security", "🛡", "Security",
        lambda d: (f"{d['open_vulns']} open", f"{d['critical_vulns']} critical · score {d['compliance_score']}"),
        ("security", {})),
    ("identity", "🪪", "Identity & Access",
        lambda d: (f"+{d['joiners_this_week']}/-{d['leavers_this_week']} this wk", f"{d['stale_accounts']} stale · MFA {d['mfa_coverage_pct']}%"),
        ("identity", {})),
    ("endpoints", "💻", "Endpoints",
        lambda d: (f"{d['managed']} managed", f"{d['unpatched_critical']} unpatched · enc {d['encryption_pct']}%"),
        ("endpoints", {})),
    ("help_desk", "🎫", "Help Desk",
        lambda d: (f"{d['open_tickets']} open", f"{d['breached_sla']} past SLA · CSAT {d['csat_30d']}"),
        ("help_desk", {})),
    ("network", "📡", "Network",
        lambda d: (f"{d['wan_sites_up']}/{d['wan_sites_total']} WAN", f"{d['isp_incidents_30d']} ISP inc · {d['mean_latency_ms']}ms"),
        ("network", {})),
    ("saas", "☁️", "SaaS Health",
        lambda d: (f"{d['apps_tracked']} apps", f"{d['apps_degraded']} degraded · {d['renewals_90d']} renewals 90d"),
        ("saas", {})),
    ("spend", "💰", "Spend",
        lambda d: (f"${d['monthly_run_rate_m']}M/mo", f"{d['vs_budget_pct']}% of budget · {d['top_mover']} top mover"),
        ("spend", {})),
    ("people", "👥", "People",
        lambda d: (f"{d['headcount']} across {d['countries']}", f"{d['open_reqs']} reqs · {d['on_call_now']} on-call now"),
        ("people", {})),
    ("compliance", "📋", "Compliance",
        lambda d: (f"{', '.join(d['frameworks'])}", f"{d['evidence_overdue']} overdue · {d['audit_in_flight']}"),
        ("compliance", {})),
    ("dr_bcp", "🛟", "DR / BCP",
        lambda d: (f"RTO {d['rto_hours']}h / RPO {d['rpo_minutes']}m", f"last drill {d['last_drill_days']}d ago · {d['backups_failed_24h']} backup fail 24h"),
        ("dr_bcp", {})),
]


def render_control_room() -> None:
    cr = load_control_room()
    domains = cr["domains"]

    _level_chip("LEVEL 1 · CONTROL ROOM")
    st.subheader("Your fleet, your team, your week — one screen.")
    _question("What needs me right now, and what can wait?")
    _narrate(cr["narrative"])

    # 4-column tile grid for 12 domains
    cols_per_row = 4
    for row_start in range(0, len(_DOMAIN_DEFS), cols_per_row):
        row = _DOMAIN_DEFS[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="small")
        for (key, icon, title, fmt, target), col in zip(row, cols):
            metric, sub = fmt(domains[key])
            with col:
                st.markdown(
                    f"""
                    <div class='wf-tile'>
                      <div class='wf-tile__head'>
                        <span>{title}</span>
                        <span class='wf-tile__icon'>{icon}</span>
                      </div>
                      <div class='wf-tile__metric'>{metric}</div>
                      <div class='wf-tile__sub'>{sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Step inside →", key=f"open-{key}", use_container_width=True):
                    level, params = target
                    if level == "infrastructure":
                        _drill("infrastructure")
                    elif level == "incidents":
                        _drill("incidents_board")
                    elif level == "people":
                        _drill("people_board")
                    elif level == "help_desk":
                        _drill("help_desk_board")
                    else:
                        _drill("placeholder", domain=key, title=title)


# ---------- level: incidents board (drill from control room) -----------------


def render_incidents_board() -> None:
    from .data import _all_incidents
    df = _all_incidents().sort_values(
        ["status", "severity"],
        key=lambda c: c.map({"active": 0, "monitoring": 1, "resolved": 2}) if c.name == "status"
                       else c.map({"sev1": 0, "sev2": 1, "sev3": 2}),
    ).reset_index(drop=True)

    _level_chip("LEVEL 2 · INCIDENTS")
    _arrived("Stepped into Incidents board")
    st.subheader("Active and recent incidents, every region")
    _question("Which one needs me on the bridge right now?")
    active = df[df["status"] == "active"]
    sev1 = active[active["severity"] == "sev1"]
    if len(sev1) > 0:
        _narrate(
            f"{_pulse('critical')} <strong>{len(sev1)} sev-1 active.</strong> "
            f"Largest exposure: <strong>${active['arr_at_risk_usd'].max()/1e6:.1f}M ARR</strong>. "
            f"Click any row to drop into the bridge."
        )
    else:
        _narrate("No sev-1 active. Holding pattern — pick anything and review.")

    for _, r in df.iterrows():
        sev_class = f"wf-{r['severity']}"
        label = (
            f"{_pulse(r['status'])}<span class='{sev_class}'>{r['severity'].upper()}</span> "
            f"&nbsp; {r['title']} &nbsp; <span style='color:#9aa0a6'>· {r['status']} · "
            f"{r['customers_affected']} customers · ${r['arr_at_risk_usd']/1e6:.2f}M ARR</span>"
        )
        cols = st.columns([8, 2])
        cols[0].markdown(label, unsafe_allow_html=True)
        if cols[1].button("Open →", key=f"ib-{r['id']}", use_container_width=True):
            _drill("incident", incident_id=r["id"])


# ---------- level: incident bridge -------------------------------------------


def render_incident(incident_id: str) -> None:
    bundle = load_incident(incident_id)
    if not bundle:
        st.error("Unknown incident.")
        return
    meta = bundle["meta"]
    timeline = bundle["timeline"]
    customers = bundle["customers"]

    _level_chip("LEVEL · INCIDENT BRIDGE")
    _arrived(f"You're on the bridge for {meta['id']}")
    sev_class = f"wf-{meta['severity']}"
    st.markdown(
        f"### {_pulse(meta['status'])}<span class='{sev_class}'>{meta['severity'].upper()}</span> "
        f"&nbsp; {meta['title']}",
        unsafe_allow_html=True,
    )
    _question("What broke, who's leading, and what is at stake?")

    duration = (datetime(2026, 4, 29, 12, 0) - meta["started_at"]).total_seconds() / 60
    _narrate(
        f"Started <strong>{int(duration)} min</strong> ago. "
        f"Service: <strong>{meta['service_id']}</strong>. "
        f"<strong>{meta['customers_affected']}</strong> customers affected, "
        f"~<strong>${meta['arr_at_risk_usd']/1e6:.2f}M ARR</strong> exposed. "
        f"Lead: <strong>{meta['lead_id']}</strong>."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Status", meta["status"].title())
    k2.metric("Duration", f"{int(duration)} min")
    k3.metric("Customers", meta["customers_affected"])
    k4.metric("ARR at risk", f"${meta['arr_at_risk_usd']/1e6:.2f}M")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("**Timeline**")
        for _, ev in timeline.iterrows():
            st.markdown(
                f"<div style='display:flex;gap:14px;padding:6px 0;border-bottom:1px solid #1a212c;'>"
                f"<span style='color:#5b6473;font-family:ui-monospace,Menlo,monospace;'>"
                f"{ev['time'].strftime('%H:%M')}</span>"
                f"<span style='color:#9aa0a6;'>{ev['actor']}</span>"
                f"<span style='color:#d0d4da;'>{ev['event']}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Drill →**")
        c1, c2 = st.columns(2)
        if c1.button(f"Service · {meta['service_id']}", use_container_width=True):
            _drill("service", service_id=meta["service_id"])
        if c2.button(f"Lead · {meta['lead_id']}", use_container_width=True):
            _drill("person", person_id=meta["lead_id"])

    with right:
        st.markdown("**Customers affected (worst $ first)**")
        for _, c in customers.iterrows():
            cols = st.columns([5, 2])
            cols[0].markdown(
                f"{c['flag']}  **{c['name']}**  "
                f"<span style='color:#9aa0a6'>· {c['tier']} · ${c['arr_usd']:,.0f} ARR · "
                f"{c['impact_minutes']}m impacted</span>",
                unsafe_allow_html=True,
            )
            if cols[1].button("→", key=f"cust-{c['id']}-{incident_id}", use_container_width=True):
                _drill("customer", customer_id=c["id"], from_incident=incident_id)


# ---------- level: customer impact -------------------------------------------


def render_customer(customer_id: str, from_incident: str | None = None) -> None:
    customers = load_customers().set_index("id")
    if customer_id not in customers.index:
        st.error("Unknown customer.")
        return
    c = customers.loc[customer_id]
    _level_chip("LEVEL · CUSTOMER IMPACT")
    _arrived(f"You stepped from infra into the business — meet {c['name']}")
    st.subheader(f"{c['flag']}  {c['name']}")
    _question("How big a deal is this for them, and how big a deal are they for us?")
    _narrate(
        f"<strong>{c['tier'].title()}</strong> tier · "
        f"<strong>${c['arr_usd']:,.0f}</strong> annual revenue · based in {c['country']}. "
        f"Originating incident: <strong>{from_incident or '—'}</strong>."
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Tier", c["tier"].title())
    k2.metric("ARR", f"${c['arr_usd']:,.0f}")
    k3.metric("Country", f"{c['flag']} {c['country']}")
    st.caption("Drill could continue: contract terms · CSM owner · health score history · prior tickets · executive sponsor.")


# ---------- level: service ---------------------------------------------------


def render_service(service_id: str) -> None:
    svc = load_service(service_id)
    if not svc:
        st.error("Unknown service.")
        return
    _level_chip("LEVEL · SERVICE")
    _arrived(f"Stepped into business service: {svc['label']}")
    st.subheader(f"{svc['label']}  ·  {svc['tier']}")
    _question("Is this service meeting its promise, and who owns it?")
    burn = svc["error_budget_burn"]
    burn_word = "burning fast" if burn > 1.5 else "stable"
    _narrate(
        f"SLO target <strong>{svc['slo_target']}%</strong>, currently <strong>{svc['slo_actual']}%</strong>. "
        f"Error budget is <strong>{burn_word}</strong> ({burn}× normal). "
        f"{svc['deployed_30d']} deploys and {svc['incidents_30d']} incidents in the last 30 days."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SLO target", f"{svc['slo_target']}%")
    k2.metric("SLO actual", f"{svc['slo_actual']}%",
              delta=f"{svc['slo_actual'] - svc['slo_target']:+.3f}")
    k3.metric("Deploys 30d", svc["deployed_30d"])
    k4.metric("Incidents 30d", svc["incidents_30d"])

    st.markdown("**Drill →**")
    c1, c2 = st.columns(2)
    team_label = next((t[1] for t in TEAMS if t[0] == svc["team_id"]), svc["team_id"])
    if c1.button(f"Owning team · {team_label}", use_container_width=True):
        _drill("team", team_id=svc["team_id"])
    if c2.button("Dependencies graph (placeholder)", use_container_width=True):
        _drill("placeholder", domain="deps", title=f"deps of {svc['label']}")
    st.caption("Dependencies: " + ", ".join(svc["dependencies"]))


# ---------- level: team ------------------------------------------------------


def render_team(team_id: str) -> None:
    teams = load_team_rollup().set_index("id")
    if team_id not in teams.index:
        st.error("Unknown team.")
        return
    t = teams.loc[team_id]
    people = load_people()
    members = people[people["team_id"] == team_id].copy()

    _level_chip("LEVEL · TEAM")
    _arrived(f"Inside team: {t['name']}")
    st.subheader(f"{t['name']}  ·  {t['headcount']} people across {t['countries']} countries")
    _question("Who's on this team, where are they, and who is overloaded?")
    on_call = members[members["on_call_today"]]
    _narrate(
        f"Charter: {t['charter']}.  Manager: <strong>{t['manager_name']}</strong>.  "
        f"<strong>{len(on_call)}</strong> on-call today.  "
        f"<strong>{int(members['tickets_open'].sum())}</strong> open tickets across the team."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Headcount", int(t["headcount"]))
    k2.metric("Countries", int(t["countries"]))
    k3.metric("On-call now", int(t["on_call_count"]))
    k4.metric("Open tickets", int(t["open_tickets"]))

    members = members.sort_values("tickets_open", ascending=False).reset_index(drop=True)
    st.markdown("**Members · most-loaded first**")
    for _, p in members.iterrows():
        oc = "🟠 on-call" if p["on_call_today"] else ""
        label = (
            f"{p['flag']}  **{p['name']}**  "
            f"<span style='color:#9aa0a6'>· {p['role']} · {p['tickets_open']} tickets · "
            f"{p['projects_active']} projects {oc}</span>"
        )
        cols = st.columns([8, 2])
        cols[0].markdown(label, unsafe_allow_html=True)
        if cols[1].button("Open person →", key=f"team-p-{p['id']}", use_container_width=True):
            _drill("person", person_id=p["id"])


# ---------- level: person ----------------------------------------------------


def render_person(person_id: str) -> None:
    bundle = load_person(person_id)
    if not bundle:
        st.error("Unknown person.")
        return
    p = bundle["meta"]
    schedule = bundle["on_call_week"]
    projects = bundle["projects"]

    _level_chip("LEVEL · PERSON")
    _arrived(f"Stepped into a single human — {p['name']}")
    st.subheader(f"{p['flag']}  {p['name']}  ·  {p['role']}")
    _question("How loaded are they, and what are they shipping?")
    team_label = next((t[1] for t in TEAMS if t[0] == p["team_id"]), p["team_id"])
    _narrate(
        f"On <strong>{team_label}</strong>, based in <strong>{p['country']}</strong>, "
        f"joined <strong>{p['joined_year']}</strong>. "
        f"Reports to <strong>{p['manager_id'] or 'you'}</strong>. "
        f"<strong>{p['tickets_open']}</strong> open tickets, "
        f"<strong>{p['projects_active']}</strong> active projects, "
        f"{'on-call today' if p['on_call_today'] else 'not on-call today'}."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Country", f"{p['flag']} {p['country']}")
    k2.metric("Open tickets", int(p["tickets_open"]))
    k3.metric("Projects", int(p["projects_active"]))
    k4.metric("Joined", int(p["joined_year"]))

    left, right = st.columns([2, 3], gap="large")

    with left:
        st.markdown("**On-call this week**")
        for _, d in schedule.iterrows():
            badge = "🟠" if d["shift"] else "·"
            st.markdown(
                f"<div style='display:flex;gap:12px;padding:3px 0;'>"
                f"<span style='width:30px;color:#9aa0a6'>{d['day']}</span>"
                f"<span>{badge}</span></div>",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("**Active projects**")
        for _, pr in projects.iterrows():
            due = (
                f"due in {pr['due_in_days']}d" if pr["due_in_days"] >= 0
                else f"<span style='color:#e74c3c'>{abs(pr['due_in_days'])}d overdue</span>"
            )
            label = f"{_pulse(pr['status'])} **{pr['name']}** &nbsp; <span style='color:#9aa0a6'>· {due}</span>"
            cols = st.columns([8, 2])
            cols[0].markdown(label, unsafe_allow_html=True)
            if cols[1].button("Open →", key=f"pers-pr-{pr['id']}", use_container_width=True):
                _drill("project", project_id=pr["id"])


# ---------- level: project ---------------------------------------------------


def render_project(project_id: str) -> None:
    pj = load_project(project_id)
    _level_chip("LEVEL · PROJECT")
    _arrived(f"Inside project: {pj['name']}")
    st.subheader(f"{_pulse(pj['status'])}{pj['name']}")
    _question("Is this on track, what is blocking it, what is the next concrete piece of work?")
    _narrate(
        f"Sprint <strong>{pj['sprint']}</strong> · status <strong>{pj['status']}</strong> · "
        f"<strong>{pj['progress']}%</strong> complete.  "
        + (f"Blockers: {'; '.join(pj['blockers'])}." if pj["blockers"] else "No blockers reported.")
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Sprint", pj["sprint"])
    k2.metric("Progress", f"{pj['progress']}%")
    k3.metric("Status", pj["status"].title())
    st.progress(pj["progress"] / 100)

    st.markdown("**Tickets**")
    for _, tk in pj["tickets"].iterrows():
        prio_color = {"low": "#9aa0a6", "med": "#4ea1ff", "high": "#e74c3c"}[tk["priority"]]
        label = (
            f"<span style='color:{prio_color};font-weight:600'>[{tk['priority'].upper()}]</span>  "
            f"{tk['title']}  <span style='color:#9aa0a6'>· {tk['status']} · {tk['assignee']}</span>"
        )
        cols = st.columns([8, 2])
        cols[0].markdown(label, unsafe_allow_html=True)
        if cols[1].button("Open ticket →", key=f"pj-tk-{tk['id']}", use_container_width=True):
            _drill("ticket", ticket_id=tk["id"], assignee=tk["assignee"], title=tk["title"])


# ---------- level: ticket (level 10) -----------------------------------------


def render_ticket(ticket_id: str, assignee: str | None = None, title: str | None = None) -> None:
    bundle = load_ticket(ticket_id)
    _level_chip("LEVEL 10 · TICKET")
    _arrived("You're as deep as it goes — a single piece of work")
    st.subheader(title or ticket_id)
    _question("What is the actual conversation here, and what should I do about it?")
    _narrate(
        f"Ticket <strong>{ticket_id}</strong>"
        + (f" · assigned to <strong>{assignee}</strong>" if assignee else "")
        + ".  Linked PRs: "
        + (", ".join(bundle["linked_prs"]) if bundle["linked_prs"] else "none")
        + "."
    )
    st.markdown("**Thread**")
    for _, c in bundle["comments"].iterrows():
        st.markdown(
            f"<div style='background:#11151c;border:1px solid #2a313c;border-radius:6px;"
            f"padding:8px 12px;margin:6px 0;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:11px;color:#9aa0a6;'>"
            f"<span>{c['actor']}</span><span>{c['time'].strftime('%a %H:%M')}</span></div>"
            f"<div style='margin-top:4px;color:#d0d4da'>{c['body']}</div></div>",
            unsafe_allow_html=True,
        )
    if assignee:
        if st.button(f"Open assignee · {assignee}", use_container_width=True):
            _drill("person", person_id=assignee)


# ---------- placeholder for unbuilt domain entries --------------------------


def render_placeholder(domain: str, title: str) -> None:
    _level_chip(f"LEVEL · {title.upper()}")
    _arrived(f"Stepped into {title}")
    _question("This domain is part of the spatial web but not yet wired with synthetic data.")
    _narrate(
        f"You opened the <strong>{title}</strong> domain. The control room knows this exists "
        f"and surfaces a vital sign for it on the top page, but the deeper levels for this "
        f"path haven't been built yet. Two paths from here are easy to add next: "
        f"the entity list, then the entity detail."
    )
    st.info(
        "Easy wins from here: a list view (e.g. all open vulns / all renewals / all endpoints), "
        "then a detail view that itself becomes a portal to people, services, and customers."
    )


# ---------- people / help-desk boards (level 2 from control room) -----------


def render_people_board() -> None:
    _level_chip("LEVEL 2 · PEOPLE")
    _arrived("Stepped into People")
    st.subheader("Your team — 60 across 5 countries")
    _question("Where is the team, who's loaded, and who's on call right now?")
    teams = load_team_rollup()
    people = load_people()
    by_country = people.groupby(["country", "flag"]).size().reset_index(name="count")
    _narrate(
        f"<strong>{len(teams)}</strong> teams, <strong>{int(people['on_call_today'].sum())}</strong> "
        f"on call right now, across <strong>{len(by_country)}</strong> countries. "
        f"Click any team to step inside."
    )

    st.markdown("**Country headcount**")
    cols = st.columns(len(by_country))
    for col, (_, r) in zip(cols, by_country.iterrows()):
        col.markdown(
            f"<div class='wf-tile' style='text-align:center'>"
            f"<div style='font-size:28px'>{r['flag']}</div>"
            f"<div class='wf-tile__metric' style='font-size:22px'>{r['count']}</div>"
            f"<div class='wf-tile__sub'>{r['country']}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("**Teams**")
    for _, t in teams.iterrows():
        label = (
            f"**{t['name']}** &nbsp; <span style='color:#9aa0a6'>· {t['headcount']} people · "
            f"{t['countries']} countries · {t['on_call_count']} on-call · "
            f"{t['open_tickets']} open tickets</span>"
        )
        cols = st.columns([8, 2])
        cols[0].markdown(label, unsafe_allow_html=True)
        if cols[1].button("Open team →", key=f"pb-{t['id']}", use_container_width=True):
            _drill("team", team_id=t["id"])


def render_help_desk_board() -> None:
    _level_chip("LEVEL 2 · HELP DESK")
    _arrived("Stepped into Help Desk")
    st.subheader("Help desk — every country queue")
    _question("Which queue is hot and which agent is drowning?")
    people = load_people()
    support = people[people["role"] == "IT Support"].copy()
    by_country = support.groupby(["country", "flag"])["tickets_open"].agg(["sum", "count"]).reset_index()
    by_country = by_country.sort_values("sum", ascending=False)
    total = int(support["tickets_open"].sum())
    _narrate(
        f"<strong>{total}</strong> open tickets across <strong>{len(by_country)}</strong> country queues. "
        f"<strong>{by_country.iloc[0]['flag']} {by_country.iloc[0]['country']}</strong> is hottest "
        f"with <strong>{int(by_country.iloc[0]['sum'])}</strong> open."
    )
    st.dataframe(
        by_country.rename(columns={"sum": "open tickets", "count": "agents"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown("**Agents · most-loaded first**")
    for _, a in support.sort_values("tickets_open", ascending=False).iterrows():
        oc = "🟠 on-call" if a["on_call_today"] else ""
        label = f"{a['flag']}  **{a['name']}**  <span style='color:#9aa0a6'>· {a['tickets_open']} tickets {oc}</span>"
        cols = st.columns([8, 2])
        cols[0].markdown(label, unsafe_allow_html=True)
        if cols[1].button("Open agent →", key=f"hd-{a['id']}", use_container_width=True):
            _drill("person", person_id=a["id"])


# ---------- expose injection for page entry ---------------------------------


def setup_page() -> None:
    """Call once before dispatching to a level — injects shared CSS."""
    _inject_css()

