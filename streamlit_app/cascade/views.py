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

from .data import HEALTH_COLOR, load_datacenter, load_server, load_world


# ---------- shared helpers ----------------------------------------------------


def _question(text: str) -> None:
    """Big, plain-language prompt at the top of each level."""
    st.markdown(
        f"<div style='font-size:1.05rem;color:#9aa0a6;margin:-0.5rem 0 0.75rem 0;'>"
        f"<em>{text}</em></div>",
        unsafe_allow_html=True,
    )


def _drill(level: str, **kwargs) -> None:
    """Move one level deeper and rerun. Keeps state in session, not URL hash."""
    st.session_state.cascade_level = level
    for k, v in kwargs.items():
        st.session_state[f"cascade_{k}"] = v
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
