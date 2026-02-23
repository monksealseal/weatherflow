"""
Atmospheric Chemistry Transport Model
Interactive simulations of Cassandra J. Gaston's atmospheric chemistry research

Features 7 interactive simulation labs covering:
1. Research Overview - Publication database, themes, and impact
2. Transatlantic Dust Transport - Sahara to Barbados Lagrangian trajectory
3. Nighttime Chemistry - N2O5 heterogeneous uptake and ClNO2 production
4. SOA Formation - IEPOX reactive uptake on acidic aerosol
5. Nutrient Deposition - P/Fe budget for the Amazon
6. Barbados Observatory - 21-year aerosol trend analysis
7. CCN & Cloud Formation - Kappa-Kohler activation theory

All simulations use embedded data (no downloads) and lightweight box models
optimized for the free Streamlit tier (~1GB RAM).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from weatherflow.chemistry.gaston_research import (
    PUBLICATIONS,
    RESEARCH_THEMES,
    get_barbados_timeseries,
    get_dust_transport_data,
    get_nutrient_budget_data,
    get_heterogeneous_kinetics,
    get_ccn_parameters,
)
from weatherflow.chemistry.models import (
    DustTransportModel,
    N2O5BoxModel,
    IEPOXModel,
    NutrientDepositionModel,
    CCNActivationModel,
    BarbadosTrendsModel,
    GreatSaltLakeModel,
)

st.set_page_config(
    page_title="Atmospheric Chemistry",
    page_icon="",
    layout="wide",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .research-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 5px solid;
    }
    .pub-entry {
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border-left: 3px solid #4682B4;
    }
    .metric-highlight {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
    }
    .sim-header {
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.title("Atmospheric Chemistry Transport Model")
st.markdown("""
Interactive simulations of atmospheric chemistry research by
**[Cassandra J. Gaston](https://people.miami.edu/profile/cjg174@miami.edu)**,
Associate Professor at the University of Miami Rosenstiel School.
Each tab is a simulation lab demonstrating findings from her 30+ publications spanning
dust transport, heterogeneous chemistry, SOA formation, nutrient cycling, and aerosol-cloud interactions.
""")

# Show key metrics
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Publications", f"{len(PUBLICATIONS)}+")
with mcol2:
    st.metric("Research Themes", len(RESEARCH_THEMES))
with mcol3:
    st.metric("Years of Data", "21+")
with mcol4:
    st.metric("Chemical Species", "25+")

# =============================================================================
# Main Tabs
# =============================================================================
tabs = st.tabs([
    "Research Overview",
    "Dust Transport",
    "Nighttime Chemistry",
    "SOA Formation",
    "Nutrient Deposition",
    "Barbados Observatory",
    "CCN & Clouds",
])


# =============================================================================
# TAB 1: Research Overview
# =============================================================================
with tabs[0]:
    st.header("Research Overview")
    st.markdown("Explore Cassandra Gaston's research themes, publication timeline, and key findings.")

    # Theme cards
    st.subheader("Research Themes")
    theme_cols = st.columns(3)
    for i, (key, theme) in enumerate(RESEARCH_THEMES.items()):
        with theme_cols[i % 3]:
            count = sum(1 for p in PUBLICATIONS if p["theme"] == key)
            st.markdown(f"""
**{theme['name']}**

{theme['description']}

*Key species:* {', '.join(theme['key_species'][:4])}

*Papers:* {count}

---
""")

    # Publication timeline
    st.subheader("Publication Timeline")

    years = [p["year"] for p in PUBLICATIONS]
    year_counts = {}
    for y in years:
        year_counts[y] = year_counts.get(y, 0) + 1

    sorted_years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in sorted_years]

    # Color by theme
    theme_colors = {k: v["color"] for k, v in RESEARCH_THEMES.items()}

    fig_timeline = go.Figure()

    # Stacked bar by theme
    for theme_key, theme_info in RESEARCH_THEMES.items():
        theme_pubs = [p for p in PUBLICATIONS if p["theme"] == theme_key]
        theme_years = {}
        for p in theme_pubs:
            theme_years[p["year"]] = theme_years.get(p["year"], 0) + 1

        fig_timeline.add_trace(go.Bar(
            x=sorted_years,
            y=[theme_years.get(y, 0) for y in sorted_years],
            name=theme_info["name"],
            marker_color=theme_info["color"],
            hovertemplate="%{y} paper(s) in %{x}<extra>" + theme_info["name"] + "</extra>",
        ))

    fig_timeline.update_layout(
        barmode="stack",
        title="Publications by Year and Theme",
        xaxis_title="Year",
        yaxis_title="Number of Publications",
        height=400,
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Publication list
    st.subheader("Selected Publications")
    theme_filter = st.selectbox(
        "Filter by theme",
        ["All"] + [v["name"] for v in RESEARCH_THEMES.values()],
    )

    theme_key_map = {v["name"]: k for k, v in RESEARCH_THEMES.items()}

    for pub in sorted(PUBLICATIONS, key=lambda x: -x["year"]):
        if theme_filter != "All":
            if pub["theme"] != theme_key_map.get(theme_filter):
                continue
        color = theme_colors.get(pub["theme"], "#666")
        with st.container():
            st.markdown(
                f"**{pub['year']}** | {pub['authors']}, "
                f"*\"{pub['title']}\"*, {pub['journal']}."
            )
            st.caption(f"Finding: {pub['key_finding']}")


# =============================================================================
# TAB 2: Dust Transport
# =============================================================================
with tabs[1]:
    st.header("Transatlantic Dust Transport Simulator")
    st.markdown("""
    Simulate a Saharan dust parcel traveling ~5,000 km across the Atlantic to Barbados.
    Track chemical aging, nutrient solubilization, and mixing with sea salt as dust
    descends from the Saharan Air Layer (SAL) into the marine boundary layer (MBL).

    **Key references:** Gaston (2020) *Acc. Chem. Res.*, Royer et al. (2025) *ACP*,
    Shrestha et al. (2026) *ACP*, Prospero et al. (2020) *GBC*
    """)

    col_dust_config, col_dust_viz = st.columns([1, 3])

    with col_dust_config:
        st.subheader("Configuration")
        dust_conc = st.slider("Initial dust (ug/m3)", 50, 500, 200, 10, key="dust_init")
        transport_days = st.slider("Transport time (days)", 3, 10, 5, key="dust_days")
        hno3 = st.slider("HNO3 (ppb)", 0.1, 2.0, 0.5, 0.1, key="dust_hno3")
        h2so4 = st.slider("H2SO4 (ppb)", 0.01, 0.5, 0.1, 0.01, key="dust_h2so4")
        particle_size = st.slider("Particle diameter (um)", 1.0, 10.0, 3.0, 0.5, key="dust_size")

        run_dust = st.button("Run Transport Simulation", type="primary", key="dust_run")

    with col_dust_viz:
        if run_dust or "dust_result" in st.session_state:
            if run_dust:
                model = DustTransportModel(
                    transport_days=transport_days,
                    initial_dust_conc=dust_conc,
                    hno3_ppb=hno3,
                    h2so4_ppb=h2so4,
                    particle_diameter_um=particle_size,
                )
                st.session_state["dust_result"] = model.run()

            r = st.session_state["dust_result"]
            t = r["time_hours"]

            # Create multi-panel figure
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Trajectory: Altitude vs Distance",
                    "Dust Concentration",
                    "Chemical Aging (Coatings)",
                    "Nutrient Solubility",
                    "Sea Salt Mixing Fraction",
                    "Lidar Depolarization Ratio",
                ),
                vertical_spacing=0.08,
                horizontal_spacing=0.10,
            )

            # Trajectory
            fig.add_trace(go.Scatter(
                x=r["distance_km"], y=r["altitude_m"],
                mode="lines", line=dict(color="#D4A574", width=3),
                fill="tozeroy", fillcolor="rgba(212,165,116,0.15)",
                name="Trajectory", hovertemplate="Distance: %{x:.0f} km<br>Alt: %{y:.0f} m",
            ), row=1, col=1)
            fig.add_hline(y=500, line_dash="dash", line_color="steelblue",
                         annotation_text="MBL top", row=1, col=1)
            fig.add_hline(y=3000, line_dash="dash", line_color="orange",
                         annotation_text="SAL", row=1, col=1)

            # Dust concentration
            fig.add_trace(go.Scatter(
                x=t, y=r["dust_conc_ug_m3"],
                mode="lines", line=dict(color="#CD853F", width=2.5),
                fill="tozeroy", fillcolor="rgba(205,133,63,0.15)",
                name="Dust", hovertemplate="%{y:.1f} ug/m3",
            ), row=1, col=2)

            # Chemical aging
            fig.add_trace(go.Scatter(
                x=t, y=r["nitrate_coating"] * 100,
                mode="lines", line=dict(color="#e74c3c", width=2),
                name="Nitrate coating",
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=t, y=r["sulfate_coating"] * 100,
                mode="lines", line=dict(color="#3498db", width=2),
                name="Sulfate coating",
            ), row=2, col=1)

            # Nutrient solubility
            fig.add_trace(go.Scatter(
                x=t, y=r["p_solubility"] * 100,
                mode="lines", line=dict(color="#FF8C00", width=2.5),
                name="P solubility",
            ), row=2, col=2)
            fig.add_trace(go.Scatter(
                x=t, y=r["fe_solubility"] * 100,
                mode="lines", line=dict(color="#8B0000", width=2.5),
                name="Fe solubility",
            ), row=2, col=2)

            # Sea salt mixing
            fig.add_trace(go.Scatter(
                x=t, y=r["sea_salt_mixing"] * 100,
                mode="lines", line=dict(color="#20B2AA", width=2.5),
                fill="tozeroy", fillcolor="rgba(32,178,170,0.15)",
                name="Sea salt mixed",
            ), row=3, col=1)
            fig.add_hline(y=67, line_dash="dot", line_color="gray",
                         annotation_text="67% (observed)", row=3, col=1)

            # Depolarization ratio
            fig.add_trace(go.Scatter(
                x=t, y=r["depol_ratio"],
                mode="lines", line=dict(color="#9b59b6", width=2.5),
                name="Depol. ratio",
            ), row=3, col=2)
            fig.add_hline(y=0.30, line_dash="dot", line_color="gray",
                         annotation_text="Pure dust", row=3, col=2)
            fig.add_hline(y=0.10, line_dash="dot", line_color="gray",
                         annotation_text="Mixed MBL", row=3, col=2)

            fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
            fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
            fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
            fig.update_yaxes(title_text="ug/m3", row=1, col=2)
            fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
            fig.update_yaxes(title_text="Coating (%)", row=2, col=1)
            fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
            fig.update_yaxes(title_text="Solubility (%)", row=2, col=2)
            fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
            fig.update_yaxes(title_text="Mixed (%)", row=3, col=1)
            fig.update_xaxes(title_text="Time (hours)", row=3, col=2)
            fig.update_yaxes(title_text="Ratio", row=3, col=2)

            fig.update_layout(
                height=850,
                showlegend=True,
                legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
                title="Transatlantic Dust Transport: Sahara to Barbados",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Key findings summary
            final_idx = len(t) - 1
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            with fcol1:
                st.metric("Final Dust", f"{r['dust_conc_ug_m3'][final_idx]:.1f} ug/m3",
                          f"{(r['dust_conc_ug_m3'][final_idx]/dust_conc - 1)*100:.0f}%")
            with fcol2:
                st.metric("P Solubility", f"{r['p_solubility'][final_idx]*100:.1f}%",
                          f"+{(r['p_solubility'][final_idx] - 0.05)*100:.1f}%")
            with fcol3:
                st.metric("Sea Salt Mixed", f"{r['sea_salt_mixing'][final_idx]*100:.0f}%")
            with fcol4:
                st.metric("Depolarization", f"{r['depol_ratio'][final_idx]:.2f}",
                          f"{(r['depol_ratio'][final_idx] - 0.30):.2f}")
        else:
            st.info("Click **Run Transport Simulation** to trace a dust parcel from the Sahara to Barbados.")

            # Show transport map schematic
            fig_map = go.Figure()
            # Sahara
            fig_map.add_trace(go.Scattergeo(
                lon=[-5, -20, -40, -60, -59.5],
                lat=[20, 15, 13, 13, 13.2],
                mode="lines+markers+text",
                line=dict(color="#D4A574", width=4, dash="dot"),
                marker=dict(size=[12, 8, 8, 8, 14], color=["#D4A574", "#D4A574", "#D4A574", "#20B2AA", "#CD5C5C"]),
                text=["Sahara", "", "", "", "Barbados"],
                textposition="top center",
                textfont=dict(size=14),
                hoverinfo="text",
                hovertext=["Dust source", "SAL transport", "Mid-Atlantic", "MBL mixing", "Ragged Point"],
            ))
            fig_map.update_geos(
                showland=True, landcolor="rgb(243, 243, 243)",
                showocean=True, oceancolor="rgb(204, 229, 255)",
                showcoastlines=True,
                projection_type="natural earth",
                lonaxis=dict(range=[-80, 20]),
                lataxis=dict(range=[-5, 35]),
            )
            fig_map.update_layout(
                title="Transatlantic Dust Transport Route",
                height=350, margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_map, use_container_width=True)


# =============================================================================
# TAB 3: Nighttime Chemistry
# =============================================================================
with tabs[2]:
    st.header("Nighttime N2O5 Chemistry Lab")
    st.markdown("""
    Simulate nighttime heterogeneous chemistry: N2O5 reactive uptake on aerosol
    surfaces and ClNO2 production. Compare reaction kinetics across different
    aerosol types from sea salt to Great Salt Lake playa dust.

    **Key references:** Gaston & Thornton (2016) *JPC A*, Mitroo et al. (2019) *ES&T*,
    Royer et al. (2021) *ES&T*, Christie et al. (2025) *ACS Earth Space Chem.*
    """)

    n2o5_subtab1, n2o5_subtab2 = st.tabs([
        "Box Model Simulation",
        "Great Salt Lake Scenario",
    ])

    # --- Box Model ---
    with n2o5_subtab1:
        col_n2o5_config, col_n2o5_viz = st.columns([1, 3])

        with col_n2o5_config:
            st.subheader("Configuration")
            aerosol_type = st.selectbox(
                "Aerosol type",
                ["NaCl", "Sea_salt", "NH4HSO4", "(NH4)2SO4",
                 "Great_Salt_Lake_playa", "Owens_Lake_playa", "Organic_coated", "Illite_clay"],
                key="n2o5_aero",
            )
            no2_ppb = st.slider("NO2 (ppb)", 1.0, 30.0, 5.0, 0.5, key="n2o5_no2")
            o3_ppb = st.slider("O3 (ppb)", 10.0, 80.0, 30.0, 1.0, key="n2o5_o3")
            sa_input = st.slider("Surface area (um2/cm3)", 50, 1000, 200, 10, key="n2o5_sa")
            temp = st.slider("Temperature (K)", 260, 310, 298, 1, key="n2o5_temp")
            rh = st.slider("RH (%)", 10, 95, 50, 5, key="n2o5_rh")
            duration = st.slider("Duration (hours)", 4, 14, 12, 1, key="n2o5_dur")

            run_n2o5 = st.button("Run Nighttime Simulation", type="primary", key="n2o5_run")

        with col_n2o5_viz:
            if run_n2o5 or "n2o5_result" in st.session_state:
                if run_n2o5:
                    model = N2O5BoxModel(
                        T=temp, RH=rh,
                        aerosol_surface_area=sa_input * 1e-8,
                        aerosol_type=aerosol_type,
                    )
                    st.session_state["n2o5_result"] = model.run(
                        duration_hours=duration, no2_ppb=no2_ppb, o3_ppb=o3_ppb,
                    )
                    st.session_state["n2o5_type"] = aerosol_type

                r = st.session_state["n2o5_result"]
                atype = st.session_state.get("n2o5_type", aerosol_type)
                t = r["time_hours"]

                # Metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("N2O5 Lifetime", f"{r['N2O5_lifetime_min']:.1f} min")
                with mc2:
                    st.metric("Peak N2O5", f"{np.max(r['N2O5_ppt']):.0f} ppt")
                with mc3:
                    st.metric("Peak ClNO2", f"{np.max(r['ClNO2_ppt']):.0f} ppt")
                with mc4:
                    st.metric("Total HNO3", f"{r['HNO3_ppb'][-1]:.2f} ppb")

                # Main chemistry plot
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Reactive Nitrogen Species",
                        "Halogen Products",
                        "Ozone & NO2 Depletion",
                        f"Reaction Summary on {atype}",
                    ),
                    vertical_spacing=0.12,
                )

                # Panel 1: N2O5, NO3
                fig.add_trace(go.Scatter(x=t, y=r["N2O5_ppt"], name="N2O5",
                    line=dict(color="#e74c3c", width=2.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=r["NO3_ppt"], name="NO3",
                    line=dict(color="#f39c12", width=2)), row=1, col=1)

                # Panel 2: ClNO2, HNO3
                fig.add_trace(go.Scatter(x=t, y=r["ClNO2_ppt"], name="ClNO2",
                    line=dict(color="#27ae60", width=2.5)), row=1, col=2)
                fig.add_trace(go.Scatter(x=t, y=r["HNO3_ppb"] * 1000, name="HNO3 (ppt)",
                    line=dict(color="#8e44ad", width=2)), row=1, col=2)

                # Panel 3: O3 and NO2
                fig.add_trace(go.Scatter(x=t, y=r["O3_ppb"], name="O3",
                    line=dict(color="#2980b9", width=2.5)), row=2, col=1)
                fig.add_trace(go.Scatter(x=t, y=r["NO2_ppb"], name="NO2",
                    line=dict(color="#c0392b", width=2)), row=2, col=1)

                # Panel 4: Summary bar chart
                kinetics = get_heterogeneous_kinetics()
                surfaces = list(kinetics["gamma_n2o5"].keys())[:6]
                gammas = [kinetics["gamma_n2o5"][s]["value"] for s in surfaces]
                yields = [kinetics["clno2_yield"].get(s, 0) for s in surfaces]
                bar_colors = ["#e74c3c" if s == atype else "#95a5a6" for s in surfaces]

                fig.add_trace(go.Bar(
                    x=[s.replace("_", " ") for s in surfaces], y=gammas,
                    name="gamma(N2O5)", marker_color=bar_colors,
                    text=[f"{g:.3f}" for g in gammas], textposition="outside",
                ), row=2, col=2)

                fig.update_yaxes(title_text="ppt", row=1, col=1)
                fig.update_yaxes(title_text="ppt", row=1, col=2)
                fig.update_yaxes(title_text="ppb", row=2, col=1)
                fig.update_yaxes(title_text="gamma", type="log", row=2, col=2)
                for r_idx in [1, 2]:
                    for c_idx in [1, 2]:
                        fig.update_xaxes(title_text="Time (hours)", row=r_idx, col=c_idx)
                fig.update_xaxes(title_text="Aerosol Type", row=2, col=2)

                fig.update_layout(
                    height=700, showlegend=True,
                    legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                    title=f"Nighttime Chemistry on {atype.replace('_', ' ')}",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select aerosol type and click **Run Nighttime Simulation** to explore N2O5 chemistry.")

                # Show kinetics comparison
                kinetics = get_heterogeneous_kinetics()
                fig_compare = go.Figure()
                surfaces = list(kinetics["gamma_n2o5"].keys())
                gammas = [kinetics["gamma_n2o5"][s]["value"] for s in surfaces]
                yields = [kinetics["clno2_yield"].get(s, 0) * 100 for s in surfaces]

                fig_compare.add_trace(go.Bar(
                    x=[s.replace("_", " ") for s in surfaces],
                    y=gammas, name="gamma(N2O5)",
                    marker_color="#e74c3c",
                ))
                fig_compare.add_trace(go.Bar(
                    x=[s.replace("_", " ") for s in surfaces],
                    y=[y / 100 for y in yields], name="ClNO2 yield",
                    marker_color="#27ae60",
                ))
                fig_compare.update_layout(
                    barmode="group", title="N2O5 Uptake Coefficients and ClNO2 Yields",
                    yaxis_title="Value", height=400,
                    yaxis_type="log",
                )
                st.plotly_chart(fig_compare, use_container_width=True)

    # --- Great Salt Lake ---
    with n2o5_subtab2:
        st.subheader("Great Salt Lake Playa Dust: Halogen Factory")
        st.markdown("""
        As the Great Salt Lake shrinks, exposed saline playa dust becomes a source of
        reactive halogens (ClNO2, Cl2, BrCl) that impact regional air quality in
        Salt Lake City. **Christie et al. (2025)** showed these halogens boost morning
        ozone production.
        """)

        gcol1, gcol2 = st.columns([1, 3])
        with gcol1:
            shrinkage = st.slider("Lake shrinkage (%)", 10, 90, 50, 5, key="gsl_shrink")
            gsl_dust = st.slider("Dust loading (ug/m3)", 10, 200, 50, 5, key="gsl_dust")
            gsl_no2 = st.slider("Urban NO2 (ppb)", 5, 40, 15, 1, key="gsl_no2")
            gsl_o3 = st.slider("Background O3 (ppb)", 20, 60, 40, 1, key="gsl_o3")
            run_gsl = st.button("Run Great Salt Lake Simulation", type="primary", key="gsl_run")

        with gcol2:
            if run_gsl or "gsl_result" in st.session_state:
                if run_gsl:
                    model = GreatSaltLakeModel(
                        dust_loading_ug_m3=gsl_dust,
                        lake_shrinkage_pct=shrinkage,
                    )
                    st.session_state["gsl_result"] = model.run(
                        no2_ppb=gsl_no2, o3_ppb=gsl_o3,
                    )

                r = st.session_state["gsl_result"]
                t = r["time_hours"]

                gc1, gc2, gc3, gc4 = st.columns(4)
                with gc1:
                    st.metric("gamma(N2O5)", f"{r['gamma_n2o5']:.3f}")
                with gc2:
                    st.metric("ClNO2 yield", f"{r['clno2_yield']:.0%}")
                with gc3:
                    st.metric("Peak ClNO2", f"{np.max(r['ClNO2_ppt']):.0f} ppt")
                with gc4:
                    st.metric("O3 potential", f"{np.max(r['O3_production_potential_ppb']):.1f} ppb")

                fig_gsl = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Halogen Products", "Dawn Ozone Production Potential"),
                )
                fig_gsl.add_trace(go.Scatter(x=t, y=r["ClNO2_ppt"], name="ClNO2",
                    line=dict(color="#27ae60", width=2.5)), row=1, col=1)
                fig_gsl.add_trace(go.Scatter(x=t, y=r["Cl2_ppt"], name="Cl2",
                    line=dict(color="#e67e22", width=2)), row=1, col=1)
                fig_gsl.add_trace(go.Scatter(x=t, y=r["BrCl_ppt"], name="BrCl",
                    line=dict(color="#9b59b6", width=2)), row=1, col=1)
                fig_gsl.add_trace(go.Scatter(x=t, y=r["O3_production_potential_ppb"],
                    name="O3 potential", line=dict(color="#e74c3c", width=2.5),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.1)"), row=1, col=2)

                fig_gsl.update_xaxes(title_text="Time (hours)")
                fig_gsl.update_yaxes(title_text="ppt", row=1, col=1)
                fig_gsl.update_yaxes(title_text="ppb O3", row=1, col=2)
                fig_gsl.update_layout(height=400, title=f"Great Salt Lake ({shrinkage}% shrinkage)")
                st.plotly_chart(fig_gsl, use_container_width=True)

                # Shrinkage sensitivity
                st.subheader("Sensitivity to Lake Shrinkage")
                shrinkages = np.arange(10, 91, 5)
                peak_clno2 = []
                peak_o3_pot = []
                for s in shrinkages:
                    m = GreatSaltLakeModel(dust_loading_ug_m3=gsl_dust, lake_shrinkage_pct=s)
                    res = m.run(no2_ppb=gsl_no2, o3_ppb=gsl_o3)
                    peak_clno2.append(np.max(res["ClNO2_ppt"]))
                    peak_o3_pot.append(np.max(res["O3_production_potential_ppb"]))

                fig_sens = make_subplots(rows=1, cols=2,
                    subplot_titles=("Peak ClNO2 vs Shrinkage", "O3 Production Potential vs Shrinkage"))
                fig_sens.add_trace(go.Scatter(x=shrinkages, y=peak_clno2,
                    mode="lines+markers", line=dict(color="#27ae60", width=2.5),
                    marker=dict(size=6)), row=1, col=1)
                fig_sens.add_trace(go.Scatter(x=shrinkages, y=peak_o3_pot,
                    mode="lines+markers", line=dict(color="#e74c3c", width=2.5),
                    marker=dict(size=6)), row=1, col=2)
                fig_sens.update_xaxes(title_text="Lake Shrinkage (%)")
                fig_sens.update_yaxes(title_text="ppt", row=1, col=1)
                fig_sens.update_yaxes(title_text="ppb O3", row=1, col=2)
                fig_sens.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_sens, use_container_width=True)

            else:
                st.info("Adjust the lake shrinkage and click **Run Great Salt Lake Simulation**.")


# =============================================================================
# TAB 4: SOA Formation
# =============================================================================
with tabs[3]:
    st.header("IEPOX-SOA Formation Lab")
    st.markdown("""
    Simulate reactive uptake of isoprene-derived epoxydiols (IEPOX) on acidic sulfate
    aerosol. Explore how aerosol acidity (pH), organic coatings, and surface area
    control SOA production rates and composition.

    **Key references:** Gaston et al. (2014) *ES&T*, Zhang et al. (2018) *ES&T Letters*
    """)

    soa_subtab1, soa_subtab2 = st.tabs(["Uptake Simulation", "pH Sensitivity"])

    with soa_subtab1:
        scol1, scol2 = st.columns([1, 3])

        with scol1:
            st.subheader("Configuration")
            iepox_ppb = st.slider("IEPOX (ppb)", 0.1, 5.0, 1.0, 0.1, key="soa_iepox")
            soa_ph = st.slider("Aerosol pH", 0.0, 7.0, 1.0, 0.1, key="soa_ph")
            soa_sa = st.slider("Surface area (um2/cm3)", 50, 1000, 300, 10, key="soa_sa")
            org_coat = st.slider("Organic coating (nm)", 0, 50, 0, 1, key="soa_coat")
            soa_duration = st.slider("Duration (hours)", 1, 12, 6, 1, key="soa_dur")
            run_soa = st.button("Run SOA Simulation", type="primary", key="soa_run")

        with scol2:
            if run_soa or "soa_result" in st.session_state:
                if run_soa:
                    model = IEPOXModel()
                    st.session_state["soa_result"] = model.run_uptake(
                        iepox_ppb=iepox_ppb,
                        aerosol_surface_area=soa_sa * 1e-8,
                        pH=soa_ph,
                        organic_coating_nm=org_coat,
                        duration_hours=soa_duration,
                    )

                r = st.session_state["soa_result"]
                t = r["time_hours"]

                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("gamma(IEPOX)", f"{r['gamma']:.2e}")
                with sc2:
                    lt = r["lifetime_hours"]
                    lt_str = f"{lt:.1f} hr" if lt < 100 else f"{lt:.0f} hr"
                    st.metric("IEPOX Lifetime", lt_str)
                with sc3:
                    st.metric("Peak SOA", f"{np.max(r['soa_ug_m3']):.2f} ug/m3")
                with sc4:
                    st.metric("IEPOX consumed", f"{(1 - r['iepox_ppb'][-1]/iepox_ppb)*100:.1f}%")

                fig_soa = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=("IEPOX Decay", "SOA Mass Production", "SOA Product Speciation"),
                )

                fig_soa.add_trace(go.Scatter(x=t, y=r["iepox_ppb"], name="IEPOX",
                    line=dict(color="#2ecc71", width=2.5),
                    fill="tozeroy", fillcolor="rgba(46,204,113,0.1)"), row=1, col=1)

                fig_soa.add_trace(go.Scatter(x=t, y=r["soa_ug_m3"], name="Total SOA",
                    line=dict(color="#e74c3c", width=2.5)), row=1, col=2)
                fig_soa.add_trace(go.Scatter(x=t, y=r["organosulfate_ug_m3"],
                    name="Organosulfates", line=dict(color="#3498db", width=2, dash="dash")), row=1, col=2)
                fig_soa.add_trace(go.Scatter(x=t, y=r["methyltetrol_ug_m3"],
                    name="2-methyltetrols", line=dict(color="#f39c12", width=2, dash="dot")), row=1, col=2)

                # Pie chart for product speciation (final time)
                final_os = r["organosulfate_ug_m3"][-1]
                final_mt = r["methyltetrol_ug_m3"][-1]
                final_other = r["other_soa_ug_m3"][-1]
                total = final_os + final_mt + final_other

                if total > 0:
                    fig_soa.add_trace(go.Pie(
                        values=[final_os, final_mt, final_other],
                        labels=["Organosulfates", "2-Methyltetrols", "Other"],
                        marker_colors=["#3498db", "#f39c12", "#95a5a6"],
                        textinfo="percent+label",
                        domain=dict(x=[0.70, 1.0], y=[0.1, 0.9]),
                    ))

                fig_soa.update_xaxes(title_text="Time (hours)", row=1, col=1)
                fig_soa.update_xaxes(title_text="Time (hours)", row=1, col=2)
                fig_soa.update_yaxes(title_text="ppb", row=1, col=1)
                fig_soa.update_yaxes(title_text="ug/m3", row=1, col=2)

                fig_soa.update_layout(height=450, title=f"IEPOX-SOA Formation (pH={soa_ph}, coating={org_coat}nm)")
                st.plotly_chart(fig_soa, use_container_width=True)
            else:
                st.info("Adjust parameters and click **Run SOA Simulation** to explore IEPOX uptake.")

    with soa_subtab2:
        st.subheader("pH Sensitivity of IEPOX Uptake")
        st.markdown("""
        The uptake coefficient gamma(IEPOX) varies by **orders of magnitude** with aerosol
        acidity. At pH ~1 (ammonium bisulfate), gamma ~ 0.05; at pH ~5 (ammonium sulfate),
        gamma <= 1e-4. This is the key finding of **Gaston et al. (2014)**.
        """)

        coat_compare = st.slider("Compare organic coating thickness (nm)", 0, 40, 0, 5, key="soa_coat_cmp")

        model = IEPOXModel()
        pH_range, gammas_no_coat = model.ph_sensitivity(organic_coating_nm=0)
        _, gammas_with_coat = model.ph_sensitivity(organic_coating_nm=coat_compare)

        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(x=pH_range, y=gammas_no_coat, name="No coating",
            line=dict(color="#e74c3c", width=3)))
        if coat_compare > 0:
            fig_ph.add_trace(go.Scatter(x=pH_range, y=gammas_with_coat,
                name=f"{coat_compare} nm coating",
                line=dict(color="#3498db", width=3, dash="dash")))

        # Add reference points from Gaston et al. (2014)
        fig_ph.add_trace(go.Scatter(
            x=[1, 5], y=[0.05, 1e-4],
            mode="markers+text",
            marker=dict(size=14, color="#2ecc71", symbol="star"),
            text=["NH4HSO4", "(NH4)2SO4"],
            textposition="top center",
            name="Gaston et al. (2014)",
        ))

        fig_ph.update_yaxes(type="log", title="gamma(IEPOX)")
        fig_ph.update_xaxes(title="Aerosol pH")
        fig_ph.update_layout(
            height=450, title="IEPOX Reactive Uptake vs Aerosol Acidity",
            legend=dict(x=0.65, y=0.95),
        )
        st.plotly_chart(fig_ph, use_container_width=True)


# =============================================================================
# TAB 5: Nutrient Deposition
# =============================================================================
with tabs[4]:
    st.header("Amazon Nutrient Deposition Budget")
    st.markdown("""
    The Amazon rainforest depends on atmospheric deposition for phosphorus and iron.
    Cassandra Gaston's work revealed that **African biomass burning supplies up to 50%
    of phosphorus** deposited to the Amazon, overturning the long-held assumption that
    Saharan dust is the sole fertilizer.

    **Key references:** Barkley et al. (2019) *PNAS*, Prospero et al. (2020) *GBC*,
    Barkley et al. (2021) *GRL*, Elliott et al. (2024, 2025)
    """)

    ncol1, ncol2 = st.columns([1, 3])

    with ncol1:
        st.subheader("Deposition Fluxes")
        dust_flux = st.slider("Saharan dust (Tg/yr)", 50, 400, 180, 10, key="nut_dust")
        smoke_flux = st.slider("African smoke (Tg/yr)", 5, 60, 20, 1, key="nut_smoke")
        volcanic_freq = st.slider("Volcanic events/yr", 0.0, 1.0, 0.1, 0.05, key="nut_volc")
        volcanic_mass = st.slider("Ash per event (Tg)", 1.0, 20.0, 5.0, 1.0, key="nut_ash")
        run_nutrients = st.button("Compute Budget", type="primary", key="nut_run")

    with ncol2:
        if run_nutrients or "nut_result" in st.session_state:
            if run_nutrients:
                model = NutrientDepositionModel()
                st.session_state["nut_result"] = model.compute_annual_budget(
                    dust_flux_tg_yr=dust_flux,
                    smoke_flux_tg_yr=smoke_flux,
                    volcanic_events_per_year=volcanic_freq,
                    volcanic_ash_tg_event=volcanic_mass,
                )
                st.session_state["nut_seasonal"] = model.seasonal_deposition()

            r = st.session_state["nut_result"]
            pb = r["p_budget"]
            feb = r["fe_budget"]

            # Key finding metric
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                st.metric("Smoke P fraction", f"{r['smoke_p_fraction']*100:.0f}%",
                          help="Fraction of soluble P from biomass burning (Barkley et al. 2019: up to 50%)")
            with nc2:
                st.metric("Total soluble P", f"{pb['total_soluble_Tg']*1000:.1f} Gg/yr")
            with nc3:
                st.metric("Total soluble Fe", f"{feb['total_soluble_Tg']*1000:.1f} Gg/yr")

            # P budget chart
            fig_nut = make_subplots(
                rows=1, cols=3,
                subplot_titles=(
                    "Phosphorus Budget (Tg P/yr)",
                    "Iron Budget (Tg Fe/yr)",
                    "Seasonal Deposition Pattern",
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]],
            )

            # P budget bars
            p_sources = ["Dust\n(initial)", "Dust\n(aged)", "Smoke", "Volcanic"]
            p_values = [pb["dust_soluble_Tg"], pb["dust_aged_Tg"],
                       pb["smoke_soluble_Tg"], pb["volcanic_Tg"]]
            p_colors = ["#D4A574", "#CD853F", "#e74c3c", "#7f8c8d"]
            fig_nut.add_trace(go.Bar(
                x=p_sources, y=[v * 1000 for v in p_values],  # Convert to Gg
                marker_color=p_colors, name="P sources",
                text=[f"{v*1000:.2f}" for v in p_values], textposition="outside",
            ), row=1, col=1)

            # Fe budget bars
            fe_sources = ["Dust\n(initial)", "Dust\n(aged)", "Diatoms", "Smoke"]
            fe_values = [feb["dust_soluble_Tg"], feb["dust_aged_Tg"],
                        feb["diatom_Tg"], feb["smoke_soluble_Tg"]]
            fe_colors = ["#D4A574", "#CD853F", "#3498db", "#e74c3c"]
            fig_nut.add_trace(go.Bar(
                x=fe_sources, y=[v * 1000 for v in fe_values],
                marker_color=fe_colors, name="Fe sources",
                text=[f"{v*1000:.3f}" for v in fe_values], textposition="outside",
            ), row=1, col=2)

            # Seasonal deposition
            seas = st.session_state["nut_seasonal"]
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig_nut.add_trace(go.Scatter(
                x=month_names, y=seas["dust_relative"],
                name="Dust", line=dict(color="#D4A574", width=2.5),
            ), row=1, col=3)
            fig_nut.add_trace(go.Scatter(
                x=month_names, y=seas["smoke_relative"],
                name="Smoke", line=dict(color="#e74c3c", width=2.5),
            ), row=1, col=3)
            fig_nut.add_trace(go.Scatter(
                x=month_names, y=seas["total_relative"],
                name="Total", line=dict(color="black", width=2, dash="dash"),
            ), row=1, col=3)

            fig_nut.update_yaxes(title_text="Gg/yr", row=1, col=1)
            fig_nut.update_yaxes(title_text="Gg/yr", row=1, col=2)
            fig_nut.update_yaxes(title_text="Relative deposition", row=1, col=3)

            fig_nut.update_layout(
                height=450, showlegend=True,
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
                title="Amazon Basin Nutrient Deposition Budget",
            )
            st.plotly_chart(fig_nut, use_container_width=True)

            # Highlight PNAS finding
            st.markdown(f"""
---
**Barkley et al. (2019) PNAS Finding:**
African biomass burning smoke supplies **{r['smoke_p_fraction']*100:.0f}%** of soluble phosphorus
deposited to the Amazon in this scenario. Smoke P is more than 2x more soluble than dust P
(~10% vs ~5%), making biomass burning a critical and previously underestimated fertilizer.
""")
        else:
            st.info("Adjust deposition fluxes and click **Compute Budget** to see nutrient budgets.")


# =============================================================================
# TAB 6: Barbados Observatory
# =============================================================================
with tabs[5]:
    st.header("Barbados Atmospheric Chemistry Observatory")
    st.markdown("""
    Ragged Point, Barbados hosts one of the world's longest-running aerosol monitoring
    stations, founded by Joseph Prospero in 1966. Cassandra Gaston directs the
    **Barbados Atmospheric Chemistry Observatory (BACO)**, analyzing decades of data
    to reveal how clean air policies, African biomass burning, and changing emissions
    shape remote Atlantic aerosol composition.

    **Key reference:** Gaston et al. (2024) *ACP* - 21 years of diverging sulfate and nitrate trends
    """)

    @st.cache_data
    def load_barbados_data():
        model = BarbadosTrendsModel()
        return model.data, model.compute_trends(), model.seasonal_analysis()

    data, trends, seasonal = load_barbados_data()
    months = data["months"]

    # Trend metrics
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.metric("Sulfate trend",
                  f"{trends['sulfate_pct_change']:.0f}% over 21yr",
                  help="Decline due to US Clean Air Act & EU regulations")
    with tc2:
        st.metric("Nitrate trend",
                  f"{trends['nitrate_pct_change']:+.0f}% over 21yr",
                  help="Diverging: summer increase (African smoke), winter decrease")
    with tc3:
        st.metric("Station age", "60 years", help="Founded 1966 by J. Prospero")

    # Main time series
    fig_barb = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Non-Sea-Salt Sulfate", "Nitrate", "Mineral Dust"),
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    # Sulfate
    fig_barb.add_trace(go.Scatter(
        x=months, y=data["sulfate_ug_m3"],
        mode="lines", line=dict(color="#3498db", width=1),
        opacity=0.5, name="nss-SO4 monthly",
    ), row=1, col=1)
    # Add trend line
    so4_fit = np.polyval(np.polyfit(months, data["sulfate_ug_m3"], 1), months)
    fig_barb.add_trace(go.Scatter(
        x=months, y=so4_fit,
        mode="lines", line=dict(color="#e74c3c", width=3),
        name="Sulfate trend (declining)",
    ), row=1, col=1)

    # Nitrate
    fig_barb.add_trace(go.Scatter(
        x=months, y=data["nitrate_ug_m3"],
        mode="lines", line=dict(color="#27ae60", width=1),
        opacity=0.5, name="NO3 monthly",
    ), row=2, col=1)
    no3_fit = np.polyval(np.polyfit(months, data["nitrate_ug_m3"], 1), months)
    fig_barb.add_trace(go.Scatter(
        x=months, y=no3_fit,
        mode="lines", line=dict(color="#e74c3c", width=3),
        name="Nitrate trend",
    ), row=2, col=1)

    # Dust
    fig_barb.add_trace(go.Scatter(
        x=months, y=data["dust_ug_m3"],
        mode="lines", line=dict(color="#D4A574", width=1),
        fill="tozeroy", fillcolor="rgba(212,165,116,0.15)",
        name="Dust mass",
    ), row=3, col=1)

    fig_barb.update_yaxes(title_text="ug/m3", row=1, col=1)
    fig_barb.update_yaxes(title_text="ug/m3", row=2, col=1)
    fig_barb.update_yaxes(title_text="ug/m3", row=3, col=1)
    fig_barb.update_xaxes(title_text="Year", row=3, col=1)

    fig_barb.update_layout(
        height=700,
        title="Barbados Aerosol Time Series (1990-2011)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_barb, use_container_width=True)

    # Seasonal climatology
    st.subheader("Monthly Climatology")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_seas = make_subplots(rows=1, cols=3,
        subplot_titles=("Sulfate", "Nitrate", "Dust"))

    fig_seas.add_trace(go.Bar(
        x=month_names, y=seasonal["sulfate_monthly_mean"],
        marker_color="#3498db", name="nss-SO4",
    ), row=1, col=1)
    fig_seas.add_trace(go.Bar(
        x=month_names, y=seasonal["nitrate_monthly_mean"],
        marker_color="#27ae60", name="NO3",
    ), row=1, col=2)
    fig_seas.add_trace(go.Bar(
        x=month_names, y=seasonal["dust_monthly_mean"],
        marker_color="#D4A574", name="Dust",
    ), row=1, col=3)

    fig_seas.update_yaxes(title_text="ug/m3")
    fig_seas.update_layout(height=350, showlegend=False, title="Monthly Mean Concentrations")
    st.plotly_chart(fig_seas, use_container_width=True)

    st.markdown("""
---
**Key insight from Gaston et al. (2024):** The decline in non-sea-salt sulfate
at Barbados is a clear signal of the US Clean Air Act and EU emission regulations
reducing SO2 across the North Atlantic. The diverging nitrate trends point to
increasing African biomass burning influence in summer and declining shipping/industrial
NOx in winter.
""")


# =============================================================================
# TAB 7: CCN & Clouds
# =============================================================================
with tabs[6]:
    st.header("CCN Activation & Cloud Formation")
    st.markdown("""
    Cloud Condensation Nuclei (CCN) activity determines which particles form cloud droplets.
    The hygroscopicity parameter kappa varies enormously across aerosol types, from
    sea salt (kappa~1.1) to mineral dust (kappa~0.05). Explore kappa-Kohler theory
    and compare how different aerosols activate into cloud droplets.

    **Key references:** Gaston et al. (2018) *Atmosphere*, Pohlker et al. (2023)
    *Nat. Comm.*, Gaston et al. (2023) *ACP*
    """)

    ccn_subtab1, ccn_subtab2 = st.tabs([
        "Activation Spectra",
        "Kohler Curves",
    ])

    ccn_model = CCNActivationModel()
    ccn_params = get_ccn_parameters()

    with ccn_subtab1:
        st.subheader("Critical Supersaturation vs Particle Size")
        st.markdown("""
        Smaller particles and less hygroscopic compositions require higher supersaturations
        to activate. Typical cloud supersaturations range from 0.1% to 1%.
        """)

        selected_types = st.multiselect(
            "Select aerosol types to compare",
            list(ccn_params["kappa"].keys()),
            default=["Sea_salt", "Ammonium_sulfate", "Biomass_burning", "Mineral_dust", "Aged_dust_sea_salt"],
            key="ccn_types",
        )

        ss_threshold = st.slider("Cloud supersaturation threshold (%)", 0.05, 1.0, 0.3, 0.05, key="ccn_ss")

        fig_ccn = go.Figure()

        colors = px.colors.qualitative.Set2
        for i, atype in enumerate(selected_types):
            kappa = ccn_params["kappa"][atype]["mean"]
            diams, Sc = ccn_model.activation_spectrum(kappa)
            fig_ccn.add_trace(go.Scatter(
                x=diams, y=Sc,
                name=f"{atype.replace('_', ' ')} (kappa={kappa:.2f})",
                line=dict(color=colors[i % len(colors)], width=2.5),
            ))

        # Add supersaturation threshold
        fig_ccn.add_hline(y=ss_threshold, line_dash="dash", line_color="gray",
                         annotation_text=f"S = {ss_threshold}%")
        fig_ccn.add_hrect(y0=0, y1=ss_threshold, fillcolor="rgba(100,200,100,0.05)",
                         line_width=0)

        fig_ccn.update_xaxes(type="log", title="Dry Diameter (nm)")
        fig_ccn.update_yaxes(type="log", title="Critical Supersaturation (%)",
                            range=[-2, 1.5])
        fig_ccn.update_layout(
            height=550, title="CCN Activation Spectra",
            legend=dict(x=0.55, y=0.95),
        )
        st.plotly_chart(fig_ccn, use_container_width=True)

        # Comparison table at fixed size
        st.subheader("Comparison at 100 nm Dry Diameter")
        comparison = ccn_model.compare_aerosol_types(100)
        table_data = []
        for name, vals in comparison.items():
            table_data.append({
                "Aerosol Type": name.replace("_", " "),
                "kappa": f"{vals['kappa']:.2f}",
                "Sc (%)": f"{vals['Sc_percent']:.3f}",
                "Activates at 0.3%": "Yes" if vals["activates_at_0.3"] else "No",
                "Activates at 0.1%": "Yes" if vals["activates_at_0.1"] else "No",
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    with ccn_subtab2:
        st.subheader("Kohler Curves")
        st.markdown("""
        The Kohler curve shows the equilibrium supersaturation as a function of droplet
        growth factor. The peak of the curve is the critical supersaturation - the
        barrier a particle must overcome to activate into a cloud droplet.
        """)

        kcol1, kcol2 = st.columns([1, 3])
        with kcol1:
            kohler_type = st.selectbox("Aerosol type", list(ccn_params["kappa"].keys()), key="kohler_type")
            kohler_diam = st.slider("Dry diameter (nm)", 20, 500, 100, 10, key="kohler_diam")

        with kcol2:
            kappa_val = ccn_params["kappa"][kohler_type]["mean"]
            gf, ss = ccn_model.kohler_curve(kohler_diam, kappa_val)

            fig_kohler = go.Figure()
            fig_kohler.add_trace(go.Scatter(
                x=gf, y=ss,
                mode="lines", line=dict(color="#2980b9", width=3),
                name="Kohler curve",
            ))

            # Mark critical point
            max_idx = np.argmax(ss)
            fig_kohler.add_trace(go.Scatter(
                x=[gf[max_idx]], y=[ss[max_idx]],
                mode="markers+text",
                marker=dict(size=14, color="#e74c3c", symbol="star"),
                text=[f"Sc = {ss[max_idx]:.3f}%"],
                textposition="top center",
                name="Critical point",
            ))

            fig_kohler.add_hline(y=0, line_color="gray", line_dash="dash")

            fig_kohler.update_xaxes(title="Growth Factor (Dwet/Ddry)")
            fig_kohler.update_yaxes(title="Supersaturation (%)", range=[min(-0.5, min(ss)*1.1), max(ss)*1.5])
            fig_kohler.update_layout(
                height=450,
                title=f"Kohler Curve: {kohler_type.replace('_',' ')} (Dd={kohler_diam}nm, kappa={kappa_val:.2f})",
            )
            st.plotly_chart(fig_kohler, use_container_width=True)


# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em;'>
    <b>Atmospheric Chemistry Transport Model</b><br>
    Simulations based on the research of Cassandra J. Gaston, University of Miami Rosenstiel School<br>
    All models use embedded data and lightweight box models optimized for interactive exploration<br>
    <a href="https://people.miami.edu/profile/cjg174@miami.edu">Gaston Lab</a> |
    <a href="https://baco.earth.miami.edu/">Barbados Atmospheric Chemistry Observatory</a>
</div>
""", unsafe_allow_html=True)

# Code reference
with st.expander("Code Reference"):
    st.code("""
# Using the atmospheric chemistry models
from weatherflow.chemistry.models import (
    DustTransportModel, N2O5BoxModel, IEPOXModel,
    NutrientDepositionModel, CCNActivationModel,
    BarbadosTrendsModel, GreatSaltLakeModel,
)

# 1. Simulate transatlantic dust transport
dust = DustTransportModel(transport_days=5, initial_dust_conc=200)
result = dust.run()  # Returns time series of aging, nutrients, mixing

# 2. Nighttime N2O5 chemistry
n2o5 = N2O5BoxModel(aerosol_type="NaCl", aerosol_surface_area=200e-6)
result = n2o5.run(duration_hours=12, no2_ppb=5, o3_ppb=30)

# 3. IEPOX-SOA formation
iepox = IEPOXModel()
result = iepox.run_uptake(iepox_ppb=1.0, pH=1.0, organic_coating_nm=0)

# 4. Amazon nutrient budget
nutrients = NutrientDepositionModel()
budget = nutrients.compute_annual_budget(dust_flux_tg_yr=180)

# 5. CCN activation
ccn = CCNActivationModel()
diameters, Sc = ccn.activation_spectrum(kappa=0.61)  # Ammonium sulfate

# 6. Barbados trends
barbados = BarbadosTrendsModel()
trends = barbados.compute_trends()

# 7. Great Salt Lake halogens
gsl = GreatSaltLakeModel(lake_shrinkage_pct=50)
result = gsl.run(no2_ppb=15, o3_ppb=40)
    """, language="python")
