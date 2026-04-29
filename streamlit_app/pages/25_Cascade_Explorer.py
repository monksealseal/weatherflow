"""Cascade Explorer — a linked map / table / drill-down view.

Each level answers exactly one question. A single click on a marker, a
table row, or a button moves you one level deeper. The breadcrumb at
the top is always the way back.

Levels:
    1. world       — "Where should I focus my attention?"        (geographic)
    2. datacenter  — "Which servers in this building need help?" (physical, not geographic)
    3. server      — "What is happening on this machine?"        (time series)

Same pattern works for any data with a "place → thing → measurement"
hierarchy (weather stations → sensors → readings, stores → SKUs → sales,
hospitals → wards → patients, ...).
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade import render_datacenter, render_server, render_world  # noqa: E402

st.set_page_config(page_title="Cascade Explorer", page_icon="🧭", layout="wide")

st.session_state.setdefault("cascade_level", "world")
st.session_state.setdefault("cascade_dc_id", None)
st.session_state.setdefault("cascade_server_id", None)


def _go(level: str, **kwargs) -> None:
    st.session_state.cascade_level = level
    for k, v in kwargs.items():
        st.session_state[f"cascade_{k}"] = v
    st.rerun()


# ---- breadcrumb -------------------------------------------------------------

level = st.session_state.cascade_level
dc_id = st.session_state.cascade_dc_id
server_id = st.session_state.cascade_server_id

crumbs = st.container()
with crumbs:
    cols = st.columns([1, 1, 1, 6])
    if cols[0].button("🌍 World", use_container_width=True):
        _go("world", dc_id=None, server_id=None)
    if dc_id and cols[1].button(f"🏢 {dc_id}", use_container_width=True):
        _go("datacenter", server_id=None)
    if server_id and cols[2].button(f"🖥 {server_id}", use_container_width=True):
        _go("server")

st.divider()

# ---- route ------------------------------------------------------------------

if level == "world":
    render_world()
elif level == "datacenter" and dc_id:
    render_datacenter(dc_id)
elif level == "server" and dc_id and server_id:
    render_server(dc_id, server_id)
else:
    st.warning("Lost the trail — heading back to the world view.")
    _go("world", dc_id=None, server_id=None)

# ---- footer / explainer -----------------------------------------------------

with st.sidebar:
    st.markdown("### Cascade Explorer")
    st.markdown(
        "A geographic map blended with structured data. Click any marker, "
        "row, or button to **transport** one level deeper. Each level "
        "answers a single question — so you always know what you're "
        "looking at and why."
    )
    st.markdown("---")
    st.markdown("**Levels**")
    st.markdown(
        "- 🌍 **World** — where to focus\n"
        "- 🏢 **Data center** — which server needs help (no longer geographic — "
        "we're inside a building)\n"
        "- 🖥 **Server** — what's happening, is it getting worse"
    )
    st.markdown("---")
    st.caption(
        "Demo data is synthetic and deterministic. Swap `cascade/data.py` "
        "for SQL queries to point this at a real Postgres source."
    )
