"""Cascade Explorer — IT Director's Control Room.

A spatial-web exploration tool. Every level opens with one plain-language
question, every level changes shape (map → tile board → bridge → human →
ticket), and the breadcrumb is the trail you walked.

Drilling is unbounded — adding a new level means writing one render
function and adding it to LEVELS below.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade import (  # noqa: E402
    render_control_room,
    render_customer,
    render_datacenter,
    render_help_desk_board,
    render_incident,
    render_incidents_board,
    render_people_board,
    render_person,
    render_placeholder,
    render_project,
    render_server,
    render_service,
    render_team,
    render_ticket,
    render_world,
    setup_page,
)

st.set_page_config(page_title="Control Room", page_icon="🧭", layout="wide")

# ── level dispatcher ────────────────────────────────────────────────────────
# Each entry maps a level name → render function. Adding a level is one line.

LEVELS = {
    "control_room":    (render_control_room,    "🧭  Control Room"),
    "infrastructure":  (render_world,           "🌐  Infrastructure"),
    "incidents_board": (render_incidents_board, "🚨  Incidents"),
    "people_board":    (render_people_board,    "👥  People"),
    "help_desk_board": (render_help_desk_board, "🎫  Help Desk"),
    "datacenter":      (render_datacenter,      "🏢  Site"),
    "server":          (render_server,          "🖥  Server"),
    "incident":        (render_incident,        "🚨  Incident"),
    "service":         (render_service,         "⚙️  Service"),
    "team":            (render_team,            "👥  Team"),
    "person":          (render_person,          "👤  Person"),
    "project":         (render_project,         "📦  Project"),
    "ticket":          (render_ticket,          "🎟  Ticket"),
    "customer":        (render_customer,        "🏷  Customer"),
    "placeholder":     (render_placeholder,     "•   Domain"),
}


def _crumb_label(entry):
    level = entry["level"]
    params = entry["params"]
    base = LEVELS.get(level, (None, level))[1]
    if "dc_id" in params:
        return f"{base} · {params['dc_id']}"
    if "server_id" in params:
        return f"{base} · {params['server_id'].split('-')[-1]}"
    if "incident_id" in params:
        return f"{base} · {params['incident_id']}"
    if "service_id" in params:
        return f"{base} · {params['service_id']}"
    if "team_id" in params:
        return f"{base} · {params['team_id']}"
    if "person_id" in params:
        return f"{base} · {params['person_id']}"
    if "project_id" in params:
        return f"{base} · {params['project_id']}"
    if "ticket_id" in params:
        return f"{base} · {params['ticket_id'].split('-T')[-1]}"
    if "customer_id" in params:
        return f"{base} · {params['customer_id']}"
    return base


# ── session state ───────────────────────────────────────────────────────────

if "cascade_stack" not in st.session_state:
    st.session_state.cascade_stack = [{"level": "control_room", "params": {}}]

stack = st.session_state.cascade_stack

# ── breadcrumb (the trail you walked) ───────────────────────────────────────

crumb_cols = st.columns(len(stack) + 1)
for i, entry in enumerate(stack):
    label = _crumb_label(entry)
    is_last = i == len(stack) - 1
    if crumb_cols[i].button(
        label,
        key=f"crumb-{i}",
        use_container_width=True,
        type=("primary" if is_last else "secondary"),
        disabled=is_last,
    ):
        st.session_state.cascade_stack = stack[: i + 1]
        st.rerun()
if len(stack) > 1:
    if crumb_cols[-1].button("↑ Up", key="crumb-up", use_container_width=True):
        st.session_state.cascade_stack = stack[:-1]
        st.rerun()

st.divider()

# ── render current level ────────────────────────────────────────────────────

setup_page()
top = stack[-1]
render_fn, _ = LEVELS.get(top["level"], (render_placeholder, "?"))
try:
    render_fn(**top["params"])
except TypeError:
    # Fallback if a level is invoked with unexpected params during dev.
    render_fn()

# ── side panel ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Control Room")
    st.markdown(
        "A personal command surface for an IT Director — every domain you "
        "carried, one screen, drillable to any depth."
    )
    st.markdown("---")
    st.markdown(f"**Trail depth:**  {len(stack)} level{'s' if len(stack) != 1 else ''}")
    st.markdown(f"**Current:**  {_crumb_label(top)}")
    st.markdown("---")
    st.markdown(
        "**Levels reachable**\n"
        "1. 🧭  Control Room\n"
        "2. 🌐 / 🚨 / 👥 / 🎫  Domain board\n"
        "3. 🏢  Site\n"
        "4. 🖥  Server\n"
        "5. 🚨  Incident bridge\n"
        "6. ⚙️  Service · 👥 Team · 🏷 Customer\n"
        "7. 👤  Person\n"
        "8. 📦  Project\n"
        "9. 🎟  Ticket\n"
        "10. (loops back to people / services …)"
    )
    st.markdown("---")
    if st.button("Reset to Control Room", use_container_width=True):
        st.session_state.cascade_stack = [{"level": "control_room", "params": {}}]
        st.rerun()
    st.caption(
        "Demo data is synthetic and deterministic. Add a level by writing "
        "one render function and one row in `LEVELS`."
    )
