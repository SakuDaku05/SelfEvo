"""
dashboard/app.py — Rich Streamlit dashboard for the MASC/SePO framework.

Run with:
    streamlit run dashboard/app.py

Features
--------
* Overview KPI cards (total runs, error rate, correction rate, avg latency)
* Per-agent drill-down with success/failure/correction breakdowns
* Anomaly type distribution chart
* Evolution history table with prompt diffs
* Live log viewer with filtering
* MASC rule registry display
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# ── path setup so dashboard can import project modules ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import streamlit as st

# ── page config (MUST be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="MASC / SePO Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── imports after path setup ─────────────────────────────────────────────
from logs.logger import Logger
from evolution.evolution_tracker import EvolutionTracker
from interceptor.masc_validator import MASCValidator

# ======================================================================= #
# Config                                                                   #
# ======================================================================= #
LOG_PATH = os.path.join(_ROOT, "logs", "agent_logs.json")
EVOLUTION_PATH = os.path.join(_ROOT, "logs", "evolution_history.jsonl")

logger = Logger(log_path=LOG_PATH)
tracker = EvolutionTracker(history_path=EVOLUTION_PATH)
validator = MASCValidator()

# ======================================================================= #
# Styling                                                                  #
# ======================================================================= #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #a78bfa;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #8b8ba7;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 3px solid #a78bfa;
        padding-left: 10px;
        margin: 24px 0 12px 0;
    }

    /* Anomaly badge */
    .anomaly-badge {
        background: #3b1f4a;
        color: #e879f9;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Log entry row */
    .log-corrected { background-color: #1a3a2a; border-radius: 6px; padding: 4px 8px; }
    .log-error     { background-color: #3a1a1a; border-radius: 6px; padding: 4px 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================================= #
# Sidebar                                                                  #
# ======================================================================= #
with st.sidebar:
    st.markdown("## 🧠 MASC / SePO")
    st.markdown("**Agent Monitoring Dashboard**")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔎 Agent Drill-Down", "🔬 Evolution History", "📜 Live Logs", "🛡️ MASC Rules"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("🔄 Refresh Data"):
        st.rerun()

# ======================================================================= #
# Data loading                                                             #
# ======================================================================= #
@st.cache_data(ttl=10)
def load_stats() -> Dict:
    return logger.aggregate_stats()

@st.cache_data(ttl=10)
def load_logs() -> List[Dict]:
    return logger.all_logs()

@st.cache_data(ttl=10)
def load_evolution() -> Dict:
    return tracker.summary()

@st.cache_data(ttl=10)
def load_evolution_timeline() -> List[Dict]:
    return tracker.evolution_timeline()

stats = load_stats()
logs = load_logs()
evo_summary = load_evolution()
evo_timeline = load_evolution_timeline()

g = stats.get("global", {})
agents_stats = stats.get("agents", {})


# ======================================================================= #
# Helper: KPI card                                                         #
# ======================================================================= #
def kpi(col, value: Any, label: str) -> None:
    col.markdown(
        f"""<div class="kpi-card">
               <p class="kpi-value">{value}</p>
               <p class="kpi-label">{label}</p>
            </div>""",
        unsafe_allow_html=True,
    )


# ======================================================================= #
# Pages                                                                    #
# ======================================================================= #

# ─── Overview ─────────────────────────────────────────────────────────── #
if page == "📊 Overview":
    st.title("📊 System Overview")

    if not g:
        st.info("No runs logged yet.  Register some agents and send a query!")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, g.get("total_runs", 0), "Total Runs")
        kpi(c2, f"{g.get('error_rate', 0)*100:.1f}%", "Error Rate")
        kpi(c3, f"{g.get('correction_rate', 0)*100:.1f}%", "Correction Rate")
        kpi(c4, f"{g.get('avg_latency_ms', 0):.0f} ms", "Avg Latency")
        kpi(c5, evo_summary.get("total_evolutions", 0), "SePO Evolutions")

        st.markdown('<p class="section-header">Per-Agent Summary</p>', unsafe_allow_html=True)

        rows = []
        for aid, as_ in agents_stats.items():
            rows.append({
                "Agent": aid,
                "Runs": as_["total_runs"],
                "Errors": as_["error_count"],
                "Error %": f"{as_['error_rate']*100:.1f}%",
                "Corrections": as_["correction_count"],
                "Correction %": f"{as_['correction_rate']*100:.1f}%",
                "Avg Latency (ms)": as_["avg_latency_ms"],
            })
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-header">Anomaly Type Distribution</p>', unsafe_allow_html=True)
        anomaly_dist = g.get("anomaly_type_counts", {})
        if anomaly_dist:
            import pandas as pd
            df_a = pd.DataFrame(
                list(anomaly_dist.items()), columns=["Anomaly Type", "Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(df_a.set_index("Anomaly Type"))
        else:
            st.success("✅ No anomalies detected in any run.")


# ─── Agent Drill-Down ─────────────────────────────────────────────────── #
elif page == "🔎 Agent Drill-Down":
    st.title("🔎 Agent Drill-Down")

    if not agents_stats:
        st.info("No agent runs found.")
    else:
        selected = st.selectbox("Select Agent", list(agents_stats.keys()))
        a = agents_stats[selected]

        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, a["total_runs"], "Total Runs")
        kpi(c2, f"{a['error_rate']*100:.1f}%", "Error Rate")
        kpi(c3, f"{a['correction_rate']*100:.1f}%", "Correction Rate")
        kpi(c4, f"{a['avg_latency_ms']:.0f} ms", "Avg Latency")

        # Anomaly breakdown for this agent
        if a.get("anomaly_type_counts"):
            st.markdown('<p class="section-header">Anomaly Breakdown</p>', unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame(
                list(a["anomaly_type_counts"].items()),
                columns=["Anomaly Type", "Count"],
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Recent logs for this agent
        st.markdown('<p class="section-header">Recent Runs</p>', unsafe_allow_html=True)
        agent_logs = [l for l in logs if l.get("agent_id") == selected][-20:]
        for entry in reversed(agent_logs):
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            corrected = "🔧" if entry.get("corrected") else "✅"
            error = "❌" if entry.get("error") else ""
            anomaly = entry.get("anomaly") or {}
            atype = anomaly.get("type", "") if isinstance(anomaly, dict) else str(anomaly)
            with st.expander(f"{corrected}{error} {ts}  —  {entry.get('query', '')[:60]}"):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Latency:** {entry.get('latency_ms', 0):.0f} ms")
                if atype:
                    col1.markdown(f"**Anomaly:** `{atype}`")
                if entry.get("error"):
                    col2.error(entry["error"])
                col2.json(entry.get("output", {}))


# ─── Evolution History ────────────────────────────────────────────────── #
elif page == "🔬 Evolution History":
    st.title("🔬 Evolution History")

    c1, c2, c3 = st.columns(3)
    kpi(c1, evo_summary.get("total_evolutions", 0), "Total Evolutions")
    kpi(c2, len(evo_summary.get("agents_evolved", [])), "Agents Evolved")
    kpi(c3, len(evo_summary.get("anomaly_type_distribution", {})), "Distinct Anomaly Types")

    if evo_timeline:
        st.markdown('<p class="section-header">Evolution Timeline</p>', unsafe_allow_html=True)

        agent_filter = st.selectbox(
            "Filter by Agent",
            ["All"] + sorted({r["agent_id"] for r in evo_timeline}),
        )
        filtered = (
            evo_timeline
            if agent_filter == "All"
            else [r for r in evo_timeline if r["agent_id"] == agent_filter]
        )

        for ev in reversed(filtered):
            ts = (ev.get("timestamp") or "")[:19].replace("T", " ")
            with st.expander(
                f"[{ev.get('agent_id')}]  {ts}  —  anomaly: {ev.get('anomaly_type')}  "
                f"({'🤖 LLM' if ev.get('method') == 'llm' else '🔧 Heuristic'})"
            ):
                # Load full record for prompt diff
                records = tracker.for_agent(ev["agent_id"])
                match = next(
                    (r for r in records if r.get("timestamp") == ev.get("timestamp")), None
                )
                if match:
                    uid = match.get("timestamp", str(id(match)))
                    col1, col2 = st.columns(2)
                    col1.markdown("**Old Prompt**")
                    col1.text_area("", match.get("old_prompt", ""), height=120,
                                   disabled=True, key=f"old_{uid}")
                    col2.markdown("**New Prompt**")
                    col2.text_area("", match.get("new_prompt", ""), height=120,
                                   disabled=True, key=f"new_{uid}")
    else:
        st.info("No evolution events recorded yet.")


# ─── Live Logs ────────────────────────────────────────────────────────── #
elif page == "📜 Live Logs":
    st.title("📜 Live Logs")

    n_logs = st.slider("Number of recent entries", 10, 500, 50)
    filter_agent = st.text_input("Filter by agent_id (leave blank for all)")
    filter_corrected = st.checkbox("Only show corrected runs")
    filter_errors = st.checkbox("Only show runs with errors")

    recent = logger.recent(n_logs)
    if filter_agent:
        recent = [l for l in recent if filter_agent.lower() in l.get("agent_id", "").lower()]
    if filter_corrected:
        recent = [l for l in recent if l.get("corrected")]
    if filter_errors:
        recent = [l for l in recent if l.get("error")]

    st.markdown(f"**Showing {len(recent)} entries**")
    st.dataframe(
        [
            {
                "Time": (l.get("timestamp") or "")[:19].replace("T", " "),
                "Agent": l.get("agent_id"),
                "Query": str(l.get("query", ""))[:80],
                "Corrected": l.get("corrected"),
                "Anomaly": (l.get("anomaly") or {}).get("type", "") if isinstance(l.get("anomaly"), dict) else "",
                "Error": str(l.get("error") or "")[:60],
                "Latency (ms)": l.get("latency_ms"),
            }
            for l in reversed(recent)
        ],
        use_container_width=True,
        hide_index=True,
    )


# ─── MASC Rules ───────────────────────────────────────────────────────── #
elif page == "🛡️ MASC Rules":
    st.title("🛡️ Active MASC Validation Rules")
    st.markdown(
        "These rules run on every agent output in order.  "
        "Add custom rules via `MASCValidator.add_rule()`."
    )
    for i, rule in enumerate(validator._rules):
        with st.expander(f"{i+1}. `{rule.name}`"):
            st.write(rule.description or "_No description provided._")
            st.caption(f"Class: `{type(rule).__name__}`")
