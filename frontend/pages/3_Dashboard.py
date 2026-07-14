"""
frontend/pages/3_Dashboard.py

Risk Analytics Dashboard: aggregate charts over prediction history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from frontend.utils.api_client import ApiError, get_model_information, get_prediction_history
from frontend.utils.styling import apply_dark_theme, render_metric_box

st.set_page_config(page_title="Dashboard · ReturnGuard AI", page_icon="📈", layout="wide")
apply_dark_theme()

if not st.session_state.get("authenticated"):
    st.warning("Please log in from the main page first.")
    st.stop()

st.title("📈 Risk Analytics Dashboard")
st.caption("Aggregate view of fraud risk across all recorded predictions.")

try:
    history = get_prediction_history(limit=500)
    info = get_model_information()
except ApiError as exc:
    st.error(f"Could not load dashboard data: {exc.detail}")
    st.stop()
except Exception as exc:
    st.error(f"Could not reach backend: {exc}")
    st.stop()

items = history["items"]
if not items:
    st.info("No predictions recorded yet. Make some predictions first to populate this dashboard.")
    st.stop()

df = pd.DataFrame(items)
df["created_at"] = pd.to_datetime(df["created_at"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_metric_box("Total Predictions", f"{len(df):,}"), unsafe_allow_html=True)
with c2:
    high_pct = (df["risk_level"] == "HIGH").mean() * 100
    st.markdown(render_metric_box("High Risk %", f"{high_pct:.1f}%"), unsafe_allow_html=True)
with c3:
    st.markdown(render_metric_box("Avg Risk Score", f"{df['risk_score'].mean():.1f}"), unsafe_allow_html=True)
with c4:
    auto_approve_pct = (df["recommended_action"] == "AUTO_APPROVE").mean() * 100
    st.markdown(render_metric_box("Auto-Approve %", f"{auto_approve_pct:.1f}%"), unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Level Distribution")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    fig = px.pie(
        risk_counts, names="risk_level", values="count", hole=0.5,
        color="risk_level",
        color_discrete_map={"LOW": "#2ECC71", "MEDIUM": "#F1C40F", "HIGH": "#E74C3C"},
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6E6E6")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Recommended Actions")
    action_counts = df["recommended_action"].value_counts().reset_index()
    action_counts.columns = ["action", "count"]
    fig2 = px.bar(action_counts, x="action", y="count", color="action")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6E6E6", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Prediction Timeline")
timeline = df.set_index("created_at").sort_index()
timeline["rolling_avg_risk"] = timeline["risk_score"].rolling(10, min_periods=1).mean()
fig3 = px.line(timeline.reset_index(), x="created_at", y="rolling_avg_risk", title="Rolling Average Risk Score (window=10)")
fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6E6E6")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Fraud Probability Distribution")
fig4 = px.histogram(df, x="fraud_probability", nbins=30, color="risk_level",
                     color_discrete_map={"LOW": "#2ECC71", "MEDIUM": "#F1C40F", "HIGH": "#E74C3C"})
fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6E6E6")
st.plotly_chart(fig4, use_container_width=True)
