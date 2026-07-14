"""
frontend/pages/2_History.py

Prediction History: browse and filter past predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from frontend.utils.api_client import ApiError, get_prediction_history
from frontend.utils.styling import apply_dark_theme

st.set_page_config(page_title="History · ReturnGuard AI", page_icon="🕒", layout="wide")
apply_dark_theme()

if not st.session_state.get("authenticated"):
    st.warning("Please log in from the main page first.")
    st.stop()

st.title("🕒 Prediction History")
st.caption("Browse recent return fraud predictions made by the system.")

c1, c2 = st.columns([1, 3])
limit = c1.slider("Rows to load", 10, 500, 100, 10)

try:
    data = get_prediction_history(limit=limit)
except ApiError as exc:
    st.error(f"Could not load history: {exc.detail}")
    st.stop()
except Exception as exc:
    st.error(f"Could not reach backend: {exc}")
    st.stop()

items = data["items"]
if not items:
    st.info("No predictions recorded yet. Make a prediction from the Prediction page first.")
    st.stop()

df = pd.DataFrame(items)
df["created_at"] = pd.to_datetime(df["created_at"])

st.subheader(f"Showing {len(df)} of {data['total']} predictions")

risk_filter = st.multiselect("Filter by risk level", options=["LOW", "MEDIUM", "HIGH"], default=["LOW", "MEDIUM", "HIGH"])
filtered = df[df["risk_level"].isin(risk_filter)]

def _highlight_risk(row):
    color_map = {"LOW": "background-color: rgba(46,204,113,0.12)",
                 "MEDIUM": "background-color: rgba(241,196,15,0.12)",
                 "HIGH": "background-color: rgba(231,76,60,0.12)"}
    return [color_map.get(row["risk_level"], "")] * len(row)

st.dataframe(
    filtered[[
        "created_at", "request_id", "customer_code", "seller_code",
        "fraud_probability", "risk_score", "risk_level", "recommended_action", "model_used",
    ]].style.apply(_highlight_risk, axis=1),
    use_container_width=True,
    height=500,
)

with st.expander("🔎 Inspect a single prediction's explanation"):
    request_ids = filtered["request_id"].tolist()
    if request_ids:
        chosen = st.selectbox("Request ID", request_ids)
        row = filtered[filtered["request_id"] == chosen].iloc[0]
        st.write(f"**Risk Level:** {row['risk_level']}  |  **Action:** {row['recommended_action']}")
        st.write(f"**Business Explanation:** {row['business_explanation']}")
