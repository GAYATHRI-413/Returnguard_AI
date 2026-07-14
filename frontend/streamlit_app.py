"""
frontend/streamlit_app.py

ReturnGuard AI - Streamlit Frontend (entrypoint / login + home).

Run:
    streamlit run frontend/streamlit_app.py

Additional pages live in frontend/pages/ and appear automatically in the
sidebar navigation (Streamlit's native multipage app convention).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from frontend.utils.api_client import ApiError, get_health, get_model_information
from frontend.utils.styling import apply_dark_theme, render_brand_header, render_metric_box

st.set_page_config(
    page_title="ReturnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_dark_theme()

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


def login_screen() -> None:
    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
        render_brand_header(subtitle="Sign in to access the risk console")
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        st.markdown("<span class='auth-badge'>Secure Access</span>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.markdown("</div>", unsafe_allow_html=True)


def dashboard_home() -> None:
    render_brand_header(subtitle="Real-time return fraud risk intelligence for your e-commerce operations")

    try:
        health = get_health()
        info = get_model_information()
    except ApiError as exc:
        st.error(f"Could not reach the backend API: {exc.detail}")
        st.info(
            "Make sure the FastAPI backend is running:\n\n"
            "`uvicorn backend.app.main:app --reload`"
        )
        return
    except Exception as exc:
        st.error(f"Unexpected error contacting backend: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_box("System Status", "🟢 Online" if health["status"] == "ok" else "🟡 Degraded"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_box("Best Model", info["best_model"].replace("_", " ").title()), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_box("Dataset Size", f"{info['dataset_size']:,}" if info["dataset_size"] else "N/A"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_box("Base Fraud Rate", f"{info['fraud_rate']*100:.1f}%" if info["fraud_rate"] else "N/A"), unsafe_allow_html=True)

    st.divider()

    st.subheader("📋 Navigate")
    st.markdown(
        """
        Use the sidebar to access:
        - **1 · Prediction** — Score a new return request in real time
        - **2 · History** — Browse past predictions and their outcomes
        - **3 · Dashboard** — Risk analytics and trends
        - **4 · Model Metrics** — Model comparison, SHAP importance, confusion matrix
        """
    )

    st.subheader("🧠 Models Trained")
    cols = st.columns(len(info["all_models_trained"]) or 1)
    for col, model_name in zip(cols, info["all_models_trained"]):
        with col:
            is_best = model_name == info["best_model"]
            label = model_name.replace("_", " ").title()
            st.markdown(
                f"<div class='metric-box'><h2>{'⭐' if is_best else '🔹'}</h2><p>{label}</p></div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_screen()
        return

    with st.sidebar:
        st.markdown(f"👤 **{st.session_state.get('username', 'admin')}**")
        if st.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()
        st.divider()

    dashboard_home()


if __name__ == "__main__":
    main()
