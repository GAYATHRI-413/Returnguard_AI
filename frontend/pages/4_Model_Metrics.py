"""
frontend/pages/4_Model_Metrics.py

Model Metrics: comparison table across all 4 trained models, global feature
importance (SHAP), and the saved training plots (confusion matrix, ROC,
precision-recall, accuracy/loss curves).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from frontend.utils.api_client import ApiError, get_feature_importance, get_model_metrics, trigger_retrain
from frontend.utils.styling import apply_dark_theme

st.set_page_config(page_title="Model Metrics · ReturnGuard AI", page_icon="🧪", layout="wide")
apply_dark_theme()

if not st.session_state.get("authenticated"):
    st.warning("Please log in from the main page first.")
    st.stop()

st.title("🧪 Model Metrics & Explainability")

try:
    metrics_data = get_model_metrics()
except ApiError as exc:
    st.error(f"Could not load model metrics: {exc.detail}")
    st.stop()
except Exception as exc:
    st.error(f"Could not reach backend: {exc}")
    st.stop()

st.subheader("📊 Model Comparison")
df = pd.DataFrame(metrics_data["models"])
df["model_name"] = df["model_name"].str.replace("_", " ").str.title()
st.dataframe(
    df[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "is_best_model"]]
    .sort_values("roc_auc", ascending=False)
    .style.highlight_max(subset=["roc_auc"], color="rgba(46,204,113,0.25)"),
    use_container_width=True,
)

fig = px.bar(df, x="model_name", y=["accuracy", "precision", "recall", "f1_score", "roc_auc"], barmode="group")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6E6E6")
st.plotly_chart(fig, use_container_width=True)

st.caption(f"⭐ Best model currently deployed: **{metrics_data['best_model'].replace('_', ' ').title()}**")

st.divider()

st.subheader("🧠 Global Feature Importance (SHAP)")
try:
    fi_data = get_feature_importance()
    fi_df = pd.DataFrame(fi_data["features"]).head(15)
    fig2 = px.bar(fi_df, x="importance", y="business_label", orientation="h")
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6E6E6",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig2, use_container_width=True)
except ApiError as exc:
    st.warning(f"Feature importance unavailable: {exc.detail}")

st.divider()

st.subheader("📉 Training Plots")
plots_dir = Path(__file__).resolve().parents[2] / "ml" / "models" / "plots"
if plots_dir.exists():
    plot_files = sorted(plots_dir.glob("*.png"))
    if plot_files:
        tabs = st.tabs(["Confusion Matrices", "ROC Curves", "Precision-Recall", "DNN Training Curves"])
        with tabs[0]:
            cols = st.columns(2)
            for i, f in enumerate([p for p in plot_files if "confusion_matrix" in p.name]):
                cols[i % 2].image(str(f), caption=f.stem.replace("_", " ").title())
        with tabs[1]:
            cols = st.columns(2)
            for i, f in enumerate([p for p in plot_files if "roc_curve" in p.name]):
                cols[i % 2].image(str(f), caption=f.stem.replace("_", " ").title())
        with tabs[2]:
            cols = st.columns(2)
            for i, f in enumerate([p for p in plot_files if "precision_recall" in p.name]):
                cols[i % 2].image(str(f), caption=f.stem.replace("_", " ").title())
        with tabs[3]:
            cols = st.columns(2)
            for i, f in enumerate([p for p in plot_files if "training_" in p.name]):
                cols[i % 2].image(str(f), caption=f.stem.replace("_", " ").title())
    else:
        st.info("No plots found yet. Train models first: `python -m ml.training.model_comparison`")
else:
    st.info("Plots directory not found yet. Train models first.")

st.divider()

st.subheader("🔁 Retrain Models")
with st.form("retrain_form"):
    regenerate = st.checkbox("Regenerate synthetic dataset before retraining")
    num_records = st.number_input("Number of records (if regenerating)", 1000, 200000, 12000, 1000)
    go = st.form_submit_button("Start Retraining")
    if go:
        with st.spinner("Retraining all models... this may take a few minutes."):
            try:
                result = trigger_retrain(regenerate_dataset=regenerate, num_records=num_records if regenerate else None)
                st.success(result["message"])
                st.rerun()
            except ApiError as exc:
                st.error(f"Retraining failed: {exc.detail}")
            except Exception as exc:
                st.error(f"Could not reach backend: {exc}")
