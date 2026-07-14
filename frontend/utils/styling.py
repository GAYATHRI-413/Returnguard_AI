"""
frontend/utils/styling.py

Shared theme CSS + small UI helpers reused across every Streamlit page.

Design language: "Signal" - a security/intelligence dashboard aesthetic.
Deep charcoal/indigo base, a single electric-cyan accent used sparingly for
signal, and warm amber/rose only for risk states. Distinct from the generic
Streamlit-dark-mode-plus-emoji look most tutorial projects ship with.
"""
from __future__ import annotations

import streamlit as st

THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg-base: #0B0E14;
        --bg-panel: #12151E;
        --bg-panel-raised: #171B27;
        --border-soft: rgba(255,255,255,0.07);
        --accent: #4FE0C5;
        --accent-dim: rgba(79, 224, 197, 0.14);
        --text-primary: #EDEFF4;
        --text-muted: #8791A8;
        --risk-low: #4FE0C5;
        --risk-medium: #F0B94D;
        --risk-high: #F0546B;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #131826 0%, var(--bg-base) 45%);
        color: var(--text-primary);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-panel);
        border-right: 1px solid var(--border-soft);
    }

    /* Brand mark used at the top of the app */
    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom: 4px; }
    .brand-mark {
        width:38px; height:38px; border-radius:10px;
        background: linear-gradient(135deg, var(--accent), #2E8F80);
        display:flex; align-items:center; justify-content:center;
        font-weight:700; color:#0B0E14; font-family:'Space Grotesk',sans-serif; font-size:18px;
    }
    .brand-title { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:26px; margin:0; }
    .brand-subtitle { color: var(--text-muted); font-size: 14px; margin-top:-2px; }

    /* Login card */
    .auth-card {
        background: var(--bg-panel-raised);
        border: 1px solid var(--border-soft);
        border-radius: 16px;
        padding: 32px 32px 20px 32px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    .auth-badge {
        display:inline-block; font-size:11px; letter-spacing:0.08em; text-transform:uppercase;
        color: var(--accent); background: var(--accent-dim);
        padding: 4px 10px; border-radius: 999px; margin-bottom: 10px;
    }

    /* Risk cards */
    .risk-card {
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 16px;
        background: var(--bg-panel-raised);
        border: 1px solid var(--border-soft);
        border-left: 4px solid var(--text-muted);
    }
    .risk-low { border-left-color: var(--risk-low); }
    .risk-medium { border-left-color: var(--risk-medium); }
    .risk-high { border-left-color: var(--risk-high); }

    /* Metric boxes */
    .metric-box {
        background-color: var(--bg-panel-raised);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        border: 1px solid var(--border-soft);
        transition: border-color 0.15s ease;
    }
    .metric-box:hover { border-color: var(--accent); }
    .metric-box h2 { margin: 0; font-size: 26px; font-family:'Space Grotesk',sans-serif; }
    .metric-box p { margin: 4px 0 0 0; color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; }

    /* Chips */
    .feature-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 13px;
        font-weight: 500;
    }
    .chip-risk { background-color: rgba(240, 84, 107, 0.14); color: var(--risk-high); border: 1px solid rgba(240,84,107,0.4); }
    .chip-safe { background-color: rgba(79, 224, 197, 0.14); color: var(--risk-low); border: 1px solid rgba(79,224,197,0.4); }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--border-soft);
        background: var(--bg-panel-raised);
    }
    .stButton > button:hover { border-color: var(--accent); color: var(--accent); }
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, var(--accent), #2E8F80);
        color: #0B0E14; font-weight: 600; border: none;
    }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
"""


def apply_dark_theme() -> None:
    """Kept the original function name so existing page imports don't break."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_brand_header(title: str = "ReturnGuard AI", subtitle: str = "Return Fraud Risk Intelligence") -> None:
    st.markdown(
        f"""
        <div class="brand-row">
            <div class="brand-mark">RG</div>
            <div>
                <p class="brand-title">{title}</p>
                <p class="brand-subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_class(risk_level: str) -> str:
    return {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}.get(risk_level, "risk-medium")


def render_metric_box(label: str, value: str) -> str:
    return f'<div class="metric-box"><h2>{value}</h2><p>{label}</p></div>'
