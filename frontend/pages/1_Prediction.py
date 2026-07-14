"""
frontend/pages/1_Prediction.py

Prediction form: fill in a return request, get fraud probability, risk
score, recommended action, and a SHAP-based explanation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from frontend.utils.api_client import ApiError, predict
from frontend.utils.styling import apply_dark_theme, risk_badge_class

st.set_page_config(page_title="Prediction · ReturnGuard AI", page_icon="🔍", layout="wide")
apply_dark_theme()

if not st.session_state.get("authenticated"):
    st.warning("Please log in from the main page first.")
    st.stop()

st.title("🔍 Return Fraud Prediction")
st.caption("Fill in the return request details to get a real-time fraud risk assessment.")

with st.form("prediction_form"):
    st.subheader("Identifiers")
    c1, c2, c3 = st.columns(3)
    customer_id = c1.text_input("Customer ID", "CUST100045")
    seller_id = c2.text_input("Seller ID", "SELL1021")
    order_id = c3.text_input("Order ID", "ORD500123")

    st.subheader("Customer")
    c1, c2, c3, c4 = st.columns(4)
    customer_age = c1.number_input("Customer Age", 13, 100, 29)
    account_age_days = c2.number_input("Account Age (days)", 0, 20000, 120)
    total_orders = c3.number_input("Total Orders", 1, 5000, 14)
    total_returns = c4.number_input("Total Returns", 0, 5000, 6)

    c1, c2 = st.columns(2)
    customer_segment = c1.selectbox("Customer Segment", ["New", "Regular", "Loyal", "VIP"], index=1)
    days_since_last_purchase = c2.number_input("Days Since Last Purchase", 0, 5000, 12)

    st.subheader("Product & Order")
    c1, c2, c3 = st.columns(3)
    product_category = c1.selectbox(
        "Product Category",
        ["Electronics", "Fashion", "Footwear", "Home & Kitchen", "Beauty", "Sports", "Toys", "Books", "Jewelry", "Mobile Accessories"],
    )
    product_price = c2.number_input("Product Price ($)", 0.01, 100000.0, 349.99)
    order_value = c3.number_input("Order Value ($)", 0.01, 100000.0, 379.99)
    product_weight_kg = st.number_input("Product Weight (kg)", 0.01, 500.0, 1.2)

    st.subheader("Seller")
    c1, c2 = st.columns(2)
    seller_rating = c1.slider("Seller Rating", 1.0, 5.0, 4.2, 0.1)
    seller_defect_rate = c2.slider("Seller Defect Rate", 0.0, 1.0, 0.03, 0.01)

    st.subheader("Delivery")
    c1, c2, c3 = st.columns(3)
    delivery_days = c1.number_input("Delivery Days", 0, 60, 3)
    distance_km = c2.number_input("Distance (km)", 0.0, 5000.0, 45.0)
    shipping_method = c3.selectbox("Shipping Method", ["Standard", "Express", "Same Day"])

    st.subheader("Return Details")
    c1, c2, c3 = st.columns(3)
    return_reason = c1.selectbox(
        "Return Reason",
        ["Defective / Not Working", "Wrong Item Delivered", "Size Issue", "Not as Described",
         "Changed Mind", "Better Price Found", "Damaged in Transit", "Late Delivery",
         "Missing Parts", "Quality Issue"],
    )
    avg_days_to_return = c2.number_input("Days Taken to Return", 0.0, 365.0, 1.5)
    payment_method = c3.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery", "Wallet"])

    c1, c2 = st.columns(2)
    device_type = c1.selectbox("Device Type", ["Mobile App", "Desktop Web", "Mobile Web"])
    num_payment_methods_used = c2.number_input("Payment Methods Used (lifetime)", 1, 20, 3)

    st.subheader("Behavioural Flags")
    c1, c2, c3, c4, c5 = st.columns(5)
    used_multiple_addresses = c1.checkbox("Multiple Addresses")
    used_multiple_payment_methods = c2.checkbox("Multiple Payment Methods")
    return_without_tags = c3.checkbox("Return Without Tags")
    packaging_damaged_claim = c4.checkbox("Packaging Damage Claim")
    is_holiday_purchase = c5.checkbox("Holiday Purchase")

    submitted = st.form_submit_button("🚀 Predict Fraud Risk", use_container_width=True)

if submitted:
    payload = {
        "customer_id": customer_id, "seller_id": seller_id, "order_id": order_id,
        "customer_age": customer_age, "account_age_days": account_age_days,
        "total_orders": total_orders, "total_returns": min(total_returns, total_orders),
        "customer_segment": customer_segment, "days_since_last_purchase": days_since_last_purchase,
        "product_category": product_category, "product_price": product_price,
        "order_value": order_value, "product_weight_kg": product_weight_kg,
        "seller_rating": seller_rating, "seller_defect_rate": seller_defect_rate,
        "delivery_days": delivery_days, "distance_km": distance_km, "shipping_method": shipping_method,
        "return_reason": return_reason, "avg_days_to_return": avg_days_to_return,
        "payment_method": payment_method, "device_type": device_type,
        "num_payment_methods_used": num_payment_methods_used,
        "used_multiple_addresses": used_multiple_addresses,
        "used_multiple_payment_methods": used_multiple_payment_methods,
        "return_without_tags": return_without_tags,
        "packaging_damaged_claim": packaging_damaged_claim,
        "is_holiday_purchase": is_holiday_purchase,
    }

    try:
        with st.spinner("Analyzing return request..."):
            result = predict(payload)
    except ApiError as exc:
        st.error(f"Prediction failed: {exc.detail}")
        st.stop()
    except Exception as exc:
        st.error(f"Could not reach backend: {exc}")
        st.stop()

    st.divider()
    badge_class = risk_badge_class(result["risk_level"])

    st.markdown(
        f"""
        <div class="risk-card {badge_class}">
            <h2>Risk Level: {result['risk_level']}</h2>
            <p style="font-size:16px;">Fraud Probability: <b>{result['fraud_probability']*100:.2f}%</b>
            &nbsp;|&nbsp; Risk Score: <b>{result['risk_score']}/100</b></p>
            <p style="font-size:16px;">Recommended Action: <b>{result['recommended_action'].replace('_', ' ')}</b></p>
            <p style="color:#9BA3AF;">Model used: {result['model_used']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🧠 Why this prediction?")
    st.write(result["business_explanation"])

    st.subheader("📊 Top Contributing Features")
    for f in result["top_features"]:
        chip_class = "chip-risk" if f["direction"] == "increases_risk" else "chip-safe"
        arrow = "⬆️ increases risk" if f["direction"] == "increases_risk" else "⬇️ decreases risk"
        st.markdown(
            f"<span class='feature-chip {chip_class}'>{f['business_label']} "
            f"(SHAP: {f['shap_value']:+.3f}, {arrow})</span>",
            unsafe_allow_html=True,
        )

    st.caption(f"Request ID: `{result['request_id']}`")
