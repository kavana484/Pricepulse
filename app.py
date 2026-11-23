import streamlit as st
import pandas as pd
import joblib
import glob
import os

# Page configuration
st.set_page_config(
    page_title="PricePulse",
    page_icon="💰",
    layout="wide"
)

# Load trained model
model_files = glob.glob("dynamic_pricing_model*.pkl")

if not model_files:
    st.error("Model file not found!")
    st.stop()

model = joblib.load(model_files[0])

# Dashboard title
st.title("💰 PricePulse")
st.subheader("Adaptive Dynamic Pricing & Revenue Optimization Engine")

st.write(
    "Predict order totals using Machine Learning."
)

st.divider()

# Input section
st.header("Enter Order Details")

col1, col2 = st.columns(2)

with col1:
    items_price = st.number_input(
        "Items Price", min_value=0.0, value=100.0
    )

    freight_value = st.number_input(
        "Freight Value", min_value=0.0, value=15.0
    )

    n_items = st.number_input(
        "Number of Items", min_value=1, value=2
    )

    delivery_days = st.number_input(
        "Delivery Days", min_value=0.0, value=7.0
    )

    estimated_days = st.number_input(
        "Estimated Days", min_value=0, value=5
    )

with col2:
    delivery_delay_days = st.number_input(
        "Delivery Delay Days", value=2.0
    )

    late_delivery = st.selectbox(
        "Late Delivery", [False, True]
    )

    review_score = st.slider(
        "Review Score", 1.0, 5.0, 4.0
    )

    month = st.selectbox(
        "Month", list(range(1, 13))
    )

    year = st.number_input(
        "Year", min_value=2016, max_value=2030,
        value=2018
    )

st.divider()

# Prediction
if st.button("Predict Order Total", type="primary"):

    input_data = pd.DataFrame([{
        "items_price": items_price,
        "freight_value": freight_value,
        "n_items": n_items,
        "delivery_days": delivery_days,
        "estimated_days": estimated_days,
        "delivery_delay_days": delivery_delay_days,
        "late_delivery": late_delivery,
        "review_score": review_score,
        "month": month,
        "year": year
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")

        st.metric(
            "Predicted Order Total",
            f"{prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

st.caption("PricePulse | Machine Learning Project")