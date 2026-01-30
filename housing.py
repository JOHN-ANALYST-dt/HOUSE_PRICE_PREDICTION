import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prophet.serialize import model_from_json
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Kenyan Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# UTIL FUNCTIONS
# -----------------------------
@st.cache_resource
def load_ml_model():
    model = joblib.load("models/ml/house_price_model.pkl")
    preprocessor = joblib.load("models/ml/preprocessor.pkl")
    return model, preprocessor


def load_prophet_model(model_name):
    with open(f"models/prophet/{model_name}.json", "r") as f:
        return model_from_json(f.read())


def predict_future_price(model, year):
    future = pd.DataFrame({
        "ds": pd.to_datetime([f"{year}-12-31"])
    })
    forecast = model.predict(future)
    return int(forecast["yhat"].iloc[0])


# -----------------------------
# LOAD MODELS
# -----------------------------
ml_model, preprocessor = load_ml_model()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🏗️ Prediction Settings")

prediction_type = st.sidebar.radio(
    "Choose Prediction Type",
    ["Current House Price", "Future Price Forecast"]
)

city = st.sidebar.selectbox(
    "Select City",
    ["Nairobi", "Kisumu", "Mombasa", "Nakuru"]
)

county = st.sidebar.selectbox(
    "Select County",
    ["Nairobi", "Kisumu", "Mombasa", "Nakuru"]
)

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🏠 Kenyan Housing Price Intelligence Platform")
st.caption("ML + Time Series Forecasting | Built for Kenya 🇰🇪")

# ======================================================
# CURRENT HOUSE PRICE (ML)
# ======================================================
if prediction_type == "Current House Price":

    st.subheader("📌 Enter House Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)
        square_feet = st.number_input("Square Feet", 300, 10000, 1200)

    with col2:
        floors = st.number_input("Number of Floors", 1, 5, 1)
        finishing = st.selectbox(
            "Finishing Level",
            ["basic", "standard", "premium"]
        )

    with col3:
        roof = st.selectbox(
            "Roof Type",
            ["iron_sheet", "tiles", "concrete"]
        )
        urban = st.selectbox(
            "Urban or Rural",
            ["urban", "rural"]
        )

    # Build input dataframe
    input_df = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "square_feet": square_feet,
        "number_of_floors": floors,
        "finishing_level": finishing,
        "roof_type": roof,
        "urban_or_rural": urban,
        "city": city,
        "county": county
    }])

    if st.button("💰 Predict House Price"):
        X = preprocessor.transform(input_df)
        price = int(ml_model.predict(X)[0])

        st.success(f"🏠 Estimated House Price: **KES {price:,}**")

# ======================================================
# FUTURE PRICE FORECAST (PROPHET)
# ======================================================
else:
    st.subheader("📈 Future Housing Price Forecast")

    year = st.slider(
        "Select Future Year",
        datetime.now().year + 1,
        datetime.now().year + 10,
        datetime.now().year + 5
    )

    model_key = f"city_{city.lower()}"

    try:
        prophet_model = load_prophet_model(model_key)
        future_price = predict_future_price(prophet_model, year)

        st.success(
            f"📊 In **{year}**, estimated house price in **{city}** "
            f"will be **KES {future_price:,}**"
        )

    except FileNotFoundError:
        st.warning(f"No forecast model available for {city}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Powered by Machine Learning & Prophet | Real Estate Analytics Kenya")
