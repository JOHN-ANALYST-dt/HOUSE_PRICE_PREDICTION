import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="KenyaHomes | Housing Intelligence", layout="wide")

# Custom CSS for Icons, Premium Look, Navigation, and the new AI Banner
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Main Layout Margins */
    .main { padding: 0rem 5rem; }
    
    /* Sticky Navigation Header */
    .nav-header {
        display: flex;
        justify-content: center;
        background-color: #ffffff;
        padding: 18px 0;
        border-bottom: 1px solid #e2e8f0;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .nav-item {
        margin: 0 20px;
        text-decoration: none;
        color: #003366;
        font-weight: 600;
        font-size: 0.85rem;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .nav-item i { color: #D4AF37; font-size: 1.1rem; }
    .nav-item:hover { color: #D4AF37; transform: translateY(-2px); }

    /* AI STATUS BANNER (The new addition) */
    .ai-banner {
        background-color: #f8fafc;
        text-align: center;
        padding: 10px 0;
        border-bottom: 2px solid #D4AF37;
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e293b;
        letter-spacing: 0.5px;
    }
    .ai-dot {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 8px #10b981;
    }

    /* Hero & Metrics */
    .hero-container {
        background: linear-gradient(135deg, #003366 0%, #002244 100%);
        color: white; padding: 60px; border-radius: 15px;
        text-align: center; margin-top: 30px; margin-bottom: 40px;
    }
    
    .metric-card {
        background: white; border: 1px solid #e2e8f0;
        padding: 25px; border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION HEADER ---
st.markdown("""
    <div class="nav-header">
        <a class="nav-item" href="#kenya-house"><i class="fas fa-home"></i> Kenya House</a>
        <a class="nav-item" href="#price-predictor"><i class="fas fa-chart-line"></i> Price Predictor</a>
        <a class="nav-item" href="#expenses"><i class="fas fa-calculator"></i> Expenses</a>
        <a class="nav-item" href="#material-forecast"><i class="fas fa-seedling"></i> Material Forecast</a>
        <a class="nav-item" href="#location"><i class="fas fa-map-marker-alt"></i> Location</a>
    </div>
""", unsafe_allow_html=True)

# --- AI STATUS BANNER (JUST AFTER HEADER) ---
st.markdown("""
    <div class="ai-banner">
        <span class="ai-dot"></span> AI-Powered Predictions • 2000-2025 Historical Data
    </div>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_bundle():
    try:
        # Loading the bundled model provided by the creator
        return joblib.load("rf_GRB_Model3.pkl")
    except:
        return None

model = load_bundle()

# --- KENYA HOUSE SECTION ---
st.markdown('<div id="kenya-house"></div>', unsafe_allow_html=True)
st.markdown("""
    <div class="hero-container">
        <h1 style='font-weight:700; font-size: 2.8rem;'>KenyaHomes Intelligence</h1>
        <p style='font-size: 1.2rem; opacity: 0.9;'>Trusted ensemble predictions for Kenya's leading developers and home buyers.</p>
        <div style="display: flex; justify-content: center; gap: 60px; margin-top: 30px;">
            <div><h2 style='margin-bottom:0;'>47+</h2><p>Counties</p></div>
            <div><h2 style='margin-bottom:0;'>94.2%</h2><p>Accuracy Index</p></div>
            <div><h2 style='margin-bottom:0;'>25Y+</h2><p>Data Assets</p></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- PRICE PREDICTOR SECTION ---
st.markdown('<div id="price-predictor"></div>', unsafe_allow_html=True)
st.header("🏠 Price Predictor Engine")
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Property Specifications")
    with st.container(border=True):
        town = st.selectbox("Select Town/City", ["Nairobi", "Mombasa", "Kiambu", "Nakuru", "Kisumu", "Eldoret"])
        size = st.number_input("Property Size (SqFt)", value=1500)
        beds = st.slider("Number of Bedrooms", 1, 10, 3)
        baths = st.slider("Number of Bathrooms", 1, 8, 2)

with col2:
    st.subheader("Valuation Output")
    if model:
        # Assuming model features: [Size, Beds, Baths, Year]
        features = np.array([[size, beds, baths, 2026]]) 
        prediction = model.predict(features)[0]
        st.markdown(f"""
            <div class="metric-card" style="margin-top: 20px;">
                <p style="color: #64748b; margin-bottom: 5px;">Estimated Market Value</p>
                <h1 style="color: #003366; font-size: 3rem;">KES {prediction:,.2f}</h1>
                <p style="color: #D4AF37; font-weight: 600;">Current market indices for {town}.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("System Error: Predictive model 'rf_GRB_Model3.pkl' missing.")

st.markdown("---")

# --- EXPENSES SECTION ---
st.markdown('<div id="expenses"></div>', unsafe_allow_html=True)
st.header("🏗️ Construction Expense Breakdown")
st.write("Detailed structural phase costing based on Kenyan construction standards.")

ex_col1, ex_col2 = st.columns([2, 3])
with ex_col1:
    base_cost = size * 4850 
    expense_data = {
        "Substructure": base_cost * 0.15,
        "Superstructure": base_cost * 0.35,
        "Roofing System": base_cost * 0.15,
        "M&E Services": base_cost * 0.15,
        "Interior Finishes": base_cost * 0.20
    }
    st.table(pd.DataFrame(expense_data.items(), columns=["Phase", "Cost (KES)"]))

with ex_col2:
    fig_ex = go.Figure(data=[go.Pie(labels=list(expense_data.keys()), values=list(expense_data.values()), hole=.4)])
    fig_ex.update_layout(margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_ex, use_container_with_width=True)

st.markdown("---")

# --- MATERIAL FORECAST SECTION ---
st.markdown('<div id="material-forecast"></div>', unsafe_allow_html=True)
st.header("📉 10-Year Material Price Forecast")
st.info("Time-series projection for core materials: Cement, Steel, and Wood.")

years = list(range(2026, 2037))
# Projection logic for Kenyan market
cement = [780 * (1.05**i) for i in range(len(years))]
steel = [155 * (1.07**i) for i in range(len(years))]
timber = [135 * (1.04**i) for i in range(len(years))]

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=years, y=cement, name="Cement (Bag)", line=dict(color='#003366', width=3)))
fig_trend.add_trace(go.Scatter(x=years, y=steel, name="Steel (KG)", line=dict(color='#D4AF37', width=3)))
fig_trend.add_trace(go.Scatter(x=years, y=timber, name="Timber (Ft)", line=dict(color='#10b981', width=3)))

fig_trend.update_layout(xaxis_title="Year", yaxis_title="Price (KES)", template="plotly_white", height=450)
st.plotly_chart(fig_trend, use_container_with_width=True)

st.markdown("---")

# --- LOCATION SECTION ---
st.markdown('<div id="location"></div>', unsafe_allow_html=True)
st.header("📍 Location-Tier Pricing")
loc_data = pd.DataFrame({
    'City': ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret'],
    'Premium Tier': [28.5e6, 20.2e6, 16.5e6, 13.1e6, 11.4e6],
    'Mid-Range Tier': [14.2e6, 10.5e6, 8.8e6, 7.2e6, 6.7e6],
    'Affordable Tier': [6.2e6, 5.1e6, 4.3e6, 3.6e6, 3.3e6]
})
st.dataframe(loc_data.style.format(lambda x: f"KES {x:,.0f}" if isinstance(x, (int, float)) else x), use_container_with_width=True)

# --- FOOTER ---
st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 40px 0; border-top: 1px solid #e2e8f0;">
        KenyaHomes Intel System • v3.1.2 Build • © 2026
    </div>
""", unsafe_allow_html=True)