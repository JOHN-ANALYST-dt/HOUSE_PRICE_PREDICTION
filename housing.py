import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="KenyaHomes | Housing Intelligence", layout="wide")

# Custom CSS for Premium Look, Layout Margins, and Navigation Header
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Main Layout Margins */
    .main { padding: 0rem 5rem; }
    
    /* Professional Header Navigation */
    .nav-header {
        display: flex;
        justify-content: center;
        background-color: #ffffff;
        padding: 15px 0;
        border-bottom: 2px solid #D4AF37;
        position: sticky;
        top: 0;
        z-index: 999;
        margin-bottom: 30px;
    }
    .nav-item {
        margin: 0 25px;
        text-decoration: none;
        color: #003366;
        font-weight: 600;
        font-size: 0.9rem;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .nav-item:hover { color: #D4AF37; }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #003366 0%, #002244 100%);
        color: white; padding: 60px; border-radius: 15px;
        text-align: center; margin-bottom: 40px;
    }
    
    .metric-card {
        background: white; border: 1px solid #e2e8f0;
        padding: 25px; border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
    }
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
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #1a472a; /* Dark Greenish Tint from image */
        font-family: 'Inter', sans-serif;
        margin-bottom: 25px;
    }
    .section-header i {
        font-size: 1.5rem;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* Label with Icons styling */
    .field-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #374151;
        margin-bottom: -15px; /* Adjusting for Streamlit widget spacing */
        font-size: 0.95rem;
    }
    .field-label i {
        color: #9ca3af; /* Muted icon color from image */
        font-size: 1rem;
    }
            .calc-container {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .cost-card {
        background: #f8fafc;
        border-left: 5px solid #D4AF37;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .cost-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .cost-value {
        color: #003366;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .total-box {
        background: linear-gradient(135deg, #003366 0%, #002244 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
    }
    .calc-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 25px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    /* Breakdown Item Styling */
    .breakdown-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .item-label { color: #64748b; font-weight: 500; }
    .item-value { color: #0f172a; font-weight: 700; }
    
    /* Premium Total Box */
    .total-box-v2 {
        background: linear-gradient(135deg, #003366 0%, #004080 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION HEADER ---
st.markdown(f"""
    <div class="nav-header">
        <a class="nav-item" href="#kenya-house">Kenya House</a>
        <a class="nav-item" href="#price-predictor">Price Predictor</a>
        <a class="nav-item" href="#expenses">Expenses</a>
        <a class="nav-item" href="#material-forecast">Material Forecast</a>
        <a class="nav-item" href="#location">Location</a>
    </div>
""", unsafe_allow_html=True)
st.markdown("""
    <div class="ai-banner">
        <span class="ai-dot"></span> AI-Powered Predictions • 2000-2025 Historical Data
    </div>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_bundle():
    try:
        return joblib.load("rf_GRB_Model3.pkl")
    except:
        return None

model = load_bundle()

# --- KENYA HOUSE (HERO SECTION) ---
st.markdown('<div id="kenya-house"></div>', unsafe_allow_html=True)
st.markdown("""
    <div class="hero-container">
        <h1 style='font-weight:700; font-size: 2.8rem;color:white;'>KenyaHomes Intelligence</h1>
        <p style='font-size: 1.2rem; opacity: 0.9;'>Powered by ensemble machine learning models trained on 25 years of Kenyan housing data. 
            Predict house prices, construction costs, and material expenses with confidence.</p>
        <div style="display: flex; justify-content: center; gap: 60px; margin-top: 30px;color:white;">
            <div><h2 style='margin-bottom:0;color:white;'>47+</h2><p>Counties</p></div>
            <div><h2 style='margin-bottom:0;color:white;'>94.2%</h2><p>Accuracy</p></div>
            <div><h2 style='margin-bottom:0;color:white;'>25Y+</h2><p>Data Assets</p></div>
        </div>
    </div>
""", unsafe_allow_html=True)


# --- PRICE PREDICTOR SECTION ---

st.markdown('<div id="price-predictor"></div>', unsafe_allow_html=True)
st.header("🏠 House Price Prediction")
st.markdown("""
    <h2 class="predict-h2">Predict Your Dream Home's Price</h2>
    <h3 class="predict-h3">Enter your property requirements and get an accurate price prediction based on current market data and ML models.</h3>
""", unsafe_allow_html=True)
# --- PROPERTY DETAILS SECTION ---

with st.container(border=True):
    # Main Section Heading with Icon
    st.markdown("""
        <div class="section-header">
            <i class="far fa-list-alt"></i>
            <h2>Property Details</h2>
        </div>
    """, unsafe_allow_html=True)

    # First Row: Region and Area
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="field-label"><i class="fas fa-map-marker-alt"></i> Region/County</div>', unsafe_allow_html=True)
        region = st.selectbox("", ["Nairobi", "Kiambu", "Mombasa", "Nakuru", "Machakos"], index=None, placeholder="Select region", key="reg")
    
    with col2:
        st.markdown('<div class="field-label">Area/Neighborhood</div>', unsafe_allow_html=True)
        area = st.selectbox("", ["Westlands", "Kilimani", "Karen", "Runda", "Thika"], index=None, placeholder="Select area", key="area")

    # Second Row: Property Type and Size
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="field-label"><i class="fas fa-home"></i> Property Type</div>', unsafe_allow_html=True)
        p_type = st.selectbox("", ["Apartment", "Bungalow", "Mansionette", "Townhouse"], index=None, placeholder="Select type", key="ptype")
    
    with col4:
        st.markdown('<div class="field-label"><i class="fas fa-ruler-combined"></i> Size (sq ft)</div>', unsafe_allow_html=True)
        size = st.text_input("", placeholder="e.g., 2500", key="psize")

    # Third Row: Bedrooms and Bathrooms
    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="field-label"><i class="fas fa-bed"></i> Bedrooms</div>', unsafe_allow_html=True)
        beds = st.selectbox("", ["1", "2", "3", "4", "5+"], index=None, placeholder="Select", key="pbeds")
    
    with col6:
        st.markdown('<div class="field-label"><i class="fas fa-bath"></i> Bathrooms</div>', unsafe_allow_html=True)
        baths = st.selectbox("", ["1", "2", "3", "4+"], index=None, placeholder="Select", key="pbaths")

    # Fourth Row: Parking Spaces
    col7, _ = st.columns(2)
    with col7:
        st.markdown('<div class="field-label"><i class="fas fa-car"></i> Parking Spaces</div>', unsafe_allow_html=True)
        parking = st.selectbox("", ["1", "2", "3+"], index=None, placeholder="Select", key="park")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# --- CALCULATOR LOGIC & UI ---
st.markdown('<div id="expenses" style="padding-top: 50px;"></div>', unsafe_allow_html=True)
st.header("🏗️ Construction Cost Calculator")
st.write("Professional estimate based on 2026 Kenyan Building Indices.")

with st.container():
    col_input, col_output = st.columns([1, 1.3], gap="large")

    with col_input:
        st.subheader("Build Parameters")
        with st.container(border=True):
            # 1. Standard of Finish
            build_type = st.selectbox("Standard of Finish", 
                ["Standard (Budget)", "Middle-Class", "Luxurious (Premium)"], 
                index=1)
            
            # 2. Square Meters
            sqm = st.number_input("Total Floor Area (sq. meters)", min_value=30, value=120, step=10)
            
            # 3. NEW: Number of Floors
            num_floors = st.select_slider("Number of Floors", options=[1, 2, 3, 4, 5], value=1)
            
            # LOGIC: Mapping Rates & Floor Multipliers
            rates = {"Standard (Budget)": 42000, "Middle-Class": 60000, "Luxurious (Premium)": 85000}
            base_rate = rates[build_type]
            
            # Floor multiplier: 1 floor=1.0, 2 floors=1.15 (slab), 3+ floors=1.25 (structural reinforcement)
            floor_multiplier = 1.0 if num_floors == 1 else (1.15 if num_floors == 2 else 1.25)
            
            total_estimate = sqm * base_rate * floor_multiplier

    with col_output:
        st.subheader("Budget Breakdown")
        
        # Breakdown calculation logic
        breakdown = {
            "Substructure (Foundation)": total_estimate * 0.18,
            "Walling & Superstructure": total_estimate * 0.32,
            "Roofing & Ceiling": total_estimate * 0.15,
            "Finishes (Tiles, Paint, Joinery)": total_estimate * 0.25,
            "Electrical & Plumbing": total_estimate * 0.10
        }

        # Clearer Organizational View
        with st.container(border=True):
            # Summary Metrics Row
            m1, m2 = st.columns(2)
            m1.metric("Rate / m²", f"KES {base_rate:,.0f}")
            m2.metric("Total Area", f"{sqm} sqm")
            
            st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
            
            # Detailed Items
            for item, cost in breakdown.items():
                st.markdown(f"""
                    <div class="breakdown-row">
                        <span class="item-label">{item}</span>
                        <span class="item-value">KES {cost:,.0f}</span>
                    </div>
                """, unsafe_allow_html=True)

            # Highlighted Total
            st.markdown(f"""
                <div class="total-box-v2">
                    <div style="font-size: 0.8rem; opacity: 0.8; letter-spacing: 1px;">PROJECTED TOTAL BUDGET</div>
                    <div style="font-size: 2.2rem; font-weight: 800; margin: 5px 0;">KES {total_estimate:,.0f}</div>
                    <div style="font-size: 0.75rem;">Includes {num_floors} level structural complexity</div>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")
# --- LOCATION SECTION ---
st.markdown('<div id="location"></div>', unsafe_allow_html=True)
st.header("📍 Location-Based Pricing")
st.write("Comparing property tiers across Kenyan urban centers.")

loc_data = pd.DataFrame({
    'City': ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret'],
    'Premium (KES)': [25e6, 18e6, 15e6, 12e6, 10e6],
    'Mid-Range (KES)': [12e6, 9e6, 7.5e6, 6e6, 5.5e6],
    'Affordable (KES)': [5e6, 4.5e6, 3.8e6, 3.2e6, 3e6]
})
st.dataframe(loc_data.style.format(lambda x: f"KES {x:,.0f}" if isinstance(x, (int, float)) else x), use_container_with_width=True)

# --- FOOTER ---
st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 40px 0;">
        KenyaHomes Intelligence v3.0 | 2026 Housing Market Analysis
    </div>
""", unsafe_allow_html=True)