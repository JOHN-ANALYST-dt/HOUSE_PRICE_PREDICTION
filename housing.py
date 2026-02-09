import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import os
import pathlib


# Base directory setup
BASE_DIR = pathlib.Path(__file__).parent 
CSS_PATH = os.path.join(BASE_DIR, 'style.css') 

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
st.markdown ("""
<h2 class="predict-h2">🏠 House Price Prediction</h2>
             """,unsafe_allow_html=True)
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

# --- CALCULATOR LOGIC & UI ---
st.markdown('<div id="expenses" style="padding-top: 50px;"></div>', unsafe_allow_html=True)
st.header("🏗️ Construction Cost Calculator")
st.write("Professional estimate based on 2026 Kenyan Building Indices.")

with st.container(border=True):
    col_input, col_output = st.columns([1, 1.3], gap="small")

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

# --- 1. INITIALIZE INTERACTIVE STATE ---
# This tracks which material the user has clicked on
if 'selected_material' not in st.session_state:
    st.session_state.selected_material = "Cement (50kg bag)"

# --- 2. MATERIAL PREDICTION LOGIC ---
def get_material_predictions():
    years = np.arange(2025, 2036)
    materials_config = {
        "Cement (50kg bag)": {"base": 780, "rate": 0.052, "icon": "🧱", "color": "#003366"},
        "Steel Bars (12mm)": {"base": 1550, "rate": 0.068, "icon": "🔩", "color": "#D4AF37"},
        "Iron Sheets (G30)": {"base": 1280, "rate": 0.055, "icon": "🏗️", "color": "#10b981"},
        "Timber (Cypress)": {"base": 180, "rate": 0.045, "icon": "🪵", "color": "#ef4444"},
        "River Sand (Ton)": {"base": 3900, "rate": 0.075, "icon": "🏖️", "color": "#f59e0b"},
        "Ballast (Ton)": {"base": 2700, "rate": 0.048, "icon": "🪨", "color": "#6366f1"},
        "Building Bricks": {"base": 22, "rate": 0.035, "icon": "🧱", "color": "#8b5cf6"},
        "Floor Tiles (m²)": {"base": 1450, "rate": 0.050, "icon": "✨", "color": "#ec4899"}
    }
    
    forecast_data = {"Year": years}
    for name, config in materials_config.items():
        forecast_data[name] = [config["base"] * (1 + config["rate"])**i for i in range(len(years))]
        
    return pd.DataFrame(forecast_data), materials_config

df_forecast, config = get_material_predictions()

# --- 3. UI SECTION: INTERACTIVE CARDS ---
st.markdown('<div id="material-forecast" style="padding-top: 50px;"></div>', unsafe_allow_html=True)
st.header("📈 10-Year Material Price Intelligence")
st.write("Click on any material card below to view its specific 10-year forecast on the graph.")

# Predictive Cost Breakdown Grid (Clickable)
cols = st.columns(4)
for i, (name, info) in enumerate(config.items()):
    with cols[i % 4]:
        # Logic: If the card/button is clicked, update the selected material
        if st.button(f"{info['icon']} {name}", key=f"btn_{name}", use_container_width=True):
            st.session_state.selected_material = name
        
        # UI Styling for the "Selected" state
        is_selected = st.session_state.selected_material == name
        border_style = f"2px solid {info['color']}" if is_selected else "1px solid #e2e8f0"
        bg_style = "#f8fafc" if is_selected else "white"
        
        future_p = df_forecast[name].iloc[-1]
        growth = ((future_p / info['base']) - 1) * 100
        
        st.markdown(f"""
            <div style="background: {bg_style}; border: {border_style}; padding: 15px; border-radius: 12px; margin-bottom: 20px; text-align: center;">
                <div style="color: #64748b; font-size: 0.75rem;">2035 Projection</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {info['color']};">KES {future_p:,.0f}</div>
                <div style="color: #10b981; font-size: 0.7rem; font-weight: 600;">↑ {growth:.1f}% Growth</div>
            </div>
        """, unsafe_allow_html=True)

# --- 4. DYNAMIC FORECAST CHART ---
selected = st.session_state.selected_material
selected_color = config[selected]['color']

st.subheader(f"Focus Forecast: {selected}")

fig_mat = go.Figure()

# Add the selected material line (Thick and Highlighted)
fig_mat.add_trace(go.Scatter(
    x=df_forecast['Year'], 
    y=df_forecast[selected], 
    name=selected,
    mode='lines+markers+text',
    text=[f"{v:,.0f}" if y % 2 == 0 else "" for y, v in zip(df_forecast['Year'], df_forecast[selected])],
    textposition="top center",
    line=dict(width=5, color=selected_color),
    marker=dict(size=10, symbol='diamond')
))

# Add other materials as faint reference lines
for name, info in config.items():
    if name != selected:
        fig_mat.add_trace(go.Scatter(
            x=df_forecast['Year'], 
            y=df_forecast[name], 
            name=name,
            mode='lines',
            line=dict(width=1, color='#e2e8f0'),
            showlegend=False,
            hoverinfo='skip'
        ))

fig_mat.update_layout(
    xaxis_title="Forecast Year",
    yaxis_title="Price (KES)",
    template="plotly_white",
    hovermode="x",
    height=450,
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig_mat, use_container_width=True)

# 5. Data Export
with st.expander("View Full Forecast Data Table"):
    st.dataframe(df_forecast.style.format(lambda x: f"KES {x:,.2f}" if isinstance(x, (int, float)) and x > 2000 else x), use_container_width=True)