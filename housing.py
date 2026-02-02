import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
from datetime import datetime

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Housing Forecast Pro", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main-header { background-color: #1E3A8A; padding: 20px; border-radius: 10px; color: white; margin-bottom: 25px; }
    .kpi-card { background-color: #F8FAFC; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; text-align: center; }
    .kpi-value { font-size: 1.5rem; font-weight: bold; color: #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

def load_model_file(file_buffer):
    """Safely loads scikit-learn models from buffer."""
    try:
        # Attempt joblib first (standard for sklearn)
        return joblib.load(file_buffer)
    except:
        file_buffer.seek(0)
        return pickle.load(file_buffer)

def get_forecast(model, base_features, years, growth_rate, uncertainty=0.05):
    """Generates multi-year predictions based on compound growth."""
    forecast_data = []
    current_features = base_features.copy()
    
    # Identify Year column to increment it naturally
    year_col = next((c for c in current_features.columns if 'year' in c.lower()), None)
    
    for i in range(years + 1):
        # 1. Predict (Handle single vs multi-output models)
        preds = model.predict(current_features)
        h_price = preds[0][0] if preds.ndim > 1 else preds[0]
        m_cost = preds[0][1] if preds.ndim > 1 and len(preds[0]) > 1 else 0.0
        
        # 2. Store results with uncertainty bands
        row = {
            "Year": int(current_features[year_col].iloc[0]) if year_col else (datetime.now().year + i),
            "Housing Price": h_price,
            "Material Cost": m_cost,
            "Lower Bound": h_price * (1 - (uncertainty * i)),
            "Upper Bound": h_price * (1 + (uncertainty * i))
        }
        forecast_data.append(row)
        
        # 3. Advance to next year: Increment Year and apply Growth Rate to others
        if year_col:
            current_features[year_col] += 1
        
        for col in current_features.select_dtypes(include=[np.number]).columns:
            if col != year_col and "rate" not in col.lower(): # Don't grow rates/years
                current_features[col] *= (1 + growth_rate)
                
    return pd.DataFrame(forecast_data)

# --- HEADER ---
st.markdown('<div class="main-header"><h1>Strategic Housing Forecast Portal</h1><p>Predictive analytics for real estate investment.</p></div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    model_file = st.file_uploader("Upload Model (rf_GRB_Model3.pkl)", type=['pkl', 'joblib'])
    
    st.markdown("---")
    cities_raw = st.text_input("Cities (comma separated)", value="New York, Austin")
    cities = [c.strip() for c in cities_raw.split(",") if c.strip()]
    horizon = st.slider("Forecast Horizon (Years)", 5, 15, 7)
    
    scenario = st.select_slider("Scenario", options=["Pessimistic", "Baseline", "Optimistic"], value="Baseline")
    growth_map = {"Pessimistic": -0.02, "Baseline": 0.03, "Optimistic": 0.06}
    selected_growth = growth_map[scenario]

# --- MAIN LOGIC ---
if model_file:
    model = load_model_file(model_file)
    
    # Determine required features automatically
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    else:
        # Fallback if model lacks metadata
        features = ["Year", "Average_Income", "Interest_Rate"]

    st.success(f"Model Active. Features expected: {', '.join(features)}")

    # Dynamic Form Creation
    with st.form("input_form"):
        st.subheader("Baseline Market Conditions")
        input_data = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                if "year" in feat.lower():
                    input_data[feat] = st.number_input(feat, value=datetime.now().year)
                else:
                    input_data[feat] = st.number_input(f"Current {feat}", value=100.0)
        
        submitted = st.form_submit_button("Run Forecast")

    if submitted:
        base_df = pd.DataFrame([input_data])
        results = {city: get_forecast(model, base_df, horizon, selected_growth) for city in cities}
        
        # 1. KPIs (Showing first city)
        kpi_cols = st.columns(3)
        main_res = results[cities[0]]
        with kpi_cols[0]:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">${main_res["Housing Price"].iloc[0]:,.0f}</div><div>Start Price</div></div>', unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">${main_res["Housing Price"].iloc[-1]:,.0f}</div><div>Final Forecast</div></div>', unsafe_allow_html=True)
        with kpi_cols[2]:
            growth = ((main_res["Housing Price"].iloc[-1] / main_res["Housing Price"].iloc[0]) - 1) * 100
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{growth:.1f}%</div><div>Total Growth</div></div>', unsafe_allow_html=True)

        # 2. Charts
        fig = go.Figure()
        for city, df in results.items():
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Housing Price'], name=city, mode='lines+markers'))
            # Uncertainty shading
            fig.add_trace(go.Scatter(x=pd.concat([df['Year'], df['Year'][::-1]]), 
                                     y=pd.concat([df['Upper Bound'], df['Lower Bound'][::-1]]),
                                     fill='toself', fillcolor='rgba(30,58,138,0.1)', line_color='rgba(0,0,0,0)', showlegend=False))
        
        fig.update_layout(title="Projected Market Value", hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_with_width=True)
        
        # 3. Data View
        st.dataframe(pd.concat([df.assign(City=c) for c, df in results.items()]).style.format(precision=2))

else:
    st.info("Waiting for model upload...")
    st.markdown("""
        <div style="background-color: #F3F4F6; padding: 2rem; border-radius: 8px; border: 1px dashed #D1D5DB;">
            <h4 style="margin-top:0;"><i class="fas fa-info-circle"></i> Getting Started</h4>
            <ol>
                <li>Upload your regression model (e.g., <b>rf_GRB_Model3.pkl</b>).</li>
                <li>Optionally upload historical training data to auto-calculate growth rates.</li>
                <li>Enter city names and adjust the growth scenario sliders.</li>
                <li>Review the generated KPIs and interactive forecast charts.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.8rem;">
        <i class="fas fa-shield-alt"></i> Professional Forecasting Tool &nbsp; | &nbsp; 
        <i class="fas fa-envelope"></i> Contact System Admin &nbsp; | &nbsp; 
        v1.0.4 Build
    </div>
""", unsafe_allow_html=True)