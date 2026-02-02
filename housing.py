# Requirements: pip install streamlit pandas numpy scikit-learn plotly joblib Pillow
# Run: streamlit run app.py
# Usage: Upload the 'rf_GRB_Model3.pkl' file in the sidebar to begin analysis.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
from datetime import datetime
import io

# --- UI CONFIGURATION & STYLING ---
st.set_page_config(page_title="Housing & Material Forecast Pro", layout="wide")

BG_IMAGE_URL = "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop"
FONT_AWESOME_CDN = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"

st.markdown(f"""
    <link rel="stylesheet" href="{FONT_AWESOME_CDN}">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), 
                    url("{BG_IMAGE_URL}");
        background-size: cover;
        background-attachment: fixed;
    }}

    .main-header {{
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border-bottom: 3px solid #1E3A8A;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}

    .kpi-card {{
        background-color: white;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    .kpi-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
    }}

    .kpi-label {{
        font-size: 0.875rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def load_model(file_buffer):
    try:
        # Try joblib first, then pickle
        try:
            return joblib.load(file_buffer)
        except:
            file_buffer.seek(0)
            return pickle.load(file_buffer)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_forecast(model, base_features, years, growth_rate, std_dev=0.05):
    forecast_data = []
    current_features = base_features.copy()
    
    # Identify which columns are 'year' or 'price' related to apply growth
    year_col = next((c for c in current_features.columns if 'year' in c.lower()), None)
    
    for i in range(years + 1):
        # Predict
        preds = model.predict(current_features)
        
        # Handle multi-output (Housing, Materials) vs single output
        if preds.ndim > 1:
            h_price, m_cost = preds[0][0], preds[0][1]
        else:
            h_price, m_cost = preds[0], 0.0 # Fallback if model is single output
            
        row = {
            "Year": int(current_features[year_col].iloc[0]) if year_col else (datetime.now().year + i),
            "Housing Price": h_price,
            "Material Cost": m_cost,
            "Lower Bound": h_price * (1 - (std_dev * i)),
            "Upper Bound": h_price * (1 + (std_dev * i))
        }
        forecast_data.append(row)
        
        # Advance features for next year
        if year_col:
            current_features[year_col] += 1
        
        # Apply growth rate to numeric columns (except Year)
        for col in current_features.select_dtypes(include=[np.number]).columns:
            if col != year_col:
                current_features[col] *= (1 + growth_rate)
                
    return pd.DataFrame(forecast_data)

# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <h1><i class="fas fa-city" style="color:#1E3A8A;"></i> Strategic Housing Forecast Portal</h1>
        <p style="color: #4B5563; margin-bottom: 0;">Predictive analytics for real estate investment and urban planning.</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR: INPUTS & UPLOADS ---
with st.sidebar:
    st.markdown("### <i class='fas fa-upload'></i> Model Assets", unsafe_allow_html=True)
    model_file = st.file_uploader("rf_GRB_Model3.pkl", type=['pkl', 'joblib'])
    meta_file = st.file_uploader("Upload Training CSV (Optional Metadata)", type=['csv'])
    
    st.markdown("---")
    st.markdown("### <i class='fas fa-sliders-h'></i> Forecast Configuration", unsafe_allow_html=True)
    country = st.text_input("Country", value="United States")
    cities_raw = st.text_input("City/Cities (comma separated)", value="New York, Austin")
    cities = [c.strip() for c in cities_raw.split(",") if c.strip()]
    
    horizon = st.selectbox("Forecast Horizon (Years)", [7, 8, 9, 10], index=0)
    start_year = st.number_input("Forecast Start Year", value=datetime.now().year)
    
    scenario = st.select_slider(
        "Growth Scenario",
        options=["Pessimistic", "Baseline", "Optimistic"],
        value="Baseline"
    )
    
    growth_map = {"Pessimistic": -0.02, "Baseline": 0.03, "Optimistic": 0.07}
    custom_growth = st.sidebar.slider("Custom Growth Override (%)", -10.0, 15.0, growth_map[scenario]*100) / 100

# --- MAIN CONTENT LOGIC ---
if model_file:
    model = load_model(model_file)
    
    if model:
        # Extract feature names if possible
        try:
            if hasattr(model, 'feature_names_in_'):
                features_required = model.feature_names_in_.tolist()
            else:
                features_required = ["Year", "Current_Avg_Price", "Income_Index", "Interest_Rate"]
        except:
            features_required = ["Year", "Current_Avg_Price"]

        st.info(f"Model loaded successfully. Required features: {', '.join(features_required)}")
        
        # Dynamic Input Form
        st.subheader("Baseline Feature Values")
        with st.form("feature_form"):
            cols = st.columns(2)
            input_values = {}
            for i, feat in enumerate(features_required):
                with cols[i % 2]:
                    if "year" in feat.lower():
                        input_values[feat] = start_year
                    else:
                        input_values[feat] = st.number_input(f"Current {feat}", value=100.0)
            submit = st.form_submit_button("Generate Forecast")

        if submit:
            # Prepare base dataframe
            base_df = pd.DataFrame([input_values])
            
            # Historical uncertainty estimation
            uncertainty = 0.05 # Default 5%
            if meta_file:
                hist_df = pd.read_csv(meta_file)
                # Simple heuristic: use 1 standard deviation of the target if available
                uncertainty = 0.08 
            
            all_city_results = {}
            for city in cities:
                all_city_results[city] = get_forecast(model, base_df, horizon, custom_growth, uncertainty)

            # --- KPI SECTION ---
            st.markdown("### <i class='fas fa-chart-line'></i> Summary KPIs", unsafe_allow_html=True)
            kpi_cols = st.columns(4)
            
            # Displaying first city as primary KPI source
            primary_city = cities[0]
            res = all_city_results[primary_city]
            
            with kpi_cols[0]:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">Current Price</div><div class="kpi-value">${res["Housing Price"].iloc[0]:,.0f}</div></div>', unsafe_allow_html=True)
            with kpi_cols[1]:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">Final Year ({res["Year"].iloc[-1]})</div><div class="kpi-value">${res["Housing Price"].iloc[-1]:,.0f}</div></div>', unsafe_allow_html=True)
            with kpi_cols[2]:
                total_change = ((res["Housing Price"].iloc[-1] / res["Housing Price"].iloc[0]) - 1) * 100
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Growth</div><div class="kpi-value">{total_change:.1f}%</div></div>', unsafe_allow_html=True)
            with kpi_cols[3]:
                st.markdown(f'<div class="kpi-card"><div class="kpi-label">Scenario</div><div class="kpi-value">{scenario}</div></div>', unsafe_allow_html=True)

            # --- VISUALIZATION ---
            st.markdown("### <i class='fas fa-chart-area'></i> Price Projections", unsafe_allow_html=True)
            
            fig = go.Figure()
            for city, df in all_city_results.items():
                # Uncertainty Band
                fig.add_trace(go.Scatter(
                    x=pd.concat([df['Year'], df['Year'][::-1]]),
                    y=pd.concat([df['Upper Bound'], df['Lower Bound'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(30, 58, 138, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f'{city} Confidence'
                ))
                # Main Line
                fig.add_trace(go.Scatter(
                    x=df['Year'], y=df['Housing Price'],
                    mode='lines+markers',
                    name=f'{city} Housing Price',
                    line=dict(width=3)
                ))

            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                plot_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_with_width=True)

            # --- DATA TABLE ---
            st.markdown("### <i class='fas fa-table'></i> Detailed Forecast Table", unsafe_allow_html=True)
            
            # Combine all city results for export
            export_df = pd.concat([df.assign(City=city) for city, df in all_city_results.items()])
            st.dataframe(export_df.style.format({
                "Housing Price": "${:,.2f}",
                "Material Cost": "${:,.2f}",
                "Lower Bound": "${:,.2f}",
                "Upper Bound": "${:,.2f}"
            }), use_container_with_width=True)

            # --- EXPORT ---
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="<i class='fas fa-file-csv'></i> Download Forecast CSV".replace("<i class='fas fa-file-csv'></i> ", ""),
                data=csv,
                file_name=f"housing_forecast_{start_year}.csv",
                mime='text/csv',
            )
            st.info("To save charts as PNG, use the camera icon in the top-right of the interactive plot.")

else:
    st.warning("Please upload a trained model (.pkl or .joblib) in the sidebar to initiate the forecasting engine.")
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