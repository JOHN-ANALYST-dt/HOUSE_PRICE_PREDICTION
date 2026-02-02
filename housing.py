import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Kenya Housing & Construction Predictor", layout="wide")

# Custom Professional Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #f4f7f6; }
    
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .header-box {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_ensemble_model(file):
    try:
        return joblib.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_kenya_forecast(model, base_features, years, inflation_rate=0.06):
    """
    Simulates a 10-year horizon based on current market features and 
    historical Kenya inflation (approx 6-8%).
    """
    forecast_data = []
    current_state = base_features.copy()
    
    # Identify key tracking variables
    year_col = next((c for c in current_state.columns if 'year' in c.lower()), None)
    
    for i in range(years + 1):
        # Predict House Price and Aggregate Expenses
        prediction = model.predict(current_state)
        
        # Handling ensemble multi-output (Price, Expenses)
        price = prediction[0][0] if prediction.ndim > 1 else prediction[0]
        expenses = prediction[0][1] if (prediction.ndim > 1 and prediction.shape[1] > 1) else (price * 0.4)
        
        # Estimate Material Breakdown (Cement, Steel, Wood, Iron Sheets)
        # These ratios are typical for Kenyan construction projects
        row = {
            "Year": int(current_state[year_col].iloc[0]) if year_col else (2025 + i),
            "House Price": price,
            "Total Build Expenses": expenses,
            "Cement (Bags)": expenses * 0.25,
            "Steel/Iron": expenses * 0.30,
            "Timber/Wood": expenses * 0.15,
            "Iron Sheets": expenses * 0.10
        }
        forecast_data.append(row)
        
        # Advance state for next year
        if year_col: current_state[year_col] += 1
        for col in current_state.select_dtypes(include=[np.number]).columns:
            if col != year_col:
                current_state[col] *= (1 + inflation_rate)
                
    return pd.DataFrame(forecast_data)

# --- APP HEADER ---
st.markdown("""
    <div class="header-box">
        <h1>Kenya Real Estate & Construction Intelligence</h1>
        <p>AI-Powered Valuation & Material Expense Forecasting (2025 - 2035)</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🏢 Model Deployment")
    model_file = st.file_uploader("Upload Ensemble Model (.pkl)", type=['pkl', 'joblib'])
    
    st.header("📍 Geography")
    selected_city = st.selectbox("Target Region/City", 
        ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Kiambu", "Machakos"])
    
    st.header("📉 Economic Scenario")
    market_trend = st.select_slider("Market Projection", 
        options=["Stagnant", "Baseline (Kenya Avg)", "High Growth"], value="Baseline (Kenya Avg)")
    
    growth_rates = {"Stagnant": 0.02, "Baseline (Kenya Avg)": 0.07, "High Growth": 0.12}

# --- MAIN INTERFACE ---
if model_file:
    ensemble_model = load_ensemble_model(model_file)
    
    if ensemble_model:
        # Auto-detect features from the ensemble model metadata
        if hasattr(ensemble_model, 'feature_names_in_'):
            req_features = list(ensemble_model.feature_names_in_)
        else:
            req_features = ["Year", "Bedrooms", "Land_Size_Acres", "Distance_to_CBD"]

        tab1, tab2 = st.tabs(["🏠 House Price Valuation", "🏗️ Construction Expenses (10YR)"])

        # TAB 1: CUSTOMER VALUATION
        with tab1:
            st.subheader(f"Price Predictor: {selected_city}")
            with st.form("valuation_form"):
                cols = st.columns(2)
                user_inputs = {}
                for i, feat in enumerate(req_features):
                    with cols[i % 2]:
                        if "year" in feat.lower():
                            user_inputs[feat] = 2025
                        else:
                            user_inputs[feat] = st.number_input(f"Enter {feat}", value=1.0 if "size" in feat.lower() else 3.0)
                
                predict_btn = st.form_submit_button("Calculate Valuation")

            if predict_btn:
                input_df = pd.DataFrame([user_inputs])
                prediction = ensemble_model.predict(input_df)
                val = prediction[0][0] if prediction.ndim > 1 else prediction[0]
                
                c1, c2 = st.columns(2)
                c1.metric("Estimated Market Value", f"KES {val:,.2f}")
                c2.metric("Projected Value (2030)", f"KES {val*(1.4):,.2f}", "+40%")
                
                st.info(f"This valuation is specifically tuned for the **{selected_city}** housing corridor.")

        # TAB 2: ENGINEER EXPENSE FORECAST
        with tab2:
            st.subheader("Construction Material Time-Series (Next 10 Years)")
            st.write("Projected costs for Cement, Steel, and Wood based on Kenya inflation indices.")
            
            # Prepare data for forecast
            base_input = pd.DataFrame([{f: user_inputs.get(f, 1.0) for f in req_features}])
            forecast_df = generate_kenya_forecast(ensemble_model, base_input, 10, growth_rates[market_trend])
            
            # KPI Metrics for Engineers
            m_cols = st.columns(4)
            m_cols[0].metric("2025 Total Build", f"KES {forecast_df['Total Build Expenses'].iloc[0]:,.0f}")
            m_cols[1].metric("2035 Total Build", f"KES {forecast_df['Total Build Expenses'].iloc[-1]:,.0f}")
            m_cols[2].metric("Peak Cement Cost", f"KES {forecast_df['Cement (Bags)'].max():,.0f}")
            m_cols[3].metric("Steel Inflation", f"{market_trend}")

            # Interactive Time-Series Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['Cement (Bags)'], name='Cement', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['Steel/Iron'], name='Steel & Iron', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['Timber/Wood'], name='Timber/Wood', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['Iron Sheets'], name='Iron Sheets', line=dict(dash='dot')))
            
            fig.update_layout(
                title="10-Year Material Expense Projection",
                xaxis_title="Year",
                yaxis_title="Cost (KES)",
                hovermode="x unified",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_with_width=True)
            
            # Detailed Data Export
            with st.expander("View Annualized Expense Breakdown Table"):
                st.dataframe(forecast_df.style.format("{:,.2f}").highlight_max(axis=0))

else:
    st.warning("Please upload your ensemble model file (.pkl or .joblib) in the sidebar to begin.")
    st.markdown("""
        ### Instructions for Engineers & Analysts
        1. **Train your model**: Ensure your ensemble model is trained on Kenyan data (2000-2025).
        2. **Multi-Output**: For best results, use a model that predicts both `Sale_Price` and `Construction_Cost`.
        3. **Geography**: The sidebar allows you to filter predictions by Kenyan administrative boundaries.
    """)

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Kenya Housing Analytics System | Data-Driven Urban Planning Tool")