import streamlit as st
import joblib
import pandas as pd
import numpy as np
# 1. Config and layout setting
st.set_page_config(
    page_title="SmartHome AI Predictor",
    page_icon="🏡",
    layout="centered"
)

# 2. Safely load model
@st.cache_resource
def load_prediction_model():
    return joblib.load('smarthome_rf_model.pkl')

try:
    model = load_prediction_model()
except FileNotFoundError:
    st.error("Error: 'smarthome_rf_model.pkl' not detected. Make sure you execute all cells in your Jupyter Notebook first.")

# 3. Main Dashboard UI Header
st.title("🏡 SmartHome Value Estimator")
st.write("Leverage custom Machine Learning configurations to calculate smart property pricing.")
st.markdown("---")

# 4. Constructing Input Form layout
st.subheader("📊 Step 1: Input Structural Specifications")
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Total Finished Area (Sq Ft)", min_value=500, max_value=6000, value=2500, step=50)
    beds = st.slider("Number of Bedrooms", min_value=1, max_value=5, value=3)

with col2:
    age = st.number_input("Property Construction Age (Years)", min_value=0, max_value=30, value=5, step=1)
    baths = st.slider("Number of Bathrooms", min_value=1, max_value=4, value=2)

st.markdown("---")
st.subheader("💡 Step 2: System Features & Demographics")

col3, col4 = st.columns(2)
with col3:
    location = st.slider("Neighborhood Rating ⭐", min_value=1, max_value=5, value=4)
    security = st.checkbox("Integrated Smart Security System", value=True)

with col4:
    hub = st.checkbox("Central Automation Voice Hub", value=False)
    energy = st.checkbox("Eco-friendly/Energy Saving Appliances", value=True)

st.markdown("---")

# 5. Pipeline Logic Evaluation
if st.button("Calculate Property Value Evaluation", type="primary", use_container_width=True):
    
    # Pack input variables exactly mirroring the training DataFrame column scheme
    input_data = pd.DataFrame({
        'Square_Footage': [sqft],
        'Bedrooms': [beds],
        'Bathrooms': [baths],
        'Property_Age_Years': [age],
        'Smart_Security_System': [1 if security else 0],
        'Automation_Hub': [1 if hub else 0],
        'Energy_Saving_Appliances': [1 if energy else 0],
        'Location_Rating': [location]
    })
    
    # Calculate Prediction
    predicted_val = model.predict(input_data)[0]
    
    # Present Output Interface
    st.success(f"### 🎯 Estimated Market Valuation: **${predicted_val:,.2f} USD**")
    st.balloons()
