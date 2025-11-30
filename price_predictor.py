import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('best_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        final_cols = pickle.load(open('model_features.pkl', 'rb'))
        feature_options = pickle.load(open('feature_options.pkl', 'rb'))
        return model, scaler, final_cols, feature_options
    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e}. Please ensure all .pkl files are in the same directory.")
        st.stop()

model, scaler, FINAL_COLUMNS, FEATURE_OPTIONS = load_assets()

st.set_page_config(page_title="BMW Price Predictor", layout="wide")
st.title("🚗 BMW Price Predictor")
st.markdown("Enter the car specifications to predict the price.")
st.sidebar.header("Input Features")

model_type = st.sidebar.selectbox('Model', FEATURE_OPTIONS['model'])
year = st.sidebar.number_input('Year', 1990, 2025, 2018, step=1)
transmission = st.sidebar.selectbox('Transmission', FEATURE_OPTIONS['transmission'])
mileage = st.sidebar.number_input('Mileage', 0, 300000, 25000, step=500)
fuel_type = st.sidebar.selectbox('Fuel Type', FEATURE_OPTIONS['fuelType'])
tax = st.sidebar.number_input('Tax', 0, 1000, 150)
mpg = st.sidebar.number_input('MPG', 0.0, 200.0, 50.0, step=0.1)
engineSize = st.sidebar.number_input('Engine Size', 0.0, 10.0, 2.0, step=0.1)

if st.button('Predict Price'):
    
    data_dict = {
        'year': [year], 
        'mileage': [mileage], 
        'tax': [tax], 
        'mpg': [mpg],
        'engineSize': [engineSize],
        'model': [model_type], 
        'transmission': [transmission], 
        'fuelType': [fuel_type]
    }
    input_df = pd.DataFrame(data_dict)
    
    input_df = pd.get_dummies(input_df, columns=['model', 'transmission', 'fuelType'], drop_first=True)
    
    missing_cols = set(FINAL_COLUMNS) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[FINAL_COLUMNS] 
    
    scaling_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    
    input_to_scale = input_df[scaling_cols]
    input_df[scaling_cols] = scaler.transform(input_to_scale)
    
    log_price_pred = model.predict(input_df)[0]
    price_pred = np.exp(log_price_pred)

    st.subheader("Predicted BMW Price:")
    st.success(f"Estimated Price: **£{price_pred:,.2f}**")