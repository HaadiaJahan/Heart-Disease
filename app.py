import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model not found! Please train the model first.")

st.title("❤️ Heart Disease Prediction System")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Features", "13", "Clinical Parameters")
with col2:
    st.metric("Model Status", "✅ Loaded" if model_loaded else "❌ Not Found")
with col3:
    st.metric("Prediction", "Ready")
