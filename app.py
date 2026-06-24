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

model, scaler = load_model()

st.title("❤️ Heart Disease Prediction System")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Features", "13", "Clinical Parameters")
with col2:
    st.metric("Model Status", "✅ Loaded")
with col3:
    st.metric("Prediction", "Ready", "Enter patient details")

st.markdown("---")

st.sidebar.header("Patient Information")
st.sidebar.markdown("Enter the patient's clinical parameters:")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex_encoded = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
cp_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
fbs_encoded = 1 if fbs == "True" else 0

restecg = st.sidebar.selectbox(
    "Resting ECG Results",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)
restecg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)

thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang_encoded = 1 if exang == "Yes" else 0

oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)

slope = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)
slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope)

ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])

thal = st.sidebar.selectbox(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)
thal_encoded = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** All fields are required for prediction.")

input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_encoded],
    'cp': [cp_encoded],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs_encoded],
    'restecg': [restecg_encoded],
    'thalach': [thalach],
    'exang': [exang_encoded],
    'oldpeak': [oldpeak],
    'slope': [slope_encoded],
    'ca': [ca],
    'thal': [thal_encoded]
})

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Parameters")
    param_data = {
        "Parameter": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol", 
                      "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", 
                      "Exercise Angina", "ST Depression", "ST Slope", "Major Vessels", "Thalassemia"],
        "Value": [age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                  exang, oldpeak, slope, ca, thal]
    }
    param_df = pd.DataFrame(param_data)
    st.dataframe(param_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Feature Distribution")
    feature_values = [age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, 
                      restecg_encoded, thalach, exang_encoded, oldpeak, slope_encoded, ca, thal_encoded]
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    fig = go.Figure(data=go.Scatter(
        x=feature_names,
        y=feature_values,
        mode='lines+markers',
        name='Patient Values',
        line=dict(color='royalblue', width=2),
        marker=dict(size=10, color='red')
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Features",
        yaxis_title="Value",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

if st.button("🔍 Predict Heart Disease", use_container_width=True, type="primary"):
    with st.spinner("Analyzing patient data..."):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error("⚠️ **Prediction: Heart Disease Detected**")
                st.metric("Risk Level", "High", delta="⚠️ Seek Medical Attention")
            else:
                st.success("✅ **Prediction: No Heart Disease**")
                st.metric("Risk Level", "Low", delta="✅ Healthy")
        
        with col2:
            st.metric("Confidence Score", f"{probability[0][prediction[0]]*100:.2f}%")
            st.metric("Probability of Disease", f"{probability[0][1]*100:.2f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Prediction Probability")
            fig2 = go.Figure(data=[
                go.Bar(
                    x=['No Disease', 'Disease'],
                    y=[probability[0][0]*100, probability[0][1]*100],
                    marker_color=['#00cc96', '#ef553b'],
                    text=[f'{probability[0][0]*100:.1f}%', f'{probability[0][1]*100:.1f}%'],
                    textposition='auto',
                )
            ])
            fig2.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.subheader("📊 Risk Assessment")
            risk_percentage = probability[0][1] * 100
            if risk_percentage < 30:
                st.success(f"🟢 **Low Risk** ({risk_percentage:.1f}%)")
                st.progress(risk_percentage/100)
                st.info("Continue healthy lifestyle and regular checkups.")
            elif risk_percentage < 60:
                st.warning(f"🟡 **Moderate Risk** ({risk_percentage:.1f}%)")
                st.progress(risk_percentage/100)
                st.warning("Consider lifestyle changes and consult a doctor.")
            else:
                st.error(f"🔴 **High Risk** ({risk_percentage:.1f}%)")
                st.progress(risk_percentage/100)
                st.error("Please consult a cardiologist immediately!")
        
        st.markdown("---")
        
        with st.expander("📋 Detailed Patient Report"):
            st.subheader("Patient Summary")
            risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
            report_data = {
                "Category": ["Patient Age", "Gender", "Risk Assessment", "Disease Probability", 
                            "Confidence", "Prediction Status"],
                "Details": [f"{age} years", sex, risk_level, f"{probability[0][1]*100:.2f}%",
                           f"{probability[0][prediction[0]]*100:.2f}%", 
                           "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"]
            }
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)
            
            st.subheader("Key Risk Factors")
            risk_factors = []
            if age > 55:
                risk_factors.append(f"Age ({age} years) - Higher risk above 55")
            if chol > 240:
                risk_factors.append(f"High Cholesterol ({chol} mg/dl)")
            if trestbps > 140:
                risk_factors.append(f"High Blood Pressure ({trestbps} mm Hg)")
            if oldpeak > 2.0:
                risk_factors.append(f"Significant ST Depression ({oldpeak})")
            if cp_encoded >= 2:
                risk_factors.append("Atypical or Asymptomatic Chest Pain")
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("No major risk factors identified.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>⚠️ This tool is for educational purposes only. Always consult a healthcare professional for medical advice.</p>
    <p>Built with ❤️ using Machine Learning</p>
</div>
""", unsafe_allow_html=True)
    
