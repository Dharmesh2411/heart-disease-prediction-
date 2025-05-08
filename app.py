import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestClassifier  # Required for joblib to load

st.title("❤️ Heart Disease Predictor")

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="jaik256/heartDiseasePredictor", filename="heart_disease_model.joblib")
    return joblib.load(model_path)

model = load_model()

st.subheader("🧪 Enter patient details:")
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200)
chol = st.number_input("Serum Cholesterol (chol)", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", 70, 210)
exang = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

features = [[age, sex, cp, trestbps, chol, fbs, restecg,
             thalach, exang, oldpeak, slope, ca, thal]]

if st.button("🔍 Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    st.markdown("### 🩺 Result:")
    if prediction == 1:
        st.error(f"High chance of heart disease ({probability*100:.2f}%)")
    else:
        st.success(f"Low chance of heart disease ({(1-probability)*100:.2f}%)")
