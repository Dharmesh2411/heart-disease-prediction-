import os
import streamlit as st
import tempfile
import fitz  # PyMuPDF
import pandas as pd
import joblib
import openai
import base64
from dotenv import load_dotenv
from predict import predict_heart_disease, extract_features_from_report
from pdf_generator import generate_pdf_with_fitz

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load the ML model
model = joblib.load("heart_model.joblib")

# Streamlit UI
st.set_page_config(page_title="🫀 AI Medical Assistant", layout="centered")
st.title("🩺 Smart Health Checker")
st.markdown("**An AI-Powered Heart Disease Prediction & Report Summarizer**")

# Radio selector
option = st.radio("Choose Input Method", ["Enter Manually", "Upload Health Report"])

# --- OPTION 1: Enter Manually ---
if option == "Enter Manually":
    st.subheader("🧾 Enter Patient Data")
    patient_name = st.text_input("Patient Name")

    age = st.slider("Age", 18, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholestoral (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"])
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    input_data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": ["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"].index(restecg),
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
        "ca": ca,
        "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
    }

    if st.button("Predict"):
        prediction, probabilities = predict_heart_disease(model, input_data)
        st.success(f"🩺 Prediction: {prediction}")
        st.info(f"🧪 Confidence: {probabilities[1]:.2%} (Disease), {probabilities[0]:.2%} (No Disease)")

        # Optionally generate and download PDF report
        if st.button("Download PDF Report"):
            chart_path = "chart.png"  # Replace with actual chart if generated
            pdf_path = generate_pdf_with_fitz(patient_name, input_data, prediction, probabilities, chart_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Report", f, file_name="heart_report.pdf", mime="application/pdf")


# --- OPTION 2: Upload Health Report ---
elif option == "Upload Health Report":
    uploaded_file = st.file_uploader("Upload Report (TXT or PDF)", type=["txt", "pdf"])

    if uploaded_file:
        # Extract report text
        if uploaded_file.name.endswith(".txt"):
            report_text = uploaded_file.read().decode()
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            report_text = "\n".join([page.get_text() for page in doc])
            doc.close()

        st.subheader("📜 Extracted Report Text")
        st.code(report_text[:5000] + ("..." if len(report_text) > 5000 else ""), language="text")

        # AI Summary
        if st.button("Ask AI about this report"):
            with st.spinner("Contacting medical assistant..."):
                prompt = f"This is a patient's medical report:\n{report_text[:4000]}\n\nGive a summary of this report in simple terms."
                try:
                    openai.api_key = GROQ_API_KEY
                    response = openai.ChatCompletion.create(
                        model="mixtral-8x7b-32768",
                        messages=[
                            {"role": "system", "content": "You are a helpful medical assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    summary = response.choices[0].message["content"]
                    with st.expander("🧠 AI's Summary", expanded=True):
                        st.write(summary)
                except Exception as e:
                    st.error(f"API Error: {e}")

        # Predict from report
        st.info("Calling Groq API to extract features for prediction...")
        input_data = extract_features_from_report(report_text)

        if st.button("Predict from Report"):
            prediction, probabilities = predict_heart_disease(model, input_data)
            st.success(f"🩺 Prediction: {prediction}")
            st.info(f"🧪 Confidence: {probabilities[1]:.2%} (Disease), {probabilities[0]:.2%} (No Disease)")

            if st.button("Download PDF Report"):
                chart_path = "chart.png"  # Replace with actual chart if generated
                pdf_path = generate_pdf_with_fitz("Report_Patient", input_data, prediction, probabilities, chart_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name="heart_report.pdf", mime="application/pdf")
