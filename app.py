import streamlit as st
import os
import cloudpickle
import requests
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------- Load models --------------------------
@st.cache_resource
def load_models():
    model_filenames = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "KNN": "knn_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "SVM": "svm_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl"
    }
    models = {}
    for name, filename in model_filenames.items():
        model_path = hf_hub_download(repo_id="jaik256/heartDiseasePredictor", filename=filename)
        with open(model_path, "rb") as f:
            models[name] = cloudpickle.load(f)
    return models

models = load_models()

# ---------------------- UI --------------------------
st.title("❤️ Heart Disease Prediction App")
st.markdown("Upload a report or enter health data to predict heart disease risk using multiple ML models!")

option = st.radio("Choose Input Method", ["Enter Manually", "Upload Health Report"])

def extract_features_from_report(report_text):
    # Call Groq API (LLaMA3) to extract numeric features from health report
    prompt = f"""Extract the following values as numbers from the medical report below:
    - age
    - sex (0 = female, 1 = male)
    - cp (0–3: chest pain type)
    - trestbps (resting blood pressure)
    - chol (serum cholesterol)
    - fbs (fasting blood sugar > 120 mg/dl: 1, else 0)
    - restecg (resting ECG: 0–2)
    - thalach (max heart rate)
    - exang (exercise-induced angina: 1 = yes, 0 = no)
    - oldpeak
    - slope (0–2)
    - ca (number of major vessels: 0–3)
    - thal (1 = normal, 2 = fixed defect, 3 = reversible defect)

    Report:
    {report_text}

    Return as JSON dictionary with keys: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
    """

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    return json.loads(content)

if option == "Upload Health Report":
    uploaded_file = st.file_uploader("Upload Report (TXT or PDF)", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            report_text = uploaded_file.read().decode()
        else:
            import fitz  # PyMuPDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            report_text = "\n".join([page.get_text() for page in doc])
            doc.close()
        st.subheader("Extracted Report Text:")
        st.text(report_text)
        st.info("Calling Groq API to extract features...")
        input_data = extract_features_from_report(report_text)

elif option == "Enter Manually":
    input_data = {
        "age": st.number_input("Age", 20, 100, 50),
        "sex": st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female"),
        "cp": st.slider("Chest Pain Type (0–3)", 0, 3, 1),
        "trestbps": st.number_input("Resting Blood Pressure", 80, 200, 120),
        "chol": st.number_input("Cholesterol", 100, 600, 240),
        "fbs": st.selectbox("Fasting Blood Sugar > 120", [1, 0]),
        "restecg": st.slider("Resting ECG (0–2)", 0, 2, 1),
        "thalach": st.number_input("Max Heart Rate", 60, 220, 150),
        "exang": st.selectbox("Exercise Induced Angina", [1, 0]),
        "oldpeak": st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0),
        "slope": st.slider("Slope (0–2)", 0, 2, 1),
        "ca": st.slider("Major Vessels Colored (0–3)", 0, 3, 0),
        "thal": st.slider("Thal (1=Normal, 2=Fixed, 3=Reversible)", 1, 3, 2)
    }

if st.button("Predict Heart Disease"):
    features = pd.DataFrame([input_data])
    
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        predictions[name] = pred
        probabilities[name] = prob

    st.subheader("🩺 Prediction Results from Multiple Models:")
    for name in predictions:
        st.write(f"**{name}:** {'High Risk' if predictions[name]==1 else 'Low Risk'} | Probability: {probabilities[name]*100:.2f}%")

    # Find best model based on highest probability
    best_model = max(probabilities, key=probabilities.get)
    st.success(f"⭐ **Most Confident Model: {best_model} ({probabilities[best_model]*100:.2f}% probability of heart disease)**")

    # Plot comparison
    st.subheader("📊 Probability Comparison Across Models")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(probabilities.keys(), [p*100 for p in probabilities.values()], color='skyblue')
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Heart Disease Probability by Model")
    st.pyplot(fig)
