import streamlit as st
import os
import joblib
import requests
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import json
import fitz  # PyMuPDF
import datetime
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------- Load models --------------------------
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
        if os.path.getsize(model_path) == 0:
            raise ValueError(f"Model file {filename} is empty")
        with open(model_path, "rb") as f:
            models[name] = joblib.load(f)
    return models

models = load_models()

# ---------------------- Groq API for Report Parsing --------------------------
def extract_features_from_report(report_text):
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

    Return only a valid JSON object with keys: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
    No explanation or additional text.
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
    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        st.error("❌ Failed to extract features from the report.")
        st.error(f"Error: {str(e)}")
        st.stop()

# ---------------------- PDF Report Generator --------------------------
def generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path):
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()

    y = 50
    line_spacing = 20

    page.insert_text((50, y), "Heart Disease Prediction Report", fontsize=16)
    y += line_spacing * 2

    page.insert_text((50, y), f"Patient Name: {patient_name}", fontsize=12)
    y += line_spacing
    page.insert_text((50, y), f"Date: {datetime.date.today().strftime('%B %d, %Y')}", fontsize=12)
    y += line_spacing

    page.insert_text((50, y), "Input Features:", fontsize=12)
    y += line_spacing
    for key, value in input_data.items():
        page.insert_text((60, y), f"{key}: {value}", fontsize=11)
        y += line_spacing

    y += line_spacing
    page.insert_text((50, y), "Prediction Results:", fontsize=12)
    y += line_spacing
    for model_name in predictions:
        result = "High Risk" if predictions[model_name] == 1 else "Low Risk"
        prob = probabilities[model_name] * 100
        page.insert_text((60, y), f"{model_name}: {result} ({prob:.2f}%)", fontsize=11)
        y += line_spacing

    img_rect = fitz.Rect(50, y + 10, 400, y + 310)
    page.insert_image(img_rect, filename=chart_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf_doc.save(tmpfile.name)
        return tmpfile.name

# ---------------------- Streamlit UI --------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Upload a report or enter health data to predict heart disease risk using multiple ML models!")

patient_name = st.text_input("Enter Patient Name", "")

option = st.radio("Choose Input Method", ["Enter Manually", "Upload Health Report"])
input_data = {}

if option == "Upload Health Report":
    uploaded_file = st.file_uploader("Upload Report (TXT or PDF)", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            report_text = uploaded_file.read().decode()
        else:
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

# Handle missing values by prompting the user for input if any field is missing
missing_fields = [key for key in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
                  if key not in input_data or input_data[key] in [None, "", "NaN"]]

if missing_fields:
    st.warning(f"⚠️ Missing values detected for: {', '.join(missing_fields)}. Please fill them manually below.")
    for field in missing_fields:
        if field in ["sex", "fbs", "exang"]:
            input_data[field] = st.selectbox(f"Enter value for {field}", [1, 0])
        elif field in ["cp", "restecg", "slope", "ca", "thal"]:
            input_data[field] = st.slider(f"Enter value for {field}", 0, 3 if field != "thal" else 3, 1)
        elif field == "oldpeak":
            input_data[field] = st.number_input(f"Enter value for {field}", 0.0, 6.0, 1.0)
        else:
            input_data[field] = st.number_input(f"Enter value for {field}", 0)

if st.button("🔍 Predict Heart Disease"):
    if not patient_name.strip():
        st.warning("Please enter the patient's name before prediction.")
        st.stop()

    features = pd.DataFrame([input_data])
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        pred = model.predict(features)[0]
prob = model.predict_proba(features)[0][pred]
predictions[name] = pred
probabilities[name] = prob
st.subheader("Prediction Results")
for model_name in predictions:
    result = "High Risk" if predictions[model_name] == 1 else "Low Risk"
    prob = probabilities[model_name] * 100
    st.write(f"{model_name}: {result} ({prob:.2f}%)")

# Generate prediction chart
fig, ax = plt.subplots()
ax.bar(predictions.keys(), [probabilities[name] * 100 for name in predictions])
ax.set_xlabel('Model')
ax.set_ylabel('Prediction Probability (%)')
ax.set_title('Prediction Probabilities from Different Models')

chart_path = "/tmp/prediction_chart.png"
plt.savefig(chart_path)
st.pyplot(fig)

# Generate PDF report
pdf_file = generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path)
st.download_button("Download PDF Report", pdf_file, file_name=f"{patient_name}_heart_disease_report.pdf")

