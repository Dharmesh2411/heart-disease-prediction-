import streamlit as st
import joblib
import os
import openai
from huggingface_hub import hf_hub_download

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set Groq API key
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="jaik256/heartDiseasePredictor", filename="heart_disease_model.joblib")
    return joblib.load(model_path)

model = load_model()

# Clinical features expected by the model
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

st.title("🫀 Heart Disease Prediction App")
st.write("Predict the likelihood of heart disease using patient data or a medical report.")

tab1, tab2 = st.tabs(["📝 Enter Manually", "📄 Upload Medical Report"])

# --------------------- TAB 1: Manual Input ---------------------
with tab1:
    st.header("Enter Patient Details")
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"{feature.upper()}", step=1.0)

    if st.button("🔍 Predict Manually"):
        input_values = [user_input[feat] for feat in features]
        prediction = model.predict([input_values])[0]
        result_text = "⚠️ High Risk of Heart Disease!" if prediction == 1 else "✅ Low Risk of Heart Disease."
        st.success(result_text)

# --------------------- TAB 2: Report Upload ---------------------
with tab2:
    st.header("Upload or Paste Report")
    report_text = st.text_area("Paste medical report here")

    if st.button("🧠 Extract and Predict"):
        with st.spinner("Analyzing report using Groq..."):

            # Prompt LLM to extract structured values
            prompt = f"""
Extract the following clinical values from the text below and return only JSON:
{', '.join(features)}.
Report:
\"\"\"{report_text}\"\"\"
Return format:
{{
  "age": 0, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3
}}
"""

            try:
                response = openai.ChatCompletion.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                content = response['choices'][0]['message']['content']
                parsed = eval(content) if isinstance(content, str) else content
                input_list = [parsed[feat] for feat in features]
                prediction = model.predict([input_list])[0]
                result_text = "⚠️ High Risk of Heart Disease!" if prediction == 1 else "✅ Low Risk of Heart Disease."
                st.success(result_text)
                st.json(parsed)

            except Exception as e:
                st.error(f"Error: {e}")
