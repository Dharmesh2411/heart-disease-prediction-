import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tempfile
import os
import fitz  # PyMuPDF

# Load the trained models
models = {
    "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl")
}

# Streamlit app configuration
st.set_page_config(page_title="Heart Disease Detection", layout="wide")
st.title("❤️ Heart Disease Detection App")
st.markdown("Predict heart disease using multiple ML models and compare results.")

# Patient name input
patient_name = st.text_input("👤 Enter Patient Name")

# Input features
st.subheader("📋 Enter Patient Details")
age = st.slider("Age", 20, 100, 50)
sex = st.radio("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Serum Cholestoral (chol)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", ["Yes", "No"])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)
exang = st.radio("Exercise Induced Angina (exang)", ["Yes", "No"])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert categorical inputs
input_data = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,
    "restecg": restecg,
    "thalach": thalach,
    "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

# Generate PDF report
def generate_pdf_with_fitz(name, inputs, predictions, probabilities, chart_path):
    pdf_path = os.path.join(tempfile.gettempdir(), f"{name}_heart_report.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), f"Heart Disease Prediction Report", fontsize=18, fontname="helv", fill=(1, 0, 0))
    page.insert_text((50, 80), f"Patient Name: {name}", fontsize=14)

    y = 110
    page.insert_text((50, y), "Patient Details:", fontsize=12)
    for k, v in inputs.items():
        y += 20
        page.insert_text((60, y), f"{k}: {v}", fontsize=11)

    y += 40
    page.insert_text((50, y), "Prediction Results:", fontsize=12)
    for model, result in predictions.items():
        y += 20
        prob = probabilities[model]
        status = "High Risk" if result == 1 else "Low Risk"
        page.insert_text((60, y), f"{model}: {status} ({prob*100:.2f}%)", fontsize=11)

    # Insert chart image
    img_rect = fitz.Rect(50, y + 40, 400, y + 280)
    page.insert_image(img_rect, filename=chart_path)

    doc.save(pdf_path)
    return pdf_path

# Predict
if st.button("Predict Heart Disease"):
    if not patient_name.strip():
        st.warning("Please enter the patient's name before prediction.")
        st.stop()

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

    best_model = max(probabilities, key=probabilities.get)
    st.success(f"⭐ **Most Confident Model: {best_model} ({probabilities[best_model]*100:.2f}% probability of heart disease)**")

    # Plot prediction probability comparison
    st.subheader("📊 Probability Comparison Across Models")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(probabilities.keys(), [p*100 for p in probabilities.values()], color='skyblue')
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Heart Disease Probability by Model")

    chart_path = os.path.join(tempfile.gettempdir(), "model_probabilities.png")
    fig.savefig(chart_path)
    st.pyplot(fig)

    # Generate PDF report
    pdf_path = generate_pdf_with_fitz(patient_name, input_data, predictions, probabilities, chart_path)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Prediction Report", f, file_name="Heart_Disease_Report.pdf", mime="application/pdf")

    # ------------------ ACCURACY CHART ------------------
    st.subheader("📈 Model Accuracy Comparison")
    # Dummy accuracy values – replace with actual test set results if available
    accuracies = {
        "Logistic Regression": 0.82,
        "Random Forest": 0.91,
        "KNN": 0.78,
        "Decision Tree": 0.76,
        "SVM": 0.80,
        "Naive Bayes": 0.74
    }

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.bar(accuracies.keys(), [v*100 for v in accuracies.values()], color='gray')
    best_model_acc = max(accuracies, key=accuracies.get)
    bars[list(accuracies.keys()).index(best_model_acc)].set_color('green')

    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Model Accuracy Comparison")
    for i, v in enumerate(accuracies.values()):
        ax2.text(i, v*100 + 1, f"{v*100:.1f}%", ha='center', fontsize=10)
    st.pyplot(fig2)
