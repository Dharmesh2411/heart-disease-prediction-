# ❤️ Heart Disease Prediction App

A Streamlit-powered web application that predicts the risk of heart disease using multiple machine learning models. The app accepts health data either through manual input or by uploading a health report and provides predictions from various classifiers. It also generates a professional PDF report summarizing the input, predictions, and model confidence chart.

---

## 🚀 Features

- 📄 Upload medical reports (TXT or PDF) and extract features using LLaMA3 (Groq API)
- 🧠 Predict heart disease risk using:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Support Vector Machine (SVM)
  - Naive Bayes
- 📊 Visualize model prediction probabilities in a bar chart
- 📥 Download a personalized PDF report including:
  - Patient name and age
  - All input features
  - Predictions from each model
  - Bar chart showing model confidence levels

---

## 🛠️ Technologies Used

- Streamlit
- Python (joblib, scikit-learn, pandas, matplotlib)
- PyMuPDF (for PDF manipulation)
- Hugging Face Hub (for model storage)
- Groq API (for report-based input extraction)
- dotenv (for environment variable handling)

---

## 📂 File Structure

heart-disease-prediction/
│
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies for the project
├── .env # Stores API keys (not uploaded to GitHub)
└── README.md # Project documentation

---

## 📋 Input Parameters

| Feature        | Description                                     |
|----------------|-------------------------------------------------|
| Age            | Age of the patient                              |
| Sex            | 0 = Female, 1 = Male                            |
| Chest Pain (cp)| Type of chest pain (0–3)                        |
| Resting BP     | Resting blood pressure                          |
| Cholesterol    | Serum cholesterol in mg/dl                      |
| FBS            | Fasting blood sugar > 120 mg/dl (1 = True)      |
| RestECG        | Resting electrocardiographic results (0–2)      |
| Max Heart Rate | Maximum heart rate achieved                     |
| Exang          | Exercise-induced angina (1 = Yes, 0 = No)       |
| Oldpeak        | ST depression induced by exercise               |
| Slope          | Slope of peak exercise ST segment (0–2)         |
| CA             | Major vessels colored by fluoroscopy (0–3)      |
| Thal           | 1 = Normal, 2 = Fixed Defect, 3 = Reversible     |

---

## 📌 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Dharmesh2411/heart-disease-prediction-n.git
cd heart-disease-prediction
GROQ_API_KEY=your_groq_api_key
pip install -r requirements.txt
streamlit run app.py

