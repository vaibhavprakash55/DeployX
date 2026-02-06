import streamlit as st
import pickle
import pandas as pd
from ui import load_ui, section

# 🔥 CALL UI HERE
load_ui()

# Load trained model
model = pickle.load(open("best_model.pkl", "rb"))

st.set_page_config(page_title="Healthcare Predictor", layout="wide")

# Sidebar
st.sidebar.subheader("🚀 Model Info")
st.sidebar.metric("Best Model Accuracy", "~80%", "Auto-selected best model")

# Intro cards (NO MORE WHITE EMPTY BOX)
section(
    "📌 About This Project",
    "This ML system predicts patient test results using clinical and admission data."
)

section(
    "🧠 How It Works",
    "Random Forest model trained on healthcare records with automated preprocessing."
)

# ---------------- INPUT FORM ----------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    blood_type = st.selectbox("Blood Type", ["A", "B", "AB", "O"])
    condition = st.selectbox(
        "Medical Condition",
        ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Cancer", "Other"]
    )

with col2:
    billing = st.number_input("Billing Amount ($)", 0.0, value=5000.0)
    admission = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
    medication = st.selectbox(
        "Medication",
        ["Insulin", "Antibiotics", "Painkiller", "Steroids", "Other"]
    )

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Test Result"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Blood Type": blood_type,
        "Medical Condition": condition,
        "Billing Amount": billing,
        "Admission Type": admission,
        "Medication": medication
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"🧪 Predicted Test Result: **{prediction}**")
from ui import load_ui, section, info_block
st.markdown("##")  # spacing

info_block(
    "🧪 What does this prediction mean?",
    [
        "Normal: Test results are within expected medical range.",
        "Abnormal: Further clinical investigation may be required.",
        "Prediction is based on historical healthcare patterns."
    ]
)

info_block(
    "🧭 How to use this system",
    [
        "Fill patient details accurately.",
        "Click on 'Predict Test Result'.",
        "Use the result as a decision-support tool, not diagnosis."
    ]
)

info_block(
    "⚠️ Medical Disclaimer",
    [
        "This application is for educational & hackathon purposes.",
        "It does NOT replace professional medical advice.",
        "Always consult certified healthcare professionals."
    ]
)
