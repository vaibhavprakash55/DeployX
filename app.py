import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline model
model = pickle.load(open("best_model.pkl", "rb"))

st.set_page_config(page_title="Healthcare Predictor", layout="wide")

# Sidebar
st.sidebar.subheader("🚀 Model Info")
st.sidebar.metric(
    label="Best Model Accuracy",
    value="~80%",
    delta="Auto-selected best model"
)
st.sidebar.write("---")

st.title("🏥 Patient Test Result Prediction")
st.write("This ML model predicts **Patient Test Results** based on clinical details.")

# ------------------ USER INPUT ------------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    blood_type = st.selectbox("Blood Type", ["A", "B", "AB", "O"])
    condition = st.selectbox(
        "Medical Condition",
        ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "Cancer", "Other"]
    )

with col2:
    billing = st.number_input("Billing Amount ($)", min_value=0.0, value=5000.0)
    admission = st.selectbox(
        "Admission Type",
        ["Emergency", "Elective", "Urgent"]
    )
    medication = st.selectbox(
        "Medication",
        ["Insulin", "Antibiotics", "Painkiller", "Steroids", "Other"]
    )

# ------------------ PREDICTION ------------------

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
