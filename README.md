# 🏥 Healthcare Predictive Analytics System

## 📌 About the Project

This project was developed during the **DeployX ML & MLOps Hackathon 2026**, where our team **VEDA secured 1st Position 🥇**.

The goal of this application is to perform **Healthcare Predictive Analytics** by building an end-to-end Machine Learning pipeline.

It predicts patient **Test Results**:
- ✅ Normal  
- ⚠️ Abnormal  
- ❓ Inconclusive  

based on various clinical and demographic parameters.

💡 The system helps healthcare providers:
- Identify **high-risk patients**
- Take **early medical action**
- Reduce chances of **readmission**

---

## 🛠️ Technical Workflow & Implementation

### 🔹 1. Data Processing & Cleaning

Healthcare data is often noisy and inconsistent. We applied:

- **Outlier Handling (IQR Method)**  
  Removed extreme anomalies in billing data.

- **Noise Reduction**  
  Dropped irrelevant columns like Patient Name and Hospital ID.

- **Categorical Encoding**  
  Converted textual medical data into numerical form using Label Encoding.

---

### 🔹 2. Feature Engineering & Scaling

- **Robust Scaling**  
  Used `RobustScaler` to handle outliers effectively.

- **Balanced Dataset (Stratification)**  
  Ensured equal distribution of all classes in train/test sets to prevent bias.

---

### 🔹 3. Model Selection & Evaluation

We evaluated multiple ML models:

- **Logistic Regression**  
  Baseline model for linear patterns

- **Decision Tree**  
  Captured rule-based medical logic

- **🌟 Random Forest (Final Model)**  
  - Used ensemble learning (100+ trees)  
  - Reduced overfitting  
  - Achieved best and most stable accuracy  

---

### 🔹 4. Interactive Deployment

- **Streamlit Dashboard**  
  Real-time prediction system for user input

- **EDA Visualization Module**  
  Displays insights like:
  - Age vs Medical Condition
  - Data trends & patterns  

---

## 🚀 Features

- Real-time patient risk prediction  
- Clean and interactive UI (Streamlit)  
- End-to-end ML pipeline  
- Data visualization for insights  

---

## 🧠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib / Seaborn  

---

## 🎯 Outcome

- 🥇 Secured **1st Position** in Hackathon  
- Built a **production-level ML pipeline**  
- Solved a real-world healthcare problem  

---

## 💡 Future Improvements

- Deploy on Cloud (AWS / GCP)  
- Add Deep Learning models  
- Integrate with hospital systems (API-based)  
