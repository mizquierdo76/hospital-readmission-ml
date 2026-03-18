import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏥 Hospital Readmission Risk Predictor")

st.markdown("### 🔍 Predict patient readmission risk using machine learning")

st.write("Enter patient details:")

# Inputs (match your dataset roughly)
age = st.slider("Age", 20, 90, 50)
time_in_hospital = st.slider("Days in Hospital", 1, 14, 5)
num_lab_procedures = st.slider("Lab Procedures", 1, 100, 40)
num_medications = st.slider("Medications", 1, 50, 10)
number_diagnoses = st.slider("Diagnoses Count", 1, 10, 3)

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "time_in_hospital": [time_in_hospital],
    "num_lab_procedures": [num_lab_procedures],
    "num_medications": [num_medications],
    "number_diagnoses": [number_diagnoses]
})

# Predict
import numpy as np

if st.button("Predict Risk"):
    # create base input
    input_data = pd.DataFrame({
        "age": [age],
        "time_in_hospital": [time_in_hospital],
        "num_lab_procedures": [num_lab_procedures],
        "num_medications": [num_medications],
        "number_diagnoses": [number_diagnoses]
    })

    # 🔥 ADD THIS PART
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0

        # ensure correct order
        input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk ({probability:.2%})")

    # -------- EXPLANATION --------
    st.subheader("🧠 Why this prediction?")

    risk_factors = []

if age > 65:
    risk_factors.append(f"Age ({age}) is above 65 → higher risk")

if time_in_hospital > 5:
    risk_factors.append(f"Hospital stay ({time_in_hospital} days) is long")

if num_medications > 8:
    risk_factors.append(f"Medications ({num_medications} indicate complexity")

if number_diagnoses > 4:
    risk_factors.append(f"Diagnoses count ({number_diagnoses}) is high")

if len(risk_factors) > 0:
    for factor in risk_factors:
        st.write("•", factor)
else:
    st.write("Patient profile shows generally low risk indicators")
       
if number_diagnoses > 5:
    risk_factors.append("Multiple diagnoses increase complication risk")

if len(risk_factors) > 0:
    for factor in risk_factors:
        st.write("•", factor)
else:
    st.write("No major risk factors detected")
    
    st.subheader("🧠 Key Risk Drivers")

    input_df = pd.DataFrame({
        "Feature": ["Age", "Days in Hospital", "Lab Procedures", "Medications", "Diagnoses"],
        "Value": [age, time_in_hospital, num_lab_procedures, num_medications, number_diagnoses]
    })

import pandas as pd
import matplotlib.pyplot as plt


st.subheader("📊 Model Insignts (Top Drivers)")

fi = pd.read_csv("feature_importance_logreg.csv")

# 👇 Sort and take top 10 FIRST
fi_sorted = fi.sort_values(by=fi.columns[1], ascending=False).head(10)

# 👇 Clean names AFTER slicing (so lengths match)
clean_names = fi_sorted.iloc[:, 0].str.replace("cat_diag_", "", regex=False)
clean_names = clean_names.str.replace("_", " ")

fig, ax = plt.subplots(figsize=(8, 5))

ax.barh(clean_names, fi_sorted.iloc[:, 1])

ax.set_xlabel("Importance")
ax.set_title("Top Drivers of Readmission")

plt.tight_layout()

st.pyplot(fig)
