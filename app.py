import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏥 Hospital Readmission Risk Predictor")

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
    input_data = np.array([[age, days, labs, meds, diagnoses]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Readmission ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Readmission ({probability:.2%})")
