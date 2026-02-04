import streamlit as st
import requests

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000/predict/"

def get_prediction(data):
    response = requests.post(FASTAPI_URL, json=data)
    return response.json()

# Streamlit UI
st.title("Diabetes Prediction")

# Collect input data
BMI = st.number_input("Enter BMI:", min_value=0.0)
Insulin = st.number_input("Enter Insulin level:", min_value=0.0)
Glucose = st.number_input("Enter Glucose level:", min_value=0.0)
Pregnancies = st.number_input("Enter number of Pregnancies:", min_value=0)
BloodPressure = st.number_input("Enter Blood Pressure level:", min_value=0.0)
SkinThickness = st.number_input("Enter Skin Thickness:", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Enter Diabetes Pedigree Function:", min_value=0.0)
Age = st.number_input("Enter Age:", min_value=0)

# Prepare data for prediction
input_data = {
    "BMI": BMI,
    "Insulin": Insulin,
    "Glucose": Glucose,
    "Pregnancies": Pregnancies,
    "BloodPressure": BloodPressure,
    "SkinThickness": SkinThickness,
    "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    "Age": Age
}

# Make prediction on button click
if st.button("Predict"):
    result = get_prediction(input_data)
    st.write(f"Prediction: {result['prediction']}")
