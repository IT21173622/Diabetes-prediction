from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

app = FastAPI()

# Load the trained model from disk
model = pickle.load(open("models/diabetes.pkl", 'rb'))

# Define the input data model using Pydantic
class DiabetesInput(BaseModel):
    BMI: float
    Insulin: float
    Glucose: float
    Pregnancies: int
    BloodPressure: float
    SkinThickness: float
    DiabetesPedigreeFunction: float
    Age: int

    class Config:
        orm_mode = True

# Create a function to handle pre-processing similar to the pre-processing done above
def preprocess_input(input_data):
    # Creating BMI_Categorie
    if input_data.BMI < 18.5:
        BMI_Categorie = "Underweight"
    elif 18.5 <= input_data.BMI <= 24.9:
        BMI_Categorie = "Normalweight"
    elif 25 <= input_data.BMI <= 29.9:
        BMI_Categorie = "Overweight"
    else:
        BMI_Categorie = "Obesity"

    # Insulin Score
    insulin_score = "Normal" if 16 <= input_data.Insulin <= 166 else "Abnormal"
    
    # NewGlucose category
    if input_data.Glucose <= 70:
        glucose_category = "Low"
    elif 70 < input_data.Glucose <= 99:
        glucose_category = "Normal"
    elif 99 < input_data.Glucose <= 126:
        glucose_category = "Overweight"
    else:
        glucose_category = "Secret"

    # Creating the input DataFrame
    input_dict = {
        "BMI": [input_data.BMI],
        "Insulin": [input_data.Insulin],
        "Glucose": [input_data.Glucose],
        "Pregnancies": [input_data.Pregnancies],
        "BloodPressure": [input_data.BloodPressure],
        "SkinThickness": [input_data.SkinThickness],
        "DiabetesPedigreeFunction": [input_data.DiabetesPedigreeFunction],
        "Age": [input_data.Age],
        "BMI_Categorie_Obesity": [1 if BMI_Categorie == "Obesity" else 0],
        "BMI_Categorie_Overweight": [1 if BMI_Categorie == "Overweight" else 0],
        "BMI_Categorie_Underweight": [1 if BMI_Categorie == "Underweight" else 0],
        "NewInsulinScore_Normal": [1 if insulin_score == "Normal" else 0],
        "NewGlucose_Low": [1 if glucose_category == "Low" else 0],
        "NewGlucose_Normal": [1 if glucose_category == "Normal" else 0],
        "NewGlucose_Overweight": [1 if glucose_category == "Overweight" else 0],
        "NewGlucose_Secret": [1 if glucose_category == "Secret" else 0]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Apply RobustScaler
    transformer = RobustScaler()
    input_df_scaled = transformer.fit_transform(input_df)
    
    # Return the processed data
    return input_df_scaled

# Endpoint for prediction
@app.post("/predict/")
def predict_diabetes(input_data: DiabetesInput):
    # Preprocess the input
    processed_input = preprocess_input(input_data)
    
    # Make a prediction
    prediction = model.predict(processed_input)
    
    # Return the result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return {"prediction": result}

