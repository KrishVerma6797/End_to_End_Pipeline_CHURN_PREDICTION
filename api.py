from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict")
def predict(tenure: int,monthly_charges: float,total_charges: float,contract: str,payment_method: str,online_security: str,tech_support: str): 
    sample = pd.DataFrame({
        "TotalCharges": [total_charges],
        "MonthlyCharges": [monthly_charges],
        "tenure": [tenure],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "OnlineSecurity": [online_security],
        "TechSupport": [tech_support]
    })

    for col in ["Contract","PaymentMethod","OnlineSecurity","TechSupport"]:
        sample[col] = encoders[col].transform(sample[col]         )

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)[0][1]

    return {
        'Prediction' : "Churn" if prediction[0]==1 else "No Churn",
        'Churn Probability' : round(float(probability*1000),2),
        "Rish Score" : "High" if probability >=0.75 else "Medium" if probability >=0.5 else "Low"
    }
