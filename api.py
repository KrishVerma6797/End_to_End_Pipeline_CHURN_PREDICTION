from fastapi import FastAPI
import pandas as pd
import joblib
from enum import Enum


app = FastAPI()

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")


class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"
class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"
class OnlineSecurity(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"
class TechSupport(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict")
def predict(tenure: int,monthly_charges: float,total_charges: float,contract: ContractType,payment_method: PaymentMethod,online_security: OnlineSecurity,tech_support: TechSupport):    
    sample = pd.DataFrame({
        "TotalCharges": [total_charges],
        "MonthlyCharges": [monthly_charges],
        "tenure": [tenure],
        "Contract": [contract.value],
        "PaymentMethod": [payment_method.value],
        "OnlineSecurity": [online_security.value],
        "TechSupport": [tech_support.value]
    })

    for col in ["Contract","PaymentMethod","OnlineSecurity","TechSupport"]:
        sample[col] = encoders[col].transform(sample[col]         )

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)[0][1]

    return {
        'Prediction' : "Churn" if prediction[0]==1 else "No Churn",
        'Churn Probability' : round(float(probability*100),2),
        "Risk Score" : "High" if probability >=0.75 else "Medium" if probability >=0.5 else "Low"
    }
