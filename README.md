# Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![FastAPI](https://img.shields.io/badge/FastAPI-API-success)

## Overview

This project predicts whether a telecom customer is likely to churn (leave the service) using Machine Learning and XGBoost.

## Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Selection
- Feature Scaling
- Logistic Regression
- XGBoost
- Hyperparameter Tuning
- FastAPI Deployment
- Model Serialization

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

- Records: 7,043
- Features: 20
- Target Variable: Churn

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- FastAPI
- Joblib

---

## Machine Learning Pipeline

1. Data Loading
2. Data Cleaning
3. Missing Value Handling
4. Encoding Categorical Features
5. Feature Scaling
6. Feature Selection
7. Train-Test Split
8. Model Training
9. Hyperparameter Tuning
10. Model Evaluation
11. Model Saving
12. API Deployment

---

## Model Performance

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 78.42% |
| XGBoost | 75.23% |
| Tuned XGBoost | 75.16% |

### Churn Detection (Best Model)

- Accuracy: 75.15
- Recall: 78%
- Business Goal: Maximize churn detection to identify customers likely to leave.


---

## Important Features

According to XGBoost:

1. Contract
2. Online Security
3. Tech Support
4. Tenure
5. Monthly Charges
6. Payment Method
7. Total Charges

---

## API Usage

Start FastAPI server:

```bash
uvicorn api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```
Note: If deployed on a cloud platform, replace `127.0.0.1:8000` with the deployment URL.
---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── main.py
├── api.py
├── churn.csv
├── model.pkl
├── scaler.pkl
├── encoders.pkl
├── feature_importance.csv
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Streamlit Dashboard
- Docker Deployment
- Cloud Deployment (AWS/Azure)
- SHAP Explainability
- Real-Time Monitoring

---

## Author

**Krish Verma**
