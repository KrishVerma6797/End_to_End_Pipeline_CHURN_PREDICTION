# Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![FastAPI](https://img.shields.io/badge/FastAPI-API-success)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Run-blue)

## Overview

This project predicts whether a telecom customer is likely to churn (leave the service) using Machine Learning and XGBoost. The solution includes data preprocessing, model training, hyperparameter tuning, FastAPI deployment, and cloud deployment on Google Cloud Run.

---

## Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Selection
- Feature Scaling
- Logistic Regression
- XGBoost Classification
- Hyperparameter Tuning
- Model Serialization (Joblib)
- FastAPI REST API
- Google Cloud Run Deployment

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
- Google Cloud Run

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
11. Model Serialization
12. API Development
13. Cloud Deployment

---

## Model Performance

| Model | Accuracy | ROC-AUC |
|---------|---------|---------|
| Logistic Regression | 78.42% | 83.24% |
| XGBoost | 75.23% | 81.74% |
| Tuned XGBoost | 75.16% | 84.14% |

### Best Model Performance (Tuned XGBoost)

- Accuracy: **75.16%**
- ROC-AUC Score: **84.14%**
- Recall (Churn Class): **77.8%**
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

## Live Deployment

### API Endpoint

```text
https://churn-api-1016947794076.us-central1.run.app
```

### Swagger Documentation

```text
https://churn-api-1016947794076.us-central1.run.app/docs
```

The model is deployed using FastAPI and Google Cloud Run for real-time churn prediction.

---

## API Usage

Run locally:

```bash
uvicorn api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

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
├── Dockerfile
└── README.md
```

---

## Future Improvements

- Streamlit Dashboard
- SHAP Explainability
- Real-Time Monitoring
- CI/CD Pipeline
- Kubernetes Deployment

---

## Author

**Krish Verma**
