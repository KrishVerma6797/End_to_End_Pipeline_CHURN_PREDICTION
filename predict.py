# import streamlit as st
# import pandas as pd
# import joblib

# # Load model files
# model = joblib.load("model.pkl")
# encoders = joblib.load("encoders.pkl")
# scaler = joblib.load("scaler.pkl")

# st.set_page_config(
#     page_title="Customer Churn Prediction",
#     page_icon="📊"
# )

# st.title("📊 AI Customer Churn Prediction System")

# st.write(
#     "Predict whether a customer is likely to churn."
# )

# # Inputs

# tenure = st.number_input(
#     "Tenure (Months)",
#     min_value=0,
#     max_value=100,
#     value=12
# )

# monthly_charges = st.number_input(
#     "Monthly Charges",
#     min_value=0.0,
#     value=50.0
# )

# total_charges = st.number_input(
#     "Total Charges",
#     min_value=0.0,
#     value=500.0
# )

# contract = st.selectbox(
#     "Contract",
#     [
#         "Month-to-month",
#         "One year",
#         "Two year"
#     ]
# )

# payment_method = st.selectbox(
#     "Payment Method",
#     [
#         "Electronic check",
#         "Mailed check",
#         "Bank transfer (automatic)",
#         "Credit card (automatic)"
#     ]
# )

# online_security = st.selectbox(
#     "Online Security",
#     [
#         "Yes",
#         "No",
#         "No internet service"
#     ]
# )

# tech_support = st.selectbox(
#     "Tech Support",
#     [
#         "Yes",
#         "No",
#         "No internet service"
#     ]
# )

# # Prediction Button

# if st.button("Predict Churn"):

#     sample = pd.DataFrame({
#         "TotalCharges": [total_charges],
#         "MonthlyCharges": [monthly_charges],
#         "tenure": [tenure],
#         "Contract": [contract],
#         "PaymentMethod": [payment_method],
#         "OnlineSecurity": [online_security],
#         "TechSupport": [tech_support]
#     })

#     # Encode categorical columns

#     categorical_cols = [
#         "Contract",
#         "PaymentMethod",
#         "OnlineSecurity",
#         "TechSupport"
#     ]

#     for col in categorical_cols:
#         sample[col] = encoders[col].transform(
#             sample[col]
#         )

#     # Scale

#     sample_scaled = scaler.transform(sample)

#     # Prediction

#     prediction = model.predict(sample_scaled)

#     probability = model.predict_proba(
#         sample_scaled
#     )[0][1]

#     st.subheader("Prediction Result")

#     st.write(
#         f"Churn Probability: {probability*100:.2f}%"
#     )

#     if probability >= 0.75:

#         st.error(
#             "🔴 High Risk Customer"
#         )

#         st.warning(
#             "Recommended Action: Offer retention discount or contact customer."
#         )

#     elif probability >= 0.50:

#         st.warning(
#             "🟡 Medium Risk Customer"
#         )

#         st.info(
#             "Recommended Action: Monitor customer activity."
#         )

#     else:

#         st.success(
#             "🟢 Low Risk Customer"
#         )

#         st.info(
#             "Customer likely to stay."
#         )