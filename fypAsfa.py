import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# Title of the app
st.title("Loan Eligibility Prediction")

# Sidebar for user inputs
st.sidebar.header("Input Features")

def user_input_features():
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Married = st.sidebar.selectbox("Married", ["Yes", "No"])
    Dependents = st.sidebar.slider("Dependents", 0, 3)
    Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.sidebar.number_input("Applicant Income", 0, 100000)
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", 0, 50000)
    LoanAmount = st.sidebar.number_input("Loan Amount", 0, 500)
    Loan_Amount_Term = st.sidebar.slider("Loan Amount Term (in months)", 12, 480, 360, step=12)
    Credit_History = st.sidebar.selectbox("Credit History", [0, 1])
    Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    data = {
        'Gender': [1 if Gender == "Male" else 0],
        'Married': [1 if Married == "Yes" else 0],
        'Dependents': [Dependents],
        'Education': [1 if Education == "Graduate" else 0],
        'Self_Employed': [1 if Self_Employed == "Yes" else 0],
        'ApplicantIncome': [np.sqrt(ApplicantIncome)],
        'CoapplicantIncome': [np.sqrt(CoapplicantIncome)],
        'LoanAmount': [np.sqrt(LoanAmount)],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [1 if Property_Area == "Urban" else 2 if Property_Area == "Semiurban" else 0]
    }
    return pd.DataFrame(data)

input_df = user_input_features()

# Load model and scaler
model = pickle.load(open("logistic_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Prediction
scaled_data = scaler.transform(input_df)
prediction = model.predict(scaled_data)
result = "Eligible" if prediction[0] == 1 else "Not Eligible"

st.subheader("Prediction Result")
st.write(result)
