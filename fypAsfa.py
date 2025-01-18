import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('logistic_regression_model.joblib')

# App Title
st.title('Loan Eligibility Prediction :bank:')

# Input Fields
Gender = st.selectbox('Gender', ('Male', 'Female'))
Married = st.selectbox('Married', ('No', 'Yes'))
Dependents = st.selectbox('Number Of Dependents', ('0', '1', '2', '3+'))
Education = st.selectbox('Education Status', ('Graduate', 'Not Graduate'))
Self_Employed = st.selectbox('Self Employed', ('No', 'Yes'))
ApplicantIncome = st.number_input('Applicant Income', min_value=0, step=1000)
CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0, step=1000)
LoanAmount = st.number_input('Loan Amount', min_value=0, step=1000)
Loan_Amount_Term = st.selectbox(
    'Loan Amount Term', 
    ['12 Months', '36 Months', '60 Months', '84 Months', '120 Months', 
     '180 Months', '240 Months', '300 Months', '360 Months', '480 Months']
)
Credit_History = st.selectbox('Credit History (1=Good, 0=Bad)', [1, 0])
Property_Area = st.selectbox('Property Area', ('Urban', 'Rural', 'Semiurban'))

# Feature Mapping
def preprocess_inputs():
    term_mapping = {
        '12 Months': 12, '36 Months': 36, '60 Months': 60, '84 Months': 84,
        '120 Months': 120, '180 Months': 180, '240 Months': 240,
        '300 Months': 300, '360 Months': 360, '480 Months': 480
    }
    data = {
        'Gender': [1 if Gender == 'Male' else 0],
        'Married': [1 if Married == 'Yes' else 0],
        'Dependents': [Dependents],
        'Education': [1 if Education == 'Graduate' else 0],
        'Self_Employed': [1 if Self_Employed == 'Yes' else 0],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [term_mapping[Loan_Amount_Term]],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area]
    }
    return pd.DataFrame(data)

# Perform one-hot encoding for categorical features
df = pd.get_dummies(df, columns = ['Property_Area'], frop_first = False)

# Ensure all expected columns are present
expected_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                        'Credit_History', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']
for col in expected_columns:
    if col not in df.columns:
        df[col] = 0 # Add missing columns with default value

# Ensure columns are in the correct order
return df[expected_columns]

# Prediction
def predict():
    input_data = preprocess_inputs()
    
    # Ensure input data matches model training format
    input_data = pd.get_dummies(input_data)
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success('You Can Get The Loan :thumbsup:')
    else:
        st.error('Sorry, You Cannot Get The Loan :thumbsdown:')

# Button Styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)

# Predict Button
st.button('Predict', on_click=predict)
