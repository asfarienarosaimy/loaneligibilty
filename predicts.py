import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
try:
    pipeline = joblib.load('rf_model_pipeline.pkl')
    preprocessor = pipeline.named_steps['preprocessor']
    one_hor_encoder = preprocessor.named_transformer_['cat']
    categories = one_hot_encoder.categories_
except Exception as e :
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit app
st.title("Loan Eligibility Prediction using Machine Learning")
st.caption('Fill the form below to start a prediction')

# Input field for categorical feature
Gender = st.selectbox("Select Gender", categories[0])
Married = st.selectbox("Select Married", categories[1])
Dependents = st.selectbox("Select Dependents", categories[2])
LoanAmount = st.selectbox("Select LoanAmount", categories[3])
Education = st.selectbox("Select Education", categories[4])
ApplicantIncome = st.selectbox("Select ApplicantIncome", categories[5])
Self_Employed = st.selectbox("Select Self_Employed", categories[6])
Credit_History = st.selectbox("Select Credit_History", categories[7])

# Input field for numerical feature
LoanAmount = st.number_input("Enter Loan Amount", min_value=0, step=1)
ApplicantIncome = st.number_input("Enter Applicant Income", min_value=0, step=1)

# Create raw input DataFrame
input_data = pd.DataFrame({
    'Gender': [Gender],
    'Married': [Married],
    'Dependents': [Dependents],
    'LoanAmount': [LoanAmount],
    'Education': [Education],
    'ApplicantIncome': [ApplicantIncome],
    'Self_Employed': [Self_Employed],
    'Credit_History': [Credit_History]
})

if st.button("predict"):
    try:
        prediction = pipeline.predict(input_data)[0]
        st.success("Predicted {prediction}")
    except Exception as e:
        st.error("error: {e}")