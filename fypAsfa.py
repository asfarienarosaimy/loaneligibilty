import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess the dataset (if needed for metrics or additional functionality)
def load_data():
    df = pd.read_csv("loan_data_set.csv")
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

    df = pd.get_dummies(df)
    df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate',
                  'Self_Employed_No', 'Loan_Status_N'], axis=1)

    df.rename(columns={
        'Gender_Male': 'Gender', 'Married_Yes': 'Married',
        'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
        'Loan_Status_Y': 'Loan_Status'
    }, inplace=True)

    X = df.drop(["Loan_Status"], axis=1)
    y = df["Loan_Status"]

    return X, y

# Load the pre-trained model from a pickle file
loaded_model=pickle.load(open('logistic_regression_model.pkl','rb'))

# Streamlit app
def main():
    st.title("Loan Eligibility Prediction")
    st.write("This application predicts loan eligibility based on applicant details.")

    # Load the pre-trained model
    model = load_model()

    # User input form
    st.header("Enter Applicant Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0, step=1000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=1000)
    loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
    loan_amount_term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Convert user input into dataframe
    user_data = {
        "Gender": [1 if gender == "Male" else 0],
        "Married": [1 if married == "Yes" else 0],
        "Dependents_0": [1 if dependents == "0" else 0],
        "Dependents_1": [1 if dependents == "1" else 0],
        "Dependents_2": [1 if dependents == "2" else 0],
        "Dependents_3+": [1 if dependents == "3+" else 0],
        "Education": [1 if education == "Graduate" else 0],
        "Self_Employed": [1 if self_employed == "Yes" else 0],
        "ApplicantIncome": [np.sqrt(applicant_income)],
        "CoapplicantIncome": [np.sqrt(coapplicant_income)],
        "LoanAmount": [np.sqrt(loan_amount)],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area_Rural": [1 if property_area == "Rural" else 0],
        "Property_Area_Semiurban": [1 if property_area == "Semiurban" else 0],
        "Property_Area_Urban": [1 if property_area == "Urban" else 0]
    }

    user_df = pd.DataFrame.from_dict(user_data)

    # Match columns with training data
    expected_columns = [
        "Gender", "Married", "Dependents_0", "Dependents_1", "Dependents_2", "Dependents_3+",
        "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area_Rural", "Property_Area_Semiurban",
        "Property_Area_Urban"
    ]
    for col in expected_columns:
        if col not in user_df.columns:
            user_df[col] = 0  # Add missing columns with default value 0
    user_df = user_df[expected_columns]  # Ensure correct column order
    
    # Drop unnecessary columns for prediction
    user_df = user_df.drop(columns=["Property_Area_Rural", "Property_Area_Semiurban", "Property_Area_Urban"])

    # Prediction
    if st.button("Predict Loan Eligibility"):
        prediction = model.predict(user_df)
        if prediction[0] == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

if __name__ == "__main__":
    main()
