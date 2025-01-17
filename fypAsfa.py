import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess the dataset
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

    X, y = SMOTE().fit_resample(X, y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=0)

# Train the logistic regression model
def train_model(X_train, y_train):
    model = LogisticRegression(solver='saga', max_iter=500, random_state=1)
    model.fit(X_train, y_train)
    return model

# Streamlit app
def main():
    st.title("Loan Eligibility Prediction")
    st.write("This application predicts loan eligibility based on applicant details.")

    # Load data and train model
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    # Sidebar for user input
    st.sidebar.header("Applicant Details")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Married", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=1000)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, step=1000)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)
    loan_amount_term = st.sidebar.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
    credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
    property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

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

    # Drop unnecessary columns for prediction
    user_df = user_df.drop(columns=["Property_Area_Rural", "Property_Area_Semiurban", "Property_Area_Urban"])

    # Prediction
    if st.sidebar.button("Predict Loan Eligibility"):
        prediction = model.predict(user_df)
        if prediction[0] == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

if __name__ == "__main__":
    main()
    
