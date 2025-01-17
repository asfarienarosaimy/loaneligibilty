import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load('rf_model_pipeline.pkl')

# Extract available options for categorical features
preprocessor = pipeline.named_steps['preprocessor']
one_hot_encoder = preprocessor.named_transformers_['cat']
categories = one_hot_encoder.categories_

# Streamlit app
st.title("Loan Status Prediction")
st.caption('Fill the form below to start a prediction.')

