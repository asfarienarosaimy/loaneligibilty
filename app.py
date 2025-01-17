import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

_version_ = "0.4.8.3"
app_name = "BankingBuddy"

# Streamlit page configuration
st.set_page_config(
    page_title="Potatoes Prediction",
    page_icon="🥔",
    menu_items={
        'About': "# Made By Izat Hakimi"
    }
)

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
    st.title("🥔 Price Predictions System")

def login():
        with st.form("name_form"):
            name = st.text_input("Please Enter Your Name to Continue:")
            submitted = st.form_submit_button("Submit")

            # After submitting the form, store the submission state
            if submitted and name:
                st.session_state.form_submitted = True
                st.session_state.name = name
                st.rerun()
                st.session_state.logged_in = True
        


def logout():
    if st.button("Log out"):
        # Reset session states related to authentication
        st.session_state.form_submitted = False
        st.session_state.logged_in = False
        st.session_state.name = ""
        # Trigger rerun to return to the login page
        st.rerun()



login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
eda = st.Page('eda.py', title="Data Visualization", icon="📊")
prediction = st.Page('predicts.py', title='Potato Price Predictions', icon = '🥔')
home = st.Page('home.py', title='Homepage', default=True, icon=":material/home:")


if st.session_state.form_submitted:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Menu": [home,eda,prediction]
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()