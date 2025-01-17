import streamlit as st



st.set_page_config(
    page_title = 'Loan Prediction',

)

eda = st.Page('eda.py', title="Data Visualization", icon="📊")
predict = st.Page('predicts.py', title='Loan Predictions',icon='🥔')

pg = st.navigation(
    {
        "Menu":[eda,predict]
    }
)

pg.run()                 

