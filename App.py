# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:33:59 2023

@author: vksch
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.write("""

# Fair Lending Prediction App

This app predicts the probability of a customer loan application Approval using Bank past Loan approval data.

""")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.write("OR Input Your Features below")
if uploaded_file is not None:

    input_df1 = pd.read_csv(uploaded_file)
    input_df = input_df1[['loan_amount','loan_purpose','loan_term','property_value','income','debt_to_income_ratio','applicant_credit_score_type']]

else:

    def user_input():

        loan_amount = st.sidebar.slider('Loan Amount', 5000.0,500000.0, 1000.0)

        loan_purpose = st.sidebar.selectbox('Loan Purpose',('1','2','4','31','32'))

        loan_term = st.sidebar.slider('Loan Term', 12,360, 12)
        property_value = st.sidebar.slider('Property value', 5000.0,50000000.0, 1000.0)
        income = st.sidebar.slider('Income', 0.0,20000.0, 100.0)
        debt_to_income_ratio = st.sidebar.selectbox('DTI',('<20%','20%-<30%','30%-<36%','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50%-60%','>60%'))
        applicant_credit_score_type = st.sidebar.selectbox('Credit Score',('1', '2','3','8','9'))
        
        data = {'loan_amount':[loan_amount],
                'loan_purpose':[loan_purpose],
                'loan_term':[loan_term],
                'property_value':[property_value],
                'income':[income],
                'debt_to_income_ratio':[debt_to_income_ratio],
                'applicant_credit_score_type':[applicant_credit_score_type],
                }
        features = pd.DataFrame(data)

        return features

    input_df = user_input()

st.subheader('User Input features')
st.write(input_df)

load_clf = joblib.load(open('C:\\HMDA\\Model\\approval_pipeline.joblib', 'rb'))

prediction = load_clf.predict(input_df)

prediction_proba = load_clf.predict_proba(input_df)

Approval_labels = np.array(['Rejected','Approved'])
st.subheader('Model Prediction')
st.write(Approval_labels[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
