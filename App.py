# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:33:59 2023

@author: vksch
"""

import streamlit as st
import streamlit.components.v1 as components
from streamlit_shap import st_shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import joblib
import pickle
import io
import shap
import altair as alt

# Define functions for each tab's content
def tab1_content():
    # Sidebar content for Tab 1
    #st.sidebar.write("Enter Your Parameter")
    # Main content for Tab 1
    st.write("""

# Loan Application Prediction Model

Model predicts the probability of a customer loan application Approval using Bank past Loan application data

    """)
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.write("OR Input Your Features below")
    if uploaded_file is not None:

        input_df1 = pd.read_csv(uploaded_file)
        input_df = input_df1[['applicant_credit_score_type','debt_to_income_ratio','loan_amount','property_value','income','loan_purpose', 'loan_term']]
        input_df['loan_purpose'] = input_df['loan_purpose'].astype('O')
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
    with open("./approval_pipeline_tuned.pkl", 'rb') as pfile:  
                load_clf=pickle.load(pfile)
    NUMERICAL_VARIABLES = ['loan_amount', 'income','loan_term','property_value','applicant_credit_score_type']
    CATEGORICAL_VARIABLES = ['debt_to_income_ratio', 'loan_purpose']
    prediction = load_clf.predict(input_df)
    # st.write(input_df)

    prediction_proba = load_clf.predict_proba(input_df)

    Approval_labels = np.array(['Rejected','Approved'])
    st.subheader('Model Prediction')
    st.write(Approval_labels[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

def tab2_content():
    # Sidebar content for Tab 2
    #st.sidebar.write("This is the sidebar for Tab 2")
    # Main content for Tab 2
    st.write("Show Me Model Behaviour")
    # load pickle file
    # write as function so we can cache it
    @st.cache_data
    def load_model(pkl):
        return pickle.load(open(pkl, "rb"))
    model = load_model("./approval_pipeline_tuned.pkl")
    # Extract the final estimator from the pipeline
    final_estimator = model.named_steps['RF_tuned']
   # Apply the scaler to X_test
    encoder = model.named_steps['categorical_encoder']
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df1 = pd.read_csv(uploaded_file)
        X1 = input_df1[['applicant_credit_score_type','debt_to_income_ratio','loan_amount','property_value','income','loan_purpose', 'loan_term']]
        X1['loan_purpose'] = X1['loan_purpose'].astype('O')
        #take smaller sample to run the shap value analysis
        X2 = shap.sample(X1, 100)
# Reset index of the subset DataFrame
        X = X2.reset_index(drop=True) 
    else:
        st.stop()
        #X = pd.DataFrame()
        # input_df1 = pd.read_csv("C:/HMDA/Input/df1.csv")
        # X1 = input_df1[['applicant_credit_score_type','debt_to_income_ratio','loan_amount','property_value','income','loan_purpose', 'loan_term']]
        # X1['loan_purpose'] = X1['loan_purpose'].astype('O')
        # #take smaller sample to run the shap value analysis
        # X2 = shap.sample(X1, 100)
        # X = X2.reset_index(drop=True)
        # nobs = 20  # set a default value here
    with st.expander("Model Features and Data summary"):
        if not X.empty:
            nobs = st.slider(
                "Select number of observations to visually inspect", 1, X.shape[0], value=20
            )
        else:
            st.write("Upload a Data File")
        # Display data
        st.dataframe(X.head(nobs))
    
        # Conditionally calculate summary statistics
        if st.checkbox("Display summary statistics for visible sample?"):
            f"""Sample statistics based on {nobs} observations:"""
            st.dataframe(X.head(nobs).describe())
        X_encoder = encoder.transform(X)
    #with st.echo():
    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(X_encoder)   
    "### Feature importance - All"
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], X_encoder), height=350)
    st_shap(shap.summary_plot(shap_values, X_encoder))
    #st_shap(shap.force_plot(explainer.expected_value, shap_values, X_encoder), height=350)
    feature = st.selectbox("Choose Application Number", X.index.values)
    f"### Application No. {feature}: Explanation For Loan Approval"
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][feature], X_encoder.iloc[feature]), height=350)

def tab3_content():
    # Sidebar content for Tab 3
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df1 = pd.read_csv(uploaded_file)
        X1 = input_df1[['derived_msa-md','derived_race','derived_sex','applicant_age_above_62','loan_amount','interest_rate','income','action_taken']]
        action_to_filter = [1,3]
        X1 = X1[X1['action_taken'].isin(action_to_filter)]
        X1['Derived_action'] = X1['action_taken'].apply(lambda x:'Approved' if x==1 else 'Denied')
        X1.drop(columns=['action_taken'], inplace=True)
        X1['derived_sex'] = X1['derived_sex'].replace("Joint","Male")
        X1['derived_race'] = X1['derived_race'].replace("Joint","White")
        #take smaller sample to run the shap value analysis 
    else:
        st.stop()
    # Main content for Tab 3
    st.write(X1.head(3))
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    columns = ['derived_sex', 'derived_race', 'applicant_age_above_62']
    titles = ['Gender Distribution', 'Race Distribution', 'Age Above 62']
    for i, col in enumerate(columns):
        count_df = X1.groupby(col).size().reset_index(name='Count')
        #count_df = count_df.rename(columns={'derived_sex': 'Count'})
        total_count = count_df['Count'].sum()
        count_df['Percentage'] = count_df['Count'] / total_count * 100
        axs[i].pie(count_df['Percentage'], labels=count_df[col], autopct='%1.1f%%', shadow=False, startangle=0)
        axs[i].set_aspect('equal')
        axs[i].set_title(titles[i], loc='center', pad=20) 
    # Sidebar option to show/hide EDA
    show_eda = st.sidebar.checkbox("Show EDA")
    if show_eda:
        st.header("Exploratory Data Analysis Pie Chart")
        st.pyplot(fig)
    else:
    # You can add other content to the main section of the app
        st.write("Select 'Show EDA' to view the pie chart.")
# Main app structure
st.title("Fair Lending Analysis App")

# Create tabs
tabs = ["ML Model", "Model Explability", "Fair Lending Diaganosis"]
selected_tab = st.radio("Select a tab:", tabs, format_func=lambda x: x)

# Display content based on selected tab
if selected_tab == "ML Model":
    tab1_content()
elif selected_tab == "Model Explability":
    tab2_content()
else:
    tab3_content()

