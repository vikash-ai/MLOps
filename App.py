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
import seaborn as sns
import requests
import joblib
import pickle
import io
import shap
# import altair as alt

# Define a custom SessionState class
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Create a session state to store tab-specific variables
session_state = SessionState(
    uploaded_file_tab1=None,
    uploaded_file_tab2=None,
    uploaded_file_tab3=None
)
# Define functions for each tab's content
def tab1_content():
    # session_state.uploaded_file_tab2 = None
    # session_state.uploaded_file_tab3 = None
    st.write("""

# Model Inference by Loan Id

Model predicts the probability of a customer loan application Approval using historical application data
Start by entering the loan attributes in the left side panel:

    """)
    #uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.write("Input your loan application details")
    # if uploaded_file is not None:
    #     session_state.uploaded_file_tab1 = uploaded_file
    #     input_df1 = pd.read_csv(uploaded_file)
    #     input_df = input_df1[['applicant_credit_score_type','debt_to_income_ratio','loan_amount','property_value','income','loan_purpose', 'loan_term']]
    #     input_df['loan_purpose'] = input_df['loan_purpose'].astype('O')
    # else:
    input_df = pd.DataFrame()
    # def user_input():

    loan_amount = st.sidebar.slider('Loan Amount', 5000.0,500000.0, 1000.0)

    loan_purpose = st.sidebar.selectbox('Loan Purpose',('1','2','4','31','32'))

    loan_term = st.sidebar.slider('Loan Term', 12,360, 12)
    property_value = st.sidebar.slider('Property value', 5000.0,50000000.0, 1000.0)
    income = st.sidebar.slider('Income', 0.0,20000.0, 100.0)
    debt_to_income_ratio = st.sidebar.selectbox('DTI',('<20%','20%-<30%','30%-<36%','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50%-60%','>60%'))
    applicant_credit_score_type = st.sidebar.selectbox('Credit Score',('1', '2','3','8','9'))
    
    data = {'Application No': 1001,
            'loan_amount':[loan_amount],
            'loan_purpose':[loan_purpose],
            'loan_term':[loan_term],
            'property_value':[property_value],
            'income':[income],
            'debt_to_income_ratio':[debt_to_income_ratio],
            'applicant_credit_score_type':[applicant_credit_score_type],
            }
    features = pd.DataFrame(data)

    # return features
    # input_df = pd.DataFrame()
    input_df = features

    st.subheader('User Input features')
    st.dataframe(input_df, hide_index=True)
    #st.write(input_df.reset_index(drop=True))
    with open("./approval_pipeline_tuned.pkl", 'rb') as pfile:  
                load_clf=pickle.load(pfile)
    NUMERICAL_VARIABLES = ['loan_amount', 'income','loan_term','property_value','applicant_credit_score_type']
    CATEGORICAL_VARIABLES = ['debt_to_income_ratio', 'loan_purpose']
    input_df = input_df.drop('Application No', axis=1)
    prediction = load_clf.predict(input_df)
    # st.write(input_df)

    prediction_proba = load_clf.predict_proba(input_df)

    Approval_labels = np.array(['Rejected','Approved'])
    st.subheader('Model Prediction')
    st.write(Approval_labels[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
    
    final_estimator = load_clf.named_steps['RF_tuned']
   # Apply the scaler to X_test
    encoder = load_clf.named_steps['categorical_encoder']
    X_encoder = encoder.transform(input_df)
    explainer = shap.Explainer(final_estimator)
    shap_values = explainer.shap_values(X_encoder)
    st.title("Explanation for Model Prediction")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], X_encoder), height=350)
    
def tab2_content():
    session_state.uploaded_file_tab1 = None
    session_state.uploaded_file_tab3 = None
    st.write("Show Me Model Behaviour")
    # load pickle file
    # write as function so we can cache it
    #@st.cache_data
    def load_model(pkl):
        return pickle.load(open(pkl, "rb"))
    model = load_model("./approval_pipeline_tuned.pkl")
    # Extract the final estimator from the pipeline
    final_estimator = model.named_steps['RF_tuned']
   # Apply the scaler to X_test
    encoder = model.named_steps['categorical_encoder']
    uploaded_file1 = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file1 is not None:
        session_state.uploaded_file_tab2 = uploaded_file1
        session_state.uploaded_file_tab3 = None
        input_df1 = pd.read_csv(uploaded_file1)
        X1 = input_df1[['applicant_credit_score_type','debt_to_income_ratio','loan_amount','property_value','income','loan_purpose', 'loan_term']]
        X1['loan_purpose'] = X1['loan_purpose'].astype('O')
        #take smaller sample to run the shap value analysis
        X2 = shap.sample(X1, 500)
# Reset index of the subset DataFrame
        X = X2.reset_index(drop=True) 
    else:
        st.stop()
        
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
            f"""Sample statistics based on uploaded Data:"""
            #f"""Sample statistics based on {nobs} observations:"""
            st.dataframe(X1.describe())
            #st.dataframe(X.head(nobs).describe())
        X_encoder = encoder.transform(X)
    #with st.echo():
    explainer = shap.Explainer(final_estimator,X_encoder)
    shap_values = explainer(X_encoder)   
    "### Feature importance - Batch"
    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(X_encoder)
    #st_shap(shap.summary_plot(shap_values, X_encoder.iloc[0:100,:]))
    st_shap(shap.summary_plot(shap_values[10], X_encoder.iloc[0:10,:]))
    "### Feature importance - Application Number"
    feature = st.selectbox("Choose Application Number", X.index.values)
    f"###### Application No. {feature} selected "
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][feature], X_encoder.iloc[feature]), height=350)

def tab3_content():
    # Sidebar content for Tab 3
    session_state.uploaded_file_tab1 = None
    session_state.uploaded_file_tab2 = None
    uploaded_file2 = st.file_uploader("Upload your input CSV file", type=["csv"], key="uploaded_file2")
    if uploaded_file2 is not None:
        session_state.uploaded_file_tab3 = uploaded_file2
        session_state.uploaded_file_tab2 = None
        #st.session_state["uploaded_file2"]
        input_df1 = pd.read_csv(st.session_state['uploaded_file2'])
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
    
    st.dataframe(X1.sample(5), hide_index=True)
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    X11 = X1[(X1.derived_sex=='Male')|(X1.derived_sex=='Female') ]
    count_df = X11.groupby('derived_sex').size().reset_index(name='Count')
    total_count = count_df['Count'].sum()
    count_df['Percentage'] = count_df['Count'] / total_count * 100
    axs[0].pie(count_df['Percentage'], labels=count_df['derived_sex'], autopct='%1.1f%%', shadow=False, startangle=0)
    axs[0].set_aspect('equal')
    axs[0].set_title('Gender Distribution', loc='center', pad=20)
    
    X11 = X1[(X1.derived_race=='White')|(X1.derived_race=='Black or African American')|(X1.derived_race=='Asian')
             |(X1.derived_race=='American Indian or Alaska Native')|(X1.derived_race=='Native Hawaiian or Other Pacific Islander')]
    count_df = X11.groupby('derived_race').size().reset_index(name='Count')
    total_count = count_df['Count'].sum()
    count_df['Percentage'] = count_df['Count'] / total_count * 100
    axs[1].pie(count_df['Percentage'], labels=count_df['derived_race'], autopct='%1.1f%%', shadow=False, startangle=0)
    axs[1].set_aspect('equal')
    axs[1].set_title('Race Distribution', loc='center', pad=20)
    
    X11 = X1[(X1.applicant_age_above_62=='Yes')|(X1.applicant_age_above_62=='No') ]
    count_df = X11.groupby('applicant_age_above_62').size().reset_index(name='Count')
    total_count = count_df['Count'].sum()
    count_df['Percentage'] = count_df['Count'] / total_count * 100
    axs[2].pie(count_df['Percentage'], labels=count_df['applicant_age_above_62'], autopct='%1.1f%%', shadow=False, startangle=0)
    axs[2].set_aspect('equal')
    axs[2].set_title('Age Above 62', loc='center', pad=20)
    # columns = ['derived_sex', 'derived_race', 'applicant_age_above_62']
    # titles = ['Gender Distribution', 'Race Distribution', 'Age Above 62']
    # for i, col in enumerate(columns):
    #     count_df = X1.groupby(col).size().reset_index(name='Count')
    #     #count_df = count_df.rename(columns={'derived_sex': 'Count'})
    #     total_count = count_df['Count'].sum()
    #     count_df['Percentage'] = count_df['Count'] / total_count * 100
    #     axs[i].pie(count_df['Percentage'], labels=count_df[col], autopct='%1.1f%%', shadow=False, startangle=0)
    #     axs[i].set_aspect('equal')
    #     axs[i].set_title(titles[i], loc='center', pad=20) 
    # Sidebar option to show/hide EDA
    show_Loan_Population = st.checkbox("Show Loan Population")
    if show_Loan_Population:
        st.header("Loan Population By Prohibited Basis Factor")
        st.pyplot(fig)
    else:
    # You can add other content to the main section of the app
        st.write("Select Option to view the distribution plots")
    
    #APR distribution  
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    show_APR = st.checkbox("Show APR Distribution")
    if show_APR:
        st.header("Loan Density By Prohibited Basis Factor")
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        df_r = X1[(X1.derived_race=='White')|(X1.derived_race=='Black or African American') ]
        #sns.displot(data=df_r,x='interest_rate', hue='derived_race',kind='kde',fill=True,palette='tab10', ax=ax1)
        sns.kdeplot(data=df_r, x='interest_rate', hue='derived_race', fill=True, palette='tab10', ax=axes[0])
        axes[0].set_title("Loan Density Plot For Black or African American")
        axes[0].set_xlabel("APR")
        axes[0].grid(False)
        DIR_ratio = 0.12  # Replace this with your actual DIR ratio
        bar_labels = ['White', 'Black or African American']  # Labels for the bars
        bar_heights = [1.0, DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue', 'red'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].grid(False)
        st.pyplot(fig)
        
        fig1, axes = plt.subplots(1, 2, figsize=(20, 10))
        df_r = X1[(X1.derived_race=='White')|(X1.derived_race=='Asian') ]
        #sns.displot(data=df_r,x='interest_rate', hue='derived_race',kind='kde',fill=True,palette='tab10')
        sns.kdeplot(data=df_r, x='interest_rate', hue='derived_race', fill=True, palette='tab10', ax=axes[0])
        axes[0].set_title("Loan Density Plot For Asian")
        axes[0].set_xlabel("APR")
        axes[0].grid(False)
        DIR_ratio = 0.13  # Replace this with your actual DIR ratio
        bar_labels = ['White', 'Asian']  # Labels for the bars
        bar_heights = [1.0, DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue', 'red'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].grid(False)
        st.pyplot(fig1)
        
        fig2, axes = plt.subplots(1, 2, figsize=(20, 10))
        df_r = X1[(X1.derived_race=='White')|(X1.derived_race=='American Indian or Alaska Native') ]
        #sns.displot(data=df_r,x='interest_rate', hue='derived_race',kind='kde',fill=True,palette='tab10')
        sns.kdeplot(data=df_r, x='interest_rate', hue='derived_race', fill=True, palette='tab10', ax=axes[0])
        axes[0].set_title("Loan Density Plot For American Indian or Alaska Native")
        axes[0].set_xlabel("APR")
        axes[0].grid(False)
        DIR_ratio = 0.02  # Replace this with your actual DIR ratio
        bar_labels = ['White', 'American Indian or Alaska Native']  # Labels for the bars
        bar_heights = [1.0, DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue', 'red'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].grid(False)
        st.pyplot(fig2)
        
        fig3, axes = plt.subplots(1, 2, figsize=(20, 10))
        df_r = X1[(X1.derived_race=='White')|(X1.derived_race=='Native Hawaiian or Other Pacific Islander') ]
        #sns.displot(data=df_r,x='interest_rate', hue='derived_race',kind='kde',fill=True,palette='tab10')
        sns.kdeplot(data=df_r, x='interest_rate', hue='derived_race', fill=True, palette='tab10', ax=axes[0])
        axes[0].set_title("Loan Density Plot For  Other Pacific Islander")
        axes[0].set_xlabel("APR")
        axes[0].grid(False)
        DIR_ratio = 0.01  # Replace this with your actual DIR ratio
        bar_labels = ['White', 'Native Hawaiian or Other Pacific Islander']  # Labels for the bars
        bar_heights = [1.0, DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue', 'red'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].grid(False)
        st.pyplot(fig3)
        
        fig4, axes = plt.subplots(1, 2, figsize=(20, 10))
        df_r = X1[(X1.derived_sex=='Male')|(X1.derived_sex=='Female') ]
        #sns.displot(data=df_r,x='interest_rate', hue='derived_sex',kind='kde',fill=True,palette='tab10')
        sns.kdeplot(data=df_r, x='interest_rate', hue='derived_sex', fill=True, palette='tab10', ax=axes[0])
        axes[0].set_title("Loan Density Plot For Gender")
        axes[0].set_xlabel("APR")
        axes[0].grid(False)
        DIR_ratio = 0.30  # Replace this with your actual DIR ratio
        bar_labels = ['Male', 'Female']  # Labels for the bars
        bar_heights = [1.0, DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue', 'red'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].grid(False)
        st.pyplot(fig4)
        
        fig5, axes = plt.subplots(1, 2, figsize=(20, 10))
        df_r = X1[(X1.applicant_age_above_62=='Yes')|(X1.applicant_age_above_62=='No') ]
        #sns.displot(data=df_r,x='interest_rate', hue='applicant_age_above_62',kind='kde',fill=True,palette='tab10')
        sns.kdeplot(data=df_r, x='interest_rate', hue='applicant_age_above_62', fill=True, palette='tab10', ax=axes[0])
        axes[0].set_title("Loan Density Plot For Age Above 62")
        axes[0].set_xlabel("APR")
        axes[0].grid(False)
        DIR_ratio = 0.30  # Replace this with your actual DIR ratio
        bar_labels = ['No', 'Yes']  # Labels for the bars
        bar_heights = [1.0, DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue', 'red'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].grid(False)
        
        st.pyplot(fig5)
        # Filter the data by each race class and create displots for APR within each class
        # unique_races = X1['derived_race'].unique()
        # for race in unique_races:
        #     st.subheader(f"APR Distribution for Race: {race}")
        #     race_data = X1[X1['derived_race'] == race]

        #     # Create a displot for age
        #     sns.set(style="whitegrid")
        #     plt.figure(figsize=(8, 6))
        #     sns.displot(data=race_data,x='interest_rate',kind='kde',fill=True)
        #     #sns.histplot(data=race_data, x='interest_rate', kde=True, bins=10)
        #     plt.xlabel("APR")
        #     plt.ylabel("Frequency")
        #     st.pyplot()
    else:
    # You can add other content to the main section of the app
        st.stop()
# Main app structure
#sns.displot(data=X1,hue='applicant_age_above_62',x='interest_rate',kind='kde',fill=True)
st.title("Fair Lending Analysis App")

# Create tabs
tabs = ["Credit Risk Assesment - Individual Loan", "Credit Risk Assesment - Batch Upload", "Fair Lending - Early Warning"]
selected_tab = st.radio("Select a tab:", tabs, format_func=lambda x: x)

# Display content based on selected tab

if selected_tab == "Credit Risk Assesment - Individual Loan":
    tab1_content()
elif selected_tab == "Credit Risk Assesment - Batch Upload":
    tab2_content()
else:
    tab3_content()
