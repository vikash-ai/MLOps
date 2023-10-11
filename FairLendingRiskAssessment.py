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
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import mlflow
from mlflow.tracking import MlflowClient


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
    st.sidebar.subheader("Input your loan application details")
    
    input_df = pd.DataFrame()
    # def user_input():

    loan_amount = st.sidebar.slider('Loan Amount', 5000.0,500000.0, 1000.0)
    loan_term = st.sidebar.slider('Loan Term', 12,360, 12)
    property_value = st.sidebar.slider('Property value', 5000.0,50000000.0, 1000.0)
    income = st.sidebar.slider('Income', 0.0,20000.0, 100.0)
    debt_to_income_ratio = st.sidebar.selectbox('DTI',('<20%','20%-<30%','30%-<36%','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50%-60%','>60%'))
    applicant_credit_score_type = st.sidebar.selectbox('Credit Score',('1', '2','3','8','9'))
    #loan_purpose = st.sidebar.selectbox('Loan Purpose',('1','2','4','31','32'))
    loan_purpose = st.sidebar.selectbox('Loan Purpose',('Home purchase','Home improvement','Other purpose','Refinancing','Cash-out refinancing'))
    data = {'Application No': 1001,
            'loan_amount':[loan_amount],
            'loan_term':[loan_term],
            'property_value':[property_value],
            'income':[income],
            'debt_to_income_ratio':[debt_to_income_ratio],
            'applicant_credit_score_type':[applicant_credit_score_type],
            'loan_purpose':[loan_purpose],
            }
    features = pd.DataFrame(data)

    # return features
    # input_df = pd.DataFrame()
    input_df = features

    st.subheader('User Input features')
    st.dataframe(input_df, hide_index=True)
    #st.write(input_df.reset_index(drop=True))
    # with open("./approval_pipeline_tuned.pkl", 'rb') as pfile:  
    #             load_clf=pickle.load(pfile)
    model_name = "approval_pipe_RF_tuned"
    model_version = mlflow.get_latest_versions(model_name, stages=["Production"])[0].version
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    # run_id = "44c6ebb3f044459e95ef2a917f23bbed"
    # artifact_uri = mlflow.get_artifact_uri(run_id=run_id)
    # logged_model_uri = f"{artifact_uri}/RF_tuned_model"
    logged_model_uri = f"runs:/44c6ebb3f044459e95ef2a917f23bbed/RF_tuned_model:{model_version}"
    # client = MlflowClient()
    # run = client.get_run("44c6ebb3f044459e95ef2a917f23bbed")
    # base_uri = run.info.artifact_uri
    # logged_model_uri = f"{base_uri}/approval_pipe_RF_tuned"
    load_clf = mlflow.sklearn.load_model(logged_model_uri)
    NUMERICAL_VARIABLES = ['loan_amount', 'income','loan_term','property_value','applicant_credit_score_type']
    CATEGORICAL_VARIABLES = ['debt_to_income_ratio', 'loan_purpose']
    input_df = input_df.drop('Application No', axis=1)
    prediction = load_clf.predict(input_df)
    # st.write(input_df)

    #prediction_proba = load_clf.predict_proba(input_df)
    # If the model supports predict_proba, use it
    if hasattr(load_clf, 'predict_proba'):
        prediction_proba = load_clf.predict_proba(input_df)
        st.write("Class Probabilities:")
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
    Approval_labels = np.array(['Rejected','Approved'])
    st.subheader('Model Prediction')
    st.write(Approval_labels[prediction])

    # st.subheader('Prediction Probability')
    # st.write(prediction_proba)
    
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
    # def load_model(pkl):
    #     return pickle.load(open(pkl, "rb"))
    # model = load_model("./approval_pipeline_tuned.pkl")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    logged_model_uri = "mlflow-artifacts:/992808809450770313/44c6ebb3f044459e95ef2a917f23bbed/artifacts/RF_tuned_model"
    model = mlflow.sklearn.load_model(logged_model_uri)
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
        if st.checkbox("Display summary statistics for uploaded batch file"):
            f"""Sample statistics based on uploaded Data:"""
            #f"""Sample statistics based on {nobs} observations:"""
            st.dataframe(X1.describe())
            #st.dataframe(X.head(nobs).describe())
        X_encoder = encoder.transform(X)
    with st.expander("Model prediction and summary"):
        numeric_columns = X1.select_dtypes(include=[np.number])
        categorical_columns = ['debt_to_income_ratio']
        prediction1 = model.predict(X1)
        probability = model.predict_proba(X1)[:,1]
        X1['prediction'] = prediction1
        X1['probability'] = probability
        if st.checkbox("Choose Probability cutoff for application approval summary"):
            if not X.empty:
                score = st.slider(
                    "Select Probability cutoff for application approval", 0.0, 1.0, value=0.5
                )
            else:
                st.write("Choose Probability score")
            
            X1['prediction_updated'] = np.where(X1['probability'] >= score,1,0)
            numeric_means = X1.groupby("prediction_updated")[numeric_columns.columns].mean()
            categorical_modes = X1.groupby("prediction_updated")[categorical_columns].agg(lambda x: x.mode().iloc[0])
            # Count the occurrences of each group (1 and 0) in "prediction_updated"
            group_counts = X1["prediction_updated"].value_counts().reset_index()
            group_counts.columns = ["prediction_updated", "count"]
            #result = pd.concat([numeric_means, categorical_modes], axis=1)
            result = pd.merge(group_counts, numeric_means, on="prediction_updated")
            result = pd.merge(result, categorical_modes, on="prediction_updated")
            result = result.reset_index(drop=True)
            result = result.round(2)
            st.write('(Numbers below are averages for numeric variables)')
            st.dataframe(result, hide_index=True)
            X_cutoff = X1[X1['prediction_updated'] == 1]
            X_cutoff1= X_cutoff.drop(['prediction','prediction_updated'], axis=1)
            st.dataframe(X_cutoff1.describe())
            tot_approval = X_cutoff['prediction_updated'].sum()
            tot_app = X1['prediction'].size
            f"######  {tot_approval} Application Approved, out of {tot_app} total application"
    #with st.echo():
    explainer = shap.Explainer(final_estimator,X_encoder)
    shap_values = explainer(X_encoder)   
    "### Feature importance - Batch"
    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(X_encoder)
    #st_shap(shap.summary_plot(shap_values, X_encoder.iloc[0:100,:]))
    st_shap(shap.summary_plot(shap_values[1], X_encoder))
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
    #pie chart for approved cases
    fig11, axs = plt.subplots(1, 3, figsize=(20, 10))
    X12 = X1[(X1.Derived_action == 'Approved')]
    X11 = X12[(X12.derived_sex=='Male')|(X12.derived_sex=='Female') ]
    count_df = X11.groupby('derived_sex').size().reset_index(name='Count')
    total_count = count_df['Count'].sum()
    count_df['Percentage'] = count_df['Count'] / total_count * 100
    axs[0].pie(count_df['Percentage'], labels=count_df['derived_sex'], autopct='%1.1f%%', shadow=False, startangle=0)
    axs[0].set_aspect('equal')
    axs[0].set_title('Gender Distribution', loc='center', pad=20)
    
    X11 = X12[(X12.derived_race=='White')|(X12.derived_race=='Black or African American')|(X12.derived_race=='Asian')
             |(X12.derived_race=='American Indian or Alaska Native')|(X12.derived_race=='Native Hawaiian or Other Pacific Islander')]
    count_df = X11.groupby('derived_race').size().reset_index(name='Count')
    total_count = count_df['Count'].sum()
    count_df['Percentage'] = count_df['Count'] / total_count * 100
    axs[1].pie(count_df['Percentage'], labels=count_df['derived_race'], autopct='%1.1f%%', shadow=False, startangle=0)
    axs[1].set_aspect('equal')
    axs[1].set_title('Race Distribution', loc='center', pad=20)
    
    X11 = X12[(X12.applicant_age_above_62=='Yes')|(X12.applicant_age_above_62=='No') ]
    count_df = X11.groupby('applicant_age_above_62').size().reset_index(name='Count')
    total_count = count_df['Count'].sum()
    count_df['Percentage'] = count_df['Count'] / total_count * 100
    axs[2].pie(count_df['Percentage'], labels=count_df['applicant_age_above_62'], autopct='%1.1f%%', shadow=False, startangle=0)
    axs[2].set_aspect('equal')
    axs[2].set_title('Age Above 62', loc='center', pad=20)
    
    
    show_Loan_Population = st.checkbox("Show Loan Population")
    if show_Loan_Population:
        st.header("Loan application Population By Prohibited Basis Factor")
        st.pyplot(fig)
        st.header("Loan approved Population By Prohibited Basis Factor")
        st.pyplot(fig11)
    else:
    # You can add other content to the main section of the app
        # st.write("Select Option to view the distribution plots")
        st.stop()
    
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
        DIR_ratio = 1.12  # Replace this with your actual DIR ratio
        # bar_labels = ['White', 'Black or African American']  # Labels for the bars
        # bar_heights = [1.0, DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact
        bar_labels = ['Black or African American']  # Labels for the bars
        bar_heights = [DIR_ratio]
        # axes[1].bar(bar_labels, bar_heights, color=['blue', 'red'])
        axes[1].bar(bar_labels, bar_heights, color=['red'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].set_ylim(0.0, 3.0)
        axes[1].axhline(y=1.0, color='red', linestyle='--')
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
        bar_labels = ['Asian']  # Labels for the bars
        bar_heights = [DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].set_ylim(0.0, 3.0)
        axes[1].axhline(y=1.0, color='red', linestyle='--')
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
        bar_labels = ['American Indian or Alaska Native']  # Labels for the bars
        bar_heights = [DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].set_ylim(0.0, 3.0)
        axes[1].axhline(y=1.0, color='red', linestyle='--')
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
        bar_labels = ['Native Hawaiian or Other Pacific Islander']  # Labels for the bars
        bar_heights = [DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].set_ylim(0.0, 3.0)
        axes[1].axhline(y=1.0, color='red', linestyle='--')
        axes[1].grid(False)
        st.pyplot(fig3)
        
        fig4, axes = plt.subplots(1, 2, figsize=(20, 10))
        df_r = X1[(X1.derived_sex=='Male')|(X1.derived_sex=='Female') ]
        #sns.displot(data=df_r,x='interest_rate', hue='derived_sex',kind='kde',fill=True,palette='tab10')
        sns.kdeplot(data=df_r, x='interest_rate', hue='derived_sex', fill=True, palette='tab10', ax=axes[0])
        axes[0].set_title("Loan Density Plot For Gender")
        axes[0].set_xlabel("APR")
        axes[0].grid(False)
        DIR_ratio = 0.93  # Replace this with your actual DIR ratio
        bar_labels = ['Female']  # Labels for the bars
        bar_heights = [DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['red'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].set_ylim(0.0, 3.0)
        axes[1].axhline(y=1.0, color='red', linestyle='--')
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
        bar_labels = ['Yes']  # Labels for the bars
        bar_heights = [DIR_ratio]  # Heights of the bars, where 1.0 represents no disparate impact

        axes[1].bar(bar_labels, bar_heights, color=['blue'])
        axes[1].set_title("Disparate Impact Ratio")
        axes[1].set_ylabel("DIR Ratio")
        axes[1].set_ylim(0.0, 3.0)
        axes[1].axhline(y=1.0, color='red', linestyle='--')
        axes[1].grid(False)
        
        st.pyplot(fig5)
        
    else:
    # You can add other content to the main section of the app
        st.stop()
        
    show_map = st.checkbox("Show Minority Group Distribution")
    if show_map:
        st.subheader("Redlining By Census tract")
        # Load the shapefile for Illinois
        census_tract_shapefile = "./tl_2018_17_tract/tl_2018_17_tract.shp"
        il_shapefile = gpd.read_file(census_tract_shapefile)
        il_shapefile1 = pd.DataFrame(il_shapefile)
        il_shapefile1.GEOID = pd.to_numeric(il_shapefile1['GEOID'], errors='coerce')
        #il_shapefile1['geometry'] = il_shapefile1['geometry'].astype('O')
        HMDA_IL = input_df1[['census_tract','tract_minority_population_percent']]
        merged_data1 = HMDA_IL.merge(il_shapefile1, left_on="census_tract", right_on="GEOID", how="left")
        merged_data = merged_data1.sample(n=1000,random_state=25)
        #st.table(merged_data)
        #st.write(merged_data.shape)
        il_m = folium.Map(location=[39.8, -89.7])
        for idx, row in merged_data.iterrows():
            coordinates = row['geometry'].exterior.coords.xy  # Extract the coordinates
            polygon_coords = list(zip(coordinates[1], coordinates[0]))
            # Check if the minority percentage is greater than 50%
            if row['tract_minority_population_percent'] > 50:
            # If greater than 50%, highlight in red
                color = 'red'
            else:
            # Otherwise, use a different color (e.g., blue)
                color = 'rgba(0, 0, 0, 0)'
            folium.Polygon(
                locations=polygon_coords,  # Boundary coordinates for the zip code
                color=color,  # Customize the color based on minority percentage
                #fill_color=row['tract_minority_population_percent'],# Color-coded fill
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"Census tract: {row['census_tract']}, Minority %: {row['tract_minority_population_percent']}"
            ).add_to(il_m)
        st_folium(il_m, width=725)
    else:
    # You can add other content to the main section of the app
        st.stop()    
# Main app structure
#sns.displot(data=X1,hue='applicant_age_above_62',x='interest_rate',kind='kde',fill=True)
st.title("Fair Lending Risk Assessment")

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
