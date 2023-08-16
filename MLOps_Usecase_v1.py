#!/usr/bin/env python
# coding: utf-8

# 1. Use case Description- Fair Lending
# 2. Data Preparation- specific to Fair lending with both Accepted and Rejected application data
# 3. ML Model Development - MLFlow Experiment tracking & Model Registry
# 4. Model Explainability : PMML
# 5. Use Case 1: Model deployment/Web services Expose for prediction using Flask
# 5.1 Model serving UI using Streamlit and containerise it with Docker. Model will be pulled from MLFlow Model Registry and used in the UI. Model metrics and prediction result storage in MongoDb/backend
# 6. Use Case 2: Model Monitoring : Grafana/Evidently
# 7. Use Case 3: Model run orchestration/ Batch prediction and testing run, start stop services : Prefect
# 8. Use Case 4: Unit & Integration test, Logging and error handling
# 9. Use Case 5: Schema validation,Validate Input data, drift check before new workflow orchestration/new prediction: Grafana/Evidently
# 10. 

# Use your terminal to do below task
# 1. To create virtual env, 
# pip install virtualenv
# conda create -n venv python=3.9
# conda activate venv
# 2. To create requirement.txt file,
# pip freeze > requirements.txt
# 3. To install packages in any environment
# pip install -r requirements.txt

# In[1]:


#Import data analysis packages first
import os
#To change your working directory
os.chdir('C:\\HMDA')
#Data analysis packages
import pandas as pd
import numpy as np
import warnings # use only when you are sure your code is correct
warnings.filterwarnings('ignore')
#Let's import visualization library
import matplotlib.pyplot as plt
import seaborn as sns
#set option to see full data
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score
# pipeline
from sklearn.pipeline import Pipeline
# to save the model and the preprocessor
import joblib

#for MLops
import mlflow
import mlflow.sklearn


# In[2]:


#Load your input data
HMDA_IL_2020 = pd.read_csv('.\\Input\\state_IL_lei_B4TYDEB6GKMZO031MB27.csv')
#Let's see if data is loaded properly
HMDA_IL_2020.head(3)


# In[3]:


#Let's check the missing values and data type of each variable
HMDA_IL_2020.info()


# In[4]:


#Descriptions: 1. Loan originated 2. Application approved but not accepted 
#3. Application denied 4. Application withdrawn by applicant 5. File closed for incompleteness 
#6. Purchased loan 7. Preapproval request denied 8. Preapproval request approved but not accepted
HMDA_IL_2020.action_taken.value_counts(normalize=True)


# In[5]:


#check for details https://github.com/cfpb/hmda-platform/blob/master/docs/spec/Public_File_LAR_Spec.csv
#We will subset the data for easy understanding on deployment
HMDA_IL = HMDA_IL_2020[['applicant_credit_score_type','debt_to_income_ratio','action_taken','loan_amount','property_value','income','loan_purpose', 'loan_term']]
print(HMDA_IL_2020.shape)
print(HMDA_IL.shape)


# In[6]:


#we will only work with accepted and denied application
HMDA_IL = HMDA_IL[HMDA_IL['action_taken'].isin([1,3])]
print(HMDA_IL.shape)


# In[7]:


#check dependent variable distribution
HMDA_IL.action_taken.value_counts(normalize=True)


# In[8]:


#our reduced data
HMDA_IL.head()


# In[9]:


#check missing values
HMDA_IL.isnull().sum()


# In[10]:


#numeric variable distribution
HMDA_IL.describe()


# In[11]:


#loan amount distribution
print(HMDA_IL.loan_amount.min())
print(HMDA_IL.loan_amount.mean())
print(HMDA_IL.loan_amount.max())


# In[12]:


HMDA_IL.applicant_credit_score_type.value_counts()


# In[13]:


HMDA_IL.debt_to_income_ratio.value_counts()


# In[14]:


print(HMDA_IL.property_value.min())
print(HMDA_IL.property_value.mean())
print(HMDA_IL.property_value.max())


# In[15]:


#1. Home purchase 2. Home improvement 31. Refinancing 32. Cash-out refinancing 4. Other purpose 5. Not applicable
HMDA_IL.loan_purpose.value_counts()


# In[16]:


#fix the action taken/ target variable
HMDA_IL['action_taken'] = np.where(HMDA_IL['action_taken']==1, 1, 0)


# In[17]:


HMDA_IL.action_taken.value_counts(normalize=True)


# # Configuration

# In[18]:


HMDA_IL.info()


# In[19]:


# cast categorical variables as object

HMDA_IL['loan_purpose'] = HMDA_IL['loan_purpose'].astype('O')


# In[20]:


#for simplisity, fill missing with mode and mean here
HMDA_IL["debt_to_income_ratio"].fillna(HMDA_IL["debt_to_income_ratio"].mode()[0],inplace=True)
HMDA_IL["income"].fillna(HMDA_IL["income"].mean(),inplace=True)
HMDA_IL["property_value"].fillna(HMDA_IL["property_value"].mean(),inplace=True)


# In[21]:


#recheck the missing value
HMDA_IL.isnull().sum()


# In[22]:


NUMERICAL_VARIABLES = ['loan_amount', 'income','loan_term','property_value','applicant_credit_score_type']

CATEGORICAL_VARIABLES = ['debt_to_income_ratio', 'loan_purpose']


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(
    HMDA_IL.drop('action_taken', axis=1),  # predictors
    HMDA_IL['action_taken'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape


# In[24]:


#import few more library
# for the preprocessors
from sklearn.base import BaseEstimator, TransformerMixin


# for encoding categorical variables
from feature_engine.encoding import (
    OneHotEncoder
)


# In[25]:


# set up the pipeline
approval_pipe = Pipeline([


    # == CATEGORICAL ENCODING ======

    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=CATEGORICAL_VARIABLES)),

    # scale
    ('scaler', StandardScaler()),

    ('Logit', LogisticRegression(C=0.0005, random_state=0)),
])


# In[26]:


#from here, let's start tracking our model experiments
#Log in MLFlow
experiment_name = "Deployment"
run_name="Approval_Denied"
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.get_tracking_uri()


# In[27]:


#set experiment name
mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog()
with mlflow.start_run():
    # train the pipeline
    approval_pipe.fit(X_train, y_train)
    class_ = approval_pipe.predict(X_test)
    mlflow.sklearn.log_model(approval_pipe, "base_model")
    test_Accuracy = accuracy_score(y_test, class_)
    mlflow.log_metric("test_Accuracy", test_Accuracy)
    print(f"Test Accuracy: {test_Accuracy:.2%}")
    print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.set_tag("tag1", "Base Logistic Model")
mlflow.end_run()
print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))


# In[28]:


# make predictions for train set
class_ = approval_pipe.predict(X_train)
pred = approval_pipe.predict_proba(X_train)[:,1]

# determine mse and rmse
print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
print()

# make predictions for test set
class_ = approval_pipe.predict(X_test)
pred = approval_pipe.predict_proba(X_test)[:,1]

# determine mse and rmse
print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
print()


# In[29]:


# set up the pipeline
approval_pipe_RF = Pipeline([


    # == CATEGORICAL ENCODING ======

    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=CATEGORICAL_VARIABLES)),


    ('RF', RandomForestClassifier()),
])


# In[31]:


mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog()
with mlflow.start_run():
    # train the pipeline
    approval_pipe_RF.fit(X_train, y_train)
    class_ = approval_pipe_RF.predict(X_test)
    mlflow.sklearn.log_model(approval_pipe_RF, "RF_model")
    test_Accuracy = accuracy_score(y_test, class_)
    mlflow.log_metric("test_Accuracy", test_Accuracy)
    print(f"Test Accuracy: {test_Accuracy:.2%}")
    print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.set_tag("tag1", "Base RF Model")
mlflow.end_run()
print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))


# In[32]:


# make predictions for train set
class_ = approval_pipe_RF.predict(X_train)
pred = approval_pipe_RF.predict_proba(X_train)[:,1]

# determine auc and accuracy
print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
print()

# make predictions for test set
class_ = approval_pipe_RF.predict(X_test)
pred = approval_pipe_RF.predict_proba(X_test)[:,1]

# determine auc and accuracy
print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
print()


# In[33]:


# set up the pipeline, tune it before fitting this step. We are doing trial n error to save time as last RF model was overpredicting
approval_pipe_RF_tuned = Pipeline([


    # == CATEGORICAL ENCODING ======

    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=CATEGORICAL_VARIABLES)),


    ('RF_tuned', RandomForestClassifier(max_depth= 7,random_state = 222,class_weight={0:0.3, 1:0.7})),
])


# In[34]:


mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog()
with mlflow.start_run():
    # train the pipeline
    approval_pipe_RF_tuned.fit(X_train, y_train)
    class_ = approval_pipe_RF_tuned.predict(X_test)
    mlflow.sklearn.log_model(approval_pipe_RF_tuned, "RF_tuned_model")
    test_Accuracy = accuracy_score(y_test, class_)
    mlflow.log_metric("test_Accuracy", test_Accuracy)
    print(f"Test Accuracy: {test_Accuracy:.2%}")
    print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.set_tag("tag1", "RF tuned Model")
mlflow.end_run()
print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))


# In[43]:


# make predictions for train set
class_ = approval_pipe_RF_tuned.predict(X_train)
pred = approval_pipe_RF_tuned.predict_proba(X_train)[:,1]

# determine auc and accuracy
print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
print()

# make predictions for test set
class_ = approval_pipe_RF_tuned.predict(X_test)
pred = approval_pipe_RF_tuned.predict_proba(X_test)[:,1]

# determine auc and accuracy
print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
print()


# In[44]:


# now let's save the RF base model as final

joblib.dump(approval_pipe_RF_tuned, 'C:\\HMDA\\Model\\approval_pipeline_tuned.joblib') 


# In[ ]:




