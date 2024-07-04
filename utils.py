
# importing the required libraries
import os
import pickle

import pandas as pd
import numpy as np

import sklearn
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import xgboost as xgb

import sqlalchemy
from sqlalchemy import create_engine,text
from sqlalchemy.engine import URL

from snowflake.sqlalchemy import *

import snowflake.connector
from snowflake.connector.pandas_tools import pd_writer
from snowflake.connector.pandas_tools import write_pandas


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders


from creds import ACCOUNT,USERNAME,PASSWORD,MAIL_ID,MAIL_PASSWORD


from datetime import datetime, timedelta
import time
import pytz

tz_ny = pytz.timezone('Asia/Kolkata')

import warnings
warnings.filterwarnings('ignore')




""" PRE-PROCESSING UTILITY FUNCTION"""

# writing a function for preprocessing
def preprocess_data(data):
    
    # making clone of the data
    df=data.copy()
    
    # defining the columns to drop
    drop_cols = ["HOSPITAL_CODE","PATIENTID",'ADMISSION_DATE','DISCHARGE_DATE']
    # dropping the cols
    df = df.drop(columns=drop_cols)
    
    # setting the CASE_ID as index
    df.set_index("CASE_ID",inplace=True)
    
    # defining the numerical and categorical cols
    num_cols = ["LOS","AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL","ADMISSION_DEPOSIT","VISITORS_WITH_PATIENT"]
    cat_cols = [col for col in df.columns if col not in num_cols]
    
    # fixing the dtypes of the columns
    for col in cat_cols:
        df[col]=df[col].astype(object)
        
    for col in num_cols:
        df[col] = df[col].astype(int)
    
    # doing OHE
    df_ohe = pd.get_dummies(df,dtype=int)
    
    for col in df_ohe.columns:
        if df_ohe[col].dtype==bool:
            df_ohe[col] = df_ohe[col].astype(int)
    
    return df_ohe
    

    
    
    
    
    

""" DEPLOYMENT UITLITY FUNCTIONS"""

# defining the function to verify the feautures of incoming data
def check_create_model_features(data,features_lst):
    temp = pd.DataFrame()
    for col in features_lst:
        if col in data.columns.tolist():
            temp[col]=data[col]
        else:
            temp[col]=0
    return temp


# defining a function to insert the predictions into the snowflake table
def insert_predictions(data,connection,engine):
    

   # establishing the connection to the snowflake 
    conn = snowflake.connector.connect(
    user=USERNAME,
    password=PASSWORD,
    account=ACCOUNT,
    role='ACCOUNTADMIN',
    warehouse='COMPUTE_WH',
    database='HEALTH_DB',
    schema='PUBLIC'
    )
    
    
    # creating the logging table
    table = "PREDICTION_LOGGING"
    table_creation_query =f"""
                            CREATE TABLE IF NOT EXISTS {table} (
                                CASE_ID STRING,
                                HOSPITAL_CODE INT,
                                HOSPITAL_TYPE_CODE STRING,
                                CITY_CODE_HOSPITAL INT,
                                HOSPITAL_REGION_CODE STRING,
                                AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL INT,
                                DEPARTMENT STRING,
                                WARD_TYPE STRING,
                                WARD_FACILITY_CODE STRING,
                                BED_GRADE INT,
                                PATIENTID STRING,
                                CITY_CODE_PATIENT INT,
                                TYPE_OF_ADMISSION STRING,
                                SEVERITY_OF_ILLNESS STRING,
                                VISITORS_WITH_PATIENT INT,
                                AGE STRING,
                                ADMISSION_DEPOSIT FLOAT,
                                ADMISSION_DATE DATE,
                                DISCHARGE_DATE DATE,
                                ADMISSION_MONTH STRING,
                                ADMISSION_DAY STRING,
                                ADMISSION_ILLNESS STRING,
                                ILLNESS_BEDGRADE STRING,
                                DEPARTMENT_ILLNESS STRING,
                                LOS INT,
                                LOS_PREDICTED INT
                            )
                            """
    connection.execute(table_creation_query)
    
    # inserting the predictions to the table
    write_pandas(conn, data, table_name=table)
    return "Success"



# defining the function to send notofications via mail 
def send_mail(mail_string):
   
    subject = 'Patient LOS Prediction - STATUS MAIL'
    mail_content = mail_string

    username= MAIL_ID
    password= MAIL_PASSWORD
    send_from =MAIL_ID
    send_to = MAIL_ID
    Cc = ''
    
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = send_to
    msg['Cc'] = Cc
    msg['Date'] = formatdate(localtime = True)
    msg['Subject'] = subject
    msg.attach(MIMEText(mail_content, 'plain'))
    smtp = smtplib.SMTP('smtp.gmail.com',587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login(username,password)
    smtp.sendmail(send_from, send_to.split(',') + msg['Cc'].split(','), msg.as_string())
    smtp.quit()
    
    


# defining the function for batch prediction and writing it to the snowflake table
def deploy():
        
        # defining the query to load new test data from the simulation data
        QUERY="""

        WITH BASE AS (

            SELECT CASE_ID,
                   COALESCE(HOSPITAL_CODE,0) AS HOSPITAL_CODE,
                   COALESCE(HOSPITAL_TYPE_CODE,'None') AS HOSPITAL_TYPE_CODE,
                   COALESCE(CITY_CODE_HOSPITAL,0) AS CITY_CODE_HOSPITAL,
                   COALESCE(HOSPITAL_REGION_CODE,'None') AS HOSPITAL_REGION_CODE,
                   COALESCE(AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,0) AS AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,
                   COALESCE(DEPARTMENT,'None') AS DEPARTMENT,
                   COALESCE(WARD_TYPE,'None') AS WARD_TYPE,
                   COALESCE(WARD_FACILITY_CODE,'None') AS WARD_FACILITY_CODE,
                   COALESCE(BED_GRADE,0) AS BED_GRADE,
                   PATIENTID,
                   COALESCE(CITY_CODE_PATIENT,0) AS CITY_CODE_PATIENT,
                   COALESCE(TYPE_OF_ADMISSION,'None') AS TYPE_OF_ADMISSION,
                   COALESCE(SEVERITY_OF_ILLNESS,'Minor') AS SEVERITY_OF_ILLNESS,
                   COALESCE(VISITORS_WITH_PATIENT,0) AS VISITORS_WITH_PATIENT,
                   COALESCE(AGE,'None') AS AGE,
                   COALESCE(ADMISSION_DEPOSIT,0) AS ADMISSION_DEPOSIT,
                   ADMISSION_DATE,
                   DISCHARGE_DATE

            FROM HEALTH_DB.PUBLIC.SIMULATION_DATA

        ),

        BASE_WITH_FEATURES AS (

            SELECT *,
                    MONTHNAME(ADMISSION_DATE) AS ADMISSION_MONTH,
                    DAYNAME(ADMISSION_DATE) AS ADMISSION_DAY,    
                    CONCAT(TYPE_OF_ADMISSION,'-',SEVERITY_OF_ILLNESS) AS ADMISSION_ILLNESS,
                    CONCAT(SEVERITY_OF_ILLNESS,'-',BED_GRADE) AS ILLNESS_BEDGRADE,
                    CONCAT(DEPARTMENT,'-',SEVERITY_OF_ILLNESS) AS DEPARTMENT_ILLNESS,
                    DATEDIFF(day,ADMISSION_DATE,DISCHARGE_DATE) AS LOS
            FROM BASE 

        )    
        SELECT * FROM BASE_WITH_FEATURES WHERE ADMISSION_DATE=CURRENT_DATE-580


        """

        # defining an empty list to store notification for each phases
        mail_lst = []

        # Creating the connection engine (way 1)
        engine = create_engine(URL(
            account=ACCOUNT,
            user= USERNAME,
            password= PASSWORD,
            role="ACCOUNTADMIN",
            warehouse="COMPUTE_WH",
            database="HEALTH_DB",
            schema="PUBLIC"
        ))

        # Connecting to the DB and executing the query
        with engine.connect() as conn:

            # loading the test data
            result = conn.execute(text(QUERY))
            test_data = pd.DataFrame(result.fetchall())
            test_data.columns = result.keys()
            mail_lst.append("STEP-1: Successfully loaded the test data ")


            # appplying the preprocessing steps
            test_data.columns = [col.upper() for col in test_data.columns.tolist()]
            test_preprocessed = preprocess_data(test_data)
            print(test_data.columns)
            mail_lst.append("STEP-2: Successfully applied the data preprocessing on test data ")

            # applying the feature selection by calling the helper function to verify the feautures of incoming data
            final_features = pd.read_pickle("artifacts/final_features.pkl")
            test_data_final = check_create_model_features(test_preprocessed,final_features)
            mail_lst.append("STEP-3: Successfully applied the feature selection")

            # getting the predictions
            model = xgb.XGBRegressor()
            model.load_model("artifacts/xgb.model")
            test_data_final['LOS_PREDICTED'] = np.ceil(model.predict(test_data_final))
            mail_lst.append("STEP-4: Successfully got the predictions")


            # wrirting the dataframe into a table
            test_data_final.reset_index(inplace=True)
            predictions = test_data_final[['CASE_ID','LOS_PREDICTED']]
            logs = pd.merge(test_data,test_data_final[['CASE_ID','LOS_PREDICTED']],on="CASE_ID")
            status = insert_predictions(logs,conn,engine)
            mail_lst.append("STEP-5: Successfully wrote the predictions to snowflake table")       

            # creating the mail body and sending the notifications
            mail_string = ",\n".join(map(str,mail_lst))
            send_mail(mail_string)

            print(status)
            return status

        
        
        
        
""" DATA DRIFT DETECOR UTILITY FUNCTIONS"""

# defining a function which returns the query given the batch id for data drift detection
def get_data_drift_query(x):
    
    """ x : if the x is 1, then the function pulls last one week's data """
    
    query=f"""

    SELECT CASE_ID,
           COALESCE(HOSPITAL_CODE,0) AS HOSPITAL_CODE,
           COALESCE(HOSPITAL_TYPE_CODE,'None') AS HOSPITAL_TYPE_CODE,
           COALESCE(CITY_CODE_HOSPITAL,0) AS CITY_CODE_HOSPITAL,
           COALESCE(HOSPITAL_REGION_CODE,'None') AS HOSPITAL_REGION_CODE,
           COALESCE(AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,0) AS AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,
           COALESCE(DEPARTMENT,'None') AS DEPARTMENT,
           COALESCE(WARD_TYPE,'None') AS WARD_TYPE,
           COALESCE(WARD_FACILITY_CODE,'None') AS WARD_FACILITY_CODE,
           COALESCE(BED_GRADE,0) AS BED_GRADE,
           PATIENTID,
           COALESCE(CITY_CODE_PATIENT,0) AS CITY_CODE_PATIENT,
           COALESCE(TYPE_OF_ADMISSION,'None') AS TYPE_OF_ADMISSION,
           COALESCE(SEVERITY_OF_ILLNESS,'Minor') AS SEVERITY_OF_ILLNESS,
           COALESCE(VISITORS_WITH_PATIENT,0) AS VISITORS_WITH_PATIENT,
           COALESCE(AGE,'None') AS AGE,
           COALESCE(ADMISSION_DEPOSIT,0) AS ADMISSION_DEPOSIT,
           ADMISSION_DATE,
           DISCHARGE_DATE

    FROM HEALTH_DB.PUBLIC.PREDICTION_LOGGING
    WHERE ADMISSION_DATE>=CURRENT_DATE-580+{x*7} 

    """ 
    return query


# defining a function which pulls recent data and check if it is drifted
def data_drift_monitor(batch_id):
    
    """ batch_id : if the batch_id is 1, then the function pulls last one week's data """
    
    # defining the query
    query = get_data_drift_query(batch_id)

    # creating the snowflake engine
    engine = create_engine(URL(
            account=ACCOUNT,
            user= USERNAME,
            password= PASSWORD,
            role="ACCOUNTADMIN",
            warehouse="COMPUTE_WH",
            database="HEALTH_DB",
            schema="PUBLIC"
        ))
    
    # connecting to the engine
    with engine.connect() as conn:
        result = conn.execute(text(query))
        batch = pd.DataFrame(result.fetchall())
        batch.columns = result.keys()
        batch.columns = [col.upper() for col in batch.columns.tolist()]
        
    # defining numerical,id and categorical columns
    num_cols = ["AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL","ADMISSION_DEPOSIT","VISITORS_WITH_PATIENT"]
    id_cols = ["CASE_ID","PATIENTID","ADMISSION_DATE","DISCHARGE_DATE"]
    cat_cols = [col for col in batch.columns if col not in num_cols+id_cols]
    
    # filtering the data to include only cat and num cols
    batch_df = batch[num_cols + cat_cols]
    
    # loading the detector model from pickle file
    with open('artifacts/drift_detector.pkl','rb') as f:
        detector = pickle.load(f)
    
    # predicting whether there's a drift
    prediction = detector.predict(batch_df.values,drift_type='feature')
    
    # printing details
    labels=["No","Yes"]

#     for i in range(detector.n_features):
#         # determining the type of test done for the feature i
#         stat = 'Chi2' if i in list(cat_cols) else 'KS'
#         # getting the name of the feature
#         fname = batch_df.columns.tolist()[i]
#         # finding if drift is happened at the feature
#         is_drift = prediction['data']['is_drift'][i]
#         # getting the test-statistic and pvalue for the test
#         test_stat,p_value = prediction['data']['distance'][i], prediction['data']['p_val'][i]
#         print(f"{fname} \t\t--Drift? {labels[is_drift]} --{stat} {test_stat}-- p-value:{p_value}\n\n\n")
        
    
    # defining a logging data frame
    log_df = pd.DataFrame()

    log_df['Time Period'] = [str(batch['ADMISSION_DATE'].min()) + ' to ' + str(batch['ADMISSION_DATE'].max())] * detector.n_features
    log_df['Total Records'] = batch_df.shape[0]
    log_df['Features'] = batch_df.columns.tolist()
    log_df["is_Drift"] = prediction['data']['is_drift']
    log_df['Test'] = log_df['Features'].apply(lambda x:'Chi2' if x in cat_cols else "KS")
    log_df['Test stat'] = np.round(prediction['data']['distance'])
    log_df['P value'] = np.round(prediction['data']['p_val'])

    
    return log_df
    

""" MODEL DRIFT DETECTOR UTILITY FUNCTIONS"""    


# defining a function which pulls data from snowflake for model drift detection
def get_model_drift_query(x):
    
    """ x : if the x is 1, then the function pulls last one week's data """
    
    query=f"""

    SELECT CASE_ID,
           COALESCE(HOSPITAL_CODE,0) AS HOSPITAL_CODE,
           COALESCE(HOSPITAL_TYPE_CODE,'None') AS HOSPITAL_TYPE_CODE,
           COALESCE(CITY_CODE_HOSPITAL,0) AS CITY_CODE_HOSPITAL,
           COALESCE(HOSPITAL_REGION_CODE,'None') AS HOSPITAL_REGION_CODE,
           COALESCE(AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,0) AS AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,
           COALESCE(DEPARTMENT,'None') AS DEPARTMENT,
           COALESCE(WARD_TYPE,'None') AS WARD_TYPE,
           COALESCE(WARD_FACILITY_CODE,'None') AS WARD_FACILITY_CODE,
           COALESCE(BED_GRADE,0) AS BED_GRADE,
           PATIENTID,
           COALESCE(CITY_CODE_PATIENT,0) AS CITY_CODE_PATIENT,
           COALESCE(TYPE_OF_ADMISSION,'None') AS TYPE_OF_ADMISSION,
           COALESCE(SEVERITY_OF_ILLNESS,'Minor') AS SEVERITY_OF_ILLNESS,
           COALESCE(VISITORS_WITH_PATIENT,0) AS VISITORS_WITH_PATIENT,
           COALESCE(AGE,'None') AS AGE,
           COALESCE(ADMISSION_DEPOSIT,0) AS ADMISSION_DEPOSIT,
           ADMISSION_DATE,
           DISCHARGE_DATE,
           LOS,
           LOS_PREDICTED

    FROM HEALTH_DB.PUBLIC.PREDICTION_LOGGING
    WHERE ADMISSION_DATE>=CURRENT_DATE-580+{x*7} 

    """ 
    return query


# defining a function that checks whether the model is drifted by comparing the performance metrics on both train and new data
def check_model_drfit(ref_metric_dict,curr_metric_dict,type="regression",tol=0.1):
    
    """ref_metric_dict   : dictionary containing the performance metrics of model on train data
       curr_metric_dict  : dictionary containing the performance metrics of model on new unseen data
       type              : type of the problem (classification/regression)
       tolerance         : the minimum percentage difference between train and test metrics to decide the drift
       
       Returns floating values representing metrics change and a boolean variable is_model_drifted"""
    
    if type=="classification":
        
        # finding the deviation in the classification metrics
        precision_change = abs((curr_metric_dict['Precision']-ref_metric_dict['Precision'])/ref_metric_dict['Precision'])
        recall_change = abs((curr_metric_dict['Recall']-ref_metric_dict['Recall'])/ref_metric_dict['Recall'])
        roc_auc_change = abs((curr_metric_dict['Roc-Auc']-ref_metric_dict['Roc-Auc'])/ref_metric_dict['Roc-Auc'])
        
        # checking how many metrics are deviated beyond the tolerance threshold
        counter = 0
        for i in [precision_change,recall_change,roc_auc_change]:
            if i > tol:
                counter+=1
        
        if counter>0:
            print(f"ALERT ! There's a model drift")
            print("Change in Precision: "+ str(round(100*precision_change,2)) + "%" )
            print("Change in Recall: "+ str(round(100*recall_change,2)) + "%" )
            print("Change in Roc-Auc: "+ str(round(100*roc_auc_change,2)) + "%" )
            return 1,precision_change,recall_change,roc_auc_change
        else:
            print("There is no Model drift.")
            return 0,precision_change,recall_change,roc_auc_change
        
    
    elif type=="regression":
        
        # finding the deviation in the regression metrics
        rmse_change = abs((curr_metric_dict['RMSE']-ref_metric_dict['RMSE'])/ref_metric_dict['RMSE'])
        mae_change = abs((curr_metric_dict['MAE']-ref_metric_dict['MAE'])/ref_metric_dict['MAE'])
        
        # checking how many metrics are deviated beyond the tolerance threshold
        counter = 0
        for i in [rmse_change,mae_change]:
            if i > tol:
                counter+=1
        
        if counter>0:
            print(f"ALERT ! There's a model drift")
            print("Change in RMSE: "+ str(round(100*rmse_change,2)) + "%" )
            print("Change in MAE: "+ str(round(100*mae_change,2)) + "%" )
            return 1,rmse_change,mae_change
        else:
            print("There is no Model drift.")
            return 0,rmse_change,mae_change
        
            


# defining a function that monitors for model drift
def model_drift_monitor(batch_id):
    
    """ batch_id : if the batch_id is 1, then the function pulls last one week's data """
    
    # defining the query
    query = get_model_drift_query(batch_id)


    # creating the snowflake engine
    engine = create_engine(URL(
            account=ACCOUNT,
            user= USERNAME,
            password= PASSWORD,
            role="ACCOUNTADMIN",
            warehouse="COMPUTE_WH",
            database="HEALTH_DB",
            schema="PUBLIC"
        ))
    
    # connecting to the engine
    with engine.connect() as conn:
        result = conn.execute(text(query))
        batch = pd.DataFrame(result.fetchall())
        batch.columns = result.keys()
        batch.columns = [col.upper() for col in batch.columns.tolist()]
        
        
    # getting the actual LOS and predicted los
    actual = batch['LOS']
    predicted = batch['LOS_PREDICTED']
    
    # computing the metrics
    rmse = root_mean_squared_error(actual,predicted)
    mae = mean_absolute_error(actual,predicted)
    
    # storing the metrics in a dictionary for detecting model drift
    scoring_ref_metrics = dict()
    scoring_ref_metrics['RMSE'] = rmse
    scoring_ref_metrics['MAE'] = mae
          
          
    # loading the model_ref_metrics which conatains the metrics of model on train data
    with open('artifacts/model_ref_metrics.pkl','rb') as f:
          model_ref_metrics = pickle.load(f)
          
          
    # calling the check_model_drfit to compare the performance metrics
    model_drift,rmse_change,mae_change = check_model_drfit(model_ref_metrics,scoring_ref_metrics,type="regression",tol=0.1)
          
    # creating a log in the form of dictionary
    log = dict()
    log['Time-Period'] = str(batch['ADMISSION_DATE'].min()) + ' to ' + str(batch['ADMISSION_DATE'].max())
    log['Total records'] = batch.shape[0]
    log['Scoring metrics'] = scoring_ref_metrics
    log['Training metrics'] = model_ref_metrics
    log['Model Drift '] = model_drift
    log['RMSE change'] = rmse_change
    log['MAE change'] = mae_change
          
    return log




"""MODEL RETRAINING UTILITY FUNCTIONS"""

# Orders the features in the dataset, if unknown new features are added then it assings 0 for all values in that feature"""
def validate_features(df,features_list):
    
    test = pd.DataFrame()
    for col in features_list:
        if col in df.columns.tolist():
            test[col]=df[col]
        else:
            test[col]=0
    return test


# Selects the best features for model building
def select_features(df):
    
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    import xgboost as xgb

    
    # splitting the target and input features
    x = df.drop(columns=['LOS'])
    y = df[['LOS']]

    # train test splitting
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y)
    

    # 1. DECISION TREE
    
    dtree = DecisionTreeRegressor()
    # fitting the tree
    dtree.fit(x_train,y_train)
    # Checking the feature importance
    zipper = dict(zip(x_train.columns,dtree.feature_importances_))
    # making it as a dataframe
    feature_importances= pd.DataFrame.from_dict(zipper,orient='index').reset_index().rename(columns={"index":"feature",0:"importance"}).sort_values(by='importance',ascending=False)
    # taking the features which have importance greater than threshold 0.01
    dtree_features = feature_importances[feature_importances['importance']>0.01]['feature'].values.tolist()
    
    # 2. XGBOOST
    
    xgb_ = xgb.XGBRegressor()
    xgb_.fit(x_train,y_train)
    xgb_.score(x_train,y_train)
    # Checking the feature importance
    zipper = dict(zip(x_train.columns,xgb_.feature_importances_))
    # making it as a dataframe
    xgb_feature_importances= pd.DataFrame.from_dict(zipper,orient='index').reset_index().rename(columns={"index":"feature",0:"importance"}).sort_values(by='importance',ascending=False)
    # taking the features which have importance greater than threshold 0.01
    xgb_features = feature_importances[xgb_feature_importances['importance']>0.01]['feature'].values.tolist()
    
    
    
    # Joining the features from both dtree and xgboost
    final_features = list(set(dtree_features).union(set(xgb_features)))
    

    # exporting the list of final features for future predictions
    with open('retraining_artifacts/retrained_final_features.pkl', 'wb') as f:
        pickle.dump(final_features, f)
        
    return final_features


# defining a function to create query to fetch the data for retraining
def get_retraining_query(max_date):
    
    # defining the query (taking old training data from HEALTH_DATA table and new data from PREDICTION_LOGGING table )
    query =f"""

            WITH TRAIN_BASE AS (

                SELECT CASE_ID,
                       COALESCE(HOSPITAL_CODE,0) AS HOSPITAL_CODE,
                       COALESCE(HOSPITAL_TYPE_CODE,'None') AS HOSPITAL_TYPE_CODE,
                       COALESCE(CITY_CODE_HOSPITAL,0) AS CITY_CODE_HOSPITAL,
                       COALESCE(HOSPITAL_REGION_CODE,'None') AS HOSPITAL_REGION_CODE,
                       COALESCE(AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,0) AS AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,
                       COALESCE(DEPARTMENT,'None') AS DEPARTMENT,
                       COALESCE(WARD_TYPE,'None') AS WARD_TYPE,
                       COALESCE(WARD_FACILITY_CODE,'None') AS WARD_FACILITY_CODE,
                       COALESCE(BED_GRADE,0) AS BED_GRADE,
                       PATIENTID,
                       COALESCE(CITY_CODE_PATIENT,0) AS CITY_CODE_PATIENT,
                       COALESCE(TYPE_OF_ADMISSION,'None') AS TYPE_OF_ADMISSION,
                       COALESCE(SEVERITY_OF_ILLNESS,'Minor') AS SEVERITY_OF_ILLNESS,
                       COALESCE(VISITORS_WITH_PATIENT,0) AS VISITORS_WITH_PATIENT,
                       COALESCE(AGE,'None') AS AGE,
                       COALESCE(ADMISSION_DEPOSIT,0) AS ADMISSION_DEPOSIT,
                       ADMISSION_DATE,
                       DISCHARGE_DATE

                FROM HEALTH_DB.PUBLIC.HEALTH_DATA

            ),

            TRAIN_BASE_WITH_FEATURES AS (

                SELECT *,
                        MONTHNAME(ADMISSION_DATE) AS ADMISSION_MONTH,
                        DAYNAME(ADMISSION_DATE) AS ADMISSION_DAY,    
                        CONCAT(TYPE_OF_ADMISSION,'-',SEVERITY_OF_ILLNESS) AS ADMISSION_ILLNESS,
                        CONCAT(SEVERITY_OF_ILLNESS,'-',BED_GRADE) AS ILLNESS_BEDGRADE,
                        CONCAT(DEPARTMENT,'-',SEVERITY_OF_ILLNESS) AS DEPARTMENT_ILLNESS,
                        DATEDIFF(day,ADMISSION_DATE,DISCHARGE_DATE) AS LOS
                FROM TRAIN_BASE 

            ),
            
            NEW_DATA_WITH_FEATURES AS (
            
                SELECT CASE_ID,
                       COALESCE(HOSPITAL_CODE,0) AS HOSPITAL_CODE,
                       COALESCE(HOSPITAL_TYPE_CODE,'None') AS HOSPITAL_TYPE_CODE,
                       COALESCE(CITY_CODE_HOSPITAL,0) AS CITY_CODE_HOSPITAL,
                       COALESCE(HOSPITAL_REGION_CODE,'None') AS HOSPITAL_REGION_CODE,
                       COALESCE(AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,0) AS AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL,
                       COALESCE(DEPARTMENT,'None') AS DEPARTMENT,
                       COALESCE(WARD_TYPE,'None') AS WARD_TYPE,
                       COALESCE(WARD_FACILITY_CODE,'None') AS WARD_FACILITY_CODE,
                       COALESCE(BED_GRADE,0) AS BED_GRADE,
                       PATIENTID,
                       COALESCE(CITY_CODE_PATIENT,0) AS CITY_CODE_PATIENT,
                       COALESCE(TYPE_OF_ADMISSION,'None') AS TYPE_OF_ADMISSION,
                       COALESCE(SEVERITY_OF_ILLNESS,'Minor') AS SEVERITY_OF_ILLNESS,
                       COALESCE(VISITORS_WITH_PATIENT,0) AS VISITORS_WITH_PATIENT,
                       COALESCE(AGE,'None') AS AGE,
                       COALESCE(ADMISSION_DEPOSIT,0) AS ADMISSION_DEPOSIT,
                       ADMISSION_DATE,
                       DISCHARGE_DATE,
                       ADMISSION_MONTH,
                       ADMISSION_DAY,
                       ADMISSION_ILLNESS,
                       ILLNESS_BEDGRADE,
                       DEPARTMENT_ILLNESS,
                       LOS
                FROM HEALTH_DB.PUBLIC.PREDICTION_LOGGING
                WHERE ADMISSION_DATE<='{max_date}'
                
            
            )
            
            SELECT * FROM TRAIN_BASE_WITH_FEATURES
            UNION ALL
            SELECT * FROM NEW_DATA_WITH_FEATURES
           
            """
    return query


# defining the function for retraining
def retrain_model(cut_off_date):

    # Creating the connection engine (way 1)
    engine = create_engine(URL(
            account=ACCOUNT,
            user= USERNAME,
            password= PASSWORD,
            role="ACCOUNTADMIN",
            warehouse="COMPUTE_WH",
            database="HEALTH_DB",
            schema="PUBLIC"
        ))
    
    # getting the query 
    query = get_retraining_query(cut_off_date)
    
    # connecting to the engine
    with engine.connect() as conn:
        result = conn.execute(text(query))
        data = pd.DataFrame(result.fetchall())
        data.columns = result.keys()
        data.columns = [col.upper() for col in data.columns.tolist()]
    
    print("Successfully fetched data from snowflake\n")
    print("Shape of fetched data is ",data.shape)
    
    # defining the max and min dates for data splitting 
    max_date = data['ADMISSION_DATE'].max()
    min_date = max_date-timedelta(days=7)
    
    # splitting the data into train and test
    d_train = data[data['ADMISSION_DATE']<=min_date]
    d_test = data[(data['ADMISSION_DATE']>min_date) & (data['ADMISSION_DATE']<=max_date)]
    
    print("Train, Test split done\n")
        
    # applying the preprocess steps to both train and test
    df_train = preprocess_data(d_train)
    df_test = preprocess_data(d_test)
    
    print("Preprocessing of Train and test data is done\n")
    
    
    # selecting features
    final_features = select_features(df_train)
    df_test_processed = validate_features(df_test,final_features)
    
    print("Feature selection executed\n")

    
    # Model building
    import xgboost as xgb
    from sklearn.metrics import root_mean_squared_error,mean_absolute_error
    
    xgb_ = xgb.XGBRegressor()
    xgb_.fit(df_train[final_features],df_train['LOS'])
    
    # getting the predictions for the test data (last 1 week's data)
    y_test_pred = np.ceil(xgb_.predict(df_test_processed))
    
    # computing the performance metrics
    rmse = root_mean_squared_error(y_test_pred,df_test['LOS'])
    mae = mean_absolute_error(y_test_pred,df_test['LOS'])
    print("Test performance of new retrained model \n")
    print(f"RMSE is {rmse}")
    print(f"MAE is {mae}\n\n")
    
    # storinig the performance metrics for future use
    retrained_model_metrics = dict()
    retrained_model_metrics['RMSE']=rmse
    retrained_model_metrics['MAE']=mae
    import pickle
    with open('retraining_artifacts/retrained_model_metrics.pkl', 'wb') as f:
        pickle.dump(retrained_model_metrics, f)
    
    # saving the model
    booster = xgb_.get_booster()
    booster.save_model('retraining_artifacts/xgb_retrained.model')
    print("Successfully saved the retrained model\n")
    
    # loading the old model
    old_model = xgb.XGBRegressor()
    old_model.load_model('artifacts/xgb.model')
    print("Sucessfully loaded old model")
    
    # loading the selected features for our old model
    with open('artifacts/final_features.pkl','rb') as f:
        final_features_old = pickle.load(f)
    
    # getting predictions the test data (last 1 week's data) from our old model
    df_test_processed_old = validate_features(df_test,final_features_old)
    y_test_pred_old = np.ceil(old_model.predict(df_test_processed_old))
    
    # computing the performance metrics
    rmse_old = root_mean_squared_error(y_test_pred_old,df_test['LOS'])
    mae_old = mean_absolute_error(y_test_pred_old,df_test['LOS'])
    print("Test performance of old existing model")
    print(f"RMSE is {rmse_old}")
    print(f"MAE is {mae_old}")
    
    # storinig the performance metrics for future use
    old_model_metrics = dict()
    old_model_metrics['RMSE']=rmse_old
    old_model_metrics['MAE']=mae_old
    
    return retrained_model_metrics,old_model_metrics


# defining a function which chooses the model to deploy based on the performance metric
def finalize_model(old_model_metrics,new_model_metrics):
    count=0
    
    # checking if the RMSE and MAE of new model metric is lesser than t
    for metric in new_model_metrics.keys():
        if new_model_metrics[metric]<old_model_metrics[metric]:
            count+=1
    
    if count>0:
        return 'New Model '
    else:
        return 'Old Model'


# defining the function to deploy the model
def deploy_model(selector="Old Model"):
    
    if selector!="Old Model":
        
        # LOADING THE OLD MODEL ARTIFACTS
        with open('artifacts/final_features.pkl','rb') as f:
            old_model_features = pickle.load(f)
        with open('artifacts/model_ref_metrics.pkl','rb') as f:
            old_model_metrics = pickle.load(f)
        old_model = xgb.Booster()
        old_model.load_model('artifacts/xgb.model')
        
        
        # MOVING THE OLD MODEL ARTIFACTS TO THE ARCHIVE FOLDER
        with open('archive/old_model_features.pkl','wb') as f:
            pickle.dump(old_model_features,f)
        with open('archive/old_model_metrics.pkl','wb') as f:
            pickle.dump(old_model_metrics,f)
        old_model.save_model('archive/old_model.model')
            
        # LOADING THE NEW MODEL
        with open('retraining_artifacts/retrained_final_features.pkl','rb') as f:
            new_model_features = pickle.load(f)
        with open('retraining_artifacts/retrained_model_metrics.pkl','rb') as f:
            new_model_metrics = pickle.load(f)
        new_model = xgb.Booster()
        new_model.load_model('retraining_artifacts/xgb_retrained.model')
            
        
        # REPLACING THE OLD MODEL WITH THE NEW RETRAINED MODEL
        with open('artifacts/final_features.pkl','wb') as f:
            pickle.dump(new_model_features,f)
        with open('artifacts/model_ref_metrics.pkl','wb') as f:
            pickle.dump(new_model_metrics,f)
        new_model.save_model('artifacts/xgb.model')
            
        print("Deployment New Model Successfully")
        
    else:
        print("Keeping the Old Model")
    
    return "Deployment Succesful"
    
            
    
