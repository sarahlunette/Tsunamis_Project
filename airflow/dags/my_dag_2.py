# imports
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import dagshub
import os
import sys
from dagshub import get_repo_bucket_client

sys.path.append('/opt/airflow/scripts')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print(sys.path)


# Get a boto3.client object
token = '9502d9affc30a33129d5c9ca267e2f3e79219c87'
dagshub.auth.add_app_token(token=token)
username= 'sarahlunette'
repo_name = 'Data_Atelier'
s3 = get_repo_bucket_client("sarahlunette/Data_Atelier")

# TODO changer le path
path = "opt/airflow/scripts/raw/"
# dagspath = 'raw/'

# TODO: check the imports
#from src.data.interim.scrape import scrape, load_data
#from load_data_s3_dagshub_airflow import upload_all
#from preprocessing_interim_dagshub_airflow import write_data
#from tsunamis_shortened_airflow import train_model

import dagshub
import sys
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import mlflow
from mlflow import MlflowClient
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import json
import subprocess
import dagshub

# TODO save best model to model/model.pkl
import pickle as pkl

    
def train_model():
    # TODO: In all scripts change repo_name in environment variables
    repo_name = "Data_Atelier"
    # token = "69afbf94b8af88d6647f88056bbbfbc775afd83e"
    token = '9502d9affc30a33129d5c9ca267e2f3e79219c87'
    experiment_name = "tsunamis_n_perplexity_k_neighbors"
    dagshub.auth.add_app_token(token)
    dagshub.init(repo_owner="sarahlunette", repo_name=repo_name, mlflow=True)

    # TODO: a try for the experiment client if already existing
    client = MlflowClient()
    mlflow.set_experiment(experiment_name)
    print(client)

    url = (
        "https://dagshub.com/sarahlunette/"
        + repo_name
        + "/raw/"
        + token
        + "/s3:/"
        + repo_name
        + "/"
    )

    def get_data(table):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(url + table)
        df.columns = df.columns.str.lower()
        print(df)
        # Display the DataFrame
        return df

    human_damages = get_data("human_damages.csv")
    houses_damages = get_data("houses_damages.csv")

    # Adding noise for augmentation
    def add_noise(X, noise_level=0.001):
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise

    # predicting the with the output of a kmeans
    def predict_kmeans(df, i, k):

        # Preprocess
        # X, y
        X = df.drop("human_damages", axis=1)
        y = df["human_damages"]

        # Kmeans
        km = KMeans(n_clusters=k + 1, n_init=10)
        centroids = km.fit(X)

        # Adding column
        X["clustering"] = pd.Series(km.labels_)
        df_processed = pd.concat([X, y], axis=1).dropna()
        X = df_processed.drop("human_damages", axis=1)
        y = df_processed["human_damages"]

        # Augment data
        X_noisy = pd.concat([X, X.apply(add_noise)], axis=0).reset_index(drop=True)
        y_noisy = pd.concat([y, y], axis=0).reset_index(drop=True)

        # train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_noisy, y_noisy, test_size=0.2, random_state=42
        )
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Train the regression model
        gbr = GradientBoostingRegressor()
        gbr.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = gbr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return km, gbr, r2

    print(human_damages.shape)
    print(human_damages.head())

    # experiment on best model with parameters perplexity and n_neigbors (we could also go up 80/90 if number of records went up)
    def run_mlflow_experiment():
        mlflow.set_tracking_uri(
            "https://dagshub.com/sarahlunette/" + repo_name + ".mlflow"
        )
        # TODO: data version tag for experiments
        '''mlflow.set_tag("data_version", data_version)'''
        artifact_path = "models"
        for k in tqdm(range(2, 40, 10), desc="Outer loop"):
            for i in tqdm(range(1, 40, 10), desc="Inner loop"):
                run_name = f"tsunamis_n_perplexity_{str(i + 1)}_n_clusters_{str(k + 1)}"
                with mlflow.start_run(run_name=run_name) as run:
                    human_damages.dropna(inplace=True)
                    km, gbr, r2 = predict_kmeans(human_damages, i, k)
                    print(km)
                    try:
                        mlflow.sklearn.log_model(km, "kmeans")
                        mlflow.sklearn.log_model(gbr, "GBR")
                    except Exception as e:
                        print(f"Error logging model: {e}")
                    mlflow.log_metric("r2", r2)
                    mlflow.log_params({"perplexity": i + 1, "n_neighbors": k + 1})
                    mlflow.set_tag("mlflow.runName", run_name)

    try:
        run_mlflow_experiment()
    except Exception as e:
        print(f"An error occurred: {e}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 2),
}

# TODO: changer les paths en fonction de AIRFLOW_HOME (qui était jusqu'à présent dans sarahlenet et qui est dans Data Atelier mtn)

# Define your DAG
dag = DAG(
    'my_dag_2',
    start_date=datetime(2024, 10, 27),
    schedule ='@daily',
    catchup=True
)
# Make sure to version changes
'''task1 = PythonOperator(
    task_id='load_data',
    python_callable=upload_all,
    dag=dag,
    provide_context=True,
)'''

# Make sure to version changes with dvc
'''task2 = PythonOperator(
    task_id='preprocessing',
    python_callable=write_data,
    dag=dag,
    provide_context=True,
)'''

task3 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
    provide_context=True
)
# TODO: rajouter task 1
# task2 >> task3
task3



