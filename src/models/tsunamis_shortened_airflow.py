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

# TODO save best model to model/model.pkl
import pickle as pkl


'''def get_data_version(dvc_file_path):
    """
    Get the data version from the DVC metadata file.
    """
    with open(dvc_file_path, 'r') as f:
        dvc_metadata = json.load(f)
    return dvc_metadata.get('outs')[0].get('version', 'unknown')

# TODO: get data version from dagshub: Specify the path to your DVC file
dvc_file_path = '/usr/local/data/processed/human_damages.csv.dvc'

# Get the data version from DVC
data_version = get_data_version(dvc_file_path)'''

    
def train_model():
    # TODO: In all scripts change repo_name in environment variables
    repo_name = "Data_Atelier"
    token = "9502d9affc30a33129d5c9ca267e2f3e79219c87"
    experiment_name = "tsunamis_n_perplexity_k_neighbors"
    dagshub.auth.add_app_token(token)
    dagshub.init(repo_owner="sarahlunette", repo_name=repo_name, mlflow=True)

    # TODO: a try for the experiment client if already existing
    client = MlflowClient()
    mlflow.set_experiment(experiment_name)
    
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

