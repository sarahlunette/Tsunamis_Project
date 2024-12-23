import sys
import os
import time
from prometheus_client import start_http_server, Counter, generate_latest, Histogram


os.environ["MLFLOW_PYTHON_ENV"] = '/opt/anaconda3/bin/python3.11.7'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle as pkl
import numpy as np
from sklearn.neighbors import NearestNeighbors
import dagshub
token = '9502d9affc30a33129d5c9ca267e2f3e79219c87'
dagshub.auth.add_app_token(token)


api = FastAPI()

import pandas as pd
import pickle as pkl

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import os

# Set DagsHub credentials (replace with your credentials) TODO: put environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = "sarahlunette"
os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    "d1e822ec803a1fc64d9063837dce3ce746"  # Ensure this is the correct token
)


# Set the MLflow tracking URI to point to your DagsHub repository. TODO: transform in f string
repo_name = "Data_Atelier"
dagshub.init(repo_owner="sarahlunette", repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sarahlunette/Data_Atelier.mlflow")

# Initialize the MLflow client
client = MlflowClient()


# TODO: has to change this, according to different experiments in retrainings (each retraining is an experiment ?)
# Replace with your experiment name or ID
experiment_name = "tsunamis_n_perplexity_k_neighbors"

# get all the experiments
experiments = mlflow.search_experiments()

for i in range(len(experiments)):
    if experiments[i].name == experiment_name:
        experiment_id = experiments[i].experiment_id

# Fetch all runs from the experiment
runs = client.search_runs(experiment_ids=[experiment_id])

# Define the metric you want to use to select the best model
metric_name = "r2"  # Replace with your desired metric

# Find the run with the best (highest) metric value
best_run = max(runs, key=lambda run: run.data.metrics.get(metric_name, float("-inf")))

# Get the run_id of the best run
best_run_id = best_run.info.run_id

best_run_id = '6ba3bbe66f0f41c7a0ee8efae45d10d6'
# Retrieve the model artifact from the best run
model_uri = f"runs:/{best_run_id}/GBR/"

# TODO the problem of artifact

# Load the best model
best_model = mlflow.pyfunc.load_model(model_uri)
model = best_model


class InputData(BaseModel):
    month: int
    day: int
    country: str
    period: int
    latitude: float
    longitude: float
    runup_ht: float
    runup_ht_r: float
    runup_hori: float
    dist_from_: float
    hour: float
    cause_code: float
    event_vali: float
    eq_mag_unk: float
    eq_mag_mb: float
    eq_mag_ms: float
    eq_mag_mw: float
    eq_mag_mfa: float
    eq_magnitu: float
    eq_magni_1: float
    eq_depth: float
    max_event_: float
    ts_mt_ii: float
    ts_intensi: float
    num_runup: float
    num_slides: float
    map_slide_: float
    map_eq_id: float


columns = [
    "month",
    "day",
    "country",
    "period",
    "latitude",
    "longitude",
    "runup_ht",
    "runup_ht_r",
    "runup_hori",
    "dist_from_",
    "hour",
    "cause_code",
    "event_vali",
    "eq_mag_unk",
    "eq_mag_mb",
    "eq_mag_ms",
    "eq_mag_mw",
    "eq_mag_mfa",
    "eq_magnitu",
    "eq_magni_1",
    "eq_depth",
    "max_event_",
    "ts_mt_ii",
    "ts_intensi",
    "num_runup",
    "num_slides",
    "map_slide_",
    "map_eq_id",
]

columns_final = [
    "month",
    "day",
    "period",
    "latitude",
    "longitude",
    "runup_ht",
    "runup_ht_r",
    "runup_hori",
    "dist_from_",
    "hour",
    "cause_code",
    "event_vali",
    "eq_mag_unk",
    "eq_mag_mb",
    "eq_mag_ms",
    "eq_mag_mw",
    "eq_mag_mfa",
    "eq_magnitu",
    "eq_magni_1",
    "eq_depth",
    "max_event_",
    "ts_mt_ii",
    "ts_intensi",
    "num_runup",
    "num_slides",
    "map_slide_",
    "map_eq_id",
    "gdp_per_capita",
    "country_bangladesh",
    "country_canada",
    "country_chile",
    "country_china",
    "country_colombia",
    "country_costa rica",
    "country_dominican republic",
    "country_ecuador",
    "country_egypt",
    "country_el salvador",
    "country_fiji",
    "country_france",
    "country_french polynesia",
    "country_greece",
    "country_haiti",
    "country_india",
    "country_indonesia",
    "country_italy",
    "country_jamaica",
    "country_japan",
    "country_kenya",
    "country_madagascar",
    "country_malaysia",
    "country_maldives",
    "country_mexico",
    "country_micronesia",
    "country_myanmar",
    "country_new caledonia",
    "country_new zealand",
    "country_nicaragua",
    "country_norway",
    "country_pakistan",
    "country_panama",
    "country_papua new guinea",
    "country_peru",
    "country_philippines",
    "country_portugal",
    "country_russia",
    "country_samoa",
    "country_solomon islands",
    "country_somalia",
    "country_south korea",
    "country_spain",
    "country_sri lanka",
    "country_taiwan",
    "country_tanzania",
    "country_tonga",
    "country_turkey",
    "country_united kingdom",
    "country_united states",
    "country_vanuatu",
    "country_venezuela",
    "country_yemen",
]

http_requests_total = Counter('http_requests_total', 'Nombre total de requêtes HTTP')
request_duration_seconds = Histogram('request_duration_seconds', 'Durée des requêtes HTTP en secondes')  # Histogram


@api.post("/predict/")
async def predict(input_data: InputData):
    start_time = time.time()
    http_requests_total.inc()

    # Change the country name key
    country = "country_" + input_data.country
    data = input_data.dict()
    del data["country"]
    for key in data.keys():
        data[key] = [data[key]]
    data[country] = 1  # Set the corresponding country column to 1

    # Create a DataFrame with the input data
    record = pd.DataFrame.from_dict(data)
    # Ensure all columns are present, filling missing columns with 0
    record = record.reindex(columns=columns_final, fill_value=0)

    # Add KMeans for filling clustering column and the rest (save Kmeans and load model)
    record["clustering"] = 0

    try:
        prediction = model.predict(record)
        return {"prediction": str(prediction[0])}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to make prediction: {str(e)}"
        )
    duration = time.time() - start_time
    
    # Request duration in graph
    request_duration_seconds.observe(duration)


@api.route('/metrics')
def metrics():
    return generate_latest()  # Expose metrics in Prometheus format

if __name__ == "__main__":
    # Start Prometheus server
    start_http_server(8080)  # The port where Prometheus can get the metrics
    api.run(port=8080)  # L'API functions on port 8080
