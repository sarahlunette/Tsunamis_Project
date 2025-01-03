# TODO
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import start_http_server, Counter, generate_latest, Histogram, REGISTRY, Gauge
from airflow.providers.prometheus.sensors.prometheus import PrometheusSensor
import mlflow
import dagshub
from pydantic import BaseModel
import uvicorn
from metrics.r2_in_production import r2_in_production # TODO: check this

# Configuring DagsHub and MLflow
dagshub.auth.add_app_token('9502d9affc30a33129d5c9ca267e2f3e79219c87')
repo_name = "Data_Atelier"
dagshub.init(repo_owner="sarahlunette", repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sarahlunette/Data_Atelier.mlflow")

# Import MLflow model
client = mlflow.tracking.MlflowClient()
experiment_name = "tsunamis_n_perplexity_k_neighbors"
experiments = mlflow.search_experiments()
for exp in experiments:
    if exp.name == experiment_name:
        experiment_id = exp.experiment_id
runs = client.search_runs(experiment_ids=[experiment_id])
best_run = max(runs, key=lambda run: run.data.metrics.get('r2', float("-inf")))
model_uri = f"runs:/{best_run.info.run_id}/GBR/"

run_id = '53d6b2758e4d4242bdf8e834ffb6988b'  # Example run ID # TODO: Change run_id when experience is cleared and models are in 3.11
model_uri = f"runs:/{run_id}/GBR/"
model = mlflow.pyfunc.load_model(model_uri)

# Definition of Prometheus metrics
http_requests_total = Counter('http_requests_total', 'Total number of HTTP requests')
request_duration_seconds = Histogram('request_duration_seconds', 'HTTP request duration in seconds')
prediction_time = Histogram("prediction_latency_seconds", "Latency of predictions")
model_r2__in_production_gauge = Gauge('model_r2_in_production', 'R2 score of the model')


# Instantiation of the API FastAPI
app = FastAPI()

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

columns_final = [
    "month", "day", "period", "latitude", "longitude", "runup_ht", "runup_ht_r", "runup_hori",
    "dist_from_", "hour", "cause_code", "event_vali", "eq_mag_unk", "eq_mag_mb", "eq_mag_ms",
    "eq_mag_mw", "eq_mag_mfa", "eq_magnitu", "eq_magni_1", "eq_depth", "max_event_",
    "ts_mt_ii", "ts_intensi", "num_runup", "num_slides", "map_slide_", "map_eq_id",
    "gdp_per_capita", "country_bangladesh", "country_canada", "country_chile", "country_china",
    "country_colombia", "country_costa rica", "country_dominican republic", "country_ecuador",
    "country_egypt", "country_el salvador", "country_fiji", "country_france",
    "country_french polynesia", "country_greece", "country_haiti", "country_india",
    "country_indonesia", "country_italy", "country_jamaica", "country_japan", "country_kenya",
    "country_madagascar", "country_malaysia", "country_maldives", "country_mexico",
    "country_micronesia", "country_myanmar", "country_new caledonia", "country_new zealand",
    "country_nicaragua", "country_norway", "country_pakistan", "country_panama",
    "country_papua new guinea", "country_peru", "country_philippines", "country_portugal",
    "country_russia", "country_samoa", "country_solomon islands", "country_somalia",
    "country_south korea", "country_spain", "country_sri lanka", "country_taiwan",
    "country_tanzania", "country_tonga", "country_turkey", "country_united kingdom",
    "country_united states", "country_vanuatu", "country_venezuela", "country_yemen",
]

@app.post("/predict/")
async def predict(input_data: InputData):
    http_requests_total.inc()  # Increment total HTTP requests

    # Measure prediction latency using a context manager
    start_time = time.time()
    with prediction_time.time():
        # Preparing entry data
        country = "country_" + input_data.country
        data = input_data.dict()
        del data["country"]
        for key in data.keys():
            data[key] = [data[key]]
        data[country] = 1  # Filling country column with 1

        # DataFrame creation
        record = pd.DataFrame.from_dict(data)
        record = record.reindex(columns=columns_final, fill_value=0)
        record["clustering"] = 0  # Example: adding a clustering column, filled with 0 #TODO: Add real clustering logic

        # Prediction
        try:
            prediction = model.predict(record)
            duration = time.time() - start_time
            request_duration_seconds.observe(duration)  # Record request duration
            return {"prediction": str(prediction[0]), "latency": str(duration)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

import time
import threading

def update_r2():
    while True:
        model_r2_in_production = r2_in_production()  # Simulate the R² value
        model_r2_in_production_gauge.set(model_r2_in_production)
        time.sleep(60)  # Update every minute

# Start the thread to update R² in the background
threading.Thread(target=update_r2, daemon=True).start()

@app.get("/metrics")
def metrics(request: Request):
    # Generate the Prometheus metrics in the correct format
    metrics_data = generate_latest(REGISTRY)
    return Response(content=metrics_data, media_type="text/plain")

# replace these by the real values
AIRFLOW_URL = "http://airflow-webserver:8080"  # URL of Airflow webserver # TODO: add monitoring network in docker-compose
DAG_ID = "training_pipeline"  # ID of DAG to trigger
# Get Airflow username and password from environment variables
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME")  # Retrieve from environment variable
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD")  # Retrieve from environment variable

# Function to trigger the DAG using Basic Authentication
def trigger_dag():
    url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    
    # Using Basic Authentication
    response = requests.post(
        url, 
        auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),  # Add the username and password here
        json={"conf": {}}  # You can include any configuration if needed
    )
    
    if response.status_code == 200:
        print("DAG triggered successfully")
    else:
        print(f"Failed to trigger DAG: {response.status_code}, {response.text}")

@app.post("/retrain")
async def retrain_model():
    trigger_dag()
    return {"message": "Model successfully retrained and DAG triggered"}


# Entry point for both local and Docker execution
if __name__ == "__main__":
    import os
    prometheus_port = int(os.getenv("PROMETHEUS_PORT", 9090))  # Default Prometheus port: 9090
    api_host = os.getenv("API_HOST", "0.0.0.0")  # Default host: all interfaces
    api_port = int(os.getenv("API_PORT", 8000))  # Default API port: 8000

    # Start Prometheus server in a separate thread
    start_http_server(prometheus_port)
    print(f"Prometheus metrics server running on port {prometheus_port}")

    # Start FastAPI server
    uvicorn.run(app, host=api_host, port=api_port)
