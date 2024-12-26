import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import start_http_server, Counter, generate_latest, Histogram, REGISTRY, Gauge, make_asgi_app
import mlflow
import dagshub
from pydantic import BaseModel
import uvicorn

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

run_id = '53d6b2758e4d4242bdf8e834ffb6988b'
model_uri = f"runs:/{run_id}/GBR/"

model = mlflow.pyfunc.load_model(model_uri)

# Definition of Prometheus metrics
http_requests_total = Counter('http_requests_total', 'Total number of HTTP requests')
request_duration_seconds = Histogram('request_duration_seconds', 'HTTP request duration in seconds')

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

PREDICTION_TIME = Histogram(
    "prediction_latency_seconds", "Latency of predictions"
)
MODEL_ACCURACY = Gauge("model_accuracy", "Production model accuracy")

INPUT_DISTRIBUTION = Histogram(
    "input_feature_distribution", "Feature distribution", ["feature"]
) # TODO: Put in place Evidently

#TODO: check this on how to expose metrics vs the underneath function
## Expose metrics endpoint
#app.mount("/metrics", make_asgi_app())

# Endpoint prédiction
#TODO: check this for prediction latency from @prediction_time
@app.post("/predict")
@PREDICTION_TIME.time()  # automatically measures prediction duration
async def predict(request: Request):
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc() # here

    # Simulation of prediction
    start_time = time.time()
    data = await request.json()
    # Example of prediction
    prediction = sum(data["features"])  # Replace with real model
    latency = time.time() - start_time

    # Log distribution of a feature
    INPUT_DISTRIBUTION.labels(feature="feature1").observe(data["features"][0])

    # Log performance modèle (simulé ici)
    MODEL_ACCURACY.set(0.85)  # Real Accuracy

    return {"prediction": prediction, "latency": latency} 


@app.post("/predict/")
@PREDICTION_TIME.time()  # automatically measures prediction duration
async def predict(input_data: InputData):
# async def predict(request: Request): # TODO: check if I need this line
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc() # here
    start_time = time.time()
    http_requests_total.inc()
    
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
    record["clustering"] = 0  # Example: adding a clustering column, filled with 0 #TODO: real clustering logic

    # TODO: get all data new included and look at total and in prod distribution (better with evidently)
    # TODO: in the next lines: change return and add input_distribution and model_accuracy
    # Log distribution of a feature
    # INPUT_DISTRIBUTION.labels(feature="feature1").observe(data["features"][0])

    # Log performance model (simulated here)
    # MODEL_ACCURACY.set(0.85)  # Real Accuracy

    # return {"prediction": prediction, "latency": latency} 
    # Prediction
    try:
        prediction = model.predict(record)
        duration = time.time() - start_time
        request_duration_seconds.observe(duration)  # Record request duration
        return {"prediction": str(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
def metrics(request: Request):
    # Generate the Prometheus metrics in the correct format
    metrics_data = generate_latest(REGISTRY)
    
    # Return the data with the correct content type for Prometheus
    return Response(content=metrics_data, media_type="text/plain")


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
