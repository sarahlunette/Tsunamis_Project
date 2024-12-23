import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import start_http_server, Counter, generate_latest, Histogram
import mlflow
import dagshub
from pydantic import BaseModel

# Configuration DagsHub et MLflow
dagshub.auth.add_app_token('9502d9affc30a33129d5c9ca267e2f3e79219c87')
repo_name = "Data_Atelier"
dagshub.init(repo_owner="sarahlunette", repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sarahlunette/Data_Atelier.mlflow")

# Importation du modèle MLflow
client = mlflow.tracking.MlflowClient()
experiment_name = "tsunamis_n_perplexity_k_neighbors"
experiments = mlflow.search_experiments()
for exp in experiments:
    if exp.name == experiment_name:
        experiment_id = exp.experiment_id
runs = client.search_runs(experiment_ids=[experiment_id])
best_run = max(runs, key=lambda run: run.data.metrics.get('r2', float("-inf")))
model_uri = f"runs:/{best_run.info.run_id}/GBR/"
model = mlflow.pyfunc.load_model(model_uri)

# Définition des métriques Prometheus
http_requests_total = Counter('http_requests_total', 'Nombre total de requêtes HTTP')
request_duration_seconds = Histogram('request_duration_seconds', 'Durée des requêtes HTTP en secondes')

# Création de l'API FastAPI
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

@app.post("/predict/")
async def predict(input_data: InputData):
    start_time = time.time()
    http_requests_total.inc()
    
    # Préparation des données d'entrée
    country = "country_" + input_data.country
    data = input_data.dict()
    del data["country"]
    for key in data.keys():
        data[key] = [data[key]]
    data[country] = 1  # Remplir la colonne pays avec 1
    
    # Création du DataFrame
    record = pd.DataFrame.from_dict(data)
    record = record.reindex(columns=columns_final, fill_value=0)
    record["clustering"] = 0  # Exemple d'ajout d'une colonne de clustering

    # Prédiction
    try:
        prediction = model.predict(record)
        duration = time.time() - start_time
        request_duration_seconds.observe(duration)  # Enregistrer la durée de la requête
        return {"prediction": str(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
def metrics():
    return generate_latest()  # Expose les métriques au format Prometheus

if __name__ == "__main__":
    start_http_server(8080)  # Démarre le serveur de métriques Prometheus
    app.run(host="localhost", port=8000)  # Démarre l'API FastAPI