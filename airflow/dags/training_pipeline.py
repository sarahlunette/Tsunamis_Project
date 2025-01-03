# TODO
# imports
import os
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.hooks.base_hook import BaseHook
from airflow.models import Connection
from airflow.utils.dates import days_ago
from airflow.utils.decorators import task
from airflow.providers.http.hooks.http import HttpHook
from datetime import datetime
import dagshub
from dagshub import get_repo_bucket_client
# from airflow.sensors.prometheus import PrometheusSensor  # Ensure to import the correct Prometheus sensor
from airflow.providers.prometheus.sensors.prometheus import PrometheusSensor

# TODO: check usefulness of these paths appendings (I should put the scripts in a scripts folder and init and .airflowignore)
sys.path.append('/opt/airflow/src')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Get a boto3.client object
token = '9502d9affc30a33129d5c9ca267e2f3e79219c87'
dagshub.auth.add_app_token(token=token)
username= 'sarahlunette'
repo_name = 'Data_Atelier'
s3 = get_repo_bucket_client("sarahlunette/Data_Atelier")

# TODO: check the imports
# TODO: add scraping when training the model again
# TODO: write load_data script
# TODO: change names of scripts for simpler names
# TODO: check what is upload all
# TODO: check scripts and that they are reachable in this script within the container (how to load from (can I add other scripts that might not be dags ?))
# TODO: changer les paths en fonction de AIRFLOW_HOME (qui était jusqu'à présent dans sarahlenet et qui est dans Data Atelier mtn)
# TODO: __init__.py for packages
# TODO: Make sure to version changes (data and code)
# TODO: organize the data folders above (these should be called in the other scripts and should be in a volume and not within the container or on a db server and to db server)

from data.load_data_s3_dagshub_airflow import upload_all
from features.preprocessing_interim_dagshub_airflow import write_data
from models.tsunamis_shortened_airflow import train_model

# Set the Prometheus connection details programmatically
def create_prometheus_connection():
    # Check if the connection already exists to avoid duplication
    connection_id = "prometheus_default"  # Replace with your desired connection ID
    try:
        # Create the connection object
        conn = Connection(
            conn_id=connection_id,
            conn_type="http",  # Prometheus uses HTTP
            host="http://prometheus:9090",  # Replace with the correct Prometheus host and port
            schema="http",
            port=9090,
        )
        # Add the connection to Airflow's connection database
        session = BaseHook.get_session()
        existing_conn = session.query(Connection).filter_by(conn_id=connection_id).first()

        if existing_conn is None:
            session.add(conn)
            session.commit()
            print(f"Connection '{connection_id}' created successfully.")
        else:
            print(f"Connection '{connection_id}' already exists.")
    except Exception as e:
        print(f"Error creating connection: {e}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 2),
}

# Define your DAG
dag = DAG(
    'training_pipeline',
    start_date=datetime(2024, 10, 27),
    schedule ='@daily',
    catchup=True
)
# Upload new data version / scraping might be done in this task as well
data = PythonOperator(
    task_id='load_data',
    python_callable=upload_all,
    dag=dag,
    provide_context=True,
)

# Create processed dataset
preprocessing = PythonOperator(
    task_id='preprocessing',
    python_callable=write_data,
    dag=dag,
    provide_context=True,
)

# Train model with mlflow experiments
train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
    provide_context=True
)

create_prometheus_connection_task = task(create_prometheus_connection)

r2_sensor = PrometheusSensor(
    task_id='r2_sensor',
    metric='model_r2_in_production',
    threshold=0.5,  # Set your threshold value # TODO: when better models, change threshold
    mode='less',  # Trigger when value is less than threshold
    timeout=600,  # Timeout after 10 minutes
    prometheus_conn_id=connection_id,  # Define a connection to Prometheus
    poke_interval=60,  # Check every minute
    dag=dag,
)

create_prometheus_connection_task >> r2_sensor >> train_model

data >> preprocessing >> train_model