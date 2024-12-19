# imports
import os
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import dagshub
from dagshub import get_repo_bucket_client

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


from src.data.load_data_s3_dagshub_airflow import upload_all
from src.features.preprocessing_interim_dagshub_airflow import write_data
from src.models.tsunamis_shortened_airflow import train_model


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 2),

# Define your DAG
dag = DAG(
    'my_dag_2',
    start_date=datetime(2024, 10, 27),
    schedule ='@daily',
    catchup=True
)
# Upload new data version / scraping might be done in this task as well
task1 = PythonOperator(
    task_id='load_data',
    python_callable=upload_all,
    dag=dag,
    provide_context=True,
)

# Create processed dataset
task2 = PythonOperator(
    task_id='preprocessing',
    python_callable=write_data,
    dag=dag,
    provide_context=True,
)

# Train model with mlflow experiments
task3 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
    provide_context=True
)

tas1 >> task2 >> task3