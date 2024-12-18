from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define a simple Python function
def print_current_datetime():
    print(f"Current date and time: {datetime.now()}")

# Define your DAG
dag = DAG(
    'simple_dag',  # Unique identifier for the DAG
    schedule_interval='@daily',  # Schedule to run daily
    start_date=datetime(2024, 10, 27),  # When to start the DAG
    catchup=False  # Do not run missed intervals
)

# Define a task using PythonOperator
task1 = PythonOperator(
    task_id='print_datetime',  # Unique identifier for the task
    python_callable=print_current_datetime,  # Function to call
    dag=dag,  # DAG to which the task belongs
)

# Set the task dependencies if you have multiple tasks
# In this case, we only have one task, so there's no dependency to set
