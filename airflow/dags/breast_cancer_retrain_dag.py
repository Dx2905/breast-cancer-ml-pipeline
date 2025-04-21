from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'gaurav',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'breast_cancer_weekly_retrain',
    default_args=default_args,
    description='Weekly retraining of Breast Cancer ML model',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'retraining'],
) as dag:

    ingest_data = BashOperator(
        task_id='ingest_data',
        bash_command='echo "Ingesting data (skipped – using existing wdbc.csv)"'

    )

    retrain_model = BashOperator(
    task_id='retrain_model',
    bash_command="""
    source /Users/gaurav/Documents/Breast-Cancer-ML-Pipeline/venv/bin/activate &&
    cd /Users/gaurav/Documents/Breast-Cancer-ML-Pipeline &&
    python train.py > airflow_retrain_log.txt 2>&1
    """
)
    ingest_data >> retrain_model
