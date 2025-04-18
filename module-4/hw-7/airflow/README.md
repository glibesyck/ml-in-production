## Deploying Airflow pipelines using Python SDK


1. Install Airflow:

```bash
pip install apache-airflow
```

### Setup Airflow Environment

2. Set the Airflow home directory:

```bash
export AIRFLOW_HOME=~/airflow
```

3. Initialize the Airflow database:

```bash
airflow db init
```

4. Create an admin user:

```bash
airflow users create \
    --username admin \
    --firstname Your \
    --lastname Name \
    --role Admin \
    --email your.email@example.com \
    --password admin
```

5. Copy the DAG file to your Airflow dags folder:

```bash
cp dags.py $AIRFLOW_HOME/dags/
```

6. Start the Airflow webserver:

```bash
airflow webserver --port 8080
```

7. In a new terminal, start the Airflow scheduler:

```bash
airflow scheduler
```

8. Access the Airflow UI at:

```
http://localhost:8080
```


The results are stored in the `/tmp/airflow/iris_pipeline` directory.