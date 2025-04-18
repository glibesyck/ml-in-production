from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def prepare_data(output_path):
    """
    Load Iris dataset and prepare it for processing
    """
    # Load dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df = df.dropna()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df.to_csv(f'{output_path}/final_df.csv', index=False)
    return output_path


def split_data(input_path, output_path):
    """
    Split data into train and test sets
    """
    final_data = pd.read_csv(f'{input_path}/final_df.csv')
    
    target_column = 'species'
    X = final_data.loc[:, final_data.columns != target_column]
    y = final_data.loc[:, final_data.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=47)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    np.save(f'{output_path}/X_train.npy', X_train)
    np.save(f'{output_path}/X_test.npy', X_test)
    np.save(f'{output_path}/y_train.npy', y_train)
    np.save(f'{output_path}/y_test.npy', y_test)
    return output_path


def train_model(input_path, output_path):
    """
    Train a logistic regression model
    """
    X_train = np.load(f'{input_path}/X_train.npy', allow_pickle=True)
    y_train = np.load(f'{input_path}/y_train.npy', allow_pickle=True)
    
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, y_train)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'{output_path}/model.pkl', "wb") as f:
        pickle.dump(classifier, f)
    
    return output_path


def predict(model_path, data_path, output_path):
    """
    Make predictions on test data
    """
    with open(f'{model_path}/model.pkl', 'rb') as f:
        logistic_reg_model = pickle.load(f)
        
    X_test = np.load(f'{data_path}/X_test.npy', allow_pickle=True)
    y_pred = logistic_reg_model.predict(X_test)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(f'{output_path}/y_pred.npy', y_pred)
    return output_path


# Define DAG parameters
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define base paths for each step
base_path = '/tmp/airflow/iris_pipeline'
data_path = f'{base_path}/data'
split_path = f'{base_path}/split'
model_path = f'{base_path}/model'
prediction_path = f'{base_path}/prediction'

# Create DAG
dag = DAG(
    'iris_pipeline',
    default_args=default_args,
    description='A simple Iris ML pipeline',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['ml', 'iris'],
)

# Define tasks
prepare_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    op_kwargs={'output_path': data_path},
    dag=dag,
)

split_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    op_kwargs={'input_path': data_path, 'output_path': split_path},
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    op_kwargs={'input_path': split_path, 'output_path': model_path},
    dag=dag,
)

predict_task = PythonOperator(
    task_id='predict',
    python_callable=predict,
    op_kwargs={
        'model_path': model_path, 
        'data_path': split_path, 
        'output_path': prediction_path
    },
    dag=dag,
)

# Define task dependencies
prepare_task >> split_task >> train_task >> predict_task