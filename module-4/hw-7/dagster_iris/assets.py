import os
import numpy as np
import pandas as pd
import pickle
from dagster import asset, Config, AssetExecutionContext, Output, MetadataValue
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class IrisConfig(Config):
    test_size: float = 0.3
    random_state: int = 47
    max_iter: int = 500

@asset
def raw_iris_data():
    # Load dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df = df.dropna()
    
    return df

@asset(deps=["raw_iris_data"])
def train_test_data(context: AssetExecutionContext, raw_iris_data, config: IrisConfig):
    # Split data into train and test sets
    target_column = 'species'
    X = raw_iris_data.loc[:, raw_iris_data.columns != target_column]
    y = raw_iris_data.loc[:, raw_iris_data.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.test_size,
        stratify=y, 
        random_state=config.random_state
    )
    
    context.log.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

@asset(deps=["train_test_data"])
def trained_model(context: AssetExecutionContext, train_test_data, config: IrisConfig):
    # Train the model
    X_train = train_test_data["X_train"]
    y_train = train_test_data["y_train"]
    
    classifier = LogisticRegression(max_iter=config.max_iter)
    classifier.fit(X_train, y_train)
    
    # Serialize the model
    model_path = os.path.join(context.instance.storage_directory(), "model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)
    
    return Output(
        value=classifier,
        metadata={
            "model_path": MetadataValue.path(model_path),
            "model_type": "LogisticRegression"
        }
    )

@asset(deps=["trained_model", "train_test_data"])
def model_predictions(context: AssetExecutionContext, trained_model, train_test_data):
    # Make predictions on test data
    X_test = train_test_data["X_test"]
    y_test = train_test_data["y_test"]
    
    # Generate predictions
    y_pred = trained_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = (y_pred == y_test.values.ravel()).mean()
    
    context.log.info(f"Model accuracy: {accuracy:.4f}")
    
    return Output(
        value=list(y_pred),
        metadata={
            "accuracy": float(accuracy),
            "test_size": len(y_test)
        }
    )