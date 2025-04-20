from dagster import Definitions
from assets import raw_iris_data, train_test_data, trained_model, model_predictions, IrisConfig
from jobs import iris_pipeline_job

defs = Definitions(
    assets=[raw_iris_data, train_test_data, trained_model, model_predictions],
    jobs=[iris_pipeline_job],
    resources={
        "iris_config": IrisConfig(),
    }
)