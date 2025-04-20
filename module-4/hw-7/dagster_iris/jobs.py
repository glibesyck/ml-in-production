from dagster import define_asset_job

# Define a job that will execute all the iris pipeline assets
iris_pipeline_job = define_asset_job(
    name="iris_pipeline_job", 
    selection="*",
    description="Job that processes the Iris dataset, trains a model, and evaluates it"
)