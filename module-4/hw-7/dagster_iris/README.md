## Deploying Dagster pipelines using Python SDK


1. Run the Dagster UI:
```bash
dagster dev
dagster dev -f definitions.py -a defs
```

2. Access the Dagster UI in your browser:
```
http://localhost:3000
```

You can run the pipeline in several ways:

3. **Through the Dagster UI**:
   - Navigate to the Jobs page
   - Click on "iris_pipeline_job"
   - Click "Launch Run"

4. **Via the CLI**:
```bash
dagster job execute -f definitions.py -a defs -j iris_pipeline_job
```
