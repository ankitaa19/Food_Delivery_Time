"""
MLflow Model Registration Script for Food Delivery ETA Prediction

This script registers the best model (Linear Regression) from MLflow experiments
into the MLflow Model Registry with appropriate tags and production alias.
"""

import mlflow
from mlflow import MlflowClient
from pathlib import Path


def main():
    """Register the best model to MLflow Model Registry."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    mlflow_db_path = project_root / 'mlflow.db'
    
    # Set up MLflow
    mlflow.set_tracking_uri(f'sqlite:///{mlflow_db_path}')
    client = MlflowClient()
    
    # Get the experiment
    exp = client.get_experiment_by_name('Food_Delivery_ETA_Prediction')
    print(f'Experiment: {exp.name} (ID: {exp.experiment_id})')
    
    # Get the most recent Linear Regression run
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.mlflow.runName = 'Linear_Regression'",
        order_by=['start_time DESC'],
        max_results=1
    )
    
    if not runs:
        print("Error: No Linear Regression run found")
        return
    
    run = runs[0]
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    print(f'\nBest Linear Regression Run:')
    print(f'  Run ID: {run_id}')
    print(f'  MAE: {run.data.metrics.get("MAE"):.4f}')
    print(f'  RMSE: {run.data.metrics.get("RMSE"):.4f}')
    print(f'  R2: {run.data.metrics.get("R2"):.4f}')
    
    # Register the model
    model_name = "Food_Delivery_ETA_Model"
    print(f'\nRegistering model: {model_name}')
    
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "model_type": "Linear Regression",
            "task": "food delivery ETA prediction",
            "selection_metric": "RMSE",
            "rmse": str(run.data.metrics.get("RMSE")),
            "mae": str(run.data.metrics.get("MAE")),
            "r2": str(run.data.metrics.get("R2"))
        }
    )
    
    print(f'Created version {model_version.version}')
    
    # Set the production alias
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=model_version.version
    )
    
    print(f'Set alias "production" on version {model_version.version}')
    
    # Verify registration
    model = client.get_registered_model(model_name)
    print(f'\n=== Registration Summary ===')
    print(f'Registered Model: {model.name}')
    print(f'Version: {model_version.version}')
    print(f'Production Alias: version {model.aliases.get("production")}')
    print(f'\nTags:')
    for key, value in model.tags.items():
        print(f'  {key}: {value}')
    
    print(f'\n=== Model Registration Complete ===')
    print(f'Model URI for loading: models:/{model_name}@production')


if __name__ == "__main__":
    main()
