import mlflow
import logging
from typing import Optional
import subprocess

def setup_mlflow_run(mlflow_tracking_uri: str, 
                    mlflow_experiment: str, 
                    logger:  Optional[logging.Logger]):
    """
    Sets up MLflow tracking, ensures the experiment exists, and starts a new run.

    Parameters:
        mlflow_tracking_uri (str): The MLflow tracking URI.
        mlflow_experiment (str): The name of the MLflow experiment.

    Returns:
        str: The run_id of the newly created MLflow run.
    """
    try:
        # Set the MLflow tracking URI
        mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
        logger.info(f"Tracking URI set to {mlflow_tracking_uri}")
        # Check if the experiment already exists
        current_experiment = mlflow.get_experiment_by_name(mlflow_experiment)
        if current_experiment is None:
            mlflow.create_experiment(
                name=mlflow_experiment, 
                tags={'mlflow.note.content': "Credit Card Fraud Classification Model"}
            )
            logger.info(f"Created experiment {mlflow_experiment}...!!!")
        else:
            logger.info(f"Experiment {mlflow_experiment} already exists...!!!")
        # Set the experiment
        mlflow.set_experiment(mlflow_experiment)
        # Get the experiment details
        experiment_id = current_experiment.experiment_id
        logger.info(f"Using experiment_id: {experiment_id}")
        # Start a new MLflow run and get the run_id
        with mlflow.start_run(experiment_id=experiment_id,nested=True) as run:
            run_id = run.info.run_id
            logger.info(f"Started new MLflow run with run_id: {run_id}")
            return run_id
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError, Exception) as e:
        logger.error(f"Error occurred while setting up MLflow: {e}")
        raise e  # Re-raise the exception if needed

