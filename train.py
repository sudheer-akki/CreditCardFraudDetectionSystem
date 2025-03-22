""" Code for training ML Models """
import os
import pandas as pd
import mlflow
from dotenv import load_dotenv
from mlproject.components import load_data, \
    DataExplore, DataProcess, DataVisualize
from mlproject.model import ML_MODELS
from mlproject.logging import setup_logger
from mlproject.components import setup_mlflow_run

#Loading Input Arguments
load_dotenv()
DATASET = os.getenv('DATASET')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT= os.getenv('EXPERIMENT_NAME')
MODEL_NAME = os.getenv('MODEL_NAME')
SPLIT_RATIO = float(os.getenv('SPLIT_RATIO'))
logger = setup_logger(pkgname=MLFLOW_EXPERIMENT)

#Loading .csv file
data = load_data(file_path=DATASET)
data = data.drop(columns=["Time"])

#Setting up the MLFlow Server
run_id = setup_mlflow_run(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_experiment=MLFLOW_EXPERIMENT)

explored_data = DataExplore(
                data=data, 
                save_report=False)
explored_results = explored_data.get_results()
report = explored_results['report']

with mlflow.start_run(run_id=run_id):  # Activate the run
    for key, value in report.items():
        mlflow.log_param(key, value)
    mlflow.log_param("SPLIT RATIO", SPLIT_RATIO)
    

DataVisualize(data=data,
            save_path="plots",
            pie_chart= True,
            histogram= True,
            bar_chart = True,
            heat_map = True,
            corr_map= True,
            scatter_plot = True,
            show_outliers = False,
            run_id=run_id)
    
mlflow.log_artifact(local_path="plots",run_id=run_id)

data_proc = DataProcess(
            data=data,
            remove_duplicate = True,
            remove_null = True,
            remove_outliers = False,
            remove_missing = True,
            handle_missing= "mean",
            categorical_missing= "mode",
            scale_features=False,
            drop_column =["Class", "Amount"])

data_processed = data_proc.get_results()

models = ML_MODELS(run_id=run_id,
        target_column="Class",
        save_folder= "weights",
        save_models=True)

X_train, X_test, y_train, y_test = models.split_data(
                                        data=data_processed,
                                        split_ratio=SPLIT_RATIO)

with mlflow.start_run(run_id=run_id):
    models.decision_tree(X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test)

# with mlflow.start_run(run_id=run_id):
#     models.random_forest(X_train=X_train,
#     X_test=X_test,
#     y_train=y_train,
#     y_test=y_test)

# with mlflow.start_run(run_id=run_id):
#     models.xgboost(X_train=X_train,
#     X_test=X_test,
#     y_train=y_train,
#     y_test=y_test)

# with mlflow.start_run(run_id=run_id):
#     models.knn(X_train=X_train,
#     X_test=X_test,
#     y_train=y_train,
#     y_test=y_test)
     
logger.info("\nTraining Completed..!!!")

