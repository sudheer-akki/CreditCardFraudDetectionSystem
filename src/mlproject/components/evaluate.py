# this script belongs to only evaluation of the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow

def evaluate_model(model_name, y_true, y_pred, run_id):
    """
    function for evaluating the model 
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_score_check = f1_score(y_true, y_pred)
    mlflow.log_metric(f"{model_name}_accuracy", round(accuracy,3), run_id=run_id)
    mlflow.log_metric(f"{model_name}_accuracy", round(accuracy,3), run_id=run_id)
    mlflow.log_metric(f"{model_name}_accuracy", round(accuracy,3), run_id=run_id)
    mlflow.log_metric(f"{model_name}_precision", round(precision,3), run_id=run_id)
    mlflow.log_metric(f"{model_name}_recall", round(recall,3),run_id=run_id)
    mlflow.log_metric(f"{model_name}_F1_score", round(f1_score_check,3),run_id=run_id)
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall":recall,
        "F1-Score":f1_score_check,
    }
 

def confusion_matrix(model_name, y_true, y_pred, run_id):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    img = disp.plot().figure_
    mlflow.log_artifact(img, artifact_file="metadata/confusion_matrix.png")
    return img