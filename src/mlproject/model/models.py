import pandas as pd
from typing import Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mlproject.components import evaluate_model
import pickle
import mlflow
from sklearn.preprocessing import StandardScaler


class ML_MODELS:
    def __init__(self, 
        run_id: str,
        logger:  Optional[logging.Logger],
        target_column: str = "Class",
        save_models: bool = True,
        save_folder: str = "weights"):
        self.run_id = run_id
        self.logger = logger
        self.save_models = save_models
        self.save_folder = save_folder
        self.target_column = target_column

    def split_data(self, 
        data: pd.DataFrame, 
        split_ratio: float = 0.8,
        random_state: int = 42):
        X = data.drop(columns = [self.target_column])
        y = data[self.target_column]
        return train_test_split(X,
        y,train_size=split_ratio,
        test_size=1-float(split_ratio),
        random_state=random_state,
        stratify=y)

    def logistic_regression(self,
        X_train: pd.DataFrame,
        X_test : pd.DataFrame,
        y_train : pd.DataFrame,
        y_test : pd.DataFrame,
        standard_scaler = StandardScaler()):
        X_train_scaled = standard_scaler.fit_transform(X_train)
        X_test_scaled = standard_scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000)
        self.logger.info(f"Started {type(model).__name__} training")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        if self.save_models:
            model_path = f"{self.save_folder}/{type(model).__name__}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path=f"models/{type(model).__name__}", run_id=self.run_id)
        self.logger.info(f"Completed {type(model).__name__}")
        return evaluate_model(type(model).__name__, y_test, y_pred,self.run_id)

    def decision_tree(self,
        X_train: pd.DataFrame,
        X_test : pd.DataFrame,
        y_train : pd.DataFrame,
        y_test : pd.DataFrame):
        model = DecisionTreeClassifier()
        self.logger.info(f"Started {type(model).__name__} training")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if self.save_models:
            model_path = f"{self.save_folder}/{type(model).__name__}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path=f"models/{type(model).__name__}", run_id=self.run_id)
        self.logger.info(f"Completed {type(model).__name__}")
        return evaluate_model(type(model).__name__,y_test, y_pred,self.run_id)

    def knn(self,
        X_train: pd.DataFrame,
        X_test : pd.DataFrame,
        y_train : pd.DataFrame,
        y_test : pd.DataFrame,
        n_neighbours: int = 5):
        model = KNeighborsClassifier(n_neighbors= n_neighbours)
        self.logger.info(f"Started {type(model).__name__} training")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if self.save_models:
            model_path = f"{self.save_folder}/{type(model).__name__}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path=f"models/{type(model).__name__}", run_id=self.run_id)
        self.logger.info(f"Completed {type(model).__name__}")
        return evaluate_model(type(model).__name__,y_test, y_pred,self.run_id)

    def random_forest(self,
        X_train: pd.DataFrame,
        X_test : pd.DataFrame,
        y_train : pd.DataFrame,
        y_test : pd.DataFrame,
        n_estimators: int = 100):
        model = RandomForestClassifier(n_estimators=n_estimators)
        self.logger.info(f"Started {type(model).__name__} training")
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        if self.save_models:
            model_path = f"{self.save_folder}/{type(model).__name__}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path=f"models/{type(model).__name__}", run_id=self.run_id)
        self.logger.info(f"Completed {type(model).__name__}")
        return evaluate_model(type(model).__name__,y_test, y_pred,self.run_id)

    def xgboost(self,
        X_train: pd.DataFrame,
        X_test : pd.DataFrame,
        y_train : pd.DataFrame,
        y_test : pd.DataFrame,
        eval_metric: str = 'logloss',
        use_label_encoder: bool = True):
        model = XGBClassifier(
        use_label_encoder=use_label_encoder, 
        eval_metric=eval_metric)
        self.logger.info(f"Started {type(model).__name__} training")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if self.save_models:
            model_path = f"{self.save_folder}/{type(model).__name__}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path=f"models/{type(model).__name__}", run_id=self.run_id)
        self.logger.info(f"Completed {type(model).__name__}")
        return evaluate_model(type(model).__name__,y_test, y_pred,self.run_id)
    