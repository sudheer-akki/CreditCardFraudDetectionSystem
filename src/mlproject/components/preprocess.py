import os
import json
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from mlproject.logging import setup_logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import shapiro, normaltest

load_dotenv()
MLFLOW_EXPERIMENT= os.getenv('EXPERIMENT_NAME')
logger = setup_logger(pkgname=MLFLOW_EXPERIMENT)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class DataExplore:
    def __init__(self,
            data: pd.DataFrame, 
            save_report: bool = False):
        self.data = data
        self.results = {} 
        self.results['report'] = self.data_explore()
        if save_report:
            with open("report.json", "w") as outfile:
                json.dump(self.results['report'], outfile, cls=NpEncoder)

    def data_explore(self) -> Dict:
        report = {
            "Number of Rows": self.data.shape[0],
            "Number of Columns": self.data.shape[1],
            "Column Names": self.data.columns.to_list(),
            "Total missing values": self.data.isnull().sum().sum(),
            "Number of duplicate rows": self.data.duplicated().sum(),
            #"Number of duplicate columns": self.data.T.duplicated(keep=False).sum(),
            "Data Types": list(self.data.dtypes),
            "Categorical Column": list(self.data.select_dtypes(include=["object", "category"]).columns),
            "Numerical Column": list(self.data.select_dtypes(include=["number"]).columns)
        }
        return report
    
    def get_results(self):
        return self.results


class DataProcess:
    def __init__(self, 
                data: pd.DataFrame,
                remove_duplicate: bool = True,
                remove_null: bool = True,
                remove_missing: bool = True,
                remove_outliers: bool = False,
                handle_missing: str = "mean",
                categorical_missing: str = "mode",
                scale_features: bool = True,
                drop_column: List[str] =["Class", "Amount"]) -> None:
        self.data = data.copy()
        if remove_duplicate:
            self.data = self.remove_duplicate(data=self.data)
        if remove_null:
            self.data = self.remove_null(data=self.data)
        if remove_missing:
            self.data = self.handle_missing(data=self.data,
                            numerical_handle=handle_missing,
                            categorical_handle=categorical_missing)
        if scale_features:
            self.data = self.check_normality_and_scale(data=self.data, drop_column = drop_column)
        if remove_outliers:
            self.data = self.remove_outliers(data=self.data,drop_column = drop_column)
        logger.info("Completed preprocessing")

    def remove_duplicate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from the dataset.

        Args:
            data (pd.DataFrame): The dataset to process.

        Returns:
            pd.DataFrame: The dataset without duplicate rows.
        """
        initial_count = data.shape[0]
        data = data.drop_duplicates()
        removed_count = initial_count - data.shape[0]
        logger.info(f"Removed {removed_count} duplicate rows.")
        return data

    def remove_null(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows with any null values from the dataset.

        Args:
            data (pd.DataFrame): The dataset to process.

        Returns:
            pd.DataFrame: The dataset without null rows.
        """
        initial_count = data.shape[0]
        data = data.dropna()
        removed_count = initial_count - data.shape[0]
        logger.info(f"Removed {removed_count} rows with null values.")
        return data

    def handle_missing(self,
        data: pd.DataFrame,
        numerical_handle: Union[str] = ["mean", "median", "zero"],
        categorical_handle: str = "mode",
        columns = None) -> pd.DataFrame:
        """
        Arguments:
        categorical_handle (str): specify mode or any desired value from the column in string format"""
        if columns is None:
            columns = data.columns
        for column in columns:
            if data[column].dtype == int or data[column].dtype == float:
                if numerical_handle == "mean":
                    data[column].fillna(data[column].mean(), inplace= True)
                elif numerical_handle == "median":
                    data[column].fillna(data[column].median, inplace= True)
                elif numerical_handle == "zero":
                    data[column].fillna(0, inplace= True)
            if data[column].dtype == object:
                if categorical_handle == "mode":
                    data[column].fillna(data[column].mode()[0],inplace=True)
        logger.info(f"Filled Missing values with {numerical_handle}, {categorical_handle}")
        return data

    def remove_outliers(self, 
                data: pd.DataFrame,
                Q1: float = 0.25,
                Q3: float = 0.75,
                fence: Union[float, int] = 3,
                drop_column: Union[List[str],str,None] = None):
        outliers = []
        for col in data.select_dtypes(include="number").drop(columns=drop_column).columns:
            Q1 = data[col].quantile(Q1)
            Q3 = data[col].quantile(Q3)
            IQR = Q3 - Q1
            lower_bound = Q1 - fence * IQR
            upper_bound = Q3 + fence * IQR
            # Z-Score implementation
            threshold = 3
            #Step1: Calculated Mean
            mean = data[col].mean()
            # Step2: Squarred differences
            squared_diff = (data[col] - mean)**2
            # Step3: Divide Squarred diff with lenght of column
            variance = squared_diff.sum() / len(data[col])
            #Step4: Standard Deviation
            std = variance ** 0.5
            z_score = (data[col] - mean)/std
            outliers_zscore =  data[col] [z_score.abs() > threshold]
            #########################################################
            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            outliers_data = data[col][outlier_mask]
            data.loc[outlier_mask, col] = np.nan
            num_outliers = len(outliers_data)
            percent_outliers = (num_outliers / len(data[col])) * 100
            #if percent_outliers > 1.0:
            outliers.append([data[col].name, data[col].shape[0], 
                num_outliers,"num:",round(percent_outliers, 3), "%", len(outliers_zscore),round(lower_bound, 3),round(upper_bound,3)])
            return data
    
    def check_normality_and_scale(data: pd.DataFrame, alpha = 0.05, drop_column: Union[List[str],str,None] = None):
        """
        Parameters:
        data (pd.DataFrame): The dataset containing the features.
        column (List[str], str, or None): Specific column(s) to check and scale, or None to process all columns.
        
        Returns:
        pd.DataFrame: A DataFrame with scaled columns added for all features."""

        print("columnames:", data.columns)
        data = data.select_dtypes(include='number')

        print("list of numeric columns", data.columns)
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        if drop_column is None:
            columns = data.columns # Use all columns in the DataFrame
        elif isinstance(drop_column, str):
            columns = data[drop_column]
        else:
            columns = drop_column # Use the list of column names directly

        for col in columns:
            feature_data = data[col].dropna()
            # Normality tests
            shapiro_p = shapiro(feature_data)[1]  # Shapiro-Wilk test p-value
            dagostino_p = normaltest(feature_data)[1]  # D'Agostino's K-squared test p-value

            #print(f"Column: {col}")
            #print(f"  Shapiro-Wilk test p-value: {shapiro_p}")
            #print(f"  D'Agostino's K-squared test p-value: {dagostino_p}")

            if shapiro_p > alpha and dagostino_p > alpha:
                #print(f"  {col} is approximately normal. Applying StandardScaler...")
                # Apply StandardScaler
                scaled_data = standard_scaler.fit_transform(feature_data.values.reshape(-1, 1))
            else:
                #print(f"  {col} is not normally distributed. Applying MinMaxScaler...")
                # Apply MinMaxScaler
                scaled_data = minmax_scaler.fit_transform(feature_data.values.reshape(-1, 1)) 
            # Add scaled column to DataFrame
            #scaled_feature_name = f"{col}_scaled"
            data[col] = scaled_data
        return data
        
    def get_results(self):
        """Returns the consolidated results."""
        return self.data


def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.info(f"File not found! :{file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.info(f"The file is empty! : {file_path}")
        return None


def save_to_csv(data: pd.DataFrame, file_name: str):
    data.to_csv(file_name, index=False)

def plot_histo(data: pd.DataFrame):
    numeric_columns = data.select_dtypes(include='number').columns
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))  # Create a new figure for each plot
        sns.histplot(data[col], binwidth=20, kde=True)  # Histogram with optional KDE overlay
        plt.title(f'Histogram of {col}', fontsize=14)  # Title for the plot
        plt.xlabel(col, fontsize=12)  # X-axis label
        plt.ylabel('Frequency', fontsize=12)  # Y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)  # Add gridlines for better readability
        plt.show()

def check_outliers(data: pd.DataFrame):

    outliers = []

    for col in data.select_dtypes(include="number").drop(columns=["Amount", "Class"]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)

        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        #print(Q1,Q3,IQR, lower_bound, upper_bound)

        # Z-Score implementation
        threshold = 3
        #Step1: Calculated Mean
        mean = data[col].mean()
        # Step2: Squarred differences
        squared_diff = (data[col] - mean)**2
        # Step3: Divide Squarred diff with lenght of column
        variance = squared_diff.sum() / len(data[col])
        #Step4: Standard Deviation
        std = variance ** 0.5
        z_score = (data[col] - mean)/std
        outliers_zscore =  data[col] [z_score.abs() > threshold]
        #########################################################

        outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)

        outliers_data = data[col][outlier_mask]

        data.loc[outlier_mask, col] = np.nan

        num_outliers = len(outliers_data)
        percent_outliers = (num_outliers / len(data[col])) * 100
        #if percent_outliers > 1.0:
        outliers.append([data[col].name, data[col].shape[0], num_outliers,"num:",round(percent_outliers, 3), "%", len(outliers_zscore),round(lower_bound, 3),round(upper_bound,3)])
    return outliers, data