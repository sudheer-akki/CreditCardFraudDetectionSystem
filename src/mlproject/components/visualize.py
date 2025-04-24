import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import seaborn as sns
from mlproject.logging import setup_logger
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import shapiro, normaltest
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()
MLFLOW_EXPERIMENT= os.getenv('EXPERIMENT_NAME')
logger = setup_logger(pkgname=MLFLOW_EXPERIMENT)

class DataVisualize:
    def __init__(self, 
                data: pd.DataFrame,
                save_path: str,
                run_id: str,
                pie_chart: bool = True,
                histogram: bool = True,
                bar_chart: bool = True,
                heat_map: bool = True,
                corr_map: bool = True,
                scatter_plot: bool = True,
                show_outliers: bool = True
                ) -> None:
        self.data = data.fillna(0)
        self.savepath = save_path
        self.run_id = run_id
        if corr_map:
            self.plot_corr()
        if pie_chart:
           self.pie_chart()
        if scatter_plot:
            pass
            #self.scatter_plot()
        if bar_chart:
            self.bar_chart()
        if histogram:
            self.histogram()
        if heat_map:
            self.plot_heatmap()
        if show_outliers:
            self.show_outliers(data=self.data)

    def show_outliers(self, 
                data: pd.DataFrame,
                Q1: float = 0.25,
                Q3: float = 0.75,
                fence: Union[float, int] = 1.5,
                drop_column: Union[List[str],str,None] = ["Amount", "Class"]):
        #outliers = []
        for col in data.select_dtypes(include="number").drop(columns=drop_column).columns:
            Q1 = data[col].quantile(Q1)
            Q3 = data[col].quantile(Q3)
            IQR = Q3 - Q1
            lower_bound = Q1 - fence * IQR
            upper_bound = Q3 + fence * IQR
            # Z-Score implementation
            #threshold = 3
            #Step1: Calculated Mean
            #mean = data[col].mean()
            # Step2: Squarred differences
            #squared_diff = (data[col] - mean)**2
            # Step3: Divide Squarred diff with lenght of column
            #variance = squared_diff.sum() / len(data[col])
            #Step4: Standard Deviation
            #std = variance ** 0.5
            #z_score = (data[col] - mean)/std
            #outliers_zscore =  data[col] [z_score.abs() > threshold]
            #########################################################
            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            #outliers_data = data[col][outlier_mask]
            data.loc[outlier_mask, col] = np.nan
            #num_outliers = len(outliers_data)
            #percent_outliers = (num_outliers / len(data[col])) * 100
            #if percent_outliers > 1.0:
            # outliers.append([data[col].name, data[col].shape[0], 
            #     num_outliers,"num:",round(percent_outliers, 3), "%", 
            #     len(outliers_zscore),round(lower_bound, 3),round(upper_bound,3)])
            plt.figure(figsize=(10,8)) # widthx Height
            sns.boxplot(data=data.drop(columns=drop_column)) # ignore Nan values 
            plt.savefig(f"{self.savepath}/outliers.png")
            plt.close()

    def pie_chart(self):
        # Determine column types
        num_columns = self.data.select_dtypes(include=["number"]).columns
        cat_columns = self.data.select_dtypes(include=["object"]).columns
        bool_columns = self.data.select_dtypes(include=["bool"]).columns
        datetime_columns = self.data.select_dtypes(include=["datetime"]).columns
        other_columns = self.data.columns.difference(num_columns.union(cat_columns).union(bool_columns).union(datetime_columns))

        # Calculate percentages
        total_columns = len(self.data.columns)
        percentages = {
            "Numerical": len(num_columns) / total_columns * 100 if total_columns > 0 else 0,
            "Categorical": len(cat_columns) / total_columns * 100 if total_columns > 0 else 0,
            "Boolean": len(bool_columns) / total_columns * 100 if total_columns > 0 else 0,
            "Other": len(other_columns) / total_columns * 100 if total_columns > 0 else 0,
        }
        # Prepare data for pie chart
        data_for_pie = pd.Series(percentages)
        data_for_pie.plot(kind="pie", autopct='%1.1f%%', startangle=90, figsize=(8, 8))
        # Title and labels
        plt.title("Percentage of Column Types")
        plt.ylabel("")  # Hides the y-axis label
        # Save and close the plot
        plt.savefig(f"{self.savepath}/piechart.png")
        plt.close()
    
    def scatter_plot(self, column_x, column_y):
        sns.scatterplot(data=self.data, x=column_x, y=column_y)
        return plt.savefig(f"{self.savepath}/Scatterplot.png")
    
    def bar_chart(self, column: str = None):
        if column is None:
            logger.info(f"No Bar chart due to empty column input")
        else:
            if column in self.data.columns:
                sns.countplot(data=self.data, x = column)
            return plt.savefig(f"{self.savepath}/BarChart.png")

    def plot_corr(self, fig_size: tuple = (20,8),
                corr_with: str = "Class",
                drop_col: List[str] = [],
                ascending: bool = True):
        plt.figure(figsize=fig_size)
        correlations  = self.data.select_dtypes(include="number").drop(columns=drop_col).corr()[corr_with]
        correlations_sorted = correlations.sort_values(ascending=ascending)
        correlations_sorted.plot(kind='bar')
        plt.title(f"Correlation with {corr_with}")
        plt.xlabel("Features")
        plt.ylabel("Correlation")
        return plt.savefig(f"{self.savepath}/Correlation.png")
        
    def plot_heatmap(self, fig_size: tuple = (10,8), cmap: str = "Paired"):
        plt.figure(figsize=fig_size)
        sns.heatmap(self.data.corr(), annot=False, cmap=cmap)
        return plt.savefig(f"{self.savepath}/Heatmap.png")
    
    def histogram(self,column: str = None, bin_count: int = 20):
        if column is None:
            logger.info("Unable to plot histogram due to empty column")
        else:
            if column in self.data.columns:
                self.data[column].plot(kind='hist', bins = bin_count)
            return plt.savefig(f"{self.savepath}/Histogram.png")