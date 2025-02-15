import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from mlproject.components import load_data, DataExplore, DataProcess, DataVisualize

data = load_data(file_path="Dataset/creditcard.csv")
x = data.drop(columns=['Class'])
y = data['Class']

count_class = y.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(count_class.index, ['Class 0', 'Class 1'])
plt.show()


from imblearn.over_sampling import SMOTE

smote=SMOTE(sampling_strategy='minority') 
x,y=smote.fit_resample(x,y)
new_count = y.value_counts()
plt.bar(new_count.index, new_count.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('new_count Distribution')
plt.xticks(new_count.index, ['Class 0', 'Class 1'])
plt.show()
