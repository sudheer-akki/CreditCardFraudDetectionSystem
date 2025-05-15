import time
import numpy as np
from mlproject.components import load_data
from sklearn.metrics import accuracy_score, classification_report
import requests

data = load_data(file_path="Dataset/creditcard.csv")

for row_num, row in data.iterrows():
    payload = row.to_dict()
    num = np.random.randint(data.shape[0] - 10000, data.shape[0])

    x_test = data.iloc[num,1:-1].to_frame().T
    y_test = np.array([data.iloc[num,-1]])

    # Send to your FastAPI model endpoint
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    y_pred = np.array([1 if response.json()['fraud_prediction'] == True else 0])

    # Ground Truth
    y_test = np.array([payload['Class']])

    print("Row",row_num)

    if y_pred == 0:
        print("Pred: Not Fraud", f" GT: {'Not Fraud' if y_test[0] == 0 else 'Fraud'}")
    else:
        print("Pred: Fraud", f" GT: {'Not Fraud' if y_test[0] == 0 else 'Fraud'}")

    # Step 4: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    #print('Classification Report:')
    #print(classification_report(y_test, y_pred))
    time.sleep(5)
