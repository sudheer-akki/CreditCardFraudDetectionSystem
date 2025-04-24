import numpy as np
from mlproject.components import load_data
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the .pkl file
with open('weights/RandomForestClassifier.pkl', 'rb') as file:
    model = pickle.load(file)


data = load_data(file_path="Dataset/creditcard.csv")

for i in range(100):
    num = np.random.randint(data.shape[0] - 10000, data.shape[0])

    x_test = data.iloc[num,1:-1].to_frame().T
    y_test = np.array([data.iloc[num,-1]])

    # Step 3: Make predictions
    y_pred = model.predict(x_test)

    if y_pred == 0:
        print("Pred: Not Fraud", f" GT: {'Not Fraud' if y_test[0] == 0 else 'Fraud'}")
    else:
        print("Pred: Fraud", f" GT: {'Not Fraud' if y_test[0] == 0 else 'Fraud'}")

    # Step 4: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))