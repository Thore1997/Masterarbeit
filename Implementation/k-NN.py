import scipy.io
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from pyod.models.knn import KNN

# 1. Load the data
file_path = os.path.join('..', 'Reproduction', 'Data', 'wineori.mat')

if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}")
else:
    mat_data = scipy.io.loadmat(file_path)
    X = mat_data['X']
    y = mat_data['y'].ravel()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Min-Max Normalization
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Initialize PyOD k-NN
    contam = (y_train == 1).sum() / len(y_train)
    clf = KNN(n_neighbors=3, contamination=contam, method='largest')

    # 5. Train
    clf.fit(X_train)

    # 6. Predict
    predictions = clf.predict(X_test)
    test_scores = clf.decision_function(X_test)

    # 7. Output Scores
    print(f"--- PyOD k-NN Evaluation (Normalized) ---")
    print(f"F1 Score (Macro): {f1_score(y_test, predictions, average='macro'):.4f}")
    print(f"ROC-AUC Score:    {roc_auc_score(y_test, test_scores):.4f}")
    print(f"AUPRC Score:      {average_precision_score(y_test, test_scores):.4f}")