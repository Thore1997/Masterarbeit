import scipy.io
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from pyod.models.knn import KNN

# 1. Define the exact path
file_path = os.path.join('Reproduction', 'Data', 'wineori.mat')

if not os.path.exists(file_path):
    print(f"❌ Error: Could not find {file_path}")
else:
    # --- EVERYTHING HAPPENS INSIDE THIS BLOCK ---
    mat_data = scipy.io.loadmat(file_path)

    # Standardize keys: your file has 'X' and 'y'
    X = mat_data['X']
    y = mat_data['y'].ravel()

    # 2. Split Strategy: Train ONLY on Normals (Semi-Supervised)
    # This teaches the model what "Normal" looks like.
    X_normals = X[y == 0]
    X_anomalies = X[y == 1]

    # Use 70% of normals for training, 30% for testing
    X_train, X_test_norm = train_test_split(X_normals, test_size=0.5, random_state=42)

    # The Test set = the remaining 30% normals + ALL anomalies
    X_test = np.vstack([X_test_norm, X_anomalies])
    y_test = np.hstack([np.zeros(len(X_test_norm)), np.ones(len(X_anomalies))])

    # 3. Min-Max Normalization
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Initialize PyOD k-NN
    # Try n_neighbors=10 or 20; '3' is often too small for wineori benchmarks
    clf = KNN(n_neighbors=3, method='largest')

    # 5. Train (Only on the clean 'Normal' data)
    clf.fit(X_train)

    # 6. Predict
    predictions = clf.predict(X_test)  # Binary labels (0 or 1)
    test_scores = clf.decision_function(X_test)  # Raw anomaly scores

    # 7. Output Scores
    print(f"--- PyOD k-NN Evaluation (Semi-Supervised) ---")
    # Binary F1 is the standard for anomaly detection papers
    print(f"F1 Score (Binary): {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC Score:     {roc_auc_score(y_test, test_scores):.4f}")
    print(f"AUPRC Score:       {average_precision_score(y_test, test_scores):.4f}")