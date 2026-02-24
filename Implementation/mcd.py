import scipy.io
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score # Added AUPRC
from pyod.models.mcd import MCD

# 1. Load Data
file_path = os.path.join('..', 'Reproduction', 'Data', 'thyroid_processed_dataset.mat')

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    data = scipy.io.loadmat(file_path)
    X = data['X']
    y = data['Y'].ravel()

    # 2. Split (50/50 as per your preference)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # 3. Calculate Contamination
    contam = (y_train == 1).sum() / len(y_train)

    # 4. Initialize MCD
    clf = MCD(contamination=contam, random_state=42)

    # 5. Fit
    clf.fit(X_train)

    # 6. Predict and Score
    predictions = clf.predict(X_test)
    test_scores = clf.decision_function(X_test) # Raw anomaly scores

    # 7. Metrics
    f1_macro = f1_score(y_test, predictions, average='macro')
    auc_roc = roc_auc_score(y_test, test_scores)
    # Average Precision represents the Area Under the Precision-Recall Curve (AUPRC)
    auprc = average_precision_score(y_test, test_scores)

    print(f"--- MCD Results ---")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"ROC-AUC Score:    {auc_roc:.4f}")
    print(f"AUPRC Score:      {auprc:.4f}")