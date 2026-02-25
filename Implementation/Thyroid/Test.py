import pandas as pd
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler

# --- 1. Data Loading ---
file_path = '../../Reproduction/Data/Thyroid/thyroid0387.data'
df = pd.read_csv(file_path, sep=',', header=None, na_values='?')

# Clean data
df_final = df.dropna(subset=[19]).copy()

# --- 2. Fill missing values with median ---
target_columns = [17, 19, 21, 23, 25, 0] # Standard order for thyroid.mat features
df_final[target_columns] = df_final[target_columns].fillna(df_final[target_columns].median())

# --- 3. Target Selection (Y) ---
# Identify outliers based on columns 9 and 10 ('t' in either means abnormal)
Y_raw = df_final.iloc[:, [9, 10]].values
Y_binary = np.where(Y_raw == 't', 1, 0)
Y_full = Y_binary.max(axis=1) # Boolean array of all labels

# --- 4. Random Undersampling to 2.5% Contamination ---
contamination = 0.025
inlier_indices = np.where(Y_full == 0)[0]
outlier_indices = np.where(Y_full == 1)[0]

n_inliers = len(inlier_indices)
n_outliers_target = int((contamination * n_inliers) / (1 - contamination))

# Randomly pick the target number of outliers
np.random.seed(42) # For reproducibility
selected_outlier_indices = np.random.choice(outlier_indices, n_outliers_target, replace=False)

# Combine indices
final_indices = np.concatenate([inlier_indices, selected_outlier_indices])
np.random.shuffle(final_indices) # Shuffle so they aren't all at the end

# Filter the dataframe
df_sampled = df_final.iloc[final_indices].copy()

# --- 5. Log-Transform and Scaling (Crucial for "Clumping") ---
# Column 17 is typically TSH, which is extremely skewed.
df_sampled[17] = np.log1p(df_sampled[17])

# Select features for X
X = df_sampled[target_columns].values

# Scale the data (Z-Score Normalization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Update Y to match the sampled rows
Y = Y_full[final_indices].reshape(-1, 1)

# --- 6. Save to .mat ---
data_to_save = {
    "X": X,
    "Y": Y.astype(float) # ODDS files usually use float/double for Y
}

scipy.io.savemat("../../Reproduction/Data/Thyroid/Test.mat", data_to_save)

# --- 7. Kontrolle ---
print(f"Datei erfolgreich erstellt!")
print(f"Shapes: X={X.shape}, Y={Y.shape}")
unique, counts = np.unique(Y, return_counts=True)
actual_ratio = (counts[1] / len(Y)) * 100
print(f"Verteilung in Y: {dict(zip(unique, counts))}")
print(f"Aktuelle Kontamination: {actual_ratio:.2f}%")