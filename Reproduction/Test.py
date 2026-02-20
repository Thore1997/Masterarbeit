import pandas as pd
import numpy as np
import scipy.io

# --- 1. Data Loading ---
file_path = 'data/thyroid0387.data'
df = pd.read_csv(file_path, sep=',', header=None, na_values='?')

# Clean data (dropping rows where column 19 is NaN)
df_final = df.dropna(subset=[19]).copy()

# --- 2. Correct Selection ---
# iloc[rows, columns] -> [:, [indices]] means "All rows, these specific columns"
X = df_final.iloc[:, [0, 17, 19, 21, 23, 25]].values
Y = df_final.iloc[:, [9, 10]].values

# --- 3. Optional: Convert 'f'/'t' to 0/1 ---
# Since you mentioned Y is binary, MATLAB usually prefers numbers over strings
Y = np.where(Y == 't', 1, 0)

# --- 4. Save to .mat ---
data_to_save = {
    "X": X,
    "Y": Y
}

scipy.io.savemat("my_dataset.mat", data_to_save)

print(f"File created! Shapes: X={X.shape}, Y={Y.shape}")