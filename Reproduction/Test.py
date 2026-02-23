import pandas as pd
import numpy as np
import scipy.io

# --- 1. Data Loading ---
file_path = 'data/thyroid0387.data'
df = pd.read_csv(file_path, sep=',', header=None, na_values='?')

# Clean data (dropping rows where column 19 is NaN)
df_final = df.dropna(subset=[19]).copy()

# --- 2. Fehlende Werte mit dem Median füllen ---
# Wir füllen die Lücken in den relevanten Spalten direkt im DataFrame
target_columns = [0, 17, 19, 21, 23, 25]
df_final[target_columns] = df_final[target_columns].fillna(df_final[target_columns].median())

# --- 3. Feature Selection (X) ---
X = df_final.iloc[:, target_columns].values

# --- 4. Target Selection & Merge (Y) ---
# Zuerst 't'/'f' in 0 und 1 umwandeln für beide Spalten (9 und 10)
Y_raw = df_final.iloc[:, [9, 10]].values
Y_binary = np.where(Y_raw == 't', 1, 0)

# Logisches ODER: Wenn in einer der beiden Spalten eine 1 steht, wird das Resultat 1.
# .max(axis=1) pickt die 1 heraus, falls vorhanden, sonst bleibt es 0.
# .reshape(-1, 1) sorgt dafür, dass es eine Spalte bleibt (n x 1 Matrix).
Y = Y_binary.max(axis=1).reshape(-1, 1)

# --- 5. Save to .mat ---
data_to_save = {
    "X": X,
    "Y": Y
}

scipy.io.savemat("my_dataset.mat", data_to_save)

# --- 6. Kontrolle ---
print(f"Datei erfolgreich erstellt!")
print(f"Shapes: X={X.shape}, Y={Y.shape}")
print(f"NaNs in X: {np.isnan(X).sum()}")

# Kurze Übersicht der Verteilung (Wieviele Kranke vs. Gesunde?)
unique, counts = np.unique(Y, return_counts=True)
print(f"Verteilung in Y: {dict(zip(unique, counts))}")