import pandas as pd
import numpy as np
import scipy.io

file_path = 'data/thyroid0387.data'
df = pd.read_csv(file_path, sep=',', header=None, na_values='?')


df_final = df.dropna(subset=[19]).copy()


target_columns = [0, 17, 19, 21, 23, 25]
df_final[target_columns] = df_final[target_columns].fillna(df_final[target_columns].median())

X = df_final.iloc[:, target_columns].values

Y_raw = df_final.iloc[:, [9, 10]].values
Y_binary = np.where(Y_raw == 't', 1, 0)

Y = Y_binary.max(axis=1).reshape(-1, 1)

data_to_save = {
    "X": X,
    "Y": Y
}

scipy.io.savemat("test_data.mat", data_to_save)

# --- 6. Kontrolle ---
print(f"Datei erfolgreich erstellt!")
print(f"Shapes: X={X.shape}, Y={Y.shape}")
print(f"NaNs in X: {np.isnan(X).sum()}")

# Kurze Übersicht der Verteilung (Wieviele Kranke vs. Gesunde?)
unique, counts = np.unique(Y, return_counts=True)
print(f"Verteilung in Y: {dict(zip(unique, counts))}")