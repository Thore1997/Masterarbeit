import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Neu für den Vergleich

# Load data
wine = sio.loadmat('Reproduction/Data/wineori.mat')
X = wine['X']
df = pd.DataFrame(X)

# --- NEU: Ground Truth laden ---
# In vielen .mat Datensätzen sind Ausreißer in 'y' markiert (oft 1 für Outlier, 0 für Normal)
y_true = wine['y'].flatten()
# Wir extrahieren die Indizes, an denen die Ground Truth einen Ausreißer meldet
true_outlier_indices = np.where(y_true == 1)[0].tolist()


# -------------------------------

#def plot_boxplots(df):
    # ... (dein restlicher Code bleibt identisch)
#    numeric_cols = df.select_dtypes(include=['number']).columns
#    sns.set_theme(style="whitegrid")
#    n_cols = 13
#    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
#    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
#    axes = axes.flatten()
#    for i, col in enumerate(numeric_cols):
#        data = df[col]
#        sns.boxplot(y=data, ax=axes[i], color='lightgreen', showfliers=True)
#        col_mean = data.mean()
#        col_median = data.median()
#        axes[i].axhline(col_mean, color='red', linestyle='--', linewidth=1)
#        axes[i].axhline(col_median, color='green', linestyle='-', linewidth=1)
#        axes[i].set_title(f'Feature {i + 1}', fontsize=14)
#        axes[i].set_ylabel('Value')
#        if i % n_cols == 0:
#            axes[i].set_ylabel('Value', fontsize=12, fontweight='bold')
#        else:
#            axes[i].set_ylabel('')
#    for j in range(i + 1, len(axes)):
#        fig.delaxes(axes[j])
#    plt.tight_layout()
#    plt.show()


#plot_boxplots(df)


def get_outlier_indices(df):
    outlier_report = {}
    for i, col in enumerate(df.columns):
        data = df[col]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(data < lower_bound) | (data > upper_bound)].index.tolist()
        outlier_report[f"Feature {col}"] = outliers
    return outlier_report


# Run the detection
all_outliers = get_outlier_indices(df)

# --- NEU: Abgleich mit Ground Truth ---
print(f"--- GROUND TRUTH VERGLEICH ---")
print(f"Echte Ausreißer laut Ground Truth (Indizes): {true_outlier_indices}")
print(f"Anzahl echter Ausreißer: {len(true_outlier_indices)}\n")

for feature, indices in all_outliers.items():
    # Schnittmenge finden: Welche vom Algorithmus gefundenen Indizes sind auch in Ground Truth?
    true_positives = set(indices).intersection(set(true_outlier_indices))

    print(f"{feature}: {len(indices)} statistische Ausreißer gefunden.")
    print(f"-> Davon sind {len(true_positives)} echte Ausreißer (Treffer).")

    # Optional: Falsch-Positive (statistisch ja, aber laut GT nein)
    false_positives = set(indices) - set(true_outlier_indices)
    print(f"-> {len(false_positives)} statistische Abweichungen, die keine echten Outlier sind.")
    print(f"Instanzen (Treffer): {list(true_positives)}\n")