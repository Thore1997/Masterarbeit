import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Processing ---
file_path = 'data/thyroid0387.data'
df = pd.read_csv(file_path, sep=',', header=None, na_values='?')

column_mapping = {0: 'Age', 17: 'TSH', 19: 'T3', 21: 'TT4', 23: 'T4U', 25: 'FTI'}

# Select and rename
df_selected = df[list(column_mapping.keys())].rename(columns=column_mapping)

# Convert all selected columns to numeric
df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

# --- 2. Handling Missing Values ---
# First: Remove rows where T3 is missing (per your requirement)
df_final = df_selected.dropna(subset=['T3']).copy()

# Second: Fill remaining missing values in other dimensions with their respective medians
# We calculate medians from the dataframe after the T3 drop to keep them representative
df_final = df_final.fillna(df_final.median())

features = ['Age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

# Check instances and verification
print(f"Number of instances in df_final: {len(df_final)}")
print("\nMissing values remaining:")
print(df_final.isnull().sum())

# --- 3. Visualization: Individual Scales without dots ---
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(features):
    # showfliers=False hides the dots but keeps the data in the background
    sns.boxplot(y=df_final[col], ax=axes[i], color='skyblue', width=0.4, showfliers=False)

    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value')
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()