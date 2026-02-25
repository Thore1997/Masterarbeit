import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
wine = sio.loadmat('../../Reproduction/Data/wineori.mat')
X = wine['X']
df = pd.DataFrame(X)


def plot_boxplots(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    sns.set_theme(style="whitegrid")

    n_cols = 13
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        data = df[col]

        sns.boxplot(y=data, ax=axes[i], color='lightgreen', showfliers=True)

        col_mean = data.mean()
        col_median = data.median()

        axes[i].axhline(col_mean, color='red', linestyle='--', linewidth=1)
        axes[i].axhline(col_median, color='green', linestyle='-', linewidth=1)
        axes[i].set_title(f'Feature {i+1}', fontsize=14)
        axes[i].set_ylabel('Value')

        if i % n_cols == 0:
            axes[i].set_ylabel('Value', fontsize=12, fontweight='bold')
        else:
            axes[i].set_ylabel('')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

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

        # Store in dictionary (Feature 1, Feature 2, etc.)
        outlier_report[f"Feature {col}"] = outliers

    return outlier_report


# Run the detection
all_outliers = get_outlier_indices(df)

# Print the results for each feature
for feature, indices in all_outliers.items():
    print(f"{feature}: {len(indices)} outliers found.")
    print(f"Indices: {indices}\n")