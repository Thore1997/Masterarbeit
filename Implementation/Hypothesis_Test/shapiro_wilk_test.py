import scipy.io
import pandas as pd
from scipy import stats


# Load the .mat file
mat = scipy.io.loadmat('Reproduction/Data/wineori.mat')

print(mat.keys())
data_array = mat['X']
df = pd.DataFrame(data_array)

results = {}

for column in df.columns:
    stat, p_value = stats.shapiro(df[column])
    results[column] = {'Statistic': stat, 'p-value': p_value, 'Normal': p_value > 0.05}

shapiro_results = pd.DataFrame(results).T
print(shapiro_results)