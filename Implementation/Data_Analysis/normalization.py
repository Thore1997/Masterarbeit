import pandas as pd
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler

mat = scipy.io.loadmat('Reproduction/Data/wineori.mat')

print(f"Original keys: {mat.keys()}")


X = mat['X']
if 'Y' in mat:
    Y = mat['Y']
elif 'y' in mat:
    Y = mat['y']
else:
    # If there is truly no Y, your Data_Loader logic will need
    # to be told how to handle unlabeled data.
    print("Warning: No label key found in original file!")
    Y = np.zeros((X.shape[0], 1)) # Creating dummy labels as a fallback

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

data_to_save = {
    'X': X_norm,
    'Y': Y
}

scipy.io.savemat('Reproduction/Data/wine_normalized.mat', data_to_save)
print("Successfully saved wine_normalized.mat with both X and Y keys.")