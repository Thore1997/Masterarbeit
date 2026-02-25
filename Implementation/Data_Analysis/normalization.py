import pandas as pd
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler

mat = scipy.io.loadmat('../../Reproduction/Data/wineori.mat')
data = mat['X']

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(data)
data_to_save = {'X': X_norm}
scipy.io.savemat('../../Reproduction/Data/wine_normalized.mat', data_to_save)

