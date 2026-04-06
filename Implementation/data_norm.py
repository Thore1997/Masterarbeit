import scipy.io
import os
from sklearn.preprocessing import RobustScaler

# 1. Load the original data
input_path = os.path.join('Reproduction', 'Data', 'wine.mat')
mat_data = scipy.io.loadmat(input_path)

X = mat_data['X']
y = mat_data['y'] # Keep as original (usually 2D in .mat files)

# 2. Apply Robust Scaling
# This scales data based on the 25th and 75th percentiles
scaler = RobustScaler()
X_robust = scaler.fit_transform(X)

# 3. Create the new data dictionary
# We keep the same keys ('X' and 'y') so your other scripts don't break
new_mat_data = {
    'X': X_robust,
    'y': y
}

# 4. Save to a new .mat file
output_path = os.path.join('Reproduction', 'Data', 'wine_robust.mat')
scipy.io.savemat(output_path, new_mat_data)

print(f"Successfully created: {output_path}")