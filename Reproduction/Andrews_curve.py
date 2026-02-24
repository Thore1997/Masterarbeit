import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import radviz
from sklearn.preprocessing import MinMaxScaler

# 1. Daten laden
data = scipy.io.loadmat('Data/thyroid.mat')
X = data['X']
y = data['y'].flatten()

# 2. WICHTIG für RadViz: Normalisierung
# RadViz funktioniert am besten, wenn alle Werte zwischen 0 und 1 liegen.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. DataFrame erstellen
df = pd.DataFrame(X_scaled, columns=[f'Dim_{i+1}' for i in range(X.shape[1])])
df['label'] = y

# Sortieren für die Farbreihenfolge (Inlier zuerst, Outlier zuletzt)
df = df.sort_values(by='label')

# 4. Plotten
plt.figure(figsize=(10, 10))

# Farbschema definieren (Grau für Inlier, Rot für Outlier)
colors = ['lightgrey', 'red']

# RadViz zeichnen
# alpha=0.3 hilft bei der hohen Anzahl an Instanzen (6500)
radviz(df, 'label', color=colors, alpha=0.3, s=10)

plt.title('RadViz: Inlier (grau) vs. Outlier (rot)')
plt.legend(['Inlier', 'Outlier'], loc='upper right', bbox_to_anchor=(1.2, 1))

plt.show()