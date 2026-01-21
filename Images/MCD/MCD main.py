import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

df = pd.read_csv('weight-height.csv')
persons = df.head(50)

df = pd.read_excel('deine_datei.xlsx', sheet_name='Tabellenblatt2')
df_person = pd.DataFrame(persons)


x = df_person['Height']
y = df_person['Weight']
mu = np.mean(df_person, axis=0)
covariance = np.cov(x, y)
inv_covariance = np.linalg.inv(covariance)


def calculate_mahalanobis(row, mu, inv_cov):
    diff = row - mu
    return np.sqrt(diff.dot(inv_cov).dot(diff))
df_person['mahalanobis'] = df.apply(lambda row: calculate_mahalanobis(row[['Height', 'Weight']], mu, inv_covariance), axis=1)


def draw_ellipse(ax, mu, cov, n_std, label, edgecolor='red'):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mu, width=width, height=height, angle=theta,
                  edgecolor=edgecolor, facecolor='none', lw=2, label=label)
    ax.add_patch(ell)


fig, ax = plt.subplots(figsize=(8, 6))

# Scatterplot: Punkte werden nach ihrer Mahalanobis-Distanz gefärbt
scatter = ax.scatter(df_person['Height'], df_person['Weight'], c=df_person['mahalanobis'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Mahalanobis Distanz')

# Ellipsen einzeichnen (z.B. für 95% Konfidenzintervall)
# Bei 2 Freiheitsgraden entspricht eine Mahalanobis-Distanz von ca. 2.45 dem 95% Bereich
dist_95 = np.sqrt(chi2.ppf(0.95, df=2))
draw_ellipse(ax, mu, covariance, n_std=dist_95, label='95% Konfidenz', edgecolor='red')

# Achsen und Beschriftung
ax.set_xlabel('Variable X')
ax.set_ylabel('Variable Y')
ax.set_title('Visualisierung der Mahalanobis-Distanz')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)



#plt.figure(figsize=[20,10])
#plt.scatter(persons['Weight'], persons['Height'])
plt.savefig('MCD.png')