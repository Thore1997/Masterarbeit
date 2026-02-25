import scipy.io
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

mat = scipy.io.loadmat('Reproduction/Data/wineori.mat')
data = mat['X']
num_features = data.shape[1]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))
axes = axes.flatten()


for i in range(num_features):

    (osm, osr), (slope, intercept, r) = stats.probplot(data[:, i], dist="norm", plot=axes[i])
    dots = axes[i].get_lines()[0]


    dots.set_markerfacecolor('lightgreen')
    dots.set_markeredgecolor('darkgreen')  # Darker edge makes them pop
    dots.set_markersize(4)  # Optional: make them slightly smaller for clarity

    axes[i].set_title(f"Feature {i + 1}")


for j in range(num_features, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()