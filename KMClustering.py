import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

data = pd.read_csv('data.csv')

data.rename(columns={'Heigth (cm)': 'Height (cm)'}, inplace=True)
data.loc[data['fruits'] == 'chanh 2', ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Perimeter']] /= 10

data['Volume'] = (
    data['Length (cm)'] *
    data['Width (cm)'] *
    data['Height (cm)']
)

X = data[['Weight (g)', 'Volume']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)
kmeans.fit(X_scaled)
labels = kmeans.labels_

x1_min, x1_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
x2_min, x2_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

xx1, xx2 = np.meshgrid(
    np.arange(x1_min, x1_max, 0.02),
    np.arange(x2_min, x2_max, 0.02)
)

# Predict cluster for every point
Z = kmeans.predict(
    np.array([xx1.ravel(), xx2.ravel()]).T
)
Z = Z.reshape(xx1.shape)

# Background colors
cmap_light = ListedColormap([
    '#FFCCCC',
    '#CCFFCC',
    '#CCCCFF'
])

colors = ['red', 'green', 'blue']

plt.figure(figsize=(10, 6))
plt.contourf(
    xx1,
    xx2,
    Z,
    alpha=0.3,
    cmap=cmap_light
)

for i in range(3):

    plt.scatter(
        X_scaled[labels == i, 0],
        X_scaled[labels == i, 1],
        color=colors[i],
        edgecolors='black',
        label=f'Cluster {i}'
    )

plt.xlabel('Weight (Standardized)')
plt.ylabel('Volume (Standardized)')
plt.title('K-Means Clustering Decision Regions')
plt.legend()
plt.grid(True)
plt.show()