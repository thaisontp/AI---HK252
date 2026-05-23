import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')

df.loc[
    df['fruits'] == 'chanh 2',
    ['Length (cm)', 'Width (cm)', 'Heigth (cm)', 'Perimeter']
] /= 10
X = df[['Length (cm)', 'Width (cm)']]
y = df['fruits']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

k = int(len(X_train) ** 0.5)
print("k =", k)

model = KNeighborsClassifier(
    n_neighbors=k,
    metric='euclidean'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy =", accuracy)

sample = [[7.4, 7.3]]
sample = scaler.transform(sample)
prediction = model.predict(sample)
print("\nPredicted Fruit =", prediction[0])


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Convert labels to numbers
label_map = {
    'cam': 0,
    'chanh': 1,
    'chanh 2': 2
}
Z = np.array([label_map[i] for i in Z])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3)
for fruit in y.unique():
    idx = y == fruit
    plt.scatter(
        X[idx, 0],
        X[idx, 1],
        label=fruit,
        s=80
    )
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('KNN Multiclass Classification')
plt.legend()
plt.show()