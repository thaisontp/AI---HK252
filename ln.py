from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data.csv')

data_clean = data.drop(columns=['id','Perimeter','fruits'])
#print(data_clean)

data_volume = data_clean
print(data_volume)

volume = []
for i in range(len(data_volume)):
    v = (data_volume.loc[i, 'Length (cm)'] * data_volume.loc[i, 'Width (cm)'] * data_volume.loc[i, 'Heigth (cm)'])
    volume.append(v)

# gán vào DataFrame nếu cần
data_volume['Volume'] = volume
#print(data_volume)

data_new = data_volume.drop(columns=['Length (cm)','Width (cm)','Heigth (cm)'])
print(data_new)

# Chọn biến
x = data_new.drop(columns=['Weight (g)'])   # input
y = data_new['Weight (g)']      # target

# Khởi tạo model
model = LinearRegression()

# Train
model.fit(x, y)

# Hệ số
beta_0 = model.intercept_
beta_1 = model.coef_[0]

print("Intercept (beta_0):", beta_0)
print("Slope (beta_1):", beta_1)

# Dự đoán
y_pred = model.predict(x)

# Vẽ
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel('Volume')
plt.ylabel('Weight (g)')
plt.title('Linear Regression: Weight vs Volume')
plt.show()