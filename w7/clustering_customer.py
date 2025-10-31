import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv("data_cluster_CustomerBehavior.csv")

data = dataset[['Annual_Income', 'Spending_Score']]

# Thiet lap so cum
kmeans = KMeans(n_clusters=7, random_state=42)

# Thuc hien
dataset['Cum'] = kmeans.fit_predict(data)

dataset.to_csv("cluster_ketqua.csv")

# Hien thi
plt.figure(figsize=(8,6))
plt.scatter(dataset['Annual_Income'], dataset['Spending_Score'], c=dataset['Cum'], cmap='viridis')
plt.xlabel('Thu nhap')
plt.ylabel('Diem tieu dung')
plt.title('Ket qua phan cum')
plt.show()