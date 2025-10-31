import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Đọc dữ liệu nhập
df = pd.read_csv("data_cluster_hand.csv")

# Chỉ lấy cột x và y để phân cụm
df_cluster = df[['x', 'y']]

# Thiết lập số cụm k = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Thực hiện phân cụm
df['Cum'] = kmeans.fit_predict(df_cluster)

# Lấy tọa độ các điểm trung tâm
centroids = kmeans.cluster_centers_
print("Diem trung tam:")
print(centroids)

# Hiển thị kết quả
plt.figure(figsize=(8, 6))
plt.scatter(df['x'], df['y'], c=df['Cum'], cmap='viridis', label="Điểm dữ liệu")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Trung tâm cụm")

# Gán nhãn cho từng điểm dữ liệu
for i, txt in enumerate(df['Diem']):
    plt.annotate(txt, (df['x'][i], df['y'][i]), fontsize=12, xytext=(5, 5), textcoords='offset points')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Kết quả phân cụm K-Means')
plt.legend()
plt.grid(True)
plt.show()
