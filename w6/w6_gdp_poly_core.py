import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import polyfit as MoHinhDuDoan # import trực tiếp polyfit

# Đọc dữ liệu
dl = pd.read_csv('w6_data_gdp.csv')

# Lọc Việt Nam
dl_vn = dl[dl["Country Code"] == "VNM"]

# Chuẩn bị X, Y (numpy array)
X = dl_vn["Year"].to_numpy(dtype=float)
Y = dl_vn["Value"].to_numpy(dtype=float)

# --- MÔ HÌNH POLY BẬC 2: Y = a2*X^2 + a1*X + a0 ---
a2, a1, a0 = MoHinhDuDoan(X, Y, 2)

print("Hệ số mô hình (bậc 2):")
print(f"a2 = {a2}")
print(f"a1 = {a1}")
print(f"a0 = {a0}")

# Hàm dự đoán
def du_doan(x):
    return a2 * x**2 + a1 * x + a0

# Năm cần dự đoán
x_test = np.array([2024, 2025, 2026, 2027], dtype=float)
y_predict = du_doan(x_test)

print("Kết quả dự đoán:")
for x, yhat in zip(x_test, y_predict):
    print(f"  Năm {int(x)} -> {yhat:.2f}")

# Vẽ dữ liệu & đường cong fitted
plt.figure()
plt.scatter(X, Y, label="Dữ liệu thực tế")

#x_line = np.linspace(X.min(), X.max(), 200)
#y_line = du_doan(x_line)
#plt.plot(x_line, y_line, label="Hồi quy đa thức bậc 2")
plt.plot(x_test, y_predict, marker='o', color="red", linestyle='--', label="Điểm dự đoán")

plt.xlabel("Năm")
plt.ylabel("GDP")
plt.title("Việt Nam - Hồi quy đa thức bậc 2")
plt.legend()
plt.tight_layout()
plt.show()