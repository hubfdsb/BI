# Import thư viện PuLP
import pulp

# --- Phần 1: Thiết lập Mô hình ---

# 1. Khởi tạo mô hình
# Chúng ta đặt tên cho mô hình và xác định mục tiêu là Tối đa hóa (LpMaximize)
model = pulp.LpProblem("Toi_uu_Ngan_sach_Marketing", pulp.LpMaximize)

# 2. Định nghĩa các biến quyết định (Decision Variables)
# Chúng ta muốn tìm số lượng leads từ mỗi kênh.
# Vì "leads" là số người, chúng ta nên đặt là biến số nguyên (cat='Integer')
# lowBound=0 đảm bảo ràng buộc không âm
# upBound là giới hạn tối đa của mỗi kênh

L_G = pulp.LpVariable("Leads_Google", lowBound=0, upBound=1500, cat='Integer')
L_L = pulp.LpVariable("Leads_LinkedIn", lowBound=0, upBound=1000, cat='Integer')
L_W = pulp.LpVariable("Leads_Webinar", lowBound=0, upBound=2000, cat='Integer')
L_E = pulp.LpVariable("Leads_Email", lowBound=0, upBound=800, cat='Integer')

# 3. Định nghĩa Hàm mục tiêu (Objective Function)
# Mục tiêu là Tối đa hóa Tổng số Leads
model += L_G + L_L + L_W + L_E, "Tong so Leads"

# 4. Định nghĩa các Ràng buộc (Constraints)
# Để dễ đọc, chúng ta có thể dùng dấu _ cho các số lớn
# Ràng buộc 1: Tổng ngân sách
model += (L_G * 800_000 + L_L * 1_200_000 + L_W * 500_000 + L_E * 300_000) <= 2_000_000_000, "Rang buoc Ngan sach"

# Ràng buộc 3 (a): Yêu cầu chiến lược - Google Ads ít nhất 30% tổng ngân sách
# 30% của 2 tỷ là 600 triệu
model += (L_G * 800_000) >= 600_000_000, "Rang buoc Chien luoc Google"

# Ràng buộc 3 (b): Yêu cầu chiến lược - Chi tiêu LinkedIn <= Chi tiêu Google
model += (L_L * 1_200_000) <= (L_G * 800_000), "Rang buoc Chien luoc LinkedIn_vs_Google"

# Ràng buộc 2 (Giới hạn kênh) & 4 (Không âm) đã được định nghĩa
# khi chúng ta tạo biến (upBound và lowBound)

# --- Phần 2: Giải bài toán ---

# In ra mô hình để kiểm tra (tùy chọn)
# print(model)

# Giải mô hình
model.solve()

# --- Phần 3: In kết quả ---

# In trạng thái của giải pháp
print("Trạng thái:", pulp.LpStatus[model.status])
print("-" * 30)

# In kết quả tối ưu của các biến quyết định
print("Kết quả phân bổ Leads tối ưu:")
for v in model.variables():
    print(f"{v.name} = {v.varValue}")

print("-" * 30)

# In giá trị tối ưu của hàm mục tiêu
tong_leads_toi_uu = pulp.value(model.objective)
print(f"Tổng số Leads tối đa có thể đạt được: {tong_leads_toi_uu}")

# Tính toán tổng chi phí thực tế
tong_chi_phi = (L_G.varValue * 800_000 + 
                L_L.varValue * 1_200_000 + 
                L_W.varValue * 500_000 + 
                L_E.varValue * 300_000)
print(f"Tổng chi phí thực tế: {tong_chi_phi:,.0f} VNĐ")