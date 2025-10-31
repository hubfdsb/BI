import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path

# 1. Đọc dữ liệu giao dịch
txt_path = Path("data_rule.txt")  # đường dẫn file đính kèm
with open(txt_path, "r", encoding="utf-8-sig") as f:
    lines = [ln.strip() for ln in f.readlines()]

transactions = []
for ln in lines:
    if not ln or ln.lstrip().startswith("#"):
        continue
    items = [it.strip() for it in ln.split(",") if it.strip()]
    # Khử trùng lặp trong cùng 1 giao dịch, giữ nguyên thứ tự xuất hiện
    items = list(dict.fromkeys(items))
    if items:
        transactions.append(items)

print(transactions)

# 2. Tiền xử lý dữ liệu: chuyển đổi sang định dạng one-hot encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print("Dữ liệu sau khi one-hot encoding:")
print(df)

# 3. Tìm frequent itemsets với thuật toán Apriori (min_support = 0.4)
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# 4. Trích xuất các luật kết hợp (min_confidence = 0.6)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print("\nAssociation Rules:")
print(rules)
