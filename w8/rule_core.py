import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Tạo tập dữ liệu giao dịch mẫu mở rộng
transactions = [
    ['Bread', 'Butter', 'Milk'],
    ['Bread', 'Diaper', 'Beer', 'Eggs'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diaper', 'Beer'],
    ['Bread', 'Butter', 'Diaper', 'Milk', 'Eggs'],
    ['Eggs', 'Milk', 'Cheese'],
    ['Bread', 'Cheese', 'Butter'],
    ['Diaper', 'Beer', 'Cheese', 'Cola'],
    ['Bread', 'Butter', 'Cheese', 'Milk'],
    ['Bread', 'Milk', 'Butter', 'Diaper'],
    ['Beer', 'Eggs', 'Cola'],
    ['Bread', 'Butter', 'Eggs'],
    ['Bread', 'Butter', 'Milk', 'Cheese'],
    ['Diaper', 'Milk', 'Cheese', 'Cola'],
    ['Milk', 'Cheese', 'Butter']
]

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
