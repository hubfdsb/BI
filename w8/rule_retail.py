import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

# Đọc dữ liệu
df = pd.read_csv('rule_retail.csv', sep=',')
print(df.head())
print(df.shape)

# Tìm tập sản phẩm
items = (df['0'].unique())
print(items)

# Tien xu lý data, và chuyển đổi data 
encoded_vals = []
for index, row in df.iterrows(): 
    labels = {}
    uncommons = list(set(items) - set(row))
    commons = list(set(items).intersection(row))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)

ohe_df = pd.DataFrame(encoded_vals)
print(ohe_df)

# Tim tap muc
freq_items = apriori(ohe_df, min_support = 0.2, use_colnames = True, verbose = 1)
print(freq_items.head())

# Tìm luật kết hợp
tap_luat = association_rules(freq_items, metric = "confidence", min_threshold = 0.6)
print(tap_luat)