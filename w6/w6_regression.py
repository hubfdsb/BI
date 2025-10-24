import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('w6_data.csv')

df= df.drop(columns=['Radio', 'Newspaper'], axis=1)
#print(df.head())
#df.info()
#df.corr()
#ax = df.plot.scatter(x='TV', y='Sales')
#plt.tight_layout()
#plt.show()

X = df['TV']
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.15)

#plt.scatter(X_train, y_train, alpha=0.7, label='Training data', color='r')
#plt.scatter(X_test, y_test, alpha=0.7, label='Testing data', color='g')
#plt.legend()
#plt.show()

hoiQuy = LinearRegression()
hoiQuy.fit(X_train.values.reshape(-1,1), y_train.values)

y_pred = hoiQuy.predict(X_test.values.reshape(-1,1))

plt.plot(X_test, y_pred, label='Hoi quy tuyen tinh', color='r')
plt.scatter(X_test, y_test, label='Du lieu thuc', color='g')
plt.legend()
#plt.savefig('w6_hinh', dpi=300, facecolor='white',bbox_inches='tight' )
plt.show()

#hoiQuy.predict(np.array([[150]]))[0]

from sklearn.metrics import mean_squared_error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
