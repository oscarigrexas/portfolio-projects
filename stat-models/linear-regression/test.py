import numpy as np
from sklearn.linear_model import LinearRegression

x = np.linspace(0, 100, 100)
x = x[:, np.newaxis]
y = 0.33*x + 3 + np.random.randn(100, 1)
lr = LinearRegression()
lr.fit(X=x, y=y)
print(lr.coef_)
