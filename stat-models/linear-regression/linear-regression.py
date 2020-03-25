# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:
class LinearRegression:
    """Multiple linear regression model
    """

    def __init__(self, lrp=0.00001, max_iter=10000):
        self.lrp = lrp
        self.max_iter = max_iter

    def fit(self, X, y):
        n = X.shape[0]
        p = X.shape[1]
        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        # initialize all coefficients as 0
        self.coefs = np.zeros((p + 1, 1))
        for _ in range(self.max_iter):
            h = np.matmul(X, self.coefs)
            residuals = h - y
            self.coefs -= self.lrp/n*np.matmul(X.T, residuals)
        return self

    def predict(self, X):
        pass


# Tests and execution

# In[ ]:
n = 20
x = np.linspace(0, 100, n)
x = x[:, np.newaxis]
y = 2*x + 50 + np.random.randn(n, 1)*1
print(x)

# In[ ]:
lr = LinearRegression()
lr.fit(X=x, y=y)
print(lr.coefs)

# In[ ]:
fig, ax = plt.subplots()
sns.scatterplot(x=x[:,0], y=y[:,0], ax=ax)
ax.plot(x[:,0], lr.coefs[0] + lr.coefs[1]*x[:,0])
plt.show()
