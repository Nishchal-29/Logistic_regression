'''Training a logistic regression classifier that will classify a plant with features
   like sepal length, sepal width, petal length and petal width as Iris-Setosa or not'''

#importing tools
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

#Loading dataset
iris = datasets.load_iris()
x=iris['data']
y=(iris['target']==0).astype(np.int64)

#Predicting for same dataset
clsfr = LogisticRegression()
clsfr.fit(x,y)
y_pred = clsfr.predict(x)
print(y_pred)

#Plotting the prediction with respect to petal width
X=iris['data'][:, 3:]
clsfr.fit(X,y)
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clsfr.predict_proba(x_new)
plt.plot(x_new, y_prob[:,1], "g-", label="Iris-Setosa")
plt.show()