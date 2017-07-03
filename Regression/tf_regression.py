import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

X = diabetes_X[:-20]
X=np.column_stack((np.ones((X.shape[0],1)), X))
y = diabetes.target[:-20]#.reshape(-1,1)
plt.scatter(X[:,1], y,  color='black')
m,n=X.shape
theta = np.zeros((X.shape[1],1))
num_iters=1000
alpha=0.1

for i in range(num_iters):
    theta = theta -alpha/2/m*np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
