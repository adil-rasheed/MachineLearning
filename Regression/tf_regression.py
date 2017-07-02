# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:34:21 2017

@author: arash
"""
import numpy as np
import tensorflow as tf
def featureNormalize(X):
    X_norm = X
    mu    = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    for i in range(X.shape[1]):
        mu[:,i] = np.mean(X[:,i])
        sigma[:,i] = np.std(X[:,i])
        X_norm[:,i] = (X[:,i] - float(mu[:,i]))/float(sigma[:,i])
    return X_norm, mu, sigma

data = np.loadtxt('data.txt', delimiter=",")
X = data[:,:2]
y = data[:,2]
y=y.reshape(-1,1)
X_norm, mu, sigma = featureNormalize(X)
mx,nx = X.shape
my,ny = y.shape
x_ph = tf.placeholder(tf.float32, shape = [None, nx])
y_ph = tf.placeholder(tf.float32, shape = [None, ny])

W = tf.Variable(tf.random_normal([nx,1]), name="Weight")
b = tf.Variable(tf.random_normal([ny,1]), name="Intercept")

hypothesis = tf.matmul(x_ph, W) + b
init = tf.global_variables_initializer()
Loss = tf.reduce_mean(tf.square(y_ph - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(Loss)
sess = tf.Session()
# Initialize all the variables
sess.run(init)
#sess.run(train,feed_dict = {x_ph:X, y_ph:y})
cost_val, hy_val, W_val, b_val, _ = sess.run([Loss, hypothesis, W, b, train],feed_dict = {x_ph:X, y_ph:y})
house_norm_padded = np.array([1650, 3])
#for epochs in range(40000):
#    cost_val, hy_val, W_val, b_val, _ = sess.run([Loss, hypothesis, W, b, train],feed_dict = {x_ph:X_norm, y_ph:y})
#    
#    if epochs % 100 == 0:
#        print(epochs, cost_val, W_val, b_val)
        
price2 = np.array(house_norm_padded).dot(W_val)+b_val
print("Predicted price of a 1650 sq-ft, 3 br house (using normal equation):", price2)