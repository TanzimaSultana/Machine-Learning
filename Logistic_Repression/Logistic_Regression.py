#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:





# In[2]:


PATH = os.getcwd() + "\\"
MEN = 1
WOMEN = 0


# In[3]:


def load_train_data():
    
    data_path = PATH + "train_data.csv"
    data = pd.read_csv(data_path, header = None)
    
    return data


# In[4]:


def load_test_data():
    data_path = PATH + "test_data.csv"
    data = pd.read_csv(data_path, header = None)
    
    return data


# In[5]:


def feature_normalization(X):
    
    mu = np.mean(X)
    sigma = np.std(X)
    X = (X - mu)/sigma
    X_norm = X
    
    return X_norm


# In[6]:


def prediction(x_test, theta):
    m = len(x_test)
    ones = np.ones((m,1))
    x_test = np.column_stack((ones, x_test))  
    
    z = np.dot(x_test, theta)
    y = (sigmoid(z) >= 0.5).astype(int)
    
    return y


# In[7]:


## --- Sigmoid Function --- ##
def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h


# In[8]:


def process_data():
    # Load train data
    df1 = load_train_data()
    train_data = df1.to_numpy()
    
    #print(train_data)
    
    X = train_data[:,:-1].astype(int)
    Y = train_data[:,-1:]
    
    #print(X)
    #print(Y)
    
    Y1 = (Y == 'M').astype(int)
    #print(Y1)
    
    # Test data
    df2 = load_test_data()
    test_data = df2.to_numpy()
    x_test = test_data[:,:-1].astype(int)
    y_test = test_data[:,-1:]
    
    y1_test = (y_test == 'M').astype(int)
    
    return X, Y1, x_test, y1_test


# In[9]:


def plot_3D(X, Y, theta):
    
    # Add column of 1's to X
    m = len(X)
    ones = np.ones((m,1))
    X = np.column_stack((ones, X))
    #X = X[np.argsort(X[:, 1])]
    
    # create x,y
    xx, yy = np.meshgrid(X[:,1], X[:,2])
    
    # calculate corresponding zz
    # h = theta_0 + theta_1*x1 + theta_2*x2 + theta_3*x3
    # x3 = -(theta_0 + theta_1*x1 + theta_2*x2) / theta_3
    theta_0 = theta[0]
    theta_1 = theta[1]
    theta_2 = theta[2]
    theta_3 = theta[3]
    #zz = (-normal1[0]*xx - normal1[1]*yy - d1)*1./normal1[2]
    
    zz = -(theta_0 + theta_1*xx + theta_2*yy)*1./theta_3

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    
    # Scatter target ouput
    i = 0
    for x in X:
        x1 = x[1]
        y1 = x[2]
        z1 = x[3]
        target = Y[i]
        if(target == MEN):
            m = plt3d.scatter(x1, y1, z1, marker='o', color='green', s=20)
        else:
            w = plt3d.scatter(x1, y1, z1, marker='o', color='red', s=20)      
        i = i + 1
    
    plt.legend((m, w), ('Man', 'Woman'),scatterpoints=1, loc='upper left',fontsize=8)
    # Plot function
    plt3d.plot_surface(xx,yy,zz, color='blue')
    plt3d.set_xlabel('Height')
    plt3d.set_ylabel('Weight')
    plt3d.set_zlabel('Age')
    plt.show()


# In[10]:


class LogisticRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.no_iter = 50
        
        # 1. Initialize x and theta
        self.m = len(x)
        ones = np.ones((self.m,1))
        self.x = np.column_stack((ones, self.x))  

        no_feature = self.x.shape[1]
        self.theta = np.zeros((no_feature, 1))
        
        # Keep cost with each iteration
        self.J = np.ones((self.no_iter, 1))
        
        #print(self.x)
        #print("Initial theta ", self.theta)
        
        self.theta = self.gradient_descent(self.theta)
    
    ## --- Gradient Descent --- ##
    def gradient_descent(self, theta):
        alpha = 0.1
        
        #print("Initial theta ", theta)
        for i in range(0, self.no_iter):
            # 2. Sigmoid function
            z = np.dot(self.x, theta)
            h = sigmoid(z)
            #print("Hypothesis", h.shape)
            #print(h)
            
            # 3. Cost function
            #cost = -y * log(h) + (1 - y) * log(1 - h)
            c1 = np.dot(self.y.T, np.log(h))
            c2 = np.dot((1 - self.y).T, np.log(1 - h))
        
            cost = -(1 / self.m) * (c1 + c2)
            #print("Cost ", cost.shape)
            #print("i = ", i, ", Cost : ", cost)
            self.J[i] = cost
            
            # 4. Update theta
            theta = theta + alpha * np.dot(self.x.T, self.y - h)
            #print("Theta", theta.shape)
            #print(theta)
            
        #print(theta)
        #print(self.J)
        return theta


# In[11]:


def calculate_accuracy(y_test, y_pred):
    length = len(y_test)
    correct = y_test == y_pred
    accuracy = (np.sum(correct) / length)*100
    print("Accuracy : ", accuracy, "%")


# In[ ]:





# In[12]:


def main():
    
    # Data
    X, Y, x_test, y_test = process_data()
    
    # Logistic Regression
    X = feature_normalization(X)
    logistic_reg = LogisticRegression(X, Y)
    theta = logistic_reg.theta
    print("Theta")
    print(theta)
    
    # Prediction 
    #print(logistic_reg.theta)
    x_test = feature_normalization(x_test)
    y_pred = prediction(x_test, theta)
    #print("Prediction")
    #print(y_pred)
    
    # Calculate accuracy
    calculate_accuracy(y_test, y_pred)
    
    # Plot function
    plot_3D(X, Y, logistic_reg.theta)
    
    # Compare with SKLEARN
    #compare(X, Y, x_test)
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




