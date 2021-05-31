#!/usr/bin/env python
# coding: utf-8

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
MEN = 0
WOMEN = 1
NO_OF_CLASS = 2


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


def process_data():
    # Load train data
    df1 = load_train_data()
    train_data = df1.to_numpy()
    
    #print(train_data)
    
    X = train_data[:,:-1].astype(int)
    Y = train_data[:,-1:]
    
    #print(X)
    #print(Y)
    X1 = feature_normalization(X)
    Y1 = (Y == 'M').astype(int)
    #print(Y1)
    
    # Test data
    df2 = load_test_data()
    test_data = df2.to_numpy()
    x_test = test_data[:,:-1].astype(int)
    y_test = test_data[:,-1:]
    
    x1_test = feature_normalization(x_test)
    y1_test = (y_test == 'M').astype(int)
    
    return X1, Y1, x1_test, y1_test


# In[7]:


def add_ones_column(x):
    m = len(x)
    ones = np.ones((m,1))
    x = np.column_stack((ones, x))
    return x


# In[8]:


def prediction(x_test, theta):
    x_test = add_ones_column(x_test)
    z = np.dot(x_test, theta)
    y = (sigmoid(z) >= 0.5).astype(int)
    
    return y


# In[9]:


## --- Sigmoid Function --- ##
def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h


# In[10]:


def class_count(y):
    
    (unique, counts) = np.unique(y, return_counts = True)
    frequencies = np.asarray((unique, counts)).T
    
    if(frequencies[0, 0] == MEN):
        c0 = frequencies[0, 1]
        c1 = frequencies[1, 1]
    else:
        c1 = frequencies[0, 1]
        c0 = frequencies[1, 1]
        
    return c0, c1


# In[11]:


class LinearDiscriminatAnalysis():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x = add_ones_column(self.x)
        
        self.class_count = np.zeros((2, 1))
        self.no_feature = self.x.shape[1]
        
        self.eigen_pairs = []
        
        # 1. Mean vector with features with respect to each class
        self.mean = self.feature_mean(self.x, self.y)
        #print("Mean : ", self.mean)
        
        # 2. Within-class scatter matrix
        self.SW = self.within_class_scatter_matrix(MEN) + self.within_class_scatter_matrix(WOMEN)
        #print("withinclass_scatter_matrix ", self.SW)
        
        # 3. Between class scatter matrix
        self.SB = self.between_class_scatter_matrix(MEN) + self.between_class_scatter_matrix(WOMEN)
        #print("between_class_scatter_matrix ", self.SB)
        
        # 4. New feature
        self.weight_matrix = self.weight_matrix(self.SW, self.SB)
        #print("Weight Matrix : ", self.weight_matrix)
    
    # Mean vector
    def feature_mean(self, x, y):
    
        row = x.shape[0]
        col = x.shape[1]

        # Initiate mean
        mean = np.zeros((NO_OF_CLASS, col))

        # mean[0] = Mean of every feature for class 'M'
        # mean[1] = Mean of every feature for class 'W'
        # Same for variance

        self.class_count[MEN], self.class_count[WOMEN] = class_count(y)
        no_feature = col

        # Mean
        for i in range(0, row):
            for j in range(0, no_feature):

                if(y[i] == MEN):
                    mean[MEN, j] += x[i, j]
                else:
                    mean[WOMEN, j] += x[i, j]
        
        mean[MEN] = mean[MEN] / self.class_count[MEN]
        mean[WOMEN] = mean[WOMEN] / self.class_count[WOMEN]

        return mean
    
    # Within class scatter matrix
    def within_class_scatter_matrix(self, c):
        mean = self.mean[c].reshape(self.no_feature, 1)
        
        class_mean = np.zeros((self.no_feature, self.no_feature))
        for i in range(0, len(self.x)):
            if(self.y[i] == c):
                class_mean += (self.x[i] - mean) * (self.x[i] - mean).T
                #print("i : ", i, class_mean)

        sw = np.zeros((self.no_feature, self.no_feature))
        sw += class_mean
        return sw
        
    # Between class scatter matrix
    def between_class_scatter_matrix(self, c):
        mean = self.mean[c]
        total_mean = np.zeros((1, self.x.shape[1]))
        
        for i in range(0, len(self.x)):
            #print("xi : ", self.x[i])
            total_mean += self.x[i]
            #print("Mean : ", total_mean)
        
        total_mean = total_mean/len(self.x)
        sb = self.class_count[c] * (mean - total_mean) * (mean - total_mean).T
        return sb
    
    # Weight Matrix
    def weight_matrix(self, sw, sb):
        #print(np.linalg.inv(sw))
        #print(sb)
        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(sw) * sb)
        
        self.eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
        self.eigen_pairs = sorted(self.eigen_pairs, key=lambda x: x[0], reverse=True)
        
        #for pair in eigen_pairs:
            #print(pair[0])
            
        #sum_of_eigen_values = sum(eigen_values)
        #print('Explained Variance')
        #for i, pair in enumerate(eigen_pairs):
            #print('Eigenvector {}: {}'.format(i, (pair[0]/sum_of_eigen_values).real))
        
        w1 = self.eigen_pairs[0][1]
        #w2 = eigen_pairs[1][1]
        w_matrix = (w1.reshape(self.no_feature,1)).real
        
        return w_matrix


# In[12]:


def plot_1D(X, Y, theta):
    X_New = np.dot(add_ones_column(X), theta)
    
    # Scatter target ouput
    for i in range(0, len(X_New)):
        if(Y[i] == MEN):
            m = plt.scatter(X_New[i], Y[i], marker='o', color='green', s=20)
        else:
            w = plt.scatter(X_New[i], Y[i], marker='o', color='red', s=20)
            
    plt.legend((m, w), ('Man', 'Woman'),scatterpoints=1, loc='upper left',fontsize=8)
    # Plot decision boundary
    #print(theta)
    constant = theta[1]
    xx = X_New
    yy = -(theta[0]/(theta[1]+theta[2]+theta[3]))*xx + constant
    plt.plot(xx, yy, 'k-')
    
    plt.xlabel("X")
    plt.ylabel("h(X)")
    plt.show()


# In[13]:


def plot_data(X, Y, theta):
    
    #print("Original data")
    # create x,y
    xx, yy = np.meshgrid(X[:,1], X[:,2])
    
    # Original data
    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    
    plot2 = plt.figure(2)
    # Scatter target ouput
    i = 0
    for x in X:
        x1 = x[1]
        y1 = x[2]
        z1 = Y[i]
        if(z1 == MEN):
            m = plt3d.scatter(x1, y1, z1, marker='o', color='green', s=20)
        else:
            w = plt3d.scatter(x1, y1, z1, marker='o', color='red', s=20)      
        i = i + 1
    
    plt3d.set_title("Original data")
    plt3d.legend((m, w), ('Man', 'Woman'),scatterpoints=1, loc='upper left',fontsize=8)
    plt3d.set_xlabel('Height')
    plt3d.set_ylabel('Weight')
    plt3d.set_zlabel('Gender')
    plt.show()
    
    plot2 = plt.figure(2)
    X_New = np.dot(add_ones_column(X), theta)
    
    # Scatter target ouput
    for i in range(0, len(X_New)):
        if(Y[i] == MEN):
            m = plt.scatter(X_New[i], Y[i], marker='o', color='green', s=20)
        else:
            w = plt.scatter(X_New[i], Y[i], marker='o', color='red', s=20)
    
    plt.title("Generated data")
    plt.legend((m, w), ('Man', 'Woman'),scatterpoints=1, loc='upper left',fontsize=8)
    plt.xlabel("X")
    plt.ylabel("Gender")
    plt.show()


# In[14]:


def calculate_accuracy(y_test, y_pred):
    length = len(y_test)
    correct = y_test == y_pred
    accuracy = (np.sum(correct) / length)*100
    print("Accuracy : ", accuracy, "%")


# In[15]:


def main():
    
    print("Linear Discriminant Analysis")
    # Data
    X, Y, x_test, y_test = process_data()
    # Linear Discriminant Analysis
    # For train data
    lda1 = LinearDiscriminatAnalysis(X, Y)
    theta = lda1.weight_matrix
    #print(theta)
    
    # Prediciton  
    y_pred = prediction(x_test, theta)
    print("Prediciton")
    print(y_pred)
    
    # Calculate accuracy
    calculate_accuracy(y_test, y_pred)
    
    # Plot function
    plot_1D(X, Y, theta)
    
    # Plot original & generated data
    plot_data(X, Y, theta)
    
    # Compare with SKLEARN  
    #compare(X, Y, x_test)
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




