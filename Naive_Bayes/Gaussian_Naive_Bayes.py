#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
import os


# In[2]:


MEN = 0
WOMEN = 1
NO_OF_CLASS = 2
PI = 3.14


PATH = os.getcwd() + "\\"


def load_train_data():

    data_path = PATH + "train_data.csv"
    data = pd.read_csv(data_path)
    
    return data


# In[4]:


def load_test_data():
    
    test_data = np.array([
        [162, 53, 28],
        [168, 75, 32], 
        [175, 70, 30],
        [180, 85, 29]
    ])
    
    #print(test_data[0])
    
    return test_data


# In[5]:


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


# In[6]:


def mean_variance(x, y):
    
    row = x.shape[0]
    col = x.shape[1]
    
    # Initiate mean & variance vector
    
    mean = np.zeros((NO_OF_CLASS, col))
    variance = np.zeros((NO_OF_CLASS, col))
    
    # mean[0] = Mean of every feature for class 'M'
    # mean[1] = Mean of every feature for class 'W'
    # Same for variance

    c0, c1 = class_count(y)
    
    no_feature = col
    
    # Mean
    for i in range(0, row):
        for j in range(0, no_feature):
            
            if(y[i] == 'M'):
                mean[MEN, j] += x[i, j]
            else:
                mean[WOMEN, j] += x[i, j]
    
    mean[MEN] = mean[MEN] / c0
    mean[WOMEN] = mean[WOMEN] / c1
    
    # Variance
    for i in range(0, row):
        for j in range(0, no_feature):
            
            if(y[i] == 'M'):
                variance[MEN, j] += np.power((x[i, j] - mean[MEN, j]), 2)
            else:
                variance[WOMEN, j] += np.power((x[i, j] - mean[WOMEN, j]), 2)
                
    variance[MEN] = variance[MEN] / (c0 - 1)
    variance[WOMEN] = variance[WOMEN] / (c1 - 1)
    
    #print(mean)
    #print(variance)
    
    return mean, variance


# In[7]:


def prior_probability(y):
    
    # Prior probability for each class
    
    # Initiate prior probability
    # prior_prob[0] = For 'M'
    # prior_prob[1] = For 'W'
    
    prior_prob = np.zeros(NO_OF_CLASS)
    
    c0, c1 = class_count(y)
    
    prior_prob[MEN] = c0 / (c0 + c1)
    prior_prob[WOMEN] = c1 / (c0 + c1)
    
    #print(prior_prob)
    
    return prior_prob


# In[8]:


def posterior_probability(mean, variance, test_data):
    
    # Posterior Probability for each data feature given class
    # Row = No of class
    # Column = No of feature
    
    no_test_data = test_data.shape[0]
    no_feature = test_data.shape[1]
    
    # Initiate posterior probability
    
    post_prob = np.zeros((no_test_data, NO_OF_CLASS))
    
    # For each test data
    for t in range(0, no_test_data): # Loop 1

        temp = test_data[t]
        #print(temp)
        
        # For each class
        pp = np.zeros(NO_OF_CLASS)
        for i in range(0, NO_OF_CLASS): # Loop 2
            
            mul = 1
            
            # For each feature
            for j in range(0, no_feature): # Loop 3
                mul = mul * (1 / np.sqrt(2*PI * variance[i][j])) * np.exp(-0.5
                                  * pow(( temp[j] - mean[i][j] ), 2) / variance[i][j])
                #print(mul)
                
            pp[i] = mul # Loop 2
            
        post_prob[t] = pp # Loop 1
    
    #print(post_prob)
    
    return post_prob
    


# In[14]:


def conditional_probability(prior, post):
    
    # Initiate conditional probability
    cond_prob = np.ones(NO_OF_CLASS)
    
    total_prob = 0
    for i in range(0, 2):
        total_prob = total_prob + (prior[i] * post[i])
        
    for i in range(0, 2):
        cond_prob[i] = (prior[i] * post[i])/total_prob
    
    #print(cond_prob)
    
    return cond_prob
        


# In[15]:


def Gaussian_Naive_Bayes(x, y, test_data):
    
    ## Step 1: Mean & Veriance Calculation
    mean, variance = mean_variance(x, y)
    
    ### Step 2: Prior probability of the 2 classes
    prior_prob = prior_probability(y)
    
    ### Step 3: Posterior probability for each test data for each class
    post_prob = posterior_probability(mean, variance, test_data)
    
    #### Step 4: Conditional Probability & Prediciton
    for i in range(0, post_prob.shape[0]):
        
        print(test_data[i])
        
        cond_prob = conditional_probability(prior_prob, post_prob[i])
        prediction = int(cond_prob.argmax())
        
        if(prediction == 0):
            print("Expected Gender: M")
        else:
            print("Expected Gender: W")
        


# In[16]:


def main():

    # Load data
    df = load_train_data()
    
    # Feature vector
    x = df[['Height', 'Weight', 'Age']].to_numpy()
    
    # Output vector, the last column of the dataset
    y = df[['Gender']].to_numpy()
    
    # Load test data
    test_data = load_test_data()
    
    # Gaussian Naive Bayes Algorithm
    Gaussian_Naive_Bayes(x, y, test_data)

    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




