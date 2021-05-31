#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import csv


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


PATH = os.getcwd()
DATA_LABELED = "data_labeled.csv"
DATA_UNLABELED = "data_unlabeled.csv"
DATA_TEST = "data_test.csv"


# In[ ]:


def load_data(path):
    data_path = os.path.join(PATH , path)
    df = pd.read_csv(data_path, header = None)
    data = df.to_numpy()
    
    X = data[:,:-1]
    Y = data[:,-1:]
    
    return X, Y


# In[ ]:


def load_unlabeled_data():
    data_path = os.path.join(PATH , DATA_UNLABELED)
    df = pd.read_csv(data_path, header = None)
    data = df.to_numpy()

    return data


# In[ ]:


def Logistic_Regression(x, y):
    # Train Logistic Regression
    logisticRegr = LogisticRegression(solver='liblinear', max_iter = 100)
    model = logisticRegr.fit(x, y.ravel())
    return model


# In[ ]:


class Self_Training(object):
    def __init__(self, x, y,  x_test, y_test, unlabel_data):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.unlabel_data = unlabel_data
        self.K = 3
        
        self.logisticRegr = LogisticRegression(solver='liblinear', max_iter = 100)
        self.model = self.self_training(self.x, self.y, self.unlabel_data)
    
    def self_training(self, x, y, unlabel_data):
        while(len(unlabel_data) > 1):
            #print("\nTrain data : ", len(x))
            #print("Unlabeled data : ", len(unlabel_data))
            
            # Train Logistic Regression
            model = self.logisticRegr.fit(x, y.ravel())

            # Predict with unlabeled data
            pred_y = model.predict(unlabel_data)
            #print(pred_y)
            
            # Generate probabilities for each prediction
            prob = model.predict_proba(unlabel_data)
            # Confidence value
            confidence = prob.max(axis = 1)
            #print(confidence)

            # Stack & sort data from max to min
            data = np.column_stack((unlabel_data, pred_y))
            data = np.column_stack((data, confidence))
            #print(data)
            data = data[np.argsort(-data[:, data.shape[1] - 1])]
            #print(data)

            # Pick the first K rows into a vector
            commit_data = data[:self.K, :];

            # Check confidence value grater than 0.90
            #if any(c > 0.90 for c in commit_data[:,-1:]):
            #print("New added data with label & confidence value")
            #print(commit_data)
            data = np.delete(data, slice(0, self.K), axis = 0)
            #print(data, data.shape)
            # Drop columns
            data = np.delete(data, data.shape[1] - 1, 1)
            data = np.delete(data, data.shape[1] - 1, 1)
            #print(data, data.shape)
            commit_data = np.delete(commit_data, commit_data.shape[1] - 1, 1)
            #print(commit_data)

            # Add commit data to label data dataset
            commit_x = commit_data[:,:-1]
            commit_y = commit_data[:,-1:]
            #print(commit_x)
            #print(commit_y)
            x = np.vstack((x, commit_x))
            y = np.vstack((y, commit_y))
            unlabel_data = data

            #print(x.shape)
            #print(y.shape)
            #print(data.shape)
            #else:
                #break
                
        return model   


# In[ ]:


def compare(train_model, base_model, x_test, y_test):
    score1 = train_model.score(x_test, y_test)
    print("Semi-supervised model score : ", score1)
    
    score2 = base_model.score(x_test, y_test)
    print("Logistic Regresseion model score :" , score2)


# In[ ]:


def main():
    
    x, y = load_data(DATA_LABELED)
    x_test, y_test = load_data(DATA_TEST)
    unlabel_data = load_unlabeled_data()
    
    semiSupervised = Self_Training(x, y, x_test, y_test, unlabel_data)
    model1 = semiSupervised.model
    model2 = Logistic_Regression(x, y)
    
    compare(model1, model2, x_test, y_test)
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




